import os
import torch
import gc
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from tqdm import tqdm

# --- ENVIRONMENT & AUTH ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print("🔑 Logging into Hugging Face Hub...")
    login(token=hf_token)
else:
    print("⚠️ No HF_TOKEN found in .env! If the repos are private, this will fail.")

# --- DATASET PREPARATION ---
class LogsDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split by the double newlines separating the conversational entries
        raw_entries = [e.strip() for e in text.split("\n\n") if e.strip()]
        
        self.encodings = []
        for entry in raw_entries:
            # Append EOS token so the model learns the stopping condition
            entry_with_eos = entry + tokenizer.eos_token
            
            enc = tokenizer(
                entry_with_eos, 
                truncation=True, 
                max_length=max_length, 
                padding="max_length",
                return_tensors="pt"
            )
            self.encodings.append({
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0]
            })
            
    def __len__(self):
        return len(self.encodings)
        
    def __getitem__(self, idx):
        return self.encodings[idx]


def train_model(model_name, hf_repo_id, output_dir, logs_file, max_length, batch_size, epochs, lr, lora_r, lora_alpha, tag_weight=25.0, think_space_weight=5.0, speak_space_weight=2.0):
    print("\n" + "="*60)
    print(f"⚠️ Training {model_name} with LoRA, HF: {hf_repo_id}...")
    print(f"Params: LR={lr}, Epochs={epochs}, BatchSize={batch_size}, MaxLen={max_length}, LoRA R={lora_r}")
    print("="*60)

    # --- LOAD MODEL & TOKENIZER FROM HUGGING FACE ---
    print(f"Downloading/Loading tokenizer for {hf_repo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # --- ADD SPECIAL TOKENS ---
    special_tags = ["<think>", "</think>"]
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tags})
    if num_added > 0:
        print(f"Added {num_added} special tokens to tokenizer: {special_tags}")

    print(f"Downloading/Loading base model weights for {hf_repo_id}...")
    # RTX 4090 Optimization: Use bfloat16 natively
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo_id, 
        torch_dtype=torch.bfloat16
    ).to("cuda")

    if num_added > 0:
        print("Resizing model embeddings for new special tokens...")
        model.resize_token_embeddings(len(tokenizer))

    # --- APPLY LoRA (PEFT) ---
    print(f"Applying LoRA configuration...")
    
    # We must explicitly train the embedding layer and LM head because we added new tokens!
    if model.config.model_type == "gpt2":
        modules_to_save = ["wte", "lm_head"]
    elif "llama" in model.config.model_type or "mistral" in model.config.model_type:
        modules_to_save = ["embed_tokens", "lm_head"]
    else:
        modules_to_save = None
        print("⚠️ Warning: Unknown model architecture. Using PEFT defaults, which might not train embeddings!")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        modules_to_save=modules_to_save
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- SETUP CUSTOM LOSS WEIGHTS ---
    vocab_size = len(tokenizer)
    
    # Get IDs for our special tags to dynamically build position masks later
    tag_ids = tokenizer.convert_tokens_to_ids(special_tags)
    think_open_id = tag_ids[0]
    think_close_id = tag_ids[1]
        
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    print("Preparing dataset...")
    dataset = LogsDataset(logs_file, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Note: Only the LoRA adapter weights (and embeddings) will receive gradients!
    optimizer = AdamW(model.parameters(), lr=lr)

    print(f"Training on {len(dataset)} distinct dialogue entries...")
    model.train()

    # --- TRAINING LOOP ---
    avg_loss_display = "N/A (0 epochs)"
    for epoch in range(epochs):
        epoch_loss = 0
        progress = tqdm(dataloader, desc=f"[{model_name}] Epoch {epoch+1}/{epochs}")
        
        for batch in progress:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            
            # Set labels to -100 for padding so it doesn't skew loss
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # Use automatic mixed precision (bfloat16)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Pass input_ids and attention_mask, but NOT labels so we can compute custom loss
                outputs = model(input_ids, attention_mask=attention_mask)
                
                # Shift logits and labels for causal LM next-token prediction
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # --- DYNAMIC THOUGHT-SPACE WEIGHTING ---
                # Find positions of tags
                open_mask = (input_ids == think_open_id).long()
                close_mask = (input_ids == think_close_id).long()
                
                # Using cumsum to figure out what is "inside" the tags
                open_cumsum = open_mask.cumsum(dim=1)
                close_cumsum = close_mask.cumsum(dim=1)
                inside_mask = (open_cumsum - close_cumsum) > 0
                
                # Base weights (outside / speak space)
                position_weights = torch.full_like(input_ids, speak_space_weight, dtype=torch.float)
                # Apply inside weights (think space)
                position_weights[inside_mask] = think_space_weight
                # Overwrite tag weights (make sure the tags themselves get the massive penalty)
                position_weights[open_mask == 1] = tag_weight
                position_weights[close_mask == 1] = tag_weight
                
                # Shift position weights to match shift_labels
                shift_weights = position_weights[..., 1:].contiguous()
                
                # Compute custom weighted loss
                raw_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                weighted_loss = raw_token_loss * shift_weights.view(-1)
                
                # Mean over valid non-padded tokens
                valid_tokens = (shift_labels.view(-1) != -100).sum()
                loss = weighted_loss.sum() / valid_tokens
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        avg_loss_display = f"{avg_loss:.4f}"
        print(f"[{model_name}] Epoch {epoch+1} Complete | Average Loss: {avg_loss_display}")

    # --- SAVE ---
    os.makedirs(output_dir, exist_ok=True)
    # For a PEFT model, save_pretrained ONLY saves the small LoRA adapter weights, not the whole 7B+ model!
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ {model_name} LoRA adapter successfully saved to {output_dir}\n")

    # --- QUICK INFERENCE TEST ---
    inference_params = {
        "max_new_tokens": 70,
        "do_sample": True,
        "temperature": 0.95,
        "top_p": 0.95,
        "repetition_penalty": 1.3,
        "pad_token_id": tokenizer.eos_token_id
    }

    print("\n" + "="*60)
    print(f"🧪 Quick Inference Test for: {model_name} (LoRA: r={lora_r}, alpha={lora_alpha})")
    print(f"   Training Params: LR={lr}, Epochs={epochs}, BatchSize={batch_size}, MaxLen={max_length}")
    print(f"   Space Weights  : Tag={tag_weight}x, Think={think_space_weight}x, Speak={speak_space_weight}x")
    print(f"   Final Epoch Loss: {avg_loss_display}")
    print(f"   Inference Params: {inference_params}")
    print("="*60 + "\n")
    
    model.eval()
    
    test_prompts = [
        # Basic cold start
        f"[{model_name}]",
        
        # Mundane small talk
        f"[OTHER] what is over there?\n[{model_name}]",
        
        # Random life complaint
        f"[OTHER] i can no longer wait.\n[{model_name}]<think>",
        
        # Basic tech/help question (forcing a thought)
        f"[OTHER] i have to go now.\n[{model_name}]\n<think>",
        
        # Simple greetings from the other persona
        f"[ABACI] morning everyone.\n[",
        f"[MAUK] hey, are you around?\n[",
        f"[ABACI] another fine morning\n[{model_name}]",
        f"[MAUK] greetings\n[{model_name}]",
        
        # Testing a partial thought on a normal topic
        f"[OTHER] what are you up to tonight?\n[{model_name}]\n<think>i need",
        f"[OTHER] what are you you looking at?\n[{model_name}]\n<think>"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs_gen = model.generate(
                **inputs,
                **inference_params
            )
        print(f"PROMPT: {repr(prompt)}")
        print(f"OUTPUT:\n{tokenizer.decode(outputs_gen[0], skip_special_tokens=False)}\n" + "-"*40)
    print("\n")

    # --- MEMORY CLEANUP ---
    try:
        del input_ids, attention_mask, labels, outputs, loss, batch, progress
    except NameError:
        pass
    del model
    del tokenizer
    del optimizer
    del dataloader
    del dataset
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Point to the local self-contained dataset
    LOGS_FILE = "./data/bashforever_mixed_train_think_only.txt"
    
    # ==============================
    # ⚙️ MAUK LoRA CONFIGURATION
    # ==============================
    MAUK_CONFIG = {
        "model_name": "MAUK",
        "hf_repo_id": "brick-factorial/mauk_v1",
        "output_dir": "./output_mauk_lora",
        "logs_file": LOGS_FILE,
        "max_length": 600,
        "batch_size": 8,
        "epochs": 10,          
        "lr": 1e-4, # LoRA typically uses higher learning rates than full finetuning
        "lora_r": 32,
        "lora_alpha": 32
    }

    # ==============================
    # ⚙️ ABACI LoRA CONFIGURATION
    # ==============================
    ABACI_CONFIG = {
        "model_name": "ABACI",
        "hf_repo_id": "brick-factorial/abaci_v1",
        "output_dir": "./output_abaci_lora",
        "logs_file": LOGS_FILE,
        "max_length": 600,
        "batch_size": 8,
        "epochs": 12,          
        "lr": 2e-4, 
        "lora_r": 32,
        "lora_alpha": 32
    }

    # Run sequential training
    train_model(**MAUK_CONFIG)
    train_model(**ABACI_CONFIG)
    
    print("🎉 All LoRA runs complete! The small adapters are saved locally.")
