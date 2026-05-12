import os
import torch
import gc
import re
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from tqdm import tqdm

# --- ENVIRONMENT & AUTH ---
# For Kaggle: ensure you've added HF_TOKEN to your secrets or environment
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print("🔑 Logging into Hugging Face Hub...")
    login(token=hf_token)
else:
    print("⚠️ No HF_TOKEN found! Private repos will fail.")

# --- DATASET PREPARATION ---
class IdentityDataset(Dataset):
    def __init__(self, file_path, tokenizer, target_bot, max_length=512, name_weight=50.0, content_weight=5.0, context_weight=1.0):
        self.tokenizer = tokenizer
        self.target_bot = f"[{target_bot}]"
        self.max_length = max_length
        self.name_weight = name_weight
        self.content_weight = content_weight
        self.context_weight = context_weight
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split by double newlines into conversation chunks
        raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        
        self.samples = []
        for chunk in raw_chunks:
            # Normalize: remove colons after tags for consistency [MAUK]: -> [MAUK]
            chunk = re.sub(r'\[(MAUK|ABACI)\]:', r'[\1]', chunk)
            
            # Split the chunk into individual turns to apply weighting
            # Pattern matches [NAME] followed by everything until the next [NAME]
            turns = re.split(r'(\[(?:MAUK|ABACI)\])', chunk)
            # Result of split will be ['', '[NAME]', 'content', '[NAME]', 'content', ...]
            
            full_ids = []
            full_weights = []
            
            for i in range(1, len(turns), 2):
                name_tag = turns[i]
                content = turns[i+1] if (i+1) < len(turns) else ""
                
                # Encode name tag
                name_ids = tokenizer.encode(name_tag, add_special_tokens=False)
                # Encode content + EOS
                content_ids = tokenizer.encode(content + tokenizer.eos_token, add_special_tokens=False)
                
                # Determine weights
                if name_tag == self.target_bot:
                    n_w = [self.name_weight] * len(name_ids)
                    c_w = [self.content_weight] * len(content_ids)
                else:
                    n_w = [self.context_weight] * len(name_ids)
                    c_w = [self.context_weight] * len(content_ids)
                
                full_ids.extend(name_ids)
                full_weights.extend(n_w)
                full_ids.extend(content_ids)
                full_weights.extend(c_w)
            
            # Truncate/Pad
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
                full_weights = full_weights[:max_length]
            
            padding_len = max_length - len(full_ids)
            input_ids = full_ids + [tokenizer.pad_token_id] * padding_len
            # Weight 0.0 for padding so it doesn't affect loss
            loss_weights = full_weights + [0.0] * padding_len
            
            # Attention mask: 1 for real tokens, 0 for padding
            attn_mask = [1] * len(full_ids) + [0] * padding_len
            
            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
                "loss_weights": torch.tensor(loss_weights, dtype=torch.float)
            })

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def train_model(bot_name, hf_repo_id, output_dir, data_file, epochs=10, batch_size=4, lr=5e-5, max_length=512):
    print(f"\n🚀 Training {bot_name} LoRA (Repo: {hf_repo_id})")
    
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add identity tags as special tokens
    special_tokens = ["[MAUK]", "[ABACI]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # 2. Model
    # Using float16 for T4 compatibility
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo_id,
        torch_dtype=torch.float16,
        device_map="auto" # This handles dual T4 distribution automatically
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # 3. LoRA Configuration
    # We must save embeddings/head because we added special tokens
    modules_to_save = ["wte", "lm_head"] if "gpt2" in model.config.model_type else ["embed_tokens", "lm_head"]
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        modules_to_save=modules_to_save
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Data
    dataset = IdentityDataset(data_file, tokenizer, bot_name, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 5. Optimizer & Loss
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    scaler = torch.cuda.amp.GradScaler() # For FP16 stability
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            loss_weights = batch["loss_weights"].to("cuda")
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Shift for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_weights = loss_weights[..., 1:].contiguous()
                
                # Custom Weighted Loss
                raw_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                weighted_loss = raw_loss * shift_weights.view(-1)
                
                # Mean only over non-padding tokens
                valid_mask = (shift_labels.view(-1) != tokenizer.pad_token_id)
                loss = weighted_loss[valid_mask].mean() if valid_mask.any() else weighted_loss.mean()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Avg Epoch Loss: {epoch_loss/len(dataloader):.4f}")
        
    # 6. Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ {bot_name} LoRA saved to {output_dir}")
    
    # Cleanup for next bot
    del model, tokenizer, optimizer, dataloader, dataset
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Update this path to where you upload your synthetic_convos.md on Kaggle
    DATA_PATH = "/kaggle/input/vat-synthetic-data/synthetic_convos.md"
    
    # MAUK Training
    train_model(
        bot_name="MAUK",
        hf_repo_id="brick-factorial/mauk_v1",
        output_dir="./lora_mauk_identity",
        data_file=DATA_PATH,
        epochs=15,
        batch_size=4
    )
    
    # ABACI Training
    train_model(
        bot_name="ABACI",
        hf_repo_id="brick-factorial/abaci_v1",
        output_dir="./lora_abaci_identity",
        data_file=DATA_PATH,
        epochs=15,
        batch_size=4
    )
