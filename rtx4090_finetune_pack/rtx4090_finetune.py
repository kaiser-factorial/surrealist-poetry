import os
import torch
import gc
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
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
            enc = tokenizer(
                entry, 
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


def train_model(model_name, hf_repo_id, output_dir, logs_file, max_length, batch_size, epochs, lr):
    print("\n" + "="*60)
    print(f"⚠️ Training {model_name}, HF: {hf_repo_id}...")
   # print(f"Saving to: {output_dir}")
    print(f"Params: LR={lr}, Epochs={epochs}, BatchSize={batch_size}, MaxLen={max_length}")
    print("="*60)

    # --- LOAD MODEL & TOKENIZER FROM HUGGING FACE ---
    print(f"Downloading/Loading tokenizer for {hf_repo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    print(f"Downloading/Loading model weights for {hf_repo_id}...")
    # RTX 4090 Optimization: Use bfloat16 natively
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo_id, 
        torch_dtype=torch.bfloat16
    ).to("cuda")

    print("Preparing dataset...")
    dataset = LogsDataset(logs_file, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    print(f"Training on {len(dataset)} distinct dialogue entries...")
    model.train()

    # --- TRAINING LOOP ---
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
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"[{model_name}] Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")

    # --- SAVE ---
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ {model_name} successfully saved to {output_dir}\n")

    # --- QUICK INFERENCE TEST ---
    print(f"--- Quick Inference Test for {model_name} ---")
    model.eval()
    
    test_prompts = [
        f"[ABACI] Tell me about it. \n[{model_name}]",
        f"[MAUK] Tell me about it. \n[{model_name}]",
        f"[OTHER] why are you like this\n[{model_name}]",
        f"[OTHER] wait, who are you?\n[{model_name}]\n<think-in>",
        f"<think-in>why are you like this<think-",
        f"[{model_name}]",
        f"[{model_name}]",
        f"[{model_name}]"


    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs_gen = model.generate(
                **inputs,
                max_new_tokens=70,
                do_sample=True,
                temperature=0.95,
                top_p=0.95,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"PROMPT: {repr(prompt)}")
        print(f"OUTPUT:\n{tokenizer.decode(outputs_gen[0], skip_special_tokens=False)}\n" + "-"*40)
    print("\n")

    # --- MEMORY CLEANUP ---
    # Crucial step for RTX 4090 so the VRAM clears before loading the next model
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
    # Point to the local self-contained dataset we copied into this pack
    LOGS_FILE = "./data/bashforever_mixed_train_no_closers.txt"
    
    # ==============================
    # ⚙️ MAUK CONFIGURATION
    # ==============================
    MAUK_CONFIG = {
        "model_name": "MAUK",
        "hf_repo_id": "brick-factorial/mauk_v1",
        "output_dir": "./output_mauk_v1_finetuned",
        "logs_file": LOGS_FILE,
        "max_length": 256,
        "batch_size": 16,
        "epochs": 8,          
        "lr": 5e-5   
    }

    # ==============================
    # ⚙️ ABACI CONFIGURATION
    # ==============================
    ABACI_CONFIG = {
        "model_name": "ABACI",
        "hf_repo_id": "brick-factorial/abaci_v1",
        "output_dir": "./output_abaci_v1_finetuned",
        "logs_file": LOGS_FILE,
        "max_length": 256,
        "batch_size": 8,
        "epochs": 4,          
        "lr": 3e-5            
    }

    # Run sequential training
    train_model(**MAUK_CONFIG)
    train_model(**ABACI_CONFIG)
    
    print("🎉 All fine-tuning runs complete! Both models are updated and saved locally.")
