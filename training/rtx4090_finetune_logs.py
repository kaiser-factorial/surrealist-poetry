import os
import torch
import gc
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm

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


def train_model(model_name, checkpoint_dir, output_dir, logs_file, max_length, batch_size, epochs, lr):
    print("="*60)
    print(f"🚀 Starting training for {model_name}...")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Params: LR={lr}, Epochs={epochs}, BatchSize={batch_size}, MaxLen={max_length}")
    print("="*60)

    # --- LOAD MODEL & TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # RTX 4090 Optimization: Use bfloat16 natively
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, 
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

    # --- MEMORY CLEANUP ---
    # Crucial step for RTX 4090 so the VRAM clears before loading the next model
    del model
    del tokenizer
    del optimizer
    del dataloader
    del dataset
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    BASE_DIR = "/Users/corinakaiser/Desktop/VAT/brain-vat-training"
    LOGS_FILE = f"{BASE_DIR}/training/gpt2/bashforever_mixed_train_no_closers.txt"
    
    # ==============================
    # ⚙️ MAUK CONFIGURATION
    # ==============================
    MAUK_CONFIG = {
        "model_name": "MAUK",
        "checkpoint_dir": f"{BASE_DIR}/model_checkpoint_mauk_1",
        "output_dir": f"{BASE_DIR}/model_checkpoint_mauk_1_logs_finetuned",
        "logs_file": LOGS_FILE,
        "max_length": 512,
        "batch_size": 8,
        "epochs": 3,
        "lr": 5e-5
    }

    # ==============================
    # ⚙️ ABACI CONFIGURATION
    # ==============================
    ABACI_CONFIG = {
        "model_name": "ABACI",
        "checkpoint_dir": f"{BASE_DIR}/model_checkpoint_abaci_1",
        "output_dir": f"{BASE_DIR}/model_checkpoint_abaci_1_logs_finetuned",
        "logs_file": LOGS_FILE,
        "max_length": 512,
        "batch_size": 8,
        "epochs": 3,          
        "lr": 5e-5            
    }

    # Run sequential training
    train_model(**MAUK_CONFIG)
    train_model(**ABACI_CONFIG)
    
    print("🎉 All fine-tuning runs complete! Both models are updated and saved.")
