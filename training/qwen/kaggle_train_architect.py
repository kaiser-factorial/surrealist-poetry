import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# TPU v5e / v3-8 Efficiency Optimization
os.environ['XLA_USE_BF16'] = '1'
os.environ['PJRT_DEVICE'] = 'TPU'

# --- UPDATE THESE PATHS IN KAGGLE ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# Make sure you upload 'prepared_chatml_dataset.jsonl' as a Kaggle dataset
DATASET_PATH = "/kaggle/input/your-dataset-name/prepared_chatml_dataset.jsonl" 
OUTPUT_DIR = "/kaggle/working/qwen2.5-architect-v5e"

def train_fn(index, flags):
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # 1. Initialize Device
    device = xm.xla_device()
    
    # 2. Load Tokenizer & Add Thinking Tags
    xm.master_print(f"Initializing tokenizer for {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    special_tokens = {
        "additional_special_tokens": ["<think-out>", "</think-out>", "<think-in>", "</think-in>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Load Model & Resize for new tokens
    xm.master_print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # Resize embeddings to match new tokenizer size (crucial for <think> tags)
    model.resize_token_embeddings(len(tokenizer))
    
    # 4. Load ChatML Dataset
    if not os.path.exists(DATASET_PATH):
        xm.master_print(f"CRITICAL ERROR: Dataset not found at {DATASET_PATH}")
        return

    # SFTTrainer will automatically use the 'messages' column if it exists
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # 5. Training Arguments (TPU Optimized with SFTConfig)
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        learning_rate=1e-5, 
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="no", # Save only at the end to minimize overhead
        bf16=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        max_seq_length=1024,
        packing=False # Better for ChatML coherence
    )
    
    # 6. Trainer Initialization
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=args
    )
    
    # 7. Execute Training
    xm.master_print("Commencing Architect Fine-Tuning...")
    trainer.train()
    
    # 8. Save Final Model (Master process only)
    xm.master_print("Training Complete. Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    xm.master_print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    # Launch on all 8 cores
    flags = {}
    xmp.spawn(train_fn, args=(flags,), nprocs=8, start_method='fork')
