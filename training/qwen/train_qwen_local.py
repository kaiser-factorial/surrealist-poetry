import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import os

# Configuration
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "training/qwen/prepared_chatml_dataset.jsonl" # Use the ChatML version
OUTPUT_DIR = "./qwen-brain-vat-checkpoint"

# Custom Tokens (only the thinking tags, roles are handled by ChatML)
SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "<think-out>", "</think-out>", 
        "<think-in>", "</think-in>"
    ]
}

def train():
    # 1. Load Tokenizer
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model
    print(f"Loading model: {MODEL_ID}")
    
    # Check for Apple Silicon (MPS)
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS) for training.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32, # MPS prefers float32 for stability
        trust_remote_code=True
    ).to(device)
    
    model.resize_token_embeddings(len(tokenizer))

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"]
    )

    # 4. Load Dataset
    print(f"Loading dataset from: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # Smaller for local memory
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        save_total_limit=2,
        use_mps_device=True if device == "mps" else False,
        report_to="none",
        remove_unused_columns=False
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        max_seq_length=1024,
    )

    print("Starting training...")
    trainer.train()
    
    # Save the final model and tokenizer
    print(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()
