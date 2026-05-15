import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load your HF_TOKEN from .env
load_dotenv()
token = os.getenv("HF_TOKEN")

def process_and_push(base_repo, adapter_path, target_repo):
    print(f"\n--- Processing {target_repo} ---")
    
    # 1. Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_repo)
    
    # 2. Resize and Tie (Critical for GPT-2)
    print("📏 Synchronizing embeddings...")
    base_model.resize_token_embeddings(len(tokenizer))
    if getattr(base_model.config, "tie_word_embeddings", False):
        base_model.tie_weights()
    
    # 3. Attach and Merge
    print(f"📎 Attaching adapter from: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("🤝 Merging weights...")
    merged_model = model.merge_and_unload()
    
    # 4. Save locally first (more robust for large uploads)
    temp_save_path = f"./temp_{target_repo.split('/')[-1]}"
    print(f"💾 Saving merged model locally to {temp_save_path}...")
    merged_model.save_pretrained(temp_save_path)
    tokenizer.save_pretrained(temp_save_path)
    
    # 5. Push Folder to Hub
    print(f"🚀 Pushing FOLDER to https://huggingface.co/{target_repo}...")
    api = HfApi()
    api.upload_folder(
        folder_path=temp_save_path,
        repo_id=target_repo,
        repo_type="model",
        token=token
    )
    print(f"✅ Success! {target_repo} is live.")

if __name__ == "__main__":
    if not token:
        print("❌ Error: HF_TOKEN not found in .env.")
    else:
        # ABACI ONLY (since MAUK was already pushed)
        if os.path.exists("./lora_abaci"):
            process_and_push("brick-factorial/abaci_v1", "./lora_abaci", "brick-factorial/abaci_v2.1")
        else:
            print("⚠️ Folder './lora_abaci' not found.")
