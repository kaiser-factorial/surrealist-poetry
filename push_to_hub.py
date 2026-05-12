import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

# Load your HF_TOKEN from .env
load_dotenv()
token = os.getenv("HF_TOKEN")

def process_and_push(base_repo, adapter_path, target_repo):
    print(f"\n--- Processing {target_repo} ---")
    
    # 1. Load tokenizer (has your new [MAUK]/[ABACI] tokens)
    print(f"🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # 2. Load the base model
    print(f"📦 Loading base model: {base_repo}...")
    base_model = AutoModelForCausalLM.from_pretrained(base_repo)
    
    # 🚨 RESIZE & TIE (Crucial for GPT-2 consistency)
    print("📏 Synchronizing embeddings...")
    base_model.resize_token_embeddings(len(tokenizer))
    if getattr(base_model.config, "tie_word_embeddings", False):
        base_model.tie_weights()
    
    # 3. Attach the adapter
    print(f"📎 Attaching adapter from: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # 4. Merge
    print("🤝 Merging weights into a single V2 model...")
    merged_model = model.merge_and_unload()
    
    # 5. Push to Hub
    print(f"🚀 Pushing to https://huggingface.co/{target_repo}...")
    # This will automatically create the repo if it doesn't exist
    merged_model.push_to_hub(target_repo, token=token)
    tokenizer.push_to_hub(target_repo, token=token)
    
    print(f"✅ Success! {target_repo} is ready for action.")

if __name__ == "__main__":
    if not token:
        print("❌ Error: HF_TOKEN not found in .env. Create one at https://huggingface.co/settings/tokens")
    else:
        # Run for MAUK
#        if os.path.exists("./lora_mauk"):
#            process_and_push("brick-factorial/mauk_v1", "./lora_mauk", "brick-factorial/mauk_v2.1")
#        else:
#            print("⚠️ Folder './lora_mauk' not found. Skipping MAUK.")
            
        # Run for ABACI
        if os.path.exists("./lora_abaci"):
            process_and_push("brick-factorial/abaci_v1", "./lora_abaci", "brick-factorial/abaci_v2.1")
        else:
            print("⚠️ Folder './lora_abaci' not found. Skipping ABACI.")
