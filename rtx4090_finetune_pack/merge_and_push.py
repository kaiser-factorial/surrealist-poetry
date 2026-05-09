import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
from dotenv import load_dotenv

def merge_and_push(base_model_id, adapter_dir, new_repo_id):
    print(f"\n" + "="*60)
    print(f"🚀 Merging {adapter_dir} into {base_model_id}")
    print(f"📦 Target Repo: {new_repo_id}")
    print("="*60)

    print("1. Loading tokenizer from adapter dir (to get the new tokens)...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    print("2. Loading base model...")
    # Using device_map="cpu" for merging is often safer for VRAM limits, 
    # but auto works if you have the 4090 free.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # IMPORTANT: Resize the base model first so the new custom tokens fit!
    print("3. Resizing token embeddings...")
    base_model.resize_token_embeddings(len(tokenizer))

    print("4. Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("5. Permanently fusing weights (merge_and_unload)...")
    merged_model = model.merge_and_unload()

    print("6. Pushing to Hugging Face Hub...")
    merged_model.push_to_hub(new_repo_id)
    tokenizer.push_to_hub(new_repo_id)
    print(f"🎉 Successfully uploaded merged model to {new_repo_id}!\n")

    # Clear memory just in case we are doing both sequentially
    del merged_model
    del model
    del base_model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️ No HF_TOKEN found in .env! Push will fail.")
        exit(1)
    else:
        login(token=hf_token)

    print("Uncomment the blocks below to run the merge and push!")

    # ==============================
    # ⚙️ MAUK MERGE
    # ==============================
    # merge_and_push(
    #     base_model_id="brick-factorial/mauk_v1",
    #     adapter_dir="./output_mauk_lora",
    #     new_repo_id="brick-factorial/mauk_v2_think_tags" # Make sure to change this!
    # )

    # ==============================
    # ⚙️ ABACI MERGE
    # ==============================
    # merge_and_push(
    #     base_model_id="brick-factorial/abaci_v1",
    #     adapter_dir="./output_abaci_lora",
    #     new_repo_id="brick-factorial/abaci_v2_think_tags" # Make sure to change this!
    # )
