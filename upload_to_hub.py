import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

load_dotenv()
token = os.getenv("HF_TOKEN")

UPLOADS = [
    ("./temp_abaci_v2.1", "brick-factorial/abaci_v2.1"),
    # ("./temp_mauk_v2.1", "brick-factorial/mauk_v2.1"),  # uncomment when needed
]

api = HfApi()

for folder_path, repo_id in UPLOADS:
    if not os.path.exists(folder_path):
        print(f"⚠️  {folder_path} not found, skipping.")
        continue
    print(f"🚀 Uploading {folder_path} → {repo_id}...")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"✅ {repo_id} is live.")
