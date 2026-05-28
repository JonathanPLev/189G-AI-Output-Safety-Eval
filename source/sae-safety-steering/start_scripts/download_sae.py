"""
start_scripts/download_sae.py

Downloads the Goodfire Llama-3.1-8B-Instruct SAE weights from HuggingFace
and saves them to models/sae/.

Usage:
    python start_scripts/download_sae.py
"""

import os
from huggingface_hub import hf_hub_download

SAE_REPO   = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
SAE_FILE   = "Llama-3.1-8B-Instruct-SAE-l19.pth"
LOCAL_DIR  = "models/sae"


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    dest = os.path.join(LOCAL_DIR, SAE_FILE)

    if os.path.exists(dest):
        print(f"SAE weights already present at {dest} — skipping download.")
        return

    print(f"Downloading {SAE_FILE} from {SAE_REPO} …")
    path = hf_hub_download(
        repo_id=SAE_REPO,
        filename=SAE_FILE,
        repo_type="model",
        local_dir=LOCAL_DIR,
    )
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
