#!/usr/bin/env python3
"""
Upload Fine-Tuned MuRIL Sarcasm Model to Hugging Face Model Hub

Usage:
    python upload_to_huggingface.py --repo_id "your-username/muril-sarcasm-detector" --token "hf_xxx"
Or run interactively:
    python upload_to_huggingface.py
"""

import argparse
import getpass
import os
import sys
from huggingface_hub import HfApi, get_token, login


def main():
    parser = argparse.ArgumentParser(description="Upload trained MuRIL model to Hugging Face")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="Target Hugging Face repo ID, e.g. 'your-username/muril-sarcasm-detector'",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face User Access Token (with WRITE permission)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./trained_muril_sarcasm",
        help="Path to the directory containing model weights and tokenizer (default: ./trained_muril_sarcasm)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Set repository visibility to private (default: public)",
    )

    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.exists(model_dir):
        print(f"[!] Error: Model directory '{model_dir}' not found.")
        sys.exit(1)

    # 1. Retrieve or prompt for Token
    token = args.token or os.environ.get("HF_TOKEN") or get_token()

    if not token:
        print("=" * 65)
        print("  Hugging Face Authentication Required")
        print("  1. Create a free account at: https://huggingface.co")
        print("  2. Generate a token (with WRITE permission) at:")
        print("     https://huggingface.co/settings/tokens")
        print("=" * 65)
        token = getpass.getpass("Enter your Hugging Face write token (hidden): ").strip()
        if not token:
            print("[!] Token cannot be empty.")
            sys.exit(1)

    # Verify Token
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info["name"]
        print(f"\n[✓] Authenticated successfully as Hugging Face user: @{username}")
    except Exception as e:
        print(f"[!] Authentication failed: {e}")
        print("    Please check that your token has 'WRITE' permission.")
        sys.exit(1)

    # 2. Determine repo_id
    repo_id = args.repo_id
    if not repo_id:
        default_repo_name = f"{username}/muril-sarcasm-detector"
        entered = input(f"\nEnter repository name [{default_repo_name}]: ").strip()
        repo_id = entered if entered else default_repo_name

    # Ensure format 'username/model-name'
    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"

    # 3. Create repo if it doesn't exist
    print(f"\n[*] Ensuring repository exists: https://huggingface.co/{repo_id}...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"[✓] Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[!] Failed to create repository: {e}")
        sys.exit(1)

    # 4. Upload model folder
    print(f"\n[*] Uploading model files from '{model_dir}'...")
    print("    (This will upload model.safetensors, config.json, tokenizer assets, and README)")
    try:
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["checkpoint-*", "*.log", "__pycache__/*"],
        )
        print("\n" + "=" * 65)
        print("  🎉 MODEL SUCCESSFULLY DEPLOYED TO HUGGING FACE!")
        print("=" * 65)
        print(f"Model URL : https://huggingface.co/{repo_id}")
        print("\nAnyone can now load and use your model directly in Python:")
        print("-" * 65)
        print(f"""from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")
""")
        print("-" * 65)
    except Exception as e:
        print(f"[!] Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
