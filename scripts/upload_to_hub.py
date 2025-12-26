#!/usr/bin/env python3
"""
Upload cloud-sec-architect-8b to Hugging Face Hub.

This script handles:
1. Loading the trained LoRA adapter
2. Optionally merging with base model
3. Uploading to Hugging Face Hub with proper model card

Usage:
    python scripts/upload_to_hub.py --repo-id your-username/cloud-sec-architect-8b
    python scripts/upload_to_hub.py --repo-id your-username/cloud-sec-architect-8b --merge
    python scripts/upload_to_hub.py --repo-id your-username/cloud-sec-architect-8b --private
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import create_repo, upload_folder
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default paths
DEFAULT_ADAPTER_PATH = "./cloud-sec-architect-8b"
DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_CARD_PATH = "./MODEL_CARD.md"


def merge_and_save(base_model_id: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter with base model and save."""
    print(f"Loading base model: {base_model_id}")

    # Load base model in full precision for merging
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)  # nosec B615
    tokenizer.save_pretrained(output_path)

    print("Merge complete!")
    return output_path


def upload_adapter(
    adapter_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload cloud-sec-architect-8b LoRA adapter",
):
    """Upload LoRA adapter to Hugging Face Hub."""
    print(f"Creating/accessing repository: {repo_id}")
    create_repo(repo_id, private=private, exist_ok=True, repo_type="model")

    # Prepare upload directory with model card
    upload_dir = Path(adapter_path)

    # Copy model card to adapter directory if it exists
    model_card_dest = upload_dir / "README.md"
    if os.path.exists(MODEL_CARD_PATH) and not model_card_dest.exists():
        print(f"Copying model card from {MODEL_CARD_PATH}")
        shutil.copy(MODEL_CARD_PATH, model_card_dest)

    print(f"Uploading to: https://huggingface.co/{repo_id}")
    upload_folder(
        folder_path=str(upload_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    print("\nUpload complete!")
    print(f"View your model at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload cloud-sec-architect-8b to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Upload LoRA adapter only (recommended for saving space)
    python scripts/upload_to_hub.py --repo-id myuser/cloud-sec-architect-8b

    # Upload merged full model (larger, but easier to use)
    python scripts/upload_to_hub.py --repo-id myuser/cloud-sec-architect-8b-merged --merge

    # Upload as private model
    python scripts/upload_to_hub.py --repo-id myuser/cloud-sec-architect-8b --private
        """,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face Hub repository ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help=f"Path to LoRA adapter (default: {DEFAULT_ADAPTER_PATH})",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model ID for merging (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA weights with base model before uploading",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--merged-output",
        type=str,
        default="./cloud-sec-architect-8b-merged",
        help="Output path for merged model (default: ./cloud-sec-architect-8b-merged)",
    )

    args = parser.parse_args()

    # Check adapter exists
    if not os.path.exists(args.adapter_path):
        print(f"Error: Adapter not found at {args.adapter_path}")
        print("Train the model first with: python train.py")
        return 1

    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("Warning: No HF_TOKEN found in environment.")
        print("Make sure you're logged in with: huggingface-cli login")

    if args.merge:
        # Merge and upload full model
        print("=" * 50)
        print("Merging LoRA adapter with base model...")
        print("=" * 50)
        merge_and_save(args.base_model, args.adapter_path, args.merged_output)

        print("\n" + "=" * 50)
        print("Uploading merged model...")
        print("=" * 50)
        upload_adapter(
            args.merged_output,
            args.repo_id,
            args.private,
            "Upload cloud-sec-architect-8b merged model",
        )
    else:
        # Upload adapter only
        print("=" * 50)
        print("Uploading LoRA adapter...")
        print("=" * 50)
        upload_adapter(args.adapter_path, args.repo_id, args.private)

    return 0


if __name__ == "__main__":
    exit(main())
