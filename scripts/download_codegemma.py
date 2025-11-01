#!/usr/bin/env python3
"""
Download Code Gemma 7B model from HuggingFace for Kaggle upload.

This script downloads the Code Gemma 7B Instruct model and saves it locally.
The model can then be uploaded to Kaggle as a dataset for offline inference.

Requirements:
    - HuggingFace transformers library
    - HuggingFace account with access to google/codegemma-7b-it
    - HF_TOKEN environment variable set
    - ~14GB disk space (model is ~7GB, but download needs extra space)
    - Internet connection (for download only)

Setup:
    1. Request access at https://huggingface.co/google/codegemma-7b-it
    2. Get token from https://huggingface.co/settings/tokens
    3. export HF_TOKEN=your_token_here

Usage:
    export HF_TOKEN=your_token_here
    uv run python scripts/download_codegemma.py [--output-dir DIR]
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers library not installed")
    print("Install with: pip install transformers torch")
    sys.exit(1)


def download_model(output_dir: Path, model_name: str = "google/codegemma-7b-it"):
    """
    Download Code Gemma model and tokenizer.

    Args:
        output_dir: Directory to save model files
        model_name: HuggingFace model identifier
    """
    print("=" * 60)
    print("Code Gemma 7B Download Script")
    print("=" * 60)

    # Check for HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\n⚠️  WARNING: HF_TOKEN environment variable not set")
        print("Code Gemma is a gated model requiring authentication.")
        print("\nTo set up authentication:")
        print("1. Go to https://huggingface.co/google/codegemma-7b-it")
        print("2. Click 'Request access' and wait for approval")
        print("3. Get token from https://huggingface.co/settings/tokens")
        print("4. Set environment: export HF_TOKEN=your_token_here")
        print("\nAlternatively, use an ungated model like google/gemma-2b")
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    print(f"Using HuggingFace token: {hf_token[:8]}...")

    # Download tokenizer
    print(f"\n[1/2] Downloading tokenizer from {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer_path = output_dir / "tokenizer"
        tokenizer.save_pretrained(tokenizer_path)
        print(f"✓ Tokenizer saved to: {tokenizer_path}")
    except Exception as e:
        print(f"✗ Failed to download tokenizer: {e}")
        return False

    # Download model
    print(f"\n[2/2] Downloading model from {model_name}...")
    print("⚠️  This will download ~7GB of data and may take 10-30 minutes...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            # Don't load to GPU during download
            device_map=None,
            # Use lower precision to save space
            torch_dtype="auto",
        )
        model_path = output_dir / "model"
        model.save_pretrained(model_path)
        print(f"✓ Model saved to: {model_path}")
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False

    # Check file sizes
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)

    total_size = 0
    print("\nModel files:")
    for path in output_dir.rglob("*"):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            if size_mb > 1:  # Only show files > 1MB
                print(f"  {path.name}: {size_mb:.1f} MB")

    print(f"\nTotal size: {total_size:.1f} MB ({total_size / 1024:.2f} GB)")

    # Next steps
    print("\n" + "=" * 60)
    print("Next Steps for Kaggle Upload:")
    print("=" * 60)
    print("\n1. Go to: https://www.kaggle.com/datasets")
    print("2. Click 'New Dataset'")
    print(f"3. Upload all files from: {output_dir}")
    print("4. Dataset name: 'codegemma-7b-instruct'")
    print("5. Make dataset public (optional)")
    print("\nIn your Kaggle notebook, reference the dataset as:")
    print("  /kaggle/input/codegemma-7b-instruct/model/")
    print("  /kaggle/input/codegemma-7b-instruct/tokenizer/")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Code Gemma 7B model for Kaggle"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/codegemma-7b"),
        help="Output directory for model files (default: models/codegemma-7b)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/codegemma-7b-it",
        help="HuggingFace model identifier (default: google/codegemma-7b-it)",
    )

    args = parser.parse_args()

    success = download_model(args.output_dir, args.model_name)

    if success:
        print("\n✅ Model download successful!")
        return 0
    else:
        print("\n❌ Model download failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
