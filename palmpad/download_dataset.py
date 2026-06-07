"""
Download the Palmpad dataset from HuggingFace.

Dataset: Teburile/Palmpad_Dataset  (~96 GB)
  32 users, 2 groups, frame-level touch/no-touch labels at 120fps.

Usage:
  python download_dataset.py --out_dir /data/palmpad_raw
"""

import argparse
import os
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",    default="data/raw",
                        help="Local directory to store the downloaded dataset")
    parser.add_argument("--token",      default=None,
                        help="HuggingFace token (if dataset is gated)")
    parser.add_argument("--resume",     action="store_true", default=True,
                        help="Resume partial download")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Downloading Teburile/Palmpad_Dataset → {args.out_dir}")
    print("Dataset size: ~96 GB. This may take a while on a slow connection.")

    path = snapshot_download(
        repo_id="Teburile/Palmpad_Dataset",
        repo_type="dataset",
        local_dir=args.out_dir,
        token=args.token,
        resume_download=args.resume,
        ignore_patterns=["*.parquet"],  # prefer raw video files
    )
    print(f"Download complete: {path}")


if __name__ == "__main__":
    main()
