"""
Download the Palmpad dataset from HuggingFace.

Dataset: Teburile/Palmpad_Dataset  (total ~96 GB, 2 groups of 16 users)
  group_1/  ~48 GB, 16 users
  group_2/  ~48 GB, 16 users

Usage:
  # group_1 only (default, ~48 GB)
  python download_dataset.py --out_dir /data/palmpad_raw

  # with HF token (recommended — avoids rate-limit stalls)
  python download_dataset.py --out_dir /data/palmpad_raw --token hf_xxx

  # both groups
  python download_dataset.py --out_dir /data/palmpad_raw --groups 1 2
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm


def fetch_file(repo_id, filename, local_dir, token):
    """Download one file; returns (filename, ok, error)."""
    try:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=local_dir,
            token=token,
        )
        return filename, True, None
    except Exception as e:
        return filename, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--groups", nargs="+", type=int, default=[1], choices=[1, 2])
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download threads")
    args = parser.parse_args()

    if args.token is None:
        print("Tip: set --token hf_xxx or export HF_TOKEN=hf_xxx to avoid rate limits\n")

    repo_id = "Teburile/Palmpad_Dataset"
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. List all files in the repo ---
    print("Fetching file list from HuggingFace...")
    try:
        all_files = list(list_repo_files(repo_id, repo_type="dataset", token=args.token))
    except Exception as e:
        print(f"Failed to list files: {e}")
        sys.exit(1)

    print(f"Total files in repo: {len(all_files)}")

    # --- 2. Filter to requested groups ---
    prefixes = tuple(f"group_{g}/" for g in args.groups)
    target_files = [f for f in all_files if f.startswith(prefixes)]

    if not target_files:
        # Fallback: dump first 20 paths so user can see real structure
        print("\nNo files matched! Actual repo paths (first 20):")
        for f in all_files[:20]:
            print(" ", f)
        sys.exit(1)

    print(f"Files to download: {len(target_files)}")

    # --- 3. Skip already-downloaded files ---
    pending = [f for f in target_files if not Path(out_dir, f).exists()]
    print(f"Already downloaded: {len(target_files) - len(pending)}  Remaining: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    # --- 4. Download in parallel ---
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_file, repo_id, f, out_dir, args.token): f
            for f in pending
        }
        with tqdm(total=len(pending), unit="file") as bar:
            for fut in as_completed(futures):
                fname, ok, err = fut.result()
                bar.update(1)
                if not ok:
                    errors.append((fname, err))
                    bar.write(f"  ERROR {fname}: {err}")

    if errors:
        print(f"\n{len(errors)} files failed. Re-run to retry (already-downloaded files are skipped).")
    else:
        print("\nDownload complete.")


if __name__ == "__main__":
    main()
