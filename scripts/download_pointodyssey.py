#!/usr/bin/env python3
"""Download PointOdyssey dataset from HuggingFace.

PointOdyssey v1.2: 159 videos (131 train, 15 val, 13 test)
- RGB images, depth maps, normal maps, instance segmentation
- 2D and 3D trajectories with visibility labels
- Camera parameters

Source: https://pointodyssey.com/
License: CC BY-NC-SA 4.0
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download PointOdyssey dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pointodyssey",
        help="Output directory (default: data/pointodyssey)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to download (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PointOdyssey Dataset Download")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Split: {args.split}")
    print()

    # Check if huggingface_hub is available
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    # HuggingFace dataset URL
    repo_id = "aharley/pointodyssey"

    print(f"Downloading from: https://huggingface.co/datasets/{repo_id}")
    print("This may take a while (~30-50 GB)...")
    print()

    # Download based on split
    if args.split == "all":
        allow_patterns = None  # Download everything
    else:
        # PointOdyssey structure: train/, val/, test/
        allow_patterns = [f"{args.split}/*"]

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            allow_patterns=allow_patterns,
        )
        print()
        print("=" * 60)
        print(f"Download complete: {local_dir}")
        print("=" * 60)

        # List downloaded contents
        print("\nContents:")
        for item in sorted(output_dir.iterdir()):
            if item.is_dir():
                count = len(list(item.iterdir()))
                print(f"  {item.name}/ ({count} items)")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.name} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"Error downloading: {e}")
        print()
        print("Alternative: Download manually from:")
        print("  https://huggingface.co/datasets/aharley/pointodyssey")
        print("  or")
        print("  https://drive.google.com/drive/folders/1W6wxsbKbTdtV8-2TwToqa_QgLqRY3ft0")
        sys.exit(1)


if __name__ == "__main__":
    main()
