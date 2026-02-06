#!/usr/bin/env python3
"""Download PointOdyssey dataset from HuggingFace.

PointOdyssey v1.2: 159 videos (131 train, 15 val, 13 test)
- RGB images, depth maps, normal maps, instance segmentation
- 2D and 3D trajectories with visibility labels
- Camera parameters

Source: https://pointodyssey.com/
License: CC BY-NC-SA 4.0

HuggingFace structure:
- val.tar.gz (~20GB)
- test.tar.gz (~18GB)
- train.tar.gz.partaa - train.tar.gz.partad (~134GB total)
- sample.tar.gz (~3GB)
"""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path


def download_and_extract_split(output_dir: Path, split: str, repo_id: str):
    """Download and extract a single split."""
    from huggingface_hub import hf_hub_download

    if split == "train":
        # Train is split into parts
        parts = [
            "train.tar.gz.partaa",
            "train.tar.gz.partab",
            "train.tar.gz.partac",
            "train.tar.gz.partad",
        ]
        print(f"Downloading train split ({len(parts)} parts)...")

        part_files = []
        for part in parts:
            print(f"  Downloading {part}...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=part,
                local_dir=str(output_dir),
            )
            part_files.append(Path(local_path))

        # Combine parts
        combined_path = output_dir / "train.tar.gz"
        print(f"Combining parts into {combined_path}...")
        with open(combined_path, "wb") as outfile:
            for part_path in part_files:
                with open(part_path, "rb") as infile:
                    outfile.write(infile.read())

        # Clean up parts
        for part_path in part_files:
            part_path.unlink()

        tar_path = combined_path
    else:
        # Single tar.gz file
        filename = f"{split}.tar.gz"
        print(f"Downloading {filename}...")
        tar_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(output_dir),
            )
        )

    # Extract
    print(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=output_dir)

    # Optionally remove tar after extraction
    # tar_path.unlink()

    print(f"  Extracted to {output_dir / split}/")


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
        choices=["train", "val", "test", "sample", "all"],
        default="all",
        help="Which split to download (default: all)",
    )
    parser.add_argument(
        "--keep-tar",
        action="store_true",
        help="Keep tar.gz files after extraction",
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
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    # HuggingFace dataset URL
    repo_id = "aharley/pointodyssey"

    print(f"Downloading from: https://huggingface.co/datasets/{repo_id}")
    print()

    # Determine splits to download
    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]

    try:
        for split in splits:
            # Check if already extracted
            split_dir = output_dir / split
            if split_dir.exists() and any(split_dir.iterdir()):
                print(f"{split}/ already exists, skipping download")
                continue

            download_and_extract_split(output_dir, split, repo_id)
            print()

        print("=" * 60)
        print("Download complete!")
        print("=" * 60)

        # List downloaded contents
        print("\nContents:")
        for item in sorted(output_dir.iterdir()):
            if item.is_dir():
                subdirs = [d for d in item.iterdir() if d.is_dir()]
                print(f"  {item.name}/ ({len(subdirs)} scenes)")
            elif item.suffix in [".gz", ".tar"]:
                size_gb = item.stat().st_size / (1024**3)
                print(f"  {item.name} ({size_gb:.1f} GB)")
            else:
                print(f"  {item.name}")

    except Exception as e:
        print(f"Error downloading: {e}")
        print()
        print("Alternative: Download manually from:")
        print("  https://huggingface.co/datasets/aharley/pointodyssey")
        print("  or")
        print("  https://drive.google.com/drive/folders/1W6wxsbKbTdtV8-2TwToqa_QgLqRY3ft0")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
