#!/usr/bin/env python3
"""Download MOVi datasets using TensorFlow Datasets

MOVi (Multi-Object Video) datasets are pre-generated Kubric synthetic datasets
provided by Google Research. Available variants:
- MOVi-A: Simple scenes with few objects (256x256)
- MOVi-B: More complex scenes
- MOVi-C: Even more complex
- MOVi-D: Indoor scenes
- MOVi-E: Outdoor scenes (512x512)

This script downloads MOVi datasets and saves them in a format compatible with
our Kubric dataset loader.

Usage:
    # Download 100 samples from MOVi-A for testing
    python scripts/download_movi.py --dataset movi_a --split train --num-samples 100

    # Download full MOVi-A training set
    python scripts/download_movi.py --dataset movi_a --split train

    # Download MOVi-A validation set
    python scripts/download_movi.py --dataset movi_a --split validation
"""

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from PIL import Image
import json


def download_movi(
    dataset_name: str = "movi_a",
    split: str = "train",
    num_samples: int = None,
    output_dir: str = "data/movi_raw",
    resolution: str = "256x256"
):
    """Download MOVi dataset using TensorFlow Datasets

    Args:
        dataset_name: MOVi variant (movi_a, movi_b, movi_c, movi_d, movi_e)
        split: Dataset split (train, validation)
        num_samples: Number of samples to download (None = all)
        output_dir: Directory to save raw MOVi data
        resolution: Resolution variant (256x256, 512x512 for movi_e)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Construct dataset name with resolution
    full_dataset_name = f"{dataset_name}/{resolution}"

    print(f"🚀 Downloading {full_dataset_name} ({split} split)...")
    print(f"   Output: {output_path}")

    # Construct split specification
    if num_samples is not None:
        split_spec = f"{split}[:{num_samples}]"
        print(f"   Samples: {num_samples} (subset)")
    else:
        split_spec = split
        print(f"   Samples: all")

    # Download dataset from Google Cloud bucket
    # MOVi datasets are stored at gs://kubric-public/tfds
    print("\n📥 Loading dataset from gs://kubric-public/tfds...")
    print("   This will download data from Google Cloud Storage")

    ds = tfds.load(
        full_dataset_name,
        split=split_spec,
        data_dir="gs://kubric-public/tfds",
        download=True,
        with_info=True,
        try_gcs=True
    )

    dataset, info = ds

    print(f"✓ Dataset info:")
    print(f"  Features: {info.features}")
    print(f"  Total examples: {info.splits[split].num_examples}")

    # Save dataset info
    info_file = output_path / f"{dataset_name}_{split}_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            'dataset': full_dataset_name,
            'split': split,
            'num_examples': info.splits[split].num_examples,
            'num_downloaded': num_samples if num_samples else info.splits[split].num_examples,
            'features': str(info.features),
        }, f, indent=2)

    print(f"\n✓ Dataset downloaded to: {output_path / 'downloads'}")
    print(f"✓ Info saved to: {info_file}")
    print(f"\n📊 Dataset structure:")
    print(f"   {output_path / 'downloads'}")
    print(f"   └── {dataset_name}/")
    print(f"       └── {resolution}/")
    print(f"           └── 3.0.0/  (TFDS version)")
    print(f"               ├── dataset_info.json")
    print(f"               └── {split}-*.tfrecord-*")

    print("\n✓ Download complete!")
    print("\nNext steps:")
    print("1. Convert MOVi format to D4RT Kubric format:")
    print(f"   python scripts/convert_movi_to_kubric.py \\")
    print(f"       --movi-dir {output_path / 'downloads'} \\")
    print(f"       --dataset {dataset_name} \\")
    print(f"       --split {split} \\")
    print(f"       --output-dir data/kubric")
    print("\n2. Train on real data:")
    print("   python scripts/train_simple.py \\")
    print("       --model-config configs/model/vit_b.yaml \\")
    print("       --training-config configs/training/debug.yaml \\")
    print("       --data-dir data/kubric")


def inspect_sample(
    dataset_name: str = "movi_a",
    split: str = "train",
    sample_idx: int = 0,
    data_dir: str = "data/movi_raw/downloads"
):
    """Inspect a single sample from downloaded MOVi dataset

    Args:
        dataset_name: MOVi variant
        split: Dataset split
        sample_idx: Index of sample to inspect
        data_dir: Directory where MOVi data is stored
    """
    print(f"🔍 Inspecting sample {sample_idx} from {dataset_name}/{split}...")

    # Load dataset
    ds = tfds.load(
        f"{dataset_name}/256x256",
        split=f"{split}[{sample_idx}:{sample_idx+1}]",
        data_dir=data_dir,
    )

    # Get first (and only) sample
    for sample in ds:
        print("\n📊 Sample structure:")
        for key in sorted(sample.keys()):
            value = sample[key]
            if isinstance(value, tf.Tensor):
                print(f"  {key:30s}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key:30s}: {type(value)}")

        # Print specific info
        if 'video' in sample:
            video = sample['video'].numpy()
            print(f"\n🎬 Video:")
            print(f"  Shape: {video.shape} (T, H, W, C)")
            print(f"  Dtype: {video.dtype}")
            print(f"  Range: [{video.min()}, {video.max()}]")

        if 'depth' in sample:
            depth = sample['depth'].numpy()
            print(f"\n📏 Depth:")
            print(f"  Shape: {depth.shape}")
            print(f"  Dtype: {depth.dtype}")
            print(f"  Range: [{depth.min():.2f}, {depth.max():.2f}]")

        if 'camera' in sample:
            print(f"\n📷 Camera:")
            camera = sample['camera']
            for cam_key in sorted(camera.keys()):
                cam_value = camera[cam_key]
                if isinstance(cam_value, tf.Tensor):
                    print(f"  {cam_key:28s}: shape={cam_value.shape}, dtype={cam_value.dtype}")

        if 'instances' in sample:
            print(f"\n🎯 Instances:")
            instances = sample['instances']
            for inst_key in sorted(instances.keys()):
                inst_value = instances[inst_key]
                if isinstance(inst_value, tf.Tensor):
                    print(f"  {inst_key:28s}: shape={inst_value.shape}, dtype={inst_value.dtype}")

        break

    print("\n✓ Inspection complete!")


def main():
    parser = argparse.ArgumentParser(description="Download MOVi datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="movi_a",
        choices=["movi_a", "movi_b", "movi_c", "movi_d", "movi_e"],
        help="MOVi dataset variant to download"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to download"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to download (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/movi_raw",
        help="Output directory for raw MOVi data"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="256x256",
        choices=["256x256", "512x512"],
        help="Resolution variant (512x512 only available for movi_e)"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect a sample instead of downloading"
    )
    parser.add_argument(
        "--inspect-idx",
        type=int,
        default=0,
        help="Sample index to inspect (used with --inspect)"
    )

    args = parser.parse_args()

    if args.inspect:
        inspect_sample(
            dataset_name=args.dataset,
            split=args.split,
            sample_idx=args.inspect_idx,
            data_dir=f"{args.output_dir}/downloads"
        )
    else:
        download_movi(
            dataset_name=args.dataset,
            split=args.split,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            resolution=args.resolution
        )


if __name__ == "__main__":
    main()
