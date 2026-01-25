#!/usr/bin/env python3
"""Convert MOVi dataset to D4RT Kubric format

This script converts MOVi datasets (loaded via TensorFlow Datasets) to the
directory structure expected by our Kubric dataset loader.

Expected Kubric format:
    data/kubric/
    ├── train/
    │   └── scene_00000/
    │       ├── rgba/
    │       │   ├── frame_00000.png
    │       │   └── ...
    │       ├── depth/
    │       │   ├── frame_00000.npy
    │       │   └── ...
    │       └── camera.json
    └── val/
        └── ...

Usage:
    # Convert 100 MOVi-A samples
    python scripts/convert_movi_to_kubric.py \\
        --dataset movi_a \\
        --split train \\
        --num-samples 100 \\
        --output-dir data/kubric

    # Convert all validation samples
    python scripts/convert_movi_to_kubric.py \\
        --dataset movi_a \\
        --split validation \\
        --output-dir data/kubric
"""

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from PIL import Image
import json


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix

    Args:
        q: Quaternion [w, x, y, z] or [x, y, z, w] (shape: [4])

    Returns:
        R: Rotation matrix (shape: [3, 3])
    """
    # Assume quaternion is [x, y, z, w] format (common in graphics)
    x, y, z, w = q[0], q[1], q[2], q[3]

    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

    return R


def compute_intrinsics_matrix(focal_length, sensor_width, image_width, image_height):
    """Compute camera intrinsics matrix from Blender camera parameters

    Args:
        focal_length: Focal length in mm
        sensor_width: Sensor width in mm
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        K: Intrinsics matrix (shape: [3, 3])
    """
    # Compute focal length in pixels
    fx = (focal_length / sensor_width) * image_width
    fy = fx  # Assume square pixels

    # Principal point (center of image)
    cx = image_width / 2.0
    cy = image_height / 2.0

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return K


def compute_extrinsics_matrix(position, quaternion):
    """Compute camera extrinsics matrix (world-to-camera transform)

    Args:
        position: Camera position in world coordinates [x, y, z]
        quaternion: Camera orientation as quaternion [x, y, z, w]

    Returns:
        T: Extrinsics matrix (shape: [4, 4])
    """
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(quaternion)

    # In computer vision, the camera coordinate system is typically:
    # +X right, +Y down, +Z forward
    # Blender uses: +X right, +Y backward, +Z up
    # We need to convert from Blender to OpenCV convention

    # Create the extrinsics matrix [R | t]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = position

    return T


def convert_movi_sample(sample, scene_idx, output_dir, split):
    """Convert a single MOVi sample to Kubric format

    Args:
        sample: MOVi dataset sample (dict of tensors)
        scene_idx: Scene index for naming
        output_dir: Base output directory
        split: Dataset split (train/val)
    """
    # Create output directories
    scene_dir = Path(output_dir) / split / f"scene_{scene_idx:05d}"
    rgba_dir = scene_dir / "rgba"
    depth_dir = scene_dir / "depth"

    rgba_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Extract video frames
    video = sample['video'].numpy()  # [T, H, W, 3]
    num_frames, height, width = video.shape[:3]

    # Save RGB frames as PNG
    for frame_idx in range(num_frames):
        frame = video[frame_idx]  # [H, W, 3], uint8
        frame_path = rgba_dir / f"frame_{frame_idx:05d}.png"
        Image.fromarray(frame).save(frame_path)

    # Extract and save depth maps
    depth = sample['depth'].numpy()  # [T, H, W, 1], uint16
    for frame_idx in range(num_frames):
        depth_frame = depth[frame_idx, :, :, 0]  # [H, W]
        # Convert from uint16 to float32 meters
        # MOVi depth is stored as uint16 with a scale factor in metadata
        depth_range = sample['metadata']['depth_range'].numpy()  # [min, max]
        depth_float = depth_frame.astype(np.float32) / 65535.0  # Normalize to [0, 1]
        depth_float = depth_float * (depth_range[1] - depth_range[0]) + depth_range[0]

        depth_path = depth_dir / f"frame_{frame_idx:05d}.npy"
        np.save(depth_path, depth_float)

    # Extract camera parameters
    camera = sample['camera']
    focal_length = float(camera['focal_length'].numpy())
    sensor_width = float(camera['sensor_width'].numpy())
    positions = camera['positions'].numpy()  # [T, 3]
    quaternions = camera['quaternions'].numpy()  # [T, 4]

    # Compute intrinsics and extrinsics for each frame
    intrinsics_list = []
    extrinsics_list = []

    for frame_idx in range(num_frames):
        # Compute intrinsics (same for all frames)
        K = compute_intrinsics_matrix(focal_length, sensor_width, width, height)
        intrinsics_list.append(K.tolist())

        # Compute extrinsics
        T = compute_extrinsics_matrix(positions[frame_idx], quaternions[frame_idx])
        extrinsics_list.append(T.tolist())

    # Save camera parameters
    camera_data = {
        'intrinsics': intrinsics_list,
        'extrinsics': extrinsics_list,
        'focal_length': focal_length,
        'sensor_width': sensor_width,
    }

    camera_path = scene_dir / "camera.json"
    with open(camera_path, 'w') as f:
        json.dump(camera_data, f, indent=2)


def convert_movi_to_kubric(
    dataset_name: str = "movi_a",
    split: str = "train",
    num_samples: int = None,
    output_dir: str = "data/kubric",
    resolution: str = "256x256"
):
    """Convert MOVi dataset to D4RT Kubric format

    Args:
        dataset_name: MOVi variant (movi_a, movi_b, etc.)
        split: Dataset split (train, validation)
        num_samples: Number of samples to convert (None = all)
        output_dir: Output directory for Kubric-formatted data
        resolution: Resolution variant (256x256, 512x512)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Map validation -> val for our dataset loader
    split_name = "val" if split == "validation" else split

    print(f"🔄 Converting {dataset_name}/{resolution} ({split} split) to Kubric format...")
    print(f"   Output: {output_path / split_name}")

    # Construct dataset name
    full_dataset_name = f"{dataset_name}/{resolution}"

    # Construct split specification
    if num_samples is not None:
        split_spec = f"{split}[:{num_samples}]"
        print(f"   Samples: {num_samples}")
    else:
        split_spec = split
        print(f"   Samples: all")

    # Load dataset from Google Cloud Storage
    print("\n📥 Loading dataset from gs://kubric-public/tfds...")

    ds = tfds.load(
        full_dataset_name,
        split=split_spec,
        data_dir="gs://kubric-public/tfds",
        try_gcs=True
    )

    # Convert samples
    print(f"\n🔄 Converting samples...")

    for scene_idx, sample in enumerate(tqdm(ds)):
        try:
            convert_movi_sample(sample, scene_idx, output_dir, split_name)
        except Exception as e:
            print(f"\n⚠ Error converting scene {scene_idx}: {e}")
            continue

    print(f"\n✓ Conversion complete!")
    print(f"   Converted {scene_idx + 1} scenes to: {output_path / split_name}")
    print(f"\n📊 Directory structure:")
    print(f"   {output_path / split_name}")
    print(f"   └── scene_00000/")
    print(f"       ├── rgba/frame_*.png  ({num_samples or 'all'} frames)")
    print(f"       ├── depth/frame_*.npy")
    print(f"       └── camera.json")

    print(f"\nNext step:")
    print(f"   Train on real MOVi data:")
    print(f"   python scripts/train_simple.py \\")
    print(f"       --model-config configs/model/vit_b.yaml \\")
    print(f"       --training-config configs/training/debug.yaml \\")
    print(f"       --data-dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert MOVi to Kubric format")
    parser.add_argument(
        "--dataset",
        type=str,
        default="movi_a",
        choices=["movi_a", "movi_b", "movi_c", "movi_d", "movi_e"],
        help="MOVi dataset variant"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to convert (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/kubric",
        help="Output directory"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="256x256",
        choices=["256x256", "512x512"],
        help="Resolution variant"
    )

    args = parser.parse_args()

    convert_movi_to_kubric(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        resolution=args.resolution
    )


if __name__ == "__main__":
    main()
