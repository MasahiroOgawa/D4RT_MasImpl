#!/usr/bin/env python3
"""Add point tracks to converted MOVi validation data

This script generates point tracks from MOVi's object_coordinates and adds them
to the validation scenes in TAP-Vid compatible format.

Usage:
    # Add tracks to first 5 validation scenes
    python scripts/add_tracks_to_movi.py \\
        --data-dir data/kubric \\
        --split val \\
        --num-samples 5 \\
        --num-points 256

    # Process all validation scenes
    python scripts/add_tracks_to_movi.py \\
        --data-dir data/kubric \\
        --split val \\
        --num-points 256
"""

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import json


def sample_points_from_segmentation(
    segmentation,  # [H, W] uint8
    object_coordinates,  # [H, W, 3] uint16 normalized object-local coords
    num_points=256,
    grid_size=16,
):
    """Sample points on objects using a grid strategy

    Args:
        segmentation: Object instance mask [H, W]
        object_coordinates: Object-local coordinates [H, W, 3] in uint16
        num_points: Number of points to sample
        grid_size: Grid sampling stride

    Returns:
        query_points: [N, 3] array of (t, y, x) query locations
        object_ids: [N] array of object IDs for each point
        obj_coords: [N, 3] array of object-local coordinates
    """
    H, W = segmentation.shape

    # Find all objects (exclude background = 0)
    object_ids = np.unique(segmentation)
    object_ids = object_ids[object_ids > 0]

    query_points = []
    query_obj_ids = []
    query_obj_coords = []

    # Sample points on a grid from each object
    for obj_id in object_ids:
        mask = (segmentation == obj_id)
        ys, xs = np.where(mask)

        if len(ys) == 0:
            continue

        # Sample on grid with some randomness
        points_per_object = max(1, num_points // max(1, len(object_ids)))

        # Create grid sampling
        step = max(1, int(np.sqrt(len(ys) / points_per_object)))
        indices = np.arange(0, len(ys), step)

        # Add random offset
        if len(indices) > 0:
            offset = np.random.randint(0, max(1, step))
            indices = indices + offset
            indices = indices[indices < len(ys)]

        # Limit to points_per_object
        if len(indices) > points_per_object:
            indices = np.random.choice(indices, points_per_object, replace=False)

        for idx in indices:
            y, x = ys[idx], xs[idx]
            # Get object-local coordinates at this pixel
            obj_coord = object_coordinates[y, x]  # [3] uint16

            query_points.append([0, y, x])  # t=0 for first frame
            query_obj_ids.append(obj_id)
            query_obj_coords.append(obj_coord)

    if len(query_points) == 0:
        # No objects found, sample from image center
        query_points = [[0, H//2, W//2]]
        query_obj_ids = [0]
        query_obj_coords = [[32768, 32768, 32768]]  # Center of uint16 range

    return (
        np.array(query_points, dtype=np.int32),
        np.array(query_obj_ids, dtype=np.uint8),
        np.array(query_obj_coords, dtype=np.uint16)
    )


def track_points_simple(
    query_obj_ids,  # [N] object IDs
    query_obj_coords,  # [N, 3] object-local coordinates (uint16)
    segmentations,  # [T, H, W] instance masks
    object_coordinates,  # [T, H, W, 3] object-local coordinates (uint16)
    depths,  # [T, H, W] depth maps (meters)
    camera_intrinsics,  # [T, 3, 3] camera intrinsics
):
    """Track points across frames using object coordinates

    Args:
        query_obj_ids: [N] Object IDs for each query point
        query_obj_coords: [N, 3] Object-local coordinates (uint16)
        segmentations: [T, H, W] Instance segmentation masks
        object_coordinates: [T, H, W, 3] Object-local coordinates (uint16)
        depths: [T, H, W] Depth maps in meters
        camera_intrinsics: [T, 3, 3] Camera intrinsics

    Returns:
        tracks_2d: [N, T, 2] 2D pixel trajectories (x, y)
        tracks_3d: [N, T, 3] 3D world coordinates (x, y, z)
        visibility: [N, T] Visibility flags (0 or 1)
    """
    T, H, W = segmentations.shape
    N = len(query_obj_ids)

    tracks_2d = np.zeros((N, T, 2), dtype=np.float32)
    tracks_3d = np.zeros((N, T, 3), dtype=np.float32)
    visibility = np.zeros((N, T), dtype=np.float32)

    for t in range(T):
        seg_t = segmentations[t]  # [H, W]
        obj_coords_t = object_coordinates[t]  # [H, W, 3]
        depth_t = depths[t]  # [H, W]
        K_t = camera_intrinsics[t]  # [3, 3]

        for n in range(N):
            obj_id = query_obj_ids[n]
            target_coord = query_obj_coords[n]  # [3] uint16

            # Find pixels belonging to this object
            obj_mask = (seg_t == obj_id)

            if not np.any(obj_mask):
                # Object not visible in this frame
                visibility[n, t] = 0.0
                continue

            # Get object coordinates for this object's pixels
            obj_pixels = obj_coords_t[obj_mask]  # [M, 3]

            # Find pixel with closest object coordinate
            distances = np.linalg.norm(
                obj_pixels.astype(np.float32) - target_coord.astype(np.float32),
                axis=1
            )

            if len(distances) == 0:
                visibility[n, t] = 0.0
                continue

            best_idx = np.argmin(distances)

            # Get pixel location
            ys, xs = np.where(obj_mask)
            y, x = ys[best_idx], xs[best_idx]

            # Record 2D position
            tracks_2d[n, t] = [x, y]
            visibility[n, t] = 1.0

            # Compute 3D position from depth
            depth_val = depth_t[y, x]

            if depth_val > 0:
                # Unproject to 3D camera coordinates
                fx = K_t[0, 0]
                fy = K_t[1, 1]
                cx = K_t[0, 2]
                cy = K_t[1, 2]

                x_3d = (x - cx) * depth_val / fx
                y_3d = (y - cy) * depth_val / fy
                z_3d = depth_val

                tracks_3d[n, t] = [x_3d, y_3d, z_3d]
            else:
                # Invalid depth
                visibility[n, t] = 0.0

    return tracks_2d, tracks_3d, visibility


def add_tracks_to_scene(scene_idx, output_dir, split, num_points=256):
    """Add point tracks to a single MOVi scene

    Args:
        scene_idx: Scene index
        output_dir: Base output directory
        split: Dataset split (train/val)
        num_points: Number of points to track
    """
    scene_dir = Path(output_dir) / split / f"scene_{scene_idx:05d}"

    if not scene_dir.exists():
        print(f"⚠ Scene directory not found: {scene_dir}")
        return False

    # Load MOVi sample from TensorFlow Datasets
    print(f"  Loading scene {scene_idx} from TFDS...")

    ds = tfds.load(
        'movi_a/256x256',
        split=f'validation[{scene_idx}:{scene_idx+1}]',
        data_dir='gs://kubric-public/tfds',
        try_gcs=True
    )

    sample = next(iter(ds))

    # Extract data
    video = sample['video'].numpy()  # [T, H, W, 3] uint8
    depth = sample['depth'].numpy()  # [T, H, W, 1] uint16
    segmentations = sample['segmentations'].numpy()  # [T, H, W, 1] uint8
    object_coordinates = sample['object_coordinates'].numpy()  # [T, H, W, 3] uint16

    # Squeeze depth and segmentation
    depth = depth[:, :, :, 0]  # [T, H, W]
    segmentations = segmentations[:, :, :, 0]  # [T, H, W]

    # Convert depth to meters
    depth_range = sample['metadata']['depth_range'].numpy()
    depth_meters = depth.astype(np.float32) / 65535.0
    depth_meters = depth_meters * (depth_range[1] - depth_range[0]) + depth_range[0]

    # Load camera parameters from saved file
    camera_file = scene_dir / 'camera.json'
    with open(camera_file, 'r') as f:
        camera_data = json.load(f)

    intrinsics = np.array(camera_data['intrinsics'], dtype=np.float32)  # [T, 3, 3]

    T, H, W, _ = video.shape

    # Sample query points from first frame
    print(f"  Sampling {num_points} query points...")
    query_points, query_obj_ids, query_obj_coords = sample_points_from_segmentation(
        segmentations[0],
        object_coordinates[0],
        num_points=num_points,
        grid_size=16
    )

    # Track points across frames
    print(f"  Tracking {len(query_points)} points across {T} frames...")
    tracks_2d, tracks_3d, visibility = track_points_simple(
        query_obj_ids,
        query_obj_coords,
        segmentations,
        object_coordinates,
        depth_meters,
        intrinsics
    )

    # Save tracks in TAP-Vid compatible format
    tracks_file = scene_dir / 'tracks.npz'
    np.savez(
        tracks_file,
        query_points=query_points,  # [N, 3] (t, y, x)
        tracks_2d=tracks_2d,  # [N, T, 2] (x, y)
        tracks_3d=tracks_3d,  # [N, T, 3] (x, y, z) in camera coords
        visibility=visibility,  # [N, T]
        query_obj_ids=query_obj_ids,  # [N]
        intrinsics=intrinsics[0],  # [3, 3] (same for all frames in MOVi)
    )

    print(f"  ✓ Saved {len(query_points)} tracks to {tracks_file}")
    print(f"    Visible points: {np.sum(visibility > 0.5)}/{visibility.size} ({100*np.mean(visibility):.1f}%)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Add point tracks to MOVi data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/kubric",
        help="Data directory with converted MOVi scenes"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of scenes to process (default: all)"
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=256,
        help="Number of points to track per scene"
    )

    args = parser.parse_args()

    # Find all scenes
    split_dir = Path(args.data_dir) / args.split
    if not split_dir.exists():
        print(f"❌ Split directory not found: {split_dir}")
        return

    scene_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

    if args.num_samples:
        scene_dirs = scene_dirs[:args.num_samples]

    print(f"🔄 Adding point tracks to {len(scene_dirs)} scenes...")
    print(f"   Split: {args.split}")
    print(f"   Points per scene: {args.num_points}")
    print()

    success_count = 0
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_idx = int(scene_dir.name.split('_')[1])

        try:
            if add_tracks_to_scene(scene_idx, args.data_dir, args.split, args.num_points):
                success_count += 1
        except Exception as e:
            print(f"\n⚠ Error processing scene {scene_idx}: {e}")
            continue

    print(f"\n✓ Successfully added tracks to {success_count}/{len(scene_dirs)} scenes")
    print(f"\nNext step: Run TAP-Vid evaluation")
    print(f"   python scripts/evaluate_tapvid.py \\")
    print(f"       --checkpoint checkpoints/checkpoint_step_0050000.pth \\")
    print(f"       --model-config configs/model/vit_b_movi.yaml \\")
    print(f"       --data-dir {args.data_dir} \\")
    print(f"       --split {args.split}")


if __name__ == "__main__":
    main()
