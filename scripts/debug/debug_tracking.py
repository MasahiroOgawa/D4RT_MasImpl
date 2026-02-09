#!/usr/bin/env python3
"""Debug script to investigate poor tracking performance."""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from d4rt.data.datasets.kubric import KubricDataset
from d4rt.models.d4rt import build_d4rt_model


def check_tracks_loaded():
    """Check if tracks.npz files are being loaded correctly."""
    print("=" * 60)
    print("1. CHECKING TRACKS LOADING")
    print("=" * 60)

    data_dir = Path("data/kubric")

    # Check validation scenes
    val_dir = data_dir / "val"
    scenes = sorted([d for d in val_dir.iterdir() if d.is_dir()])[:5]

    for scene_dir in scenes:
        tracks_file = scene_dir / "tracks.npz"
        if tracks_file.exists():
            data = np.load(tracks_file)
            keys = list(data.keys())
            if 'tracks_2d' in data:
                tracks = data['tracks_2d']
                print(f"{scene_dir.name}: tracks_2d shape = {tracks.shape}")
                # Check value ranges
                print(f"  u range: [{tracks[:,:,0].min():.1f}, {tracks[:,:,0].max():.1f}]")
                print(f"  v range: [{tracks[:,:,1].min():.1f}, {tracks[:,:,1].max():.1f}]")
            else:
                print(f"{scene_dir.name}: keys = {keys}")
        else:
            print(f"{scene_dir.name}: NO tracks.npz!")
    print()


def check_dataset_output():
    """Check what the dataset actually returns."""
    print("=" * 60)
    print("2. CHECKING DATASET OUTPUT")
    print("=" * 60)

    # Create dataset
    dataset = KubricDataset(
        data_dir="data/kubric",
        split="val",
        num_frames=24,
        resolution=(256, 256),
        num_queries=64,
    )

    # Get a sample
    sample = dataset[0]

    print(f"Sample keys: {list(sample.keys())}")
    print(f"Video shape: {sample['video'].shape}")
    print(f"Queries keys: {list(sample['queries'].keys())}")
    print(f"Targets keys: {list(sample['targets'].keys())}")

    # Check query values
    queries = sample['queries']
    print(f"\nQuery ranges:")
    print(f"  u: [{queries['u'].min():.3f}, {queries['u'].max():.3f}]")
    print(f"  v: [{queries['v'].min():.3f}, {queries['v'].max():.3f}]")
    print(f"  t_src: [{queries['t_src'].min()}, {queries['t_src'].max()}]")
    print(f"  t_tgt: [{queries['t_tgt'].min()}, {queries['t_tgt'].max()}]")

    # Check target values
    targets = sample['targets']
    print(f"\nTarget xyz:")
    xyz = targets['xyz']
    print(f"  Shape: {xyz.shape}")
    print(f"  X range: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
    print(f"  Y range: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
    print(f"  Z (depth) range: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")

    # Check visibility
    if 'visibility' in targets:
        vis = targets['visibility']
        print(f"\nVisibility: {vis.sum().item()}/{len(vis)} visible ({100*vis.float().mean():.1f}%)")

    print()
    return sample


def check_model_predictions(sample):
    """Check what the model actually predicts."""
    print("=" * 60)
    print("3. CHECKING MODEL PREDICTIONS")
    print("=" * 60)

    # Load model
    model_config = OmegaConf.load("configs/model/vit_b_movi.yaml")
    model = build_d4rt_model(model_config)

    # Load checkpoint
    ckpt_path = "checkpoints/checkpoint_step_0050000.pth"
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Prepare input
    video = sample['video'].unsqueeze(0).to(device)  # [1, C, T, H, W]
    queries = {k: v.unsqueeze(0).to(device) for k, v in sample['queries'].items()}
    targets = sample['targets']

    print(f"Input video shape: {video.shape}")
    print(f"Num queries: {queries['u'].shape[1]}")

    # Run model
    with torch.no_grad():
        outputs = model(video, queries)

    print(f"\nOutput keys: {list(outputs.keys())}")

    # Compare predictions vs ground truth
    pred_xyz = outputs['xyz'][0].cpu()  # [N, 3]
    gt_xyz = targets['xyz']  # [N, 3]

    print(f"\nPrediction xyz_3d:")
    print(f"  X range: [{pred_xyz[:, 0].min():.3f}, {pred_xyz[:, 0].max():.3f}]")
    print(f"  Y range: [{pred_xyz[:, 1].min():.3f}, {pred_xyz[:, 1].max():.3f}]")
    print(f"  Z (depth) range: [{pred_xyz[:, 2].min():.3f}, {pred_xyz[:, 2].max():.3f}]")

    print(f"\nGround truth xyz_3d:")
    print(f"  X range: [{gt_xyz[:, 0].min():.3f}, {gt_xyz[:, 0].max():.3f}]")
    print(f"  Y range: [{gt_xyz[:, 1].min():.3f}, {gt_xyz[:, 1].max():.3f}]")
    print(f"  Z (depth) range: [{gt_xyz[:, 2].min():.3f}, {gt_xyz[:, 2].max():.3f}]")

    # Compute errors
    errors = (pred_xyz - gt_xyz).abs()
    print(f"\nAbsolute errors:")
    print(f"  X: mean={errors[:, 0].mean():.4f}, max={errors[:, 0].max():.4f}")
    print(f"  Y: mean={errors[:, 1].mean():.4f}, max={errors[:, 1].max():.4f}")
    print(f"  Z: mean={errors[:, 2].mean():.4f}, max={errors[:, 2].max():.4f}")
    print(f"  3D: mean={errors.norm(dim=1).mean():.4f}, max={errors.norm(dim=1).max():.4f}")

    # Check if model is predicting constant values
    pred_std = pred_xyz.std(dim=0)
    gt_std = gt_xyz.std(dim=0)
    print(f"\nStandard deviation (variety in predictions):")
    print(f"  Pred: X={pred_std[0]:.4f}, Y={pred_std[1]:.4f}, Z={pred_std[2]:.4f}")
    print(f"  GT:   X={gt_std[0]:.4f}, Y={gt_std[1]:.4f}, Z={gt_std[2]:.4f}")

    # Check visibility predictions
    if 'visibility' in outputs:
        pred_vis = outputs['visibility'][0].cpu()
        gt_vis = targets['visibility']
        print(f"\nVisibility prediction:")
        print(f"  Pred range: [{pred_vis.min():.3f}, {pred_vis.max():.3f}]")
        print(f"  GT visible: {gt_vis.sum().item()}/{len(gt_vis)}")
        pred_vis_binary = (pred_vis > 0.5).float()
        accuracy = (pred_vis_binary == gt_vis.float()).float().mean()
        print(f"  Accuracy: {100*accuracy:.1f}%")

    print()
    return outputs, targets


def check_query_target_alignment():
    """Check if queries and targets are aligned correctly."""
    print("=" * 60)
    print("4. CHECKING QUERY-TARGET ALIGNMENT")
    print("=" * 60)

    # Load raw tracks
    tracks_file = Path("data/kubric/val/scene_00000/tracks.npz")
    if not tracks_file.exists():
        print("No tracks file found!")
        return

    tracks_data = np.load(tracks_file)
    tracks_2d = tracks_data['tracks_2d']  # [N, T, 2]
    visibility = tracks_data['visibility']  # [N, T]

    print(f"Raw tracks shape: {tracks_2d.shape}")
    print(f"Raw visibility shape: {visibility.shape}")

    # Check track motion
    track_motion = np.diff(tracks_2d, axis=1)  # [N, T-1, 2]
    motion_magnitude = np.linalg.norm(track_motion, axis=2)  # [N, T-1]

    print(f"\nTrack motion (pixels per frame):")
    print(f"  Mean: {motion_magnitude.mean():.2f}")
    print(f"  Max:  {motion_magnitude.max():.2f}")
    print(f"  Min:  {motion_magnitude.min():.2f}")

    # Check if tracks actually move
    total_displacement = np.linalg.norm(tracks_2d[:, -1] - tracks_2d[:, 0], axis=1)
    print(f"\nTotal displacement (start to end):")
    print(f"  Mean: {total_displacement.mean():.2f} pixels")
    print(f"  Max:  {total_displacement.max():.2f} pixels")
    print(f"  Static tracks (<5px): {(total_displacement < 5).sum()}/{len(total_displacement)}")

    print()


def check_coordinate_systems():
    """Check if coordinate systems match between tracks and model."""
    print("=" * 60)
    print("5. CHECKING COORDINATE SYSTEMS")
    print("=" * 60)

    # The model expects normalized (u, v) in [0, 1]
    # The tracks are in pixel coordinates

    tracks_file = Path("data/kubric/val/scene_00000/tracks.npz")
    tracks_data = np.load(tracks_file)
    tracks_2d = tracks_data['tracks_2d']  # [N, T, 2] in pixels

    print(f"Track coordinate format: pixel coordinates")
    print(f"  u (x) range: [{tracks_2d[:,:,0].min():.1f}, {tracks_2d[:,:,0].max():.1f}]")
    print(f"  v (y) range: [{tracks_2d[:,:,1].min():.1f}, {tracks_2d[:,:,1].max():.1f}]")

    # Expected for 256x256 images
    print(f"\nExpected range for 256x256 images: [0, 255]")

    # Check if tracks are in correct coordinate order (u=x, v=y)
    # In image coords: u is horizontal (column), v is vertical (row)

    print()


def main():
    print("\n" + "=" * 60)
    print("TRACKING DEBUG INVESTIGATION")
    print("=" * 60 + "\n")

    check_tracks_loaded()
    check_dataset_output()
    check_query_target_alignment()
    check_coordinate_systems()

    # Load sample for model check
    dataset = KubricDataset(
        data_dir="data/kubric",
        split="val",
        num_frames=24,
        resolution=(256, 256),
        num_queries=64,
    )
    sample = dataset[0]

    check_model_predictions(sample)

    print("=" * 60)
    print("INVESTIGATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
