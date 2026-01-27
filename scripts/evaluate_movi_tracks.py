#!/usr/bin/env python3
"""Evaluate D4RT model on MOVi validation set with TAP-Vid metrics

This script evaluates the trained D4RT model on MOVi validation scenes
with generated point tracks, using official TAP-Vid-3D evaluation metrics.

Usage:
    # Evaluate on MOVi validation set
    python scripts/evaluate_movi_tracks.py \
        --checkpoint checkpoints/checkpoint_step_0050000.pth \
        --model-config configs/model/vit_b_movi.yaml \
        --data-dir data/kubric \
        --split val \
        --max-samples 20

    # Evaluate with visualization
    python scripts/evaluate_movi_tracks.py \
        --checkpoint checkpoints/checkpoint_step_0050000.pth \
        --model-config configs/model/vit_b_movi.yaml \
        --data-dir data/kubric \
        --split val \
        --visualize \
        --output results/movi_eval_50k.json
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from omegaconf import OmegaConf
from PIL import Image

# Import D4RT components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from d4rt.models import build_d4rt_model


def load_movi_scene_with_tracks(scene_dir):
    """Load a MOVi scene with generated point tracks

    Args:
        scene_dir: Path to scene directory

    Returns:
        dict with keys:
            - video: [T, H, W, 3] RGB frames (uint8)
            - tracks_3d_gt: [N, T, 3] ground truth 3D tracks
            - visibility_gt: [N, T] ground truth visibility
            - query_points: [N, 3] query points (t, y, x)
            - intrinsics: [3, 3] camera intrinsics
    """
    scene_dir = Path(scene_dir)

    # Load RGB frames
    rgb_dir = scene_dir / 'rgb'
    frame_files = sorted(rgb_dir.glob('*.png'))

    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert('RGB')
        frames.append(np.array(img))

    video = np.stack(frames)  # [T, H, W, 3]

    # Load tracks
    tracks_file = scene_dir / 'tracks.npz'
    if not tracks_file.exists():
        raise FileNotFoundError(f"Tracks file not found: {tracks_file}")

    tracks_data = np.load(tracks_file)

    # Load camera parameters
    camera_file = scene_dir / 'camera.json'
    with open(camera_file, 'r') as f:
        camera_data = json.load(f)

    intrinsics = np.array(camera_data['intrinsics'][0], dtype=np.float32)  # [3, 3]

    return {
        'video': video,
        'tracks_3d_gt': tracks_data['tracks_3d'],  # [N, T, 3]
        'visibility_gt': tracks_data['visibility'],  # [N, T]
        'query_points': tracks_data['query_points'],  # [N, 3] (t, y, x)
        'intrinsics': intrinsics,  # [3, 3]
    }


def run_d4rt_on_movi_scene(model, scene_data, device):
    """Run D4RT inference on a MOVi scene

    Args:
        model: D4RT model
        scene_data: Scene data dict from load_movi_scene_with_tracks()
        device: torch device

    Returns:
        pred_tracks_3d: [N, T, 3] predicted 3D trajectories
        pred_visibility: [N, T] predicted visibility
    """
    model.eval()

    video = scene_data['video']  # [T, H, W, 3] uint8
    query_points = scene_data['query_points']  # [N, 3] (t, y, x)
    intrinsics = scene_data['intrinsics']  # [3, 3]

    T, H, W, _ = video.shape
    N = len(query_points)

    # Prepare video tensor
    video_tensor = torch.from_numpy(video).float() / 255.0  # [T, H, W, 3]
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, 3, H, W]
    video_tensor = video_tensor.unsqueeze(0).to(device)  # [1, T, 3, H, W]

    # Prepare intrinsics
    K = np.tile(intrinsics[None, None, :, :], (1, T, 1, 1))  # [1, T, 3, 3]
    K_tensor = torch.from_numpy(K).float().to(device)

    # For each query point, track across all frames
    all_pred_xyz = []
    all_pred_vis = []

    with torch.no_grad():
        for n in range(N):
            t_query, y_query, x_query = query_points[n]

            # Normalize to [0, 1]
            u = x_query / W
            v = y_query / H

            # Create queries for this point across all frames
            queries = {
                'u': torch.full((1, T), u, device=device),
                'v': torch.full((1, T), v, device=device),
                't_src': torch.full((1, T), t_query, device=device, dtype=torch.long),
                't_tgt': torch.arange(T, device=device).unsqueeze(0),
                't_cam': torch.arange(T, device=device).unsqueeze(0),
            }

            # Run model
            outputs = model(video_tensor, queries)

            pred_xyz = outputs['xyz'].cpu().numpy()  # [1, T, 3]
            pred_vis = torch.sigmoid(outputs['visibility']).cpu().numpy()  # [1, T, 1] or [1, T]

            # Squeeze if visibility has extra dimension
            if pred_vis.ndim == 3:
                pred_vis = pred_vis.squeeze(-1)  # [1, T]

            all_pred_xyz.append(pred_xyz[0])  # [T, 3]
            all_pred_vis.append(pred_vis[0])  # [T]

    pred_tracks_3d = np.stack(all_pred_xyz)  # [N, T, 3]
    pred_visibility = np.stack(all_pred_vis)  # [N, T]

    return pred_tracks_3d, pred_visibility


def compute_tapvid_metrics(
    gt_tracks_3d,  # [N, T, 3]
    gt_visibility,  # [N, T]
    pred_tracks_3d,  # [N, T, 3]
    pred_visibility,  # [N, T]
    intrinsics=None,  # [4] camera intrinsics (fx, fy, cx, cy)
):
    """Compute TAP-Vid-3D metrics

    Args:
        gt_tracks_3d: Ground truth 3D tracks [N, T, 3]
        gt_visibility: Ground truth visibility [N, T]
        pred_tracks_3d: Predicted 3D tracks [N, T, 3]
        pred_visibility: Predicted visibility [N, T]
        intrinsics: Camera intrinsics (fx, fy, cx, cy)

    Returns:
        dict of metrics: AJ, APD3D, OA
    """
    try:
        # Try official metrics
        from tapnet.tapvid3d.evaluation import metrics as tapvid3d_metrics

        # Convert visibility to occlusion (invert)
        gt_occluded = 1.0 - gt_visibility
        pred_occluded = 1.0 - pred_visibility

        # Default intrinsics if not provided (256x256 MOVi)
        if intrinsics is None:
            intrinsics = np.array([280.0, 280.0, 128.0, 128.0], dtype=np.float32)

        # Compute official metrics
        metrics = tapvid3d_metrics.compute_tapvid3d_metrics(
            gt_occluded=gt_occluded.astype(bool),
            gt_tracks=gt_tracks_3d,
            pred_occluded=pred_occluded > 0.5,
            pred_tracks=pred_tracks_3d,
            intrinsics_params=intrinsics,
        )

        # Extract values - may be scalars or arrays
        def extract_scalar(value):
            if hasattr(value, 'item'):
                return float(value.item())
            elif hasattr(value, '__len__'):
                return float(np.mean(value))
            else:
                return float(value)

        return {
            'average_jaccard': extract_scalar(metrics['average_jaccard']),
            'average_pts_within_thresh': extract_scalar(metrics['average_pts_within_thresh']),
            'occlusion_accuracy': extract_scalar(metrics['occlusion_accuracy']),
        }

    except (ImportError, KeyError) as e:
        print(f"⚠️  Official TAP-Vid metrics not available: {e}")
        print("   Using fallback metrics...")
        return compute_tapvid_metrics_fallback(
            gt_tracks_3d, gt_visibility, pred_tracks_3d, pred_visibility
        )


def compute_tapvid_metrics_fallback(
    gt_tracks_3d,
    gt_visibility,
    pred_tracks_3d,
    pred_visibility,
):
    """Fallback metric computation"""
    # Compute L2 distance
    dist = np.linalg.norm(pred_tracks_3d - gt_tracks_3d, axis=-1)  # [N, T]

    # APD3D: Average % within threshold (using 0.05m as threshold)
    threshold = 0.05  # meters
    within_thresh = (dist < threshold) & (gt_visibility > 0.5)
    apd3d = np.mean(within_thresh)

    # Occlusion Accuracy
    pred_occ = pred_visibility < 0.5
    gt_occ = gt_visibility < 0.5
    oa = np.mean(pred_occ == gt_occ)

    # Jaccard-like metric (intersection over union)
    correct = within_thresh
    pred_visible = pred_visibility > 0.5
    gt_visible = gt_visibility > 0.5
    intersection = correct & pred_visible & gt_visible
    union = pred_visible | gt_visible
    aj = np.sum(intersection) / (np.sum(union) + 1e-8)

    return {
        'average_jaccard': float(aj),
        'average_pts_within_thresh': float(apd3d),
        'occlusion_accuracy': float(oa),
    }


def evaluate_on_movi(
    model,
    data_dir,
    split,
    device,
    max_samples=None,
):
    """Evaluate model on MOVi validation set

    Args:
        model: D4RT model
        data_dir: Path to MOVi data directory
        split: Dataset split (train/val)
        device: torch device
        max_samples: Maximum number of samples to evaluate

    Returns:
        dict of aggregated metrics
    """
    split_dir = Path(data_dir) / split
    scene_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

    if max_samples:
        scene_dirs = scene_dirs[:max_samples]

    print(f"📊 Evaluating on {len(scene_dirs)} scenes from {split} split...")

    all_metrics = []

    for scene_dir in tqdm(scene_dirs, desc="Evaluating"):
        # Load scene
        try:
            scene_data = load_movi_scene_with_tracks(scene_dir)
        except FileNotFoundError as e:
            print(f"\n⚠️  Skipping {scene_dir.name}: {e}")
            continue

        # Run inference
        pred_tracks_3d, pred_visibility = run_d4rt_on_movi_scene(
            model, scene_data, device
        )

        # Compute metrics
        # Extract intrinsics to (fx, fy, cx, cy) format
        K = scene_data['intrinsics']
        intrinsics_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float32)

        metrics = compute_tapvid_metrics(
            gt_tracks_3d=scene_data['tracks_3d_gt'],
            gt_visibility=scene_data['visibility_gt'],
            pred_tracks_3d=pred_tracks_3d,
            pred_visibility=pred_visibility,
            intrinsics=intrinsics_params,
        )

        all_metrics.append(metrics)

    # Aggregate metrics
    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Evaluate D4RT on MOVi with TAP-Vid metrics")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-config', type=str, required=True, help='Path to model config')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to MOVi data directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to evaluate')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    # Load model config
    print(f"📁 Loading model config from {args.model_config}...")
    config = OmegaConf.load(args.model_config)

    # Build model
    print("🏗️  Building model...")
    model = build_d4rt_model(config)

    # Load checkpoint
    print(f"📦 Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    print(f"✓ Model loaded (step {checkpoint.get('step', 'unknown')})")

    # Evaluate
    results = evaluate_on_movi(
        model,
        args.data_dir,
        args.split,
        args.device,
        max_samples=args.max_samples,
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS ON MOVi")
    print("="*60)
    print(f"Dataset: {args.data_dir}/{args.split}")
    print(f"Checkpoint: {args.checkpoint} (step {checkpoint.get('step', 'unknown')})")
    print()

    for metric_name, stats in results.items():
        print(f"{metric_name}:")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Std:    {stats['std']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")

    print()
    print("Paper targets (TAPVid-3D on real datasets):")
    print("  average_jaccard: 0.304")
    print("  average_pts_within_thresh (APD3D): 0.410")
    print("  occlusion_accuracy: 0.875")
    print()
    print("Note: MOVi is synthetic data, so direct comparison may differ.")
    print("="*60)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'checkpoint': args.checkpoint,
            'step': checkpoint.get('step', None),
            'dataset': f"{args.data_dir}/{args.split}",
            'results': results,
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()
