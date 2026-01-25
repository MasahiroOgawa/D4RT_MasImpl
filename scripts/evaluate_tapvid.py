#!/usr/bin/env python3
"""Evaluate D4RT model on TAP-Vid-3D benchmark

This script evaluates a trained D4RT model on the official TAP-Vid-3D benchmark
using the official evaluation metrics from google-deepmind/tapnet.

Usage:
    # Evaluate on TAPVid-3D validation set
    python scripts/evaluate_tapvid.py \
        --checkpoint checkpoints/checkpoint_step_0005000.pth \
        --model-config configs/model/vit_b_movi.yaml \
        --dataset-dir data/tapvid3d \
        --split val

    # Evaluate on specific subset
    python scripts/evaluate_tapvid.py \
        --checkpoint checkpoints/checkpoint_step_0050000.pth \
        --model-config configs/model/vit_b_movi.yaml \
        --dataset-dir data/tapvid3d/adt \
        --split val

Metrics computed (from official TAP-Vid-3D paper):
- Average Jaccard (AJ): Joint metric for accuracy + occlusion
- APD3D: Average % of points within delta error (3D)
- Occlusion Accuracy (OA): Binary visibility prediction accuracy
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from omegaconf import OmegaConf

# Import D4RT components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from d4rt.models import build_d4rt_model


def load_tapvid3d_sample(npz_path):
    """Load a single TAP-Vid-3D sample from NPZ file

    Args:
        npz_path: Path to .npz file

    Returns:
        dict with keys:
            - video: [T, H, W, 3] RGB frames
            - tracks_xyz: [T, N, 3] 3D point trajectories (meters)
            - visibility: [T, N] binary visibility flags
            - intrinsics: [4] camera intrinsics (fx, fy, cx, cy)
            - query_points: [N, 3] query point coordinates
    """
    data = np.load(npz_path)

    # Decode JPEG frames if necessary
    if 'images_jpeg_bytes' in data:
        import io
        from PIL import Image
        frames = []
        for jpeg_bytes in data['images_jpeg_bytes']:
            img = Image.open(io.BytesIO(jpeg_bytes))
            frames.append(np.array(img))
        video = np.stack(frames)  # [T, H, W, 3]
    else:
        video = data['video']

    return {
        'video': video,
        'tracks_xyz': data['tracks_xyz'],  # [T, N, 3]
        'visibility': data['visibility'],  # [T, N]
        'intrinsics': data['intrinsics'],  # [4] or [T, 4]
        'query_points': data.get('query_points', None),  # [N, 3]
    }


def run_d4rt_inference(model, video, query_points, intrinsics, device):
    """Run D4RT inference on a video

    Args:
        model: D4RT model
        video: [T, H, W, 3] RGB frames (uint8)
        query_points: [N, 2] query points in normalized [0, 1] coordinates
        intrinsics: [4] camera intrinsics (fx, fy, cx, cy)
        device: torch device

    Returns:
        pred_tracks_xyz: [N, T, 3] predicted 3D trajectories
        pred_visibility: [N, T] predicted visibility
    """
    model.eval()

    T, H, W, _ = video.shape
    N = query_points.shape[0]

    # Prepare video tensor
    video_tensor = torch.from_numpy(video).float() / 255.0  # [T, H, W, 3]
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, 3, H, W]
    video_tensor = video_tensor.unsqueeze(0).to(device)  # [1, T, 3, H, W]

    # Prepare intrinsics (D4RT expects [1, T, 3, 3] matrix)
    if isinstance(intrinsics, np.ndarray) and intrinsics.shape == (4,):
        fx, fy, cx, cy = intrinsics
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K = np.tile(K[None, None, :, :], (1, T, 1, 1))  # [1, T, 3, 3]
    else:
        # Already in matrix form
        K = intrinsics
        if K.ndim == 2:
            K = np.tile(K[None, None, :, :], (1, T, 1, 1))

    K_tensor = torch.from_numpy(K).float().to(device)

    # For each query point, track across all frames
    all_pred_xyz = []
    all_pred_vis = []

    with torch.no_grad():
        for n in range(N):
            u, v = query_points[n]  # Normalized [0, 1] coordinates

            # Create queries for this point across all frames
            # Query format: (u, v, t_src, t_tgt, t_cam)
            queries = {
                'u': torch.full((1, T), u, device=device),
                'v': torch.full((1, T), v, device=device),
                't_src': torch.zeros((1, T), device=device, dtype=torch.long),  # Start from frame 0
                't_tgt': torch.arange(T, device=device).unsqueeze(0),  # Track to each frame
                't_cam': torch.arange(T, device=device).unsqueeze(0),  # Camera frame = target frame
            }

            # Run model
            outputs = model(
                video_tensor,
                queries,
                intrinsics=K_tensor,
                extrinsics=None,  # Will be estimated by model
            )

            pred_xyz = outputs['xyz'].cpu().numpy()  # [1, T, 3]
            pred_vis = torch.sigmoid(outputs['visibility']).cpu().numpy()  # [1, T]

            all_pred_xyz.append(pred_xyz[0])  # [T, 3]
            all_pred_vis.append(pred_vis[0])  # [T]

    pred_tracks_xyz = np.stack(all_pred_xyz)  # [N, T, 3]
    pred_visibility = np.stack(all_pred_vis)  # [N, T]

    return pred_tracks_xyz, pred_visibility


def compute_tapvid3d_metrics_official(
    gt_tracks_xyz,
    gt_visibility,
    pred_tracks_xyz,
    pred_visibility,
    intrinsics,
):
    """Compute official TAP-Vid-3D metrics using google-deepmind/tapnet code

    Args:
        gt_tracks_xyz: [T, N, 3] ground truth 3D tracks
        gt_visibility: [T, N] ground truth visibility
        pred_tracks_xyz: [N, T, 3] predicted 3D tracks
        pred_visibility: [N, T] predicted visibility
        intrinsics: [4] camera intrinsics (fx, fy, cx, cy)

    Returns:
        dict of metrics: AJ, APD3D, OA
    """
    try:
        # Try to import official TAP-Vid metrics
        from tapnet.tapvid3d.evaluation import metrics as tapvid3d_metrics

        # Reshape to match expected format: [N, T, 3]
        if gt_tracks_xyz.shape[0] == pred_tracks_xyz.shape[1]:
            # Need to transpose: [T, N, 3] -> [N, T, 3]
            gt_tracks_xyz = np.transpose(gt_tracks_xyz, (1, 0, 2))
            gt_visibility = np.transpose(gt_visibility, (1, 0))

        # Convert visibility to occlusion (invert)
        gt_occluded = 1.0 - gt_visibility
        pred_occluded = 1.0 - pred_visibility

        # Compute official metrics
        metrics = tapvid3d_metrics.compute_tapvid3d_metrics(
            gt_occluded=gt_occluded.astype(bool),
            gt_tracks=gt_tracks_xyz,
            pred_occluded=pred_occluded > 0.5,  # Threshold at 0.5
            pred_tracks=pred_tracks_xyz,
            intrinsics_params=intrinsics,
            scaling='median',  # Paper default
            order='n t',  # [N, T] format
        )

        return {
            'average_jaccard': float(metrics['average_jaccard']),
            'average_pts_within_thresh': float(metrics['average_pts_within_thresh']),
            'occlusion_accuracy': float(metrics['occlusion_accuracy']),
        }

    except ImportError:
        print("⚠️  Official TAP-Vid metrics not available. Install with:")
        print("   pip install \"git+https://github.com/google-deepmind/tapnet.git\"[tapvid3d_eval]")
        print("   Using fallback metrics instead...")
        return compute_tapvid3d_metrics_fallback(
            gt_tracks_xyz, gt_visibility, pred_tracks_xyz, pred_visibility
        )


def compute_tapvid3d_metrics_fallback(
    gt_tracks_xyz,
    gt_visibility,
    pred_tracks_xyz,
    pred_visibility,
):
    """Fallback metric computation if official code not available"""
    # Transpose if needed
    if gt_tracks_xyz.shape[0] == pred_tracks_xyz.shape[1]:
        gt_tracks_xyz = np.transpose(gt_tracks_xyz, (1, 0, 2))
        gt_visibility = np.transpose(gt_visibility, (1, 0))

    # Compute L2 distance
    dist = np.linalg.norm(pred_tracks_xyz - gt_tracks_xyz, axis=-1)  # [N, T]

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


def evaluate_on_dataset(
    model,
    dataset_dir,
    split,
    device,
    max_samples=None,
):
    """Evaluate model on full TAP-Vid-3D dataset

    Args:
        model: D4RT model
        dataset_dir: Path to TAP-Vid-3D dataset directory
        split: 'train', 'val', or 'test'
        device: torch device
        max_samples: Maximum number of samples to evaluate (None = all)

    Returns:
        dict of aggregated metrics
    """
    dataset_path = Path(dataset_dir)
    npz_files = sorted(dataset_path.glob(f"{split}/*.npz"))

    if max_samples:
        npz_files = npz_files[:max_samples]

    print(f"📊 Evaluating on {len(npz_files)} samples from {split} split...")

    all_metrics = []

    for npz_file in tqdm(npz_files, desc="Evaluating"):
        # Load sample
        sample = load_tapvid3d_sample(npz_file)

        # Run inference
        pred_tracks_xyz, pred_visibility = run_d4rt_inference(
            model,
            sample['video'],
            sample['query_points'],
            sample['intrinsics'],
            device,
        )

        # Compute metrics
        metrics = compute_tapvid3d_metrics_official(
            gt_tracks_xyz=sample['tracks_xyz'],
            gt_visibility=sample['visibility'],
            pred_tracks_xyz=pred_tracks_xyz,
            pred_visibility=pred_visibility,
            intrinsics=sample['intrinsics'],
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
        }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Evaluate D4RT on TAP-Vid-3D")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-config', type=str, required=True, help='Path to model config')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to TAP-Vid-3D dataset')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to evaluate (for debugging)')
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
    results = evaluate_on_dataset(
        model,
        args.dataset_dir,
        args.split,
        args.device,
        max_samples=args.max_samples,
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint} (step {checkpoint.get('step', 'unknown')})")
    print()

    for metric_name, stats in results.items():
        print(f"{metric_name}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Median: {stats['median']:.4f}")

    print()
    print("Paper targets (TAPVid-3D):")
    print("  average_jaccard: 0.304")
    print("  average_pts_within_thresh (APD3D): 0.410")
    print("  occlusion_accuracy: 0.875")
    print("="*60)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'checkpoint': args.checkpoint,
            'step': checkpoint.get('step', None),
            'dataset': args.dataset_dir,
            'split': args.split,
            'results': results,
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()
