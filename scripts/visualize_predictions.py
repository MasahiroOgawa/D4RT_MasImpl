#!/usr/bin/env python3
"""Visualize model predictions vs ground truth for debugging

This script loads a MOVi scene, runs inference, and visualizes:
1. Predicted 3D positions vs ground truth
2. Predicted visibility vs ground truth
3. Individual loss components

Usage:
    python scripts/visualize_predictions.py \
        --checkpoint checkpoints/checkpoint_step_0050000.pth \
        --model-config configs/model/vit_b_movi.yaml \
        --data-dir data/kubric \
        --scene-idx 0
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from omegaconf import OmegaConf

# Import D4RT components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from d4rt.models import build_d4rt_model


def load_scene_with_tracks(scene_dir):
    """Load MOVi scene with tracks"""
    from PIL import Image

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
    tracks_data = np.load(tracks_file)

    # Load camera
    camera_file = scene_dir / 'camera.json'
    with open(camera_file, 'r') as f:
        camera_data = json.load(f)

    intrinsics = np.array(camera_data['intrinsics'][0], dtype=np.float32)

    return {
        'video': video,
        'tracks_3d_gt': tracks_data['tracks_3d'],  # [N, T, 3]
        'visibility_gt': tracks_data['visibility'],  # [N, T]
        'query_points': tracks_data['query_points'],  # [N, 3] (t, y, x)
        'intrinsics': intrinsics,
    }


def run_inference(model, scene_data, device, num_points=10):
    """Run inference on a subset of points"""
    model.eval()

    video = scene_data['video']
    query_points = scene_data['query_points'][:num_points]  # Use first N points

    T, H, W, _ = video.shape
    N = len(query_points)

    # Prepare video tensor
    video_tensor = torch.from_numpy(video).float() / 255.0
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    video_tensor = video_tensor.unsqueeze(0).to(device)  # [1, T, 3, H, W]

    all_pred_xyz = []
    all_pred_vis = []

    with torch.no_grad():
        for n in range(N):
            t_query, y_query, x_query = query_points[n]

            # Normalize to [0, 1]
            u = x_query / W
            v = y_query / H

            # Create queries
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
            pred_vis = torch.sigmoid(outputs['visibility']).cpu().numpy()

            if pred_vis.ndim == 3:
                pred_vis = pred_vis.squeeze(-1)

            all_pred_xyz.append(pred_xyz[0])
            all_pred_vis.append(pred_vis[0])

    return np.stack(all_pred_xyz), np.stack(all_pred_vis)


def visualize_3d_tracks(gt_tracks, pred_tracks, gt_vis, pred_vis, output_path):
    """Visualize 3D tracks"""
    N, T, _ = gt_tracks.shape

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for n in range(min(N, 5)):  # Plot first 5 points
        # Ground truth
        visible_mask = gt_vis[n] > 0.5
        ax1.plot(gt_tracks[n, visible_mask, 0],
                gt_tracks[n, visible_mask, 1],
                gt_tracks[n, visible_mask, 2],
                'o-', alpha=0.6, label=f'GT Point {n}')

        # Predictions
        pred_visible_mask = pred_vis[n] > 0.5
        ax1.plot(pred_tracks[n, pred_visible_mask, 0],
                pred_tracks[n, pred_visible_mask, 1],
                pred_tracks[n, pred_visible_mask, 2],
                'x--', alpha=0.6, label=f'Pred Point {n}')

    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_zlabel('Z (meters)')
    ax1.set_title('3D Trajectories (GT vs Pred)')
    ax1.legend(fontsize=8)

    # 2. X-Y projection
    ax2 = fig.add_subplot(2, 3, 2)
    for n in range(min(N, 5)):
        visible_mask = gt_vis[n] > 0.5
        ax2.plot(gt_tracks[n, visible_mask, 0],
                gt_tracks[n, visible_mask, 1],
                'o-', alpha=0.6, label=f'GT {n}')
        pred_visible_mask = pred_vis[n] > 0.5
        ax2.plot(pred_tracks[n, pred_visible_mask, 0],
                pred_tracks[n, pred_visible_mask, 1],
                'x--', alpha=0.6, label=f'Pred {n}')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title('X-Y Projection')
    ax2.legend(fontsize=8)
    ax2.grid(True)

    # 3. Depth (Z) over time
    ax3 = fig.add_subplot(2, 3, 3)
    for n in range(min(N, 5)):
        ax3.plot(gt_tracks[n, :, 2], 'o-', alpha=0.6, label=f'GT {n}')
        ax3.plot(pred_tracks[n, :, 2], 'x--', alpha=0.6, label=f'Pred {n}')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Depth Z (meters)')
    ax3.set_title('Depth over Time')
    ax3.legend(fontsize=8)
    ax3.grid(True)

    # 4. L1 error per frame
    ax4 = fig.add_subplot(2, 3, 4)
    errors = np.linalg.norm(pred_tracks - gt_tracks, axis=-1)  # [N, T]
    for n in range(min(N, 5)):
        ax4.plot(errors[n], label=f'Point {n}')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('L1 Error (meters)')
    ax4.set_title('3D Position Error over Time')
    ax4.legend(fontsize=8)
    ax4.grid(True)

    # 5. Visibility comparison
    ax5 = fig.add_subplot(2, 3, 5)
    for n in range(min(N, 5)):
        ax5.plot(gt_vis[n], 'o-', alpha=0.6, label=f'GT {n}')
        ax5.plot(pred_vis[n], 'x--', alpha=0.6, label=f'Pred {n}')
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Visibility')
    ax5.set_title('Visibility over Time')
    ax5.legend(fontsize=8)
    ax5.grid(True)
    ax5.axhline(y=0.5, color='r', linestyle=':', label='Threshold')

    # 6. Statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Compute statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)

    # Visibility accuracy
    vis_correct = np.mean((pred_vis > 0.5) == (gt_vis > 0.5))

    # Check if predictions are reasonable
    gt_mean = np.mean(gt_tracks, axis=(0,1))
    gt_std = np.std(gt_tracks, axis=(0,1))
    pred_mean = np.mean(pred_tracks, axis=(0,1))
    pred_std = np.std(pred_tracks, axis=(0,1))

    stats_text = f"""
    Statistics (N={N} points, T={T} frames):

    3D Position Error:
      Mean:   {mean_error:.4f} m
      Median: {median_error:.4f} m
      Max:    {max_error:.4f} m

    Visibility Accuracy: {vis_correct*100:.1f}%

    Ground Truth Stats:
      Mean XYZ: [{gt_mean[0]:.3f}, {gt_mean[1]:.3f}, {gt_mean[2]:.3f}]
      Std XYZ:  [{gt_std[0]:.3f}, {gt_std[1]:.3f}, {gt_std[2]:.3f}]

    Prediction Stats:
      Mean XYZ: [{pred_mean[0]:.3f}, {pred_mean[1]:.3f}, {pred_mean[2]:.3f}]
      Std XYZ:  [{pred_std[0]:.3f}, {pred_std[1]:.3f}, {pred_std[2]:.3f}]

    Potential Issues:
    """

    # Check for common bugs
    if np.allclose(pred_tracks, 0.0, atol=1e-6):
        stats_text += "\n    ⚠️ PREDICTIONS ARE ALL ZEROS!"
    elif pred_std[0] < 0.01 and pred_std[1] < 0.01 and pred_std[2] < 0.01:
        stats_text += "\n    ⚠️ Predictions have very low variance (collapsed)"
    elif mean_error > 10.0:
        stats_text += "\n    ⚠️ Very large prediction errors (>10m)"

    if not np.allclose(pred_mean, gt_mean, atol=1.0):
        stats_text += f"\n    ⚠️ Prediction mean very different from GT mean"
        stats_text += f"\n       Δ = [{pred_mean[0]-gt_mean[0]:.3f}, {pred_mean[1]-gt_mean[1]:.3f}, {pred_mean[2]-gt_mean[2]:.3f}]"

    ax6.text(0.1, 0.9, stats_text, fontsize=10, family='monospace',
             verticalalignment='top', transform=ax6.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")

    return {
        'mean_error': float(mean_error),
        'median_error': float(median_error),
        'visibility_accuracy': float(vis_correct),
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--scene-idx', type=int, default=0)
    parser.add_argument('--num-points', type=int, default=10)
    parser.add_argument('--output', type=str, default='results/debug_visualization.png')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Load model
    print(f"📁 Loading model config from {args.model_config}...")
    config = OmegaConf.load(args.model_config)

    print("🏗️  Building model...")
    model = build_d4rt_model(config)

    print(f"📦 Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    print(f"✓ Model loaded (step {checkpoint.get('step', 'unknown')})")

    # Load scene
    scene_dir = Path(args.data_dir) / 'val' / f'scene_{args.scene_idx:05d}'
    print(f"\n📂 Loading scene from {scene_dir}...")
    scene_data = load_scene_with_tracks(scene_dir)

    print(f"   Video: {scene_data['video'].shape}")
    print(f"   GT tracks: {scene_data['tracks_3d_gt'].shape}")
    print(f"   Query points: {len(scene_data['query_points'])}")

    # Run inference
    print(f"\n🔮 Running inference on {args.num_points} points...")
    pred_tracks, pred_vis = run_inference(model, scene_data, args.device, args.num_points)

    # Get ground truth for same points
    gt_tracks = scene_data['tracks_3d_gt'][:args.num_points]
    gt_vis = scene_data['visibility_gt'][:args.num_points]

    # Visualize
    print("\n📊 Creating visualization...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    stats = visualize_3d_tracks(gt_tracks, pred_tracks, gt_vis, pred_vis, args.output)

    print("\n" + "="*60)
    print("DEBUGGING SUMMARY")
    print("="*60)
    print(f"Mean 3D error: {stats['mean_error']:.4f} meters")
    print(f"Median 3D error: {stats['median_error']:.4f} meters")
    print(f"Visibility accuracy: {stats['visibility_accuracy']*100:.1f}%")
    print("="*60)

    # Check for specific bugs
    if stats['mean_error'] > 5.0:
        print("\n⚠️  WARNING: Very large 3D errors (>5m)")
        print("   Possible causes:")
        print("   1. Model not learning 3D loss properly")
        print("   2. Wrong coordinate frame (camera vs world)")
        print("   3. Scale/normalization issue")

    if np.allclose(pred_tracks, 0.0, atol=1e-6):
        print("\n🔴 CRITICAL: Predictions are all zeros!")
        print("   This means the decoder is not outputting meaningful 3D positions")
        print("   Check:")
        print("   1. Decoder initialization")
        print("   2. 3D loss gradients")
        print("   3. Model forward pass")


if __name__ == '__main__':
    main()
