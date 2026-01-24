"""Script for depth reconstruction inference using D4RT."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from d4rt.inference import DepthReconstructor
from d4rt.models import build_d4rt_model
from omegaconf import OmegaConf


def load_video(video_path: str, num_frames: int = 48, resolution: int = 256) -> torch.Tensor:
    """Load and preprocess video."""
    cap = cv2.VideoCapture(video_path)

    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        frame = cv2.resize(frame, (resolution, resolution))

        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read video: {video_path}")

    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Convert to tensor [T, H, W, C] -> [T, C, H, W]
    video = torch.from_numpy(np.stack(frames))  # [T, H, W, C]
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

    return video


def visualize_depth(
    depth_map: torch.Tensor,
    save_path: str,
    colormap: str = 'viridis',
):
    """
    Visualize depth map.

    Args:
        depth_map: [H, W] depth map
        save_path: Path to save visualization
        colormap: Matplotlib colormap name
    """
    depth_np = depth_map.cpu().numpy()

    # Normalize to 0-1
    depth_min = depth_np.min()
    depth_max = depth_np.max()
    depth_normalized = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    depth_colored = cmap(depth_normalized)

    # Convert to uint8
    depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)

    # Save
    depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, depth_bgr)

    print(f"Saved depth visualization to {save_path}")
    print(f"  Depth range: [{depth_min:.3f}, {depth_max:.3f}]")


def visualize_depth_overlay(
    rgb_frame: torch.Tensor,
    depth_map: torch.Tensor,
    save_path: str,
    alpha: float = 0.5,
):
    """
    Overlay depth map on RGB frame.

    Args:
        rgb_frame: [C, H, W] RGB frame
        depth_map: [H, W] depth map
        save_path: Path to save
        alpha: Blending factor
    """
    # Convert RGB to numpy
    rgb_np = rgb_frame.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    rgb_np = (rgb_np * 255).astype(np.uint8)

    # Normalize depth
    depth_np = depth_map.cpu().numpy()
    depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

    # Apply colormap
    cmap = plt.get_cmap('plasma')
    depth_colored = cmap(depth_normalized)
    depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)

    # Blend
    blended = cv2.addWeighted(rgb_np, 1 - alpha, depth_colored, alpha, 0)

    # Save
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, blended_bgr)

    print(f"Saved depth overlay to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Depth reconstruction inference with D4RT')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to reconstruct')
    parser.add_argument('--output', type=str, default='depth_map.npz', help='Output file path')
    parser.add_argument('--visualize', type=str, default=None, help='Save depth visualization')
    parser.add_argument('--overlay', type=str, default=None, help='Save RGB+depth overlay')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--num-frames', type=int, default=48, help='Number of frames to load')
    parser.add_argument('--resolution', type=int, default=256, help='Video resolution')
    parser.add_argument('--batch-size', type=int, default=4096, help='Batch size for query processing')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("D4RT Depth Reconstruction Inference")
    print("=" * 80)

    # Load video
    print(f"\n1. Loading video from {args.video}...")
    video = load_video(args.video, num_frames=args.num_frames, resolution=args.resolution)
    print(f"   Video shape: {video.shape}")

    # Load model
    print(f"\n2. Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model config")

    config = OmegaConf.create(checkpoint['config'])
    model = build_d4rt_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Model loaded (step {checkpoint.get('step', 'unknown')})")

    # Create reconstructor
    print(f"\n3. Reconstructing depth for frame {args.frame} on {device}...")
    reconstructor = DepthReconstructor(model, device=device, batch_size=args.batch_size)
    depth_map = reconstructor.reconstruct_depth(video, frame_idx=args.frame)

    print(f"   Depth map shape: {depth_map.shape}")
    print(f"   Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

    # Save results
    print(f"\n4. Saving depth map to {args.output}...")
    reconstructor.save_depth_map(depth_map, args.output, format='npz')
    print("   ✓ Depth map saved")

    # Visualize if requested
    if args.visualize:
        print(f"\n5. Creating depth visualization...")
        visualize_depth(depth_map, args.visualize)

    # Create overlay if requested
    if args.overlay:
        print(f"\n6. Creating RGB+depth overlay...")
        rgb_frame = video[args.frame]  # [C, H, W]
        visualize_depth_overlay(rgb_frame, depth_map, args.overlay)

    print("\n" + "=" * 80)
    print("Depth reconstruction complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
