"""Script for point tracking inference using D4RT."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
import cv2
from pathlib import Path

from d4rt.inference import PointTracker
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


def parse_points(points_str: str) -> torch.Tensor:
    """Parse points from string format."""
    # Expected format: "[[u1, v1], [u2, v2], ...]"
    import ast
    points_list = ast.literal_eval(points_str)
    return torch.tensor(points_list, dtype=torch.float32)


def visualize_tracks(
    video: torch.Tensor,
    trajectories: torch.Tensor,
    visibility: torch.Tensor,
    save_path: str,
):
    """
    Visualize point tracks overlaid on video.

    Args:
        video: [T, C, H, W] video tensor
        trajectories: [N, T, 3] 3D trajectories
        visibility: [N, T] visibility scores
        save_path: Path to save output video
    """
    import matplotlib.pyplot as plt

    T, C, H, W = video.shape
    N = trajectories.shape[0]

    # Create color map for points
    colors = plt.cm.rainbow(np.linspace(0, 1, N))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 10.0, (W, H))

    for t in range(T):
        # Get frame
        frame = video[t].permute(1, 2, 0).numpy()  # [H, W, C]
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw tracks
        for n in range(N):
            if visibility[n, t] > 0.5:
                # Project 3D point to 2D (for visualization, assume intrinsics are identity scaled)
                # In practice, you'd use actual camera intrinsics
                xyz = trajectories[n, t]
                u = xyz[0] / xyz[2] if xyz[2] != 0 else 0
                v = xyz[1] / xyz[2] if xyz[2] != 0 else 0

                # Convert to pixel coordinates (simple approximation)
                x = int((u + 1) * W / 2)
                y = int((v + 1) * H / 2)

                # Clamp to image bounds
                x = max(0, min(W - 1, x))
                y = max(0, min(H - 1, y))

                # Draw point
                color = (int(colors[n][2] * 255), int(colors[n][1] * 255), int(colors[n][0] * 255))
                cv2.circle(frame, (x, y), 5, color, -1)

                # Draw trail
                if t > 0 and visibility[n, t-1] > 0.5:
                    xyz_prev = trajectories[n, t-1]
                    u_prev = xyz_prev[0] / xyz_prev[2] if xyz_prev[2] != 0 else 0
                    v_prev = xyz_prev[1] / xyz_prev[2] if xyz_prev[2] != 0 else 0
                    x_prev = int((u_prev + 1) * W / 2)
                    y_prev = int((v_prev + 1) * H / 2)
                    x_prev = max(0, min(W - 1, x_prev))
                    y_prev = max(0, min(H - 1, y_prev))
                    cv2.line(frame, (x_prev, y_prev), (x, y), color, 2)

        out.write(frame)

    out.release()
    print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Point tracking inference with D4RT')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-config', type=str, required=True, help='Path to model config file')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--points', type=str, required=True,
                       help='Points to track in format "[[u1,v1],[u2,v2],...]" (normalized 0-1)')
    parser.add_argument('--start-frame', type=int, default=0, help='Start frame for tracking')
    parser.add_argument('--output', type=str, default='tracking_results.npz', help='Output file path')
    parser.add_argument('--visualize', type=str, default=None, help='Save visualization video')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--num-frames', type=int, default=48, help='Number of frames to load')
    parser.add_argument('--resolution', type=int, default=256, help='Video resolution')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("D4RT Point Tracking Inference")
    print("=" * 80)

    # Load video
    print(f"\n1. Loading video from {args.video}...")
    video = load_video(args.video, num_frames=args.num_frames, resolution=args.resolution)
    print(f"   Video shape: {video.shape}")

    # Parse points
    print(f"\n2. Parsing points: {args.points}")
    points = parse_points(args.points)
    print(f"   Tracking {len(points)} points")

    # Load model config
    print(f"\n3. Loading model config from {args.model_config}...")
    config = OmegaConf.load(args.model_config)
    print(f"   Config loaded")

    # Build model
    print(f"   Building model...")
    model = build_d4rt_model(config)

    # Load checkpoint
    print(f"   Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Model loaded (step {checkpoint.get('step', 'unknown')})")

    # Create tracker
    print(f"\n4. Running tracking on {device}...")
    tracker = PointTracker(model, device=device)
    trajectories, visibility = tracker.track_points(video, points, start_frame=args.start_frame)

    print(f"   Trajectories shape: {trajectories.shape}")
    print(f"   Visibility shape: {visibility.shape}")

    # Save results
    print(f"\n5. Saving results to {args.output}...")
    tracker.save_trajectories(trajectories, visibility, args.output)
    print("   ✓ Results saved")

    # Visualize if requested
    if args.visualize:
        print(f"\n6. Creating visualization...")
        visualize_tracks(video, trajectories, visibility, args.visualize)

    print("\n" + "=" * 80)
    print("Tracking complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
