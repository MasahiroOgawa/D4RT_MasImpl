"""Script for camera pose estimation using D4RT."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
import cv2
from pathlib import Path

from d4rt.inference import CameraPoseEstimator
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


def rotation_matrix_to_euler(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to Euler angles (roll, pitch, yaw)."""
    sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0.0

    return torch.tensor([x, y, z])


def print_pose(R: torch.Tensor, t: torch.Tensor):
    """Print pose information."""
    # Convert rotation to Euler angles
    euler = rotation_matrix_to_euler(R)
    roll, pitch, yaw = torch.rad2deg(euler)

    print("\n  Rotation Matrix:")
    print("    " + str(R[0].cpu().numpy()))
    print("    " + str(R[1].cpu().numpy()))
    print("    " + str(R[2].cpu().numpy()))

    print(f"\n  Euler Angles (degrees):")
    print(f"    Roll:  {roll:.2f}°")
    print(f"    Pitch: {pitch:.2f}°")
    print(f"    Yaw:   {yaw:.2f}°")

    print(f"\n  Translation Vector:")
    print(f"    tx: {t[0]:.4f}")
    print(f"    ty: {t[1]:.4f}")
    print(f"    tz: {t[2]:.4f}")


def visualize_trajectory(
    rotations: torch.Tensor,
    translations: torch.Tensor,
    save_path: str,
):
    """
    Visualize camera trajectory.

    Args:
        rotations: [T, 3, 3] rotation matrices
        translations: [T, 3] translation vectors
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    translations_np = translations.cpu().numpy()

    fig = plt.figure(figsize=(12, 8))

    # 3D trajectory plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(translations_np[:, 0], translations_np[:, 1], translations_np[:, 2],
             marker='o', linestyle='-', markersize=3)
    ax1.scatter(translations_np[0, 0], translations_np[0, 1], translations_np[0, 2],
                c='g', s=100, marker='o', label='Start')
    ax1.scatter(translations_np[-1, 0], translations_np[-1, 1], translations_np[-1, 2],
                c='r', s=100, marker='o', label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Camera Trajectory (3D)')
    ax1.legend()
    ax1.grid(True)

    # Top-down view (X-Y plane)
    ax2 = fig.add_subplot(122)
    ax2.plot(translations_np[:, 0], translations_np[:, 1], marker='o', linestyle='-', markersize=3)
    ax2.scatter(translations_np[0, 0], translations_np[0, 1], c='g', s=100, marker='o', label='Start')
    ax2.scatter(translations_np[-1, 0], translations_np[-1, 1], c='r', s=100, marker='o', label='End')

    # Draw orientation arrows
    for i in range(0, len(translations_np), max(1, len(translations_np) // 10)):
        R = rotations[i].cpu().numpy()
        t = translations_np[i]

        # Forward direction (Z-axis after rotation)
        forward = R @ np.array([0, 0, 0.1])
        ax2.arrow(t[0], t[1], forward[0], forward[1],
                 head_width=0.05, head_length=0.05, fc='b', ec='b', alpha=0.5)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Camera Trajectory (Top View)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved trajectory visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Camera pose estimation with D4RT')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--target-frame', type=int, default=None, help='Target frame (default: estimate full trajectory)')
    parser.add_argument('--reference-frame', type=int, default=0, help='Reference frame')
    parser.add_argument('--output', type=str, default='camera_pose.npz', help='Output file path')
    parser.add_argument('--visualize', type=str, default=None, help='Save trajectory visualization')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--num-frames', type=int, default=48, help='Number of frames to load')
    parser.add_argument('--resolution', type=int, default=256, help='Video resolution')
    parser.add_argument('--num-points', type=int, default=256, help='Number of points for pose estimation')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("D4RT Camera Pose Estimation")
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

    # Create estimator
    estimator = CameraPoseEstimator(model, device=device, num_points=args.num_points)

    if args.target_frame is not None:
        # Estimate pose for single frame
        print(f"\n3. Estimating pose for frame {args.target_frame} (reference: {args.reference_frame})...")
        R, t = estimator.estimate_pose(video, args.target_frame, args.reference_frame)

        print_pose(R, t)

        # Save
        print(f"\n4. Saving pose to {args.output}...")
        np.savez(args.output, rotation=R.cpu().numpy(), translation=t.cpu().numpy())

    else:
        # Estimate full trajectory
        print(f"\n3. Estimating full camera trajectory (reference: {args.reference_frame})...")
        rotations, translations = estimator.estimate_trajectory(video, args.reference_frame)

        print(f"   Trajectory shape: rotations={rotations.shape}, translations={translations.shape}")

        # Print summary statistics
        translation_norms = torch.norm(translations, dim=-1)
        print(f"\n   Translation summary:")
        print(f"     Mean displacement: {translation_norms.mean():.4f}")
        print(f"     Max displacement:  {translation_norms.max():.4f}")
        print(f"     Min displacement:  {translation_norms.min():.4f}")

        # Save
        print(f"\n4. Saving trajectory to {args.output}...")
        estimator.save_trajectory(rotations, translations, args.output)

        # Visualize if requested
        if args.visualize:
            print(f"\n5. Creating trajectory visualization...")
            visualize_trajectory(rotations, translations, args.visualize)

    print("   ✓ Pose estimation saved")

    print("\n" + "=" * 80)
    print("Pose estimation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
