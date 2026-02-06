"""PointOdyssey dataset loader for D4RT.

PointOdyssey contains synthetic videos with dense 3D point tracks.
Format: anno.npz with trajs_2d, trajs_3d, visibs, intrinsics, extrinsics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PointOdysseyDataset(Dataset):
    """Dataset loader for PointOdyssey.

    Each scene contains:
    - RGB images in rgbs/
    - Depth maps in depths/
    - anno.npz with trajectories and camera params
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_frames: int = 24,
        resolution: Tuple[int, int] = (256, 256),
        num_queries: int = 64,
        frame_stride: int = 1,
    ):
        """Initialize PointOdyssey dataset.

        Args:
            data_dir: Path to pointodyssey directory
            split: 'train', 'val', 'test', or 'sample'
            num_frames: Number of frames to sample per clip
            resolution: (H, W) to resize frames to
            num_queries: Number of query points per sample
            frame_stride: Stride between frames (for longer videos)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.num_queries = num_queries
        self.frame_stride = frame_stride

        # Find all scenes
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        self.scenes = sorted(
            [d for d in split_dir.iterdir() if d.is_dir() and (d / "anno.npz").exists()]
        )

        if len(self.scenes) == 0:
            raise ValueError(f"No scenes found in {split_dir}")

        print(f"Loaded {len(self.scenes)} scenes from PointOdyssey {split} split")

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_dir = self.scenes[idx]

        # Load annotations
        anno = np.load(scene_dir / "anno.npz")
        trajs_3d = anno["trajs_3d"]  # (T, N, 3)
        trajs_2d = anno["trajs_2d"]  # (T, N, 2)
        visibs = anno["visibs"]  # (T, N)
        intrinsics = anno["intrinsics"]  # (T, 3, 3)
        extrinsics = anno["extrinsics"]  # (T, 4, 4)

        total_frames = trajs_3d.shape[0]
        num_points = trajs_3d.shape[1]

        # Sample frame indices
        max_start = total_frames - (self.num_frames * self.frame_stride)
        if max_start <= 0:
            # Video too short, use all frames with smaller stride
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            start_idx = np.random.randint(0, max_start)
            frame_indices = np.arange(
                start_idx, start_idx + self.num_frames * self.frame_stride, self.frame_stride
            )

        # Load RGB frames
        rgb_dir = scene_dir / "rgbs"
        rgb_files = sorted(rgb_dir.glob("*.jpg")) or sorted(rgb_dir.glob("*.png"))

        frames = []
        for fi in frame_indices:
            if fi < len(rgb_files):
                img = Image.open(rgb_files[fi]).convert("RGB")
                img = img.resize((self.resolution[1], self.resolution[0]), Image.BILINEAR)
                frames.append(np.array(img))
            else:
                # Repeat last frame if needed
                frames.append(
                    frames[-1] if frames else np.zeros((*self.resolution, 3), dtype=np.uint8)
                )

        video = np.stack(frames, axis=0)  # (T, H, W, 3)
        video = video.astype(np.float32) / 255.0

        # Get original resolution for coordinate scaling
        if rgb_files:
            orig_img = Image.open(rgb_files[0])
            orig_h, orig_w = orig_img.height, orig_img.width
        else:
            orig_h, orig_w = self.resolution

        scale_h = self.resolution[0] / orig_h
        scale_w = self.resolution[1] / orig_w

        # Sample query points (require finite 2D coordinates and visible at first frame)
        # Check for valid (finite) 2D coordinates across all sampled frames
        sampled_trajs_2d = trajs_2d[frame_indices]  # (T_sampled, N, 2)
        valid_2d = np.isfinite(sampled_trajs_2d).all(
            axis=(0, 2)
        )  # (N,) - valid if finite in all frames
        first_frame_vis = visibs[frame_indices[0]]

        # Points must be visible at first frame AND have valid 2D coords
        valid_indices = np.where(first_frame_vis & valid_2d)[0]

        if len(valid_indices) >= self.num_queries:
            query_indices = np.random.choice(valid_indices, self.num_queries, replace=False)
        elif len(valid_indices) > 0:
            # Repeat valid indices if not enough
            query_indices = np.random.choice(valid_indices, self.num_queries, replace=True)
        else:
            # Fallback: use any points with valid 2D at first frame only
            valid_first = np.where(np.isfinite(trajs_2d[frame_indices[0]]).all(axis=-1))[0]
            if len(valid_first) >= self.num_queries:
                query_indices = np.random.choice(valid_first, self.num_queries, replace=False)
            else:
                query_indices = np.random.choice(valid_first, self.num_queries, replace=True)

        # Extract trajectories for selected queries
        traj_3d = trajs_3d[frame_indices][:, query_indices]  # (T, Q, 3)
        traj_2d = trajs_2d[frame_indices][:, query_indices]  # (T, Q, 2)
        visib = visibs[frame_indices][:, query_indices]  # (T, Q)

        # Scale 2D coordinates to new resolution
        traj_2d_scaled = traj_2d.copy()
        traj_2d_scaled[..., 0] *= scale_w
        traj_2d_scaled[..., 1] *= scale_h

        # Normalize UV to [0, 1]
        uv_normalized = traj_2d_scaled.copy()
        uv_normalized[..., 0] /= self.resolution[1] - 1
        uv_normalized[..., 1] /= self.resolution[0] - 1

        # Get camera parameters (use first frame as reference)
        K = intrinsics[frame_indices[0]]  # (3, 3)
        # Scale intrinsics for new resolution
        K_scaled = K.copy()
        K_scaled[0, :] *= scale_w
        K_scaled[1, :] *= scale_h

        # Sample temporal indices for queries
        t_src = np.zeros(self.num_queries, dtype=np.int64)  # Start from frame 0
        t_tgt = np.random.randint(0, self.num_frames, self.num_queries)
        t_cam = t_tgt.copy()  # Camera frame same as target

        # Get query UV coordinates (at source frame, normalized)
        query_uv = uv_normalized[0]  # (Q, 2) - UV at t_src=0
        query_uv = np.clip(np.nan_to_num(query_uv, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)

        # Build output dict
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # (T, 3, H, W)

        queries = {
            "u": torch.from_numpy(query_uv[:, 0]).float(),  # (Q,) normalized x coordinate
            "v": torch.from_numpy(query_uv[:, 1]).float(),  # (Q,) normalized y coordinate
            "t_src": torch.from_numpy(t_src).float(),  # (Q,) source frame
            "t_tgt": torch.from_numpy(t_tgt).float(),  # (Q,) target frame
            "t_cam": torch.from_numpy(t_cam).float(),  # (Q,) camera frame
        }

        # Get target XYZ at t_tgt for each query
        target_xyz = np.array([traj_3d[t_tgt[i], i] for i in range(self.num_queries)])
        target_uv = np.array([uv_normalized[t_tgt[i], i] for i in range(self.num_queries)])
        target_vis = np.array([visib[t_tgt[i], i] for i in range(self.num_queries)])

        # Handle any remaining inf/nan values (mark as not visible, clamp coords)
        invalid_uv = ~np.isfinite(target_uv).all(axis=-1)
        invalid_xyz = ~np.isfinite(target_xyz).all(axis=-1)
        target_vis[invalid_uv | invalid_xyz] = False
        target_uv = np.clip(np.nan_to_num(target_uv, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        target_xyz = np.nan_to_num(target_xyz, nan=0.0, posinf=10.0, neginf=-10.0)

        targets = {
            "xyz": torch.from_numpy(target_xyz).float(),  # (Q, 3)
            "uv": torch.from_numpy(target_uv).float(),  # (Q, 2)
            "visibility": torch.from_numpy(target_vis).float().unsqueeze(-1),  # (Q, 1)
        }

        cameras = {
            "intrinsics": torch.from_numpy(K_scaled).float(),  # (3, 3)
        }

        return {
            "video": video_tensor,
            "queries": queries,
            "targets": targets,
            "cameras": cameras,
            "metadata": {
                "scene": scene_dir.name,
                "frame_indices": frame_indices.tolist(),
            },
        }


def create_pointodyssey_dataloaders(
    config: Dict,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and val dataloaders for PointOdyssey.

    Args:
        config: Dataset configuration dict
        num_workers: Number of dataloader workers

    Returns:
        train_loader, val_loader
    """
    train_dataset = PointOdysseyDataset(
        data_dir=config.get("data_dir", "data/pointodyssey"),
        split="train",
        num_frames=config.get("num_frames", 24),
        resolution=tuple(config.get("resolution", [256, 256])),
        num_queries=config.get("num_queries", 64),
    )

    val_dataset = PointOdysseyDataset(
        data_dir=config.get("data_dir", "data/pointodyssey"),
        split="val",
        num_frames=config.get("num_frames", 24),
        resolution=tuple(config.get("resolution", [256, 256])),
        num_queries=config.get("num_queries", 64),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
