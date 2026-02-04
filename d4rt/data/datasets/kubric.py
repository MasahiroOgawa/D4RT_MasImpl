"""Kubric synthetic dataset loader for D4RT."""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..base_dataset import BaseVideoDataset, CameraParameters


class KubricDataset(BaseVideoDataset):
    """
    Kubric synthetic dataset.

    Kubric provides:
    - RGB videos
    - Perfect depth maps
    - Perfect camera parameters
    - Point tracks with occlusion labels
    - Surface normals

    Expected directory structure:
    data_dir/
        train/
            scene_0000/
                rgb/           # RGB frames (00000.png, 00001.png, ...)
                depth/         # Depth maps
                camera.json    # Camera parameters
                metadata.json  # Scene metadata
            scene_0001/
            ...
        val/
            ...
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_frames: int = 48,
        resolution: Tuple[int, int] = (256, 256),
        num_queries: int = 2048,
        transform: Optional[object] = None,
    ):
        """
        Initialize Kubric dataset.

        Args:
            data_dir: Path to Kubric dataset
            split: 'train' or 'val'
            num_frames: Number of frames to load
            resolution: Target resolution (H, W)
            num_queries: Number of queries to sample
            transform: Optional transforms
        """
        super().__init__(data_dir, split, num_frames, resolution, num_queries, transform)

        # Find all scenes
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        # Filter out scenes without RGB frames
        all_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        self.samples = []
        for scene_dir in all_dirs:
            rgb_dir = scene_dir / "rgb"
            if rgb_dir.exists() and len(list(rgb_dir.glob("*.png"))) > 0:
                self.samples.append(scene_dir)

        if len(self.samples) == 0:
            raise ValueError(f"No valid scenes found in {split_dir}")

        print(f"Loaded {len(self.samples)} scenes from Kubric {split} split")

    def _load_video_data(self, idx: int) -> Dict:
        """Load video frames and metadata."""
        scene_dir = self.samples[idx]

        # Load RGB frames
        rgb_dir = scene_dir / "rgb"
        frame_files = sorted(rgb_dir.glob("*.png"))

        # Skip scenes with no frames
        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {rgb_dir}")

        if len(frame_files) < self.num_frames:
            # Repeat frames if not enough
            frame_indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
        else:
            # Sample uniformly
            frame_indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)

        frames = []
        for frame_idx in frame_indices:
            frame_path = frame_files[frame_idx]
            # Load and convert to tensor
            from PIL import Image

            img = Image.open(frame_path).convert("RGB")
            img = img.resize((self.resolution[1], self.resolution[0]))  # (W, H)
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            frames.append(img_tensor)

        frames = torch.stack(frames, dim=0)  # [T, 3, H, W]

        # Load camera parameters
        camera_file = scene_dir / "camera.json"
        if camera_file.exists():
            with open(camera_file, "r") as f:
                camera_data = json.load(f)

            # Parse camera parameters (format depends on Kubric version)
            T = len(frames)
            intrinsics = []
            extrinsics = []

            for t in range(T):
                # Default camera parameters
                H, W = self.resolution
                focal_length = W  # Approximate
                cx, cy = W / 2, H / 2

                K = CameraParameters.create_intrinsics(focal_length, cx, cy)
                intrinsics.append(torch.from_numpy(K))

                # Identity extrinsics (world = camera frame)
                T_mat = CameraParameters.create_extrinsics(np.eye(3), np.zeros(3))
                extrinsics.append(torch.from_numpy(T_mat))

            intrinsics = torch.stack(intrinsics, dim=0)
            extrinsics = torch.stack(extrinsics, dim=0)
        else:
            # Use default camera parameters
            T, _, H, W = frames.shape
            focal_length = W
            cx, cy = W / 2, H / 2

            K = CameraParameters.create_intrinsics(focal_length, cx, cy)
            intrinsics = torch.from_numpy(K).unsqueeze(0).repeat(T, 1, 1)

            T_mat = CameraParameters.create_extrinsics(np.eye(3), np.zeros(3))
            extrinsics = torch.from_numpy(T_mat).unsqueeze(0).repeat(T, 1, 1)

        return {
            "frames": frames,
            "cameras": {
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
            },
            "metadata": {
                "scene_id": scene_dir.name,
                "split": self.split,
            },
        }

    def _load_ground_truth(self, idx: int, video_data: Dict) -> Dict:
        """
        Load ground truth for queries from tracks.npz.

        Uses tracks_3d directly as ground truth instead of computing from depth maps.
        This fixes the bug where depth-computed 3D positions had invalid values
        at background pixels.

        tracks.npz format:
        - query_points: [N, 3] with format [t_src, v, u] (frame index, y, x)
        - tracks_2d: [N, T, 2] with format [u, v] (x, y)
        - tracks_3d: [N, T, 3] with format [X, Y, Z]
        - visibility: [N, T] binary mask
        """
        scene_dir = self.samples[idx]
        frames = video_data["frames"]
        T, C, H, W = frames.shape

        # Load tracks from tracks.npz (CRITICAL: use tracks_3d directly)
        tracks_file = scene_dir / "tracks.npz"
        if not tracks_file.exists():
            raise ValueError(f"tracks.npz not found in {scene_dir}. Cannot load ground truth.")

        tracks_data = np.load(tracks_file)

        # Required fields
        required_fields = ["tracks_3d", "tracks_2d", "visibility", "query_points"]
        missing = [f for f in required_fields if f not in tracks_data]
        if missing:
            raise ValueError(f"tracks.npz must contain {required_fields}, missing: {missing}")

        all_tracks_3d = torch.from_numpy(tracks_data["tracks_3d"]).float()  # [N, T_track, 3]
        all_tracks_2d = torch.from_numpy(
            tracks_data["tracks_2d"]
        ).float()  # [N, T_track, 2] format: [u, v]
        all_visibility = torch.from_numpy(tracks_data["visibility"]).float()  # [N, T_track]
        all_query_points = tracks_data["query_points"]  # [N, 3] format: [t_src, v, u]

        N_tracks, T_track, _ = all_tracks_3d.shape

        # Handle frame count mismatch (tracks.npz may have 24 frames, we want T frames)
        if T_track != T:
            # Sample frames uniformly to match our T
            frame_indices = np.linspace(0, T_track - 1, T, dtype=int)
            all_tracks_3d = all_tracks_3d[:, frame_indices, :]
            all_tracks_2d = all_tracks_2d[:, frame_indices, :]
            all_visibility = all_visibility[:, frame_indices]
            # Also update T_track for consistency
            T_track = T

        # Sample num_queries tracks
        num_queries = min(self.num_queries, N_tracks)
        track_indices = torch.randperm(N_tracks)[:num_queries]

        # Get sampled data
        tracks_3d = all_tracks_3d[track_indices]  # [num_queries, T, 3]
        tracks_2d = all_tracks_2d[track_indices]  # [num_queries, T, 2]
        visibility = all_visibility[track_indices]  # [num_queries, T]
        query_points = all_query_points[track_indices]  # [num_queries, 3] format: [t_src, v, u]

        # Sample target frames for each query
        t_tgt = torch.randint(0, T, (num_queries,))  # [num_queries]

        # Convert query_points format [t_src, v, u] to our query format
        t_src = torch.from_numpy(query_points[:, 0]).long()  # frame index
        v = torch.from_numpy(query_points[:, 1]).float()  # y coordinate (row)
        u = torch.from_numpy(query_points[:, 2]).float()  # x coordinate (col)

        # Handle frame index rescaling if T changed
        if tracks_data["tracks_3d"].shape[1] != T:
            # Rescale t_src to match our frame count
            original_T = tracks_data["tracks_3d"].shape[1]
            t_src = (t_src.float() * (T - 1) / (original_T - 1)).round().long()
            t_src = t_src.clamp(0, T - 1)

        # Normalize coordinates to [0, 1] range
        u_norm = u / (W - 1)  # Normalize u to [0, 1]
        v_norm = v / (H - 1)  # Normalize v to [0, 1]

        # Build queries dict
        # t_cam is the camera reference frame, typically same as t_tgt for point tracking
        queries = {
            "u": u_norm,  # [num_queries] normalized x coordinate
            "v": v_norm,  # [num_queries] normalized y coordinate
            "t_src": t_src.float(),  # [num_queries] source frame
            "t_tgt": t_tgt.float(),  # [num_queries] target frame
            "t_cam": t_tgt.float(),  # [num_queries] camera frame (same as t_tgt)
        }

        # Extract ground truth at target frames
        # Use batch indexing to get xyz, uv, and visibility at t_tgt
        batch_indices = torch.arange(num_queries)
        xyz = tracks_3d[batch_indices, t_tgt]  # [num_queries, 3]
        uv_pixel = tracks_2d[batch_indices, t_tgt]  # [num_queries, 2] in pixel coordinates
        vis = visibility[batch_indices, t_tgt]  # [num_queries]

        # Normalize uv to [0, 1] range for consistency
        uv_norm = torch.zeros_like(uv_pixel)
        uv_norm[:, 0] = uv_pixel[:, 0] / (W - 1)  # u (x)
        uv_norm[:, 1] = uv_pixel[:, 1] / (H - 1)  # v (y)

        targets = {
            "xyz": xyz,  # [num_queries, 3] - direct from tracks_3d
            "uv": uv_norm,  # [num_queries, 2] - normalized 2D position at t_tgt
            "visibility": vis.unsqueeze(-1),  # [num_queries, 1]
        }

        return {
            "queries": queries,
            "targets": targets,
        }

    def _depth_to_3d(
        self,
        depth_maps: torch.Tensor,  # [T, H, W]
        intrinsics: torch.Tensor,  # [T, 3, 3]
        extrinsics: torch.Tensor,  # [T, 4, 4]
    ) -> torch.Tensor:
        """
        Convert depth maps to 3D points.

        Args:
            depth_maps: [T, H, W] depth values
            intrinsics: [T, 3, 3] camera intrinsics
            extrinsics: [T, 4, 4] camera extrinsics

        Returns:
            points_3d: [T, H, W, 3] 3D positions
        """
        T, H, W = depth_maps.shape

        # Create pixel grid
        v, u = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        u = u.to(depth_maps.device)
        v = v.to(depth_maps.device)

        points_3d_list = []

        for t in range(T):
            K = intrinsics[t]
            depth = depth_maps[t]

            # Unproject to camera space
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            x = (u - cx) * depth / fx
            y = (v - cy) * depth / fy
            z = depth

            points_cam = torch.stack([x, y, z], dim=-1)  # [H, W, 3]

            # Transform to world space (if needed)
            # For simplicity, keep in camera space
            points_3d_list.append(points_cam)

        points_3d = torch.stack(points_3d_list, dim=0)  # [T, H, W, 3]

        return points_3d


def create_kubric_dataloaders(
    config: Dict,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create Kubric train and validation dataloaders.

    Args:
        config: Configuration dictionary
        num_workers: Number of dataloader workers

    Returns:
        train_loader, val_loader
    """
    from ..transforms import get_train_transforms, get_val_transforms

    # Training dataset
    train_dataset = KubricDataset(
        data_dir=config["data_dir"],
        split="train",
        num_frames=config.get("num_frames", 48),
        resolution=tuple(config.get("resolution", [256, 256])),
        num_queries=config.get("num_queries", 2048),
        transform=get_train_transforms(config),
    )

    # Validation dataset
    val_dataset = KubricDataset(
        data_dir=config["data_dir"],
        split="val",
        num_frames=config.get("num_frames", 48),
        resolution=tuple(config.get("resolution", [256, 256])),
        num_queries=config.get("num_queries", 1024),
        transform=get_val_transforms(),
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=BaseVideoDataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=BaseVideoDataset.collate_fn,
    )

    return train_loader, val_loader
