"""Kubric synthetic dataset loader for D4RT."""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

from ..base_dataset import BaseVideoDataset, CameraParameters
from ..query_sampling import QuerySampler, extract_ground_truth_at_queries


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
        split: str = 'train',
        num_frames: int = 48,
        resolution: Tuple[int, int] = (256, 256),
        num_queries: int = 2048,
        transform: Optional[object] = None,
        query_sampler: Optional[QuerySampler] = None,
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
            query_sampler: Query sampling strategy
        """
        super().__init__(data_dir, split, num_frames, resolution, num_queries, transform)

        self.query_sampler = query_sampler or QuerySampler(num_queries=num_queries)

        # Find all scenes
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        self.samples = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        if len(self.samples) == 0:
            raise ValueError(f"No scenes found in {split_dir}")

        print(f"Loaded {len(self.samples)} scenes from Kubric {split} split")

    def _load_video_data(self, idx: int) -> Dict:
        """Load video frames and metadata."""
        scene_dir = self.samples[idx]

        # Load RGB frames
        rgb_dir = scene_dir / 'rgb'
        frame_files = sorted(rgb_dir.glob('*.png'))

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
            img = Image.open(frame_path).convert('RGB')
            img = img.resize((self.resolution[1], self.resolution[0]))  # (W, H)
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            frames.append(img_tensor)

        frames = torch.stack(frames, dim=0)  # [T, 3, H, W]

        # Load camera parameters
        camera_file = scene_dir / 'camera.json'
        if camera_file.exists():
            with open(camera_file, 'r') as f:
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
                T_mat = CameraParameters.create_extrinsics(
                    np.eye(3), np.zeros(3)
                )
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
            'frames': frames,
            'cameras': {
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
            },
            'metadata': {
                'scene_id': scene_dir.name,
                'split': self.split,
            },
        }

    def _load_ground_truth(self, idx: int, video_data: Dict) -> Dict:
        """Load ground truth for queries."""
        scene_dir = self.samples[idx]
        frames = video_data['frames']
        T, C, H, W = frames.shape

        # Load depth maps
        depth_dir = scene_dir / 'depth'
        if depth_dir.exists():
            # Try PNG first, then NPY
            depth_files = sorted(depth_dir.glob('*.png'))
            if not depth_files:
                depth_files = sorted(depth_dir.glob('*.npy'))

            depth_maps = []

            for depth_file in depth_files[:T]:
                if depth_file.suffix == '.png':
                    from PIL import Image
                    depth_img = Image.open(depth_file)
                    depth_array = np.array(depth_img).astype(np.float32) / 1000.0  # Convert to meters
                elif depth_file.suffix == '.npy':
                    depth_array = np.load(depth_file).astype(np.float32)
                else:
                    continue

                depth_array = torch.from_numpy(depth_array)
                # Resize if needed
                if depth_array.shape != (H, W):
                    depth_array = torch.nn.functional.interpolate(
                        depth_array.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode='nearest',
                    ).squeeze()
                depth_maps.append(depth_array)

            if depth_maps:
                depth_maps = torch.stack(depth_maps, dim=0)  # [T, H, W]
                visibility = depth_maps > 0  # Simple visibility mask
            else:
                # No depth files found
                depth_maps = torch.ones(T, H, W) * 5.0  # Assume 5m depth
                visibility = torch.ones(T, H, W, dtype=torch.bool)
        else:
            # Generate synthetic depth if not available
            depth_maps = torch.ones(T, H, W) * 5.0  # Assume 5m depth
            visibility = torch.ones(T, H, W, dtype=torch.bool)

        # Compute 3D positions from depth
        points_3d = self._depth_to_3d(
            depth_maps,
            video_data['cameras']['intrinsics'],
            video_data['cameras']['extrinsics'],
        )  # [T, H, W, 3]

        # Sample queries
        queries = self.query_sampler.sample_queries(
            video_shape=(T, C, H, W),
            visibility_mask=visibility,
            depth_map=depth_maps,
            points_3d=points_3d,
        )

        # Extract ground truth at query locations
        targets = extract_ground_truth_at_queries(
            queries,
            points_3d,
            visibility,
            video_data['cameras']['intrinsics'],
            video_data['cameras']['extrinsics'],
        )

        return {
            'queries': queries,
            'targets': targets,
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
            indexing='ij',
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
        data_dir=config['data_dir'],
        split='train',
        num_frames=config.get('num_frames', 48),
        resolution=tuple(config.get('resolution', [256, 256])),
        num_queries=config.get('num_queries', 2048),
        transform=get_train_transforms(config),
    )

    # Validation dataset
    val_dataset = KubricDataset(
        data_dir=config['data_dir'],
        split='val',
        num_frames=config.get('num_frames', 48),
        resolution=tuple(config.get('resolution', [256, 256])),
        num_queries=config.get('num_queries', 1024),
        transform=get_val_transforms(),
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=BaseVideoDataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=BaseVideoDataset.collate_fn,
    )

    return train_loader, val_loader
