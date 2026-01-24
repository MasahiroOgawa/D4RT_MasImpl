"""Base dataset class for D4RT."""

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple
import numpy as np


class BaseVideoDataset(Dataset):
    """
    Base dataset class for D4RT video datasets.

    Subclasses should implement:
    - __len__: Return number of samples
    - _load_video_data: Load video frames and metadata
    - _load_ground_truth: Load ground truth data
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_frames: int = 48,
        resolution: Tuple[int, int] = (256, 256),
        num_queries: int = 2048,
        transform: Optional[object] = None,
    ):
        """
        Initialize base dataset.

        Args:
            data_dir: Path to dataset directory
            split: Dataset split ('train', 'val', 'test')
            num_frames: Number of frames to sample
            resolution: (H, W) target resolution
            num_queries: Number of queries to sample per video
            transform: Optional transform/augmentation
        """
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.num_queries = num_queries
        self.transform = transform

        # To be populated by subclass
        self.samples = []

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            sample: Dictionary containing:
                - video: [T, 3, H, W] video frames
                - queries: Dict with query components
                - targets: Dict with ground truth
                - cameras: Dict with camera parameters
                - metadata: Dict with additional info
        """
        # Load video data (implemented by subclass)
        video_data = self._load_video_data(idx)

        # Load ground truth (implemented by subclass)
        ground_truth = self._load_ground_truth(idx, video_data)

        # Apply transforms if provided
        if self.transform is not None:
            video_data, ground_truth = self.transform(video_data, ground_truth)

        # Package into sample dict
        sample = {
            'video': video_data['frames'],
            'queries': ground_truth['queries'],
            'targets': ground_truth['targets'],
            'cameras': video_data['cameras'],
            'metadata': video_data['metadata'],
        }

        return sample

    def _load_video_data(self, idx: int) -> Dict:
        """
        Load video frames and metadata.

        Args:
            idx: Sample index

        Returns:
            data: Dictionary with 'frames', 'cameras', 'metadata'
        """
        raise NotImplementedError("Subclass must implement _load_video_data")

    def _load_ground_truth(self, idx: int, video_data: Dict) -> Dict:
        """
        Load ground truth data.

        Args:
            idx: Sample index
            video_data: Video data from _load_video_data

        Returns:
            ground_truth: Dictionary with 'queries' and 'targets'
        """
        raise NotImplementedError("Subclass must implement _load_ground_truth")

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader.

        Args:
            batch: List of samples

        Returns:
            batched: Batched dictionary
        """
        # Stack all tensors
        batched = {
            'video': torch.stack([s['video'] for s in batch]),
            'queries': {
                'u': torch.stack([s['queries']['u'] for s in batch]),
                'v': torch.stack([s['queries']['v'] for s in batch]),
                't_src': torch.stack([s['queries']['t_src'] for s in batch]),
                't_tgt': torch.stack([s['queries']['t_tgt'] for s in batch]),
                't_cam': torch.stack([s['queries']['t_cam'] for s in batch]),
            },
            'targets': {
                'xyz': torch.stack([s['targets']['xyz'] for s in batch]),
                'uv': torch.stack([s['targets']['uv'] for s in batch]),
                'visibility': torch.stack([s['targets']['visibility'] for s in batch]),
            },
            'cameras': {
                'intrinsics': torch.stack([s['cameras']['intrinsics'] for s in batch]),
                'extrinsics': torch.stack([s['cameras']['extrinsics'] for s in batch]),
            },
            'metadata': [s['metadata'] for s in batch],  # Keep as list
        }

        # Add optional targets if present
        if 'normals' in batch[0]['targets']:
            batched['targets']['normals'] = torch.stack(
                [s['targets']['normals'] for s in batch]
            )
        if 'motion' in batch[0]['targets']:
            batched['targets']['motion'] = torch.stack(
                [s['targets']['motion'] for s in batch]
            )

        return batched


class CameraParameters:
    """Helper class for camera parameters."""

    @staticmethod
    def create_intrinsics(
        focal_length: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        """
        Create camera intrinsics matrix.

        Args:
            focal_length: Focal length in pixels
            cx: Principal point x
            cy: Principal point y

        Returns:
            K: [3, 3] intrinsics matrix
        """
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1],
        ], dtype=np.float32)
        return K

    @staticmethod
    def create_extrinsics(
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> np.ndarray:
        """
        Create camera extrinsics matrix.

        Args:
            rotation: [3, 3] rotation matrix
            translation: [3] translation vector

        Returns:
            T: [4, 4] extrinsics matrix
        """
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        return T

    @staticmethod
    def project_3d_to_2d(
        points_3d: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.

        Args:
            points_3d: [N, 3] 3D points in world coordinates
            intrinsics: [3, 3] camera intrinsics
            extrinsics: [4, 4] camera extrinsics

        Returns:
            points_2d: [N, 2] 2D image coordinates
        """
        # Transform to camera coordinates
        points_3d_hom = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
        points_cam = (extrinsics @ points_3d_hom.T).T[:, :3]

        # Project to image
        points_2d_hom = (intrinsics @ points_cam.T).T
        points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]

        return points_2d
