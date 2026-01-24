"""Camera pose estimation using D4RT."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import numpy as np
from pathlib import Path

from ..models.d4rt import D4RT


class CameraPoseEstimator:
    """
    Camera pose estimation using D4RT model.

    Estimates relative camera pose between frames using corresponding
    3D points and the Umeyama algorithm for rigid alignment.
    """

    def __init__(
        self,
        model: D4RT,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        num_points: int = 256,
    ):
        """
        Initialize camera pose estimator.

        Args:
            model: D4RT model
            device: Device to run inference on
            num_points: Number of sparse points to use for pose estimation
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.num_points = num_points

    @torch.no_grad()
    def estimate_pose(
        self,
        video: torch.Tensor,
        target_frame: int,
        reference_frame: int = 0,
        points: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate camera pose of target frame relative to reference frame.

        Args:
            video: [T, C, H, W] or [1, T, C, H, W] video tensor
            target_frame: Frame to estimate pose for
            reference_frame: Reference frame (default: 0)
            points: [N, 2] points to use (default: sample sparse grid)

        Returns:
            R: [3, 3] rotation matrix
            t: [3] translation vector
        """
        # Ensure video has batch dimension
        if video.ndim == 4:
            video = video.unsqueeze(0)

        video = video.to(self.device)

        # Sample points if not provided
        if points is None:
            points = self._sample_sparse_grid(self.num_points)
        points = points.to(self.device)

        N = points.shape[0]

        # Encode video once
        encoder_features = self.model.encode_video(video)

        # Get 3D points in reference camera frame
        queries_ref = {
            'u': points[:, 0],
            'v': points[:, 1],
            't_src': torch.full((N,), reference_frame, dtype=torch.long, device=self.device),
            't_tgt': torch.full((N,), reference_frame, dtype=torch.long, device=self.device),
            't_cam': torch.full((N,), reference_frame, dtype=torch.long, device=self.device),
        }

        outputs_ref = self.model(video, queries_ref, encoder_features=encoder_features)
        xyz_ref = outputs_ref['xyz'].squeeze(0)  # [N, 3]
        vis_ref = torch.sigmoid(outputs_ref['visibility'].squeeze(0)).squeeze(-1)  # [N]

        # Get same points in target camera frame
        # (t_src stays at reference frame, but t_tgt and t_cam are target frame)
        queries_tgt = {
            'u': points[:, 0],
            'v': points[:, 1],
            't_src': torch.full((N,), reference_frame, dtype=torch.long, device=self.device),
            't_tgt': torch.full((N,), target_frame, dtype=torch.long, device=self.device),
            't_cam': torch.full((N,), target_frame, dtype=torch.long, device=self.device),
        }

        outputs_tgt = self.model(video, queries_tgt, encoder_features=encoder_features)
        xyz_tgt = outputs_tgt['xyz'].squeeze(0)  # [N, 3]
        vis_tgt = torch.sigmoid(outputs_tgt['visibility'].squeeze(0)).squeeze(-1)  # [N]

        # Filter points by visibility
        visibility_mask = (vis_ref > 0.5) & (vis_tgt > 0.5)

        if visibility_mask.sum() < 3:
            # Not enough points for pose estimation
            return torch.eye(3, device=self.device), torch.zeros(3, device=self.device)

        xyz_ref_filtered = xyz_ref[visibility_mask]
        xyz_tgt_filtered = xyz_tgt[visibility_mask]

        # Estimate rigid transformation using Umeyama algorithm
        R, t = self._umeyama_alignment(xyz_ref_filtered, xyz_tgt_filtered)

        return R, t

    @torch.no_grad()
    def estimate_trajectory(
        self,
        video: torch.Tensor,
        reference_frame: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate camera trajectory for all frames relative to reference.

        Args:
            video: [T, C, H, W] or [1, T, C, H, W] video tensor
            reference_frame: Reference frame (default: 0)

        Returns:
            rotations: [T, 3, 3] rotation matrices
            translations: [T, 3] translation vectors
        """
        # Ensure video has batch dimension
        if video.ndim == 4:
            video = video.unsqueeze(0)

        T = video.shape[1]

        rotations = []
        translations = []

        # Sample points once
        points = self._sample_sparse_grid(self.num_points)

        for t in range(T):
            if t == reference_frame:
                # Identity transformation for reference frame
                R = torch.eye(3, device=self.device)
                t_vec = torch.zeros(3, device=self.device)
            else:
                R, t_vec = self.estimate_pose(video, target_frame=t, reference_frame=reference_frame, points=points)

            rotations.append(R)
            translations.append(t_vec)

        rotations = torch.stack(rotations, dim=0)  # [T, 3, 3]
        translations = torch.stack(translations, dim=0)  # [T, 3]

        return rotations, translations

    def _sample_sparse_grid(self, num_points: int) -> torch.Tensor:
        """
        Sample sparse grid of points.

        Args:
            num_points: Number of points to sample

        Returns:
            points: [N, 2] sampled points in normalized coordinates
        """
        # Create a stratified grid
        grid_size = int(np.sqrt(num_points))
        u = torch.linspace(0.1, 0.9, grid_size, device=self.device)
        v = torch.linspace(0.1, 0.9, grid_size, device=self.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')

        points = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=-1)

        # Trim to exact number if needed
        if len(points) > num_points:
            indices = torch.randperm(len(points))[:num_points]
            points = points[indices]

        return points

    def _umeyama_alignment(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate rigid transformation (R, t) using Umeyama algorithm.

        Solves: target = R @ source + t

        Args:
            source: [N, 3] source points
            target: [N, 3] target points

        Returns:
            R: [3, 3] rotation matrix
            t: [3] translation vector
        """
        assert source.shape == target.shape
        N, D = source.shape

        # Center the points
        source_mean = source.mean(dim=0, keepdim=True)  # [1, 3]
        target_mean = target.mean(dim=0, keepdim=True)  # [1, 3]

        source_centered = source - source_mean  # [N, 3]
        target_centered = target - target_mean  # [N, 3]

        # Compute covariance matrix
        H = source_centered.T @ target_centered / N  # [3, 3]

        # SVD
        U, S, Vt = torch.linalg.svd(H)

        # Compute rotation
        R = Vt.T @ U.T  # [3, 3]

        # Handle reflection case (determinant = -1)
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_mean.squeeze(0) - R @ source_mean.squeeze(0)  # [3]

        return R, t

    def save_trajectory(
        self,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        save_path: Union[str, Path],
    ):
        """
        Save camera trajectory to file.

        Args:
            rotations: [T, 3, 3] rotation matrices
            translations: [T, 3] translation vectors
            save_path: Path to save file (.npz)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            save_path,
            rotations=rotations.cpu().numpy(),
            translations=translations.cpu().numpy(),
        )


def estimate_pose_from_checkpoint(
    checkpoint_path: str,
    video: torch.Tensor,
    target_frame: int,
    reference_frame: int = 0,
    device: Optional[torch.device] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to estimate pose from a checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        video: [T, C, H, W] video tensor
        target_frame: Frame to estimate pose for
        reference_frame: Reference frame
        device: Device to use (default: auto-detect)
        **kwargs: Additional arguments for CameraPoseEstimator

    Returns:
        R: [3, 3] rotation matrix
        t: [3] translation vector
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Build model from config
    from ..models import build_d4rt_model
    from omegaconf import OmegaConf

    if 'config' in checkpoint:
        config = OmegaConf.create(checkpoint['config'])
        model = build_d4rt_model(config)
    else:
        raise ValueError("Checkpoint does not contain model config")

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create estimator and estimate
    estimator = CameraPoseEstimator(model, device=device, **kwargs)
    R, t = estimator.estimate_pose(video, target_frame, reference_frame)

    return R, t


def compute_pose_metrics(
    pred_rotations: torch.Tensor,
    pred_translations: torch.Tensor,
    gt_rotations: torch.Tensor,
    gt_translations: torch.Tensor,
) -> dict:
    """
    Compute pose estimation metrics.

    Args:
        pred_rotations: [T, 3, 3] predicted rotation matrices
        pred_translations: [T, 3] predicted translation vectors
        gt_rotations: [T, 3, 3] ground truth rotations
        gt_translations: [T, 3] ground truth translations

    Returns:
        metrics: Dictionary of metrics (rotation error, translation error, ATE)
    """
    # Rotation error (angle in degrees)
    # R_error = R_gt^T @ R_pred
    R_error = torch.bmm(gt_rotations.transpose(1, 2), pred_rotations)  # [T, 3, 3]

    # Trace gives the rotation angle: angle = arccos((trace - 1) / 2)
    traces = torch.diagonal(R_error, dim1=1, dim2=2).sum(dim=1)  # [T]
    angles = torch.acos(torch.clamp((traces - 1) / 2, -1, 1))  # [T]
    rotation_error = torch.rad2deg(angles).mean()

    # Translation error (L2 norm)
    translation_error = torch.norm(pred_translations - gt_translations, dim=-1).mean()

    # Absolute Trajectory Error (ATE)
    # Transform predicted points using both poses and compare
    ate = torch.norm(pred_translations - gt_translations, dim=-1).mean()

    metrics = {
        'rotation_error_deg': rotation_error.item(),
        'translation_error': translation_error.item(),
        'ate': ate.item(),
    }

    return metrics
