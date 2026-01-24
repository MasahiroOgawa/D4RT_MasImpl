"""Point tracking inference for D4RT."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union
import numpy as np
from pathlib import Path

from ..models.d4rt import D4RT


class PointTracker:
    """
    Point tracker using D4RT model.

    Tracks sparse points through a video sequence by querying the model
    at different target frames while keeping the source frame fixed.
    """

    def __init__(
        self,
        model: D4RT,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size: int = 256,
    ):
        """
        Initialize point tracker.

        Args:
            model: D4RT model
            device: Device to run inference on
            batch_size: Batch size for query processing
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def track_points(
        self,
        video: torch.Tensor,
        points: torch.Tensor,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track points through video.

        Args:
            video: [T, C, H, W] or [1, T, C, H, W] video tensor
            points: [N, 2] initial points in normalized coordinates (u, v) ∈ [0, 1]
            start_frame: Frame where points are defined
            end_frame: End frame (default: last frame)

        Returns:
            trajectories: [N, T, 3] 3D trajectories in camera coordinates
            visibility: [N, T] visibility scores (higher = more visible)
        """
        # Ensure video has batch dimension
        if video.ndim == 4:
            video = video.unsqueeze(0)  # [1, T, C, H, W]

        T = video.shape[1]
        if end_frame is None:
            end_frame = T - 1

        # Move video to device
        video = video.to(self.device)
        points = points.to(self.device)

        N = points.shape[0]
        num_frames = end_frame - start_frame + 1

        # Encode video once
        encoder_features = self.model.encode_video(video)

        # Track points through all frames
        trajectories = []
        visibility_scores = []

        for t in range(start_frame, end_frame + 1):
            # Create queries for current frame
            queries = {
                'u': points[:, 0],
                'v': points[:, 1],
                't_src': torch.full((N,), start_frame, dtype=torch.long, device=self.device),
                't_tgt': torch.full((N,), t, dtype=torch.long, device=self.device),
                't_cam': torch.full((N,), t, dtype=torch.long, device=self.device),
            }

            # Process in batches if needed
            if N <= self.batch_size:
                outputs = self.model(video, queries, encoder_features=encoder_features)
                xyz = outputs['xyz'].squeeze(0)  # [N, 3]
                vis = torch.sigmoid(outputs['visibility'].squeeze(0))  # [N, 1]
            else:
                xyz_list = []
                vis_list = []
                for i in range(0, N, self.batch_size):
                    batch_queries = {
                        k: v[i:i+self.batch_size] for k, v in queries.items()
                    }
                    batch_outputs = self.model(video, batch_queries, encoder_features=encoder_features)
                    xyz_list.append(batch_outputs['xyz'].squeeze(0))
                    vis_list.append(torch.sigmoid(batch_outputs['visibility'].squeeze(0)))

                xyz = torch.cat(xyz_list, dim=0)
                vis = torch.cat(vis_list, dim=0)

            trajectories.append(xyz)
            visibility_scores.append(vis.squeeze(-1))

        trajectories = torch.stack(trajectories, dim=1)  # [N, T, 3]
        visibility = torch.stack(visibility_scores, dim=1)  # [N, T]

        return trajectories, visibility

    @torch.no_grad()
    def track_points_bidirectional(
        self,
        video: torch.Tensor,
        points: torch.Tensor,
        query_frame: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track points bidirectionally from a query frame.

        Args:
            video: [T, C, H, W] or [1, T, C, H, W] video tensor
            points: [N, 2] query points in normalized coordinates
            query_frame: Frame where points are queried

        Returns:
            trajectories: [N, T, 3] 3D trajectories
            visibility: [N, T] visibility scores
        """
        # Ensure video has batch dimension
        if video.ndim == 4:
            video = video.unsqueeze(0)

        T = video.shape[1]

        # Track forward from query frame
        traj_forward, vis_forward = self.track_points(
            video, points, start_frame=query_frame, end_frame=T-1
        )

        if query_frame > 0:
            # Track backward to beginning
            traj_backward, vis_backward = self.track_points(
                video, points, start_frame=query_frame, end_frame=0
            )

            # Reverse backward trajectories
            traj_backward = traj_backward.flip(dims=[1])  # [N, query_frame+1, 3]
            vis_backward = vis_backward.flip(dims=[1])

            # Concatenate (remove duplicate query frame)
            trajectories = torch.cat([traj_backward[:, :-1], traj_forward], dim=1)
            visibility = torch.cat([vis_backward[:, :-1], vis_forward], dim=1)
        else:
            trajectories = traj_forward
            visibility = vis_forward

        return trajectories, visibility

    def save_trajectories(
        self,
        trajectories: torch.Tensor,
        visibility: torch.Tensor,
        save_path: Union[str, Path],
    ):
        """
        Save trajectories to file.

        Args:
            trajectories: [N, T, 3] trajectories
            visibility: [N, T] visibility scores
            save_path: Path to save file (.npz)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            save_path,
            trajectories=trajectories.cpu().numpy(),
            visibility=visibility.cpu().numpy(),
        )

    @staticmethod
    def load_trajectories(load_path: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load trajectories from file.

        Args:
            load_path: Path to .npz file

        Returns:
            trajectories: [N, T, 3] trajectories
            visibility: [N, T] visibility scores
        """
        data = np.load(load_path)
        trajectories = torch.from_numpy(data['trajectories'])
        visibility = torch.from_numpy(data['visibility'])
        return trajectories, visibility


def track_points_from_checkpoint(
    checkpoint_path: str,
    video: torch.Tensor,
    points: torch.Tensor,
    start_frame: int = 0,
    device: Optional[torch.device] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to track points from a checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        video: [T, C, H, W] video tensor
        points: [N, 2] initial points
        start_frame: Frame where points are defined
        device: Device to use (default: auto-detect)
        **kwargs: Additional arguments for PointTracker

    Returns:
        trajectories: [N, T, 3] 3D trajectories
        visibility: [N, T] visibility scores
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Build model from config
    from ..models import build_d4rt_model
    from omegaconf import OmegaConf

    if 'config' in checkpoint:
        # Build from saved config
        config = OmegaConf.create(checkpoint['config'])
        model = build_d4rt_model(config)
    else:
        raise ValueError("Checkpoint does not contain model config")

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create tracker and track
    tracker = PointTracker(model, device=device, **kwargs)
    trajectories, visibility = tracker.track_points(video, points, start_frame=start_frame)

    return trajectories, visibility


def compute_tracking_metrics(
    pred_trajectories: torch.Tensor,
    gt_trajectories: torch.Tensor,
    gt_visibility: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute tracking metrics.

    Args:
        pred_trajectories: [N, T, 3] predicted trajectories
        gt_trajectories: [N, T, 3] ground truth trajectories
        gt_visibility: [N, T] ground truth visibility (optional)

    Returns:
        metrics: Dictionary of metrics (APE, MAE, etc.)
    """
    # Average Position Error (APE)
    position_error = torch.norm(pred_trajectories - gt_trajectories, dim=-1)  # [N, T]

    if gt_visibility is not None:
        # Only compute error on visible points
        visible_mask = gt_visibility > 0.5
        ape = position_error[visible_mask].mean()
        num_visible = visible_mask.sum()
    else:
        ape = position_error.mean()
        num_visible = position_error.numel()

    # Per-coordinate MAE
    mae_x = torch.abs(pred_trajectories[..., 0] - gt_trajectories[..., 0]).mean()
    mae_y = torch.abs(pred_trajectories[..., 1] - gt_trajectories[..., 1]).mean()
    mae_z = torch.abs(pred_trajectories[..., 2] - gt_trajectories[..., 2]).mean()

    metrics = {
        'ape': ape.item(),
        'mae_x': mae_x.item(),
        'mae_y': mae_y.item(),
        'mae_z': mae_z.item(),
        'num_visible': num_visible.item() if isinstance(num_visible, torch.Tensor) else num_visible,
    }

    return metrics
