"""Depth reconstruction inference for D4RT."""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
import numpy as np
from pathlib import Path

from ..models.d4rt import D4RT


class DepthReconstructor:
    """
    Dense depth reconstruction using D4RT model.

    Reconstructs a full depth map for a frame by querying a dense grid
    of image coordinates.
    """

    def __init__(
        self,
        model: D4RT,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size: int = 4096,
    ):
        """
        Initialize depth reconstructor.

        Args:
            model: D4RT model
            device: Device to run inference on
            batch_size: Batch size for query processing (pixels per batch)
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def reconstruct_depth(
        self,
        video: torch.Tensor,
        frame_idx: int,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Reconstruct depth map for a single frame.

        Args:
            video: [T, C, H, W] or [1, T, C, H, W] video tensor
            frame_idx: Frame index to reconstruct
            resolution: Output resolution (H, W). If None, uses video resolution.

        Returns:
            depth_map: [H, W] depth map (z-coordinates in camera space)
        """
        # Ensure video has batch dimension
        if video.ndim == 4:
            video = video.unsqueeze(0)  # [1, T, C, H, W]

        video = video.to(self.device)

        # Get resolution
        if resolution is None:
            _, _, _, H, W = video.shape
        else:
            H, W = resolution

        # Create dense grid of normalized coordinates
        u_coords = torch.linspace(0, 1, W, device=self.device)
        v_coords = torch.linspace(0, 1, H, device=self.device)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')

        # Flatten grid
        u_flat = u_grid.flatten()  # [H*W]
        v_flat = v_grid.flatten()  # [H*W]

        total_pixels = H * W

        # Encode video once
        encoder_features = self.model.encode_video(video)

        # Process grid in batches
        depth_values = []

        for i in range(0, total_pixels, self.batch_size):
            batch_u = u_flat[i:i+self.batch_size]
            batch_v = v_flat[i:i+self.batch_size]
            batch_size_actual = len(batch_u)

            # Create queries (all at same frame)
            queries = {
                'u': batch_u,
                'v': batch_v,
                't_src': torch.full((batch_size_actual,), frame_idx, dtype=torch.long, device=self.device),
                't_tgt': torch.full((batch_size_actual,), frame_idx, dtype=torch.long, device=self.device),
                't_cam': torch.full((batch_size_actual,), frame_idx, dtype=torch.long, device=self.device),
            }

            # Get predictions
            outputs = self.model(video, queries, encoder_features=encoder_features)
            xyz = outputs['xyz'].squeeze(0)  # [batch_size, 3]

            # Extract depth (z-coordinate)
            depth = xyz[:, 2]
            depth_values.append(depth)

        # Concatenate and reshape
        depth_map = torch.cat(depth_values, dim=0).reshape(H, W)

        return depth_map

    @torch.no_grad()
    def reconstruct_depth_with_visibility(
        self,
        video: torch.Tensor,
        frame_idx: int,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct depth map with visibility mask.

        Args:
            video: [T, C, H, W] or [1, T, C, H, W] video tensor
            frame_idx: Frame index to reconstruct
            resolution: Output resolution (H, W)

        Returns:
            depth_map: [H, W] depth map
            visibility: [H, W] visibility scores (0-1)
        """
        # Ensure video has batch dimension
        if video.ndim == 4:
            video = video.unsqueeze(0)

        video = video.to(self.device)

        # Get resolution
        if resolution is None:
            _, _, _, H, W = video.shape
        else:
            H, W = resolution

        # Create dense grid
        u_coords = torch.linspace(0, 1, W, device=self.device)
        v_coords = torch.linspace(0, 1, H, device=self.device)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')

        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
        total_pixels = H * W

        # Encode video once
        encoder_features = self.model.encode_video(video)

        # Process in batches
        depth_values = []
        visibility_values = []

        for i in range(0, total_pixels, self.batch_size):
            batch_u = u_flat[i:i+self.batch_size]
            batch_v = v_flat[i:i+self.batch_size]
            batch_size_actual = len(batch_u)

            queries = {
                'u': batch_u,
                'v': batch_v,
                't_src': torch.full((batch_size_actual,), frame_idx, dtype=torch.long, device=self.device),
                't_tgt': torch.full((batch_size_actual,), frame_idx, dtype=torch.long, device=self.device),
                't_cam': torch.full((batch_size_actual,), frame_idx, dtype=torch.long, device=self.device),
            }

            outputs = self.model(video, queries, encoder_features=encoder_features)
            xyz = outputs['xyz'].squeeze(0)
            vis = torch.sigmoid(outputs['visibility'].squeeze(0))

            depth_values.append(xyz[:, 2])
            visibility_values.append(vis.squeeze(-1))

        depth_map = torch.cat(depth_values, dim=0).reshape(H, W)
        visibility = torch.cat(visibility_values, dim=0).reshape(H, W)

        return depth_map, visibility

    @torch.no_grad()
    def reconstruct_video_depth(
        self,
        video: torch.Tensor,
        frames: Optional[list] = None,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Reconstruct depth maps for multiple frames.

        Args:
            video: [T, C, H, W] or [1, T, C, H, W] video tensor
            frames: List of frame indices to reconstruct (default: all frames)
            resolution: Output resolution (H, W)

        Returns:
            depth_maps: [T', H, W] depth maps for selected frames
        """
        # Ensure video has batch dimension
        if video.ndim == 4:
            video = video.unsqueeze(0)

        T = video.shape[1]

        if frames is None:
            frames = list(range(T))

        depth_maps = []
        for frame_idx in frames:
            depth = self.reconstruct_depth(video, frame_idx, resolution=resolution)
            depth_maps.append(depth)

        return torch.stack(depth_maps, dim=0)

    def save_depth_map(
        self,
        depth_map: torch.Tensor,
        save_path: Union[str, Path],
        format: str = 'npz',
    ):
        """
        Save depth map to file.

        Args:
            depth_map: [H, W] depth map
            save_path: Path to save file
            format: Save format ('npz' or 'png')
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'npz':
            np.savez(save_path, depth=depth_map.cpu().numpy())
        elif format == 'png':
            # Normalize to 0-255 for visualization
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            depth_uint8 = (depth_normalized * 255).cpu().numpy().astype(np.uint8)

            from PIL import Image
            Image.fromarray(depth_uint8).save(save_path)
        else:
            raise ValueError(f"Unsupported format: {format}")


def reconstruct_depth_from_checkpoint(
    checkpoint_path: str,
    video: torch.Tensor,
    frame_idx: int,
    device: Optional[torch.device] = None,
    **kwargs
) -> torch.Tensor:
    """
    Convenience function to reconstruct depth from a checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        video: [T, C, H, W] video tensor
        frame_idx: Frame index to reconstruct
        device: Device to use (default: auto-detect)
        **kwargs: Additional arguments for DepthReconstructor

    Returns:
        depth_map: [H, W] depth map
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

    # Create reconstructor and reconstruct
    reconstructor = DepthReconstructor(model, device=device, **kwargs)
    depth_map = reconstructor.reconstruct_depth(video, frame_idx)

    return depth_map


def compute_depth_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_depth: float = 10.0,
) -> dict:
    """
    Compute depth estimation metrics.

    Args:
        pred_depth: [H, W] predicted depth
        gt_depth: [H, W] ground truth depth
        mask: [H, W] valid mask (optional)
        max_depth: Maximum depth for evaluation

    Returns:
        metrics: Dictionary of metrics (MAE, RMSE, delta thresholds)
    """
    if mask is None:
        mask = (gt_depth > 0) & (gt_depth < max_depth)
    else:
        mask = mask & (gt_depth > 0) & (gt_depth < max_depth)

    pred = pred_depth[mask]
    gt = gt_depth[mask]

    if len(pred) == 0:
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'delta_1.25': float('nan'),
            'delta_1.25^2': float('nan'),
            'delta_1.25^3': float('nan'),
        }

    # Mean Absolute Error
    mae = torch.abs(pred - gt).mean()

    # Root Mean Squared Error
    rmse = torch.sqrt(((pred - gt) ** 2).mean())

    # Delta thresholds
    ratio = torch.max(pred / gt, gt / pred)
    delta_1 = (ratio < 1.25).float().mean()
    delta_2 = (ratio < 1.25 ** 2).float().mean()
    delta_3 = (ratio < 1.25 ** 3).float().mean()

    metrics = {
        'mae': mae.item(),
        'rmse': rmse.item(),
        'delta_1.25': delta_1.item(),
        'delta_1.25^2': delta_2.item(),
        'delta_1.25^3': delta_3.item(),
    }

    return metrics
