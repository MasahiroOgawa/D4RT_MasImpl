"""Dense all-pixel tracking for D4RT.

Implements Algorithm 1 from the paper: tracking all pixels in a video
using the D4RT model.
"""

import torch
from typing import Dict, Optional, Tuple
import numpy as np

from ..models.d4rt import D4RT


class DensePixelTracker:
    """
    Track all pixels using D4RT model.

    This implements dense tracking where every pixel (at a specified stride)
    is tracked through the video sequence.
    """

    def __init__(
        self,
        model: D4RT,
        device: torch.device = None,
        batch_size: int = 2048,
    ):
        """
        Initialize dense pixel tracker.

        Args:
            model: D4RT model
            device: Device to run inference on
            batch_size: Number of queries per batch
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def track_all_pixels(
        self,
        video: torch.Tensor,
        source_frame: int = 0,
        stride: int = 1,
        target_frames: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Track all pixels from a source frame through the video.

        Args:
            video: [B, T, C, H, W] or [T, C, H, W] video tensor
            source_frame: Frame index where pixels are defined
            stride: Spatial stride for pixel sampling (1 = all pixels)
            target_frames: List of target frame indices (default: all frames)

        Returns:
            Dictionary with:
                - 'trajectories_3d': [H//stride, W//stride, T, 3] 3D trajectories
                - 'trajectories_2d': [H//stride, W//stride, T, 2] 2D projections (if available)
                - 'visibility': [H//stride, W//stride, T] visibility scores
                - 'confidence': [H//stride, W//stride, T] confidence scores
        """
        # Ensure batch dimension
        if video.dim() == 4:
            video = video.unsqueeze(0)

        video = video.to(self.device)
        B, T, C, H, W = video.shape

        if target_frames is None:
            target_frames = list(range(T))

        # Create pixel grid
        H_out = H // stride
        W_out = W // stride

        u_coords = torch.linspace(0, 1, W_out, device=self.device)
        v_coords = torch.linspace(0, 1, H_out, device=self.device)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')

        # Flatten
        u_flat = u_grid.flatten()  # [H_out * W_out]
        v_flat = v_grid.flatten()

        num_pixels = len(u_flat)
        num_frames = len(target_frames)

        # Encode video once
        encoder_features = self.model.encode_video(video)

        # Initialize outputs
        traj_3d = torch.zeros(num_pixels, num_frames, 3, device=self.device)
        traj_2d = torch.zeros(num_pixels, num_frames, 2, device=self.device)
        visibility = torch.zeros(num_pixels, num_frames, device=self.device)
        confidence = torch.zeros(num_pixels, num_frames, device=self.device)

        # Track through all target frames
        for t_idx, t_tgt in enumerate(target_frames):
            # Process in batches
            for i in range(0, num_pixels, self.batch_size):
                batch_size_actual = min(self.batch_size, num_pixels - i)

                batch_u = u_flat[i:i + batch_size_actual]
                batch_v = v_flat[i:i + batch_size_actual]

                queries = {
                    'u': batch_u.unsqueeze(0),  # [1, batch]
                    'v': batch_v.unsqueeze(0),
                    't_src': torch.full((1, batch_size_actual), source_frame,
                                        dtype=torch.long, device=self.device),
                    't_tgt': torch.full((1, batch_size_actual), t_tgt,
                                        dtype=torch.long, device=self.device),
                    't_cam': torch.full((1, batch_size_actual), t_tgt,
                                        dtype=torch.long, device=self.device),
                }

                outputs = self.model.predict_from_queries(
                    encoder_features, queries, video
                )

                # Extract outputs
                traj_3d[i:i + batch_size_actual, t_idx] = outputs['xyz'].squeeze(0)
                if 'uv' in outputs:
                    traj_2d[i:i + batch_size_actual, t_idx] = outputs['uv'].squeeze(0)
                visibility[i:i + batch_size_actual, t_idx] = torch.sigmoid(
                    outputs['visibility'].squeeze(0).squeeze(-1)
                )
                if 'confidence' in outputs:
                    confidence[i:i + batch_size_actual, t_idx] = torch.sigmoid(
                        outputs['confidence'].squeeze(0).squeeze(-1)
                    )

        # Reshape to spatial grid
        result = {
            'trajectories_3d': traj_3d.view(H_out, W_out, num_frames, 3),
            'trajectories_2d': traj_2d.view(H_out, W_out, num_frames, 2),
            'visibility': visibility.view(H_out, W_out, num_frames),
            'confidence': confidence.view(H_out, W_out, num_frames),
        }

        return result

    @torch.no_grad()
    def track_pixels_bidirectional(
        self,
        video: torch.Tensor,
        query_frame: int,
        stride: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Track all pixels bidirectionally from a query frame.

        Pixels are tracked both forward and backward in time from
        the query frame.

        Args:
            video: [B, T, C, H, W] video tensor
            query_frame: Frame where pixels are defined
            stride: Spatial stride for pixel sampling

        Returns:
            Dictionary with trajectories, visibility, and confidence
        """
        # Ensure batch dimension
        if video.dim() == 4:
            video = video.unsqueeze(0)

        T = video.shape[1]

        # Track to all frames with query_frame as source
        all_frames = list(range(T))
        return self.track_all_pixels(video, source_frame=query_frame,
                                     stride=stride, target_frames=all_frames)

    def compute_scene_flow(
        self,
        video: torch.Tensor,
        stride: int = 4,
    ) -> torch.Tensor:
        """
        Compute dense scene flow between consecutive frames.

        Args:
            video: [B, T, C, H, W] video tensor
            stride: Spatial stride

        Returns:
            flow: [T-1, H//stride, W//stride, 3] 3D motion vectors
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)

        T = video.shape[1]

        flows = []
        for t in range(T - 1):
            # Track from frame t to t+1
            result = self.track_all_pixels(
                video,
                source_frame=t,
                stride=stride,
                target_frames=[t, t + 1]
            )
            traj = result['trajectories_3d']  # [H, W, 2, 3]

            # Compute flow as difference
            flow = traj[:, :, 1] - traj[:, :, 0]  # [H, W, 3]
            flows.append(flow)

        return torch.stack(flows, dim=0)  # [T-1, H, W, 3]
