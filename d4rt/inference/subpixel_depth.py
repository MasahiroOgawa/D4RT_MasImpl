"""Sub-pixel depth reconstruction for D4RT (Figure 9).

The model's continuous coordinate space (Fourier encoding of (u,v) in [0,1]^2)
enables querying at arbitrary positions, not tied to the pixel grid.
This allows output resolution independent of input resolution.
"""

import torch
from typing import Dict, Optional, Tuple
import numpy as np

from ..models.d4rt import D4RT


class SubPixelDepthReconstructor:
    """
    Generate sub-pixel accuracy depth maps.

    Paper capability (Figure 9): Query at arbitrary continuous (u,v) coordinates
    to achieve depth resolution beyond input image resolution.
    """

    def __init__(
        self,
        model: D4RT,
        device: torch.device = None,
        batch_size: int = 4096,
    ):
        """
        Initialize sub-pixel depth reconstructor.

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
    def reconstruct_depth(
        self,
        video: torch.Tensor,
        frame_idx: int,
        output_resolution: Tuple[int, int] = (512, 512),
        input_resolution: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate high-resolution depth map using continuous query space.

        The model's Fourier encoding of (u,v) coordinates enables querying
        at arbitrary positions, allowing output resolution higher than input.

        Args:
            video: [B, T, C, H, W] or [T, C, H, W] input video
            frame_idx: Frame to reconstruct
            output_resolution: Target depth map resolution (H_out, W_out)
                              Can exceed input resolution
            input_resolution: Original video resolution (H, W) for reference

        Returns:
            Dictionary with:
                - 'depth': [H_out, W_out] depth values (Z coordinate)
                - 'confidence': [H_out, W_out] prediction confidence
                - 'normals': [H_out, W_out, 3] surface normals (if available)
        """
        # Ensure batch dimension
        if video.dim() == 4:
            video = video.unsqueeze(0)

        video = video.to(self.device)
        B, T, C, H, W = video.shape

        if input_resolution is None:
            input_resolution = (H, W)

        H_out, W_out = output_resolution

        # Create sub-pixel query grid at output resolution
        # Query at normalized (u,v) coordinates in [0,1]
        # Model's continuous coordinate space enables arbitrary resolution
        u_coords = torch.linspace(0, 1, W_out, device=self.device)
        v_coords = torch.linspace(0, 1, H_out, device=self.device)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')

        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
        num_queries = len(u_flat)

        # Encode video once
        encoder_features = self.model.encode_video(video)

        # Initialize outputs
        depth = torch.zeros(num_queries, device=self.device)
        confidence = torch.zeros(num_queries, device=self.device)
        normals = torch.zeros(num_queries, 3, device=self.device)

        # Process in batches
        for i in range(0, num_queries, self.batch_size):
            batch_size_actual = min(self.batch_size, num_queries - i)

            batch_u = u_flat[i:i + batch_size_actual]
            batch_v = v_flat[i:i + batch_size_actual]

            queries = {
                'u': batch_u.unsqueeze(0),
                'v': batch_v.unsqueeze(0),
                't_src': torch.full((1, batch_size_actual), frame_idx,
                                    dtype=torch.long, device=self.device),
                't_tgt': torch.full((1, batch_size_actual), frame_idx,
                                    dtype=torch.long, device=self.device),
                't_cam': torch.full((1, batch_size_actual), frame_idx,
                                    dtype=torch.long, device=self.device),
            }

            outputs = self.model.predict_from_queries(
                encoder_features, queries, video
            )

            # Extract depth (Z coordinate)
            xyz = outputs['xyz'].squeeze(0)  # [batch, 3]
            depth[i:i + batch_size_actual] = xyz[:, 2]

            if 'confidence' in outputs:
                confidence[i:i + batch_size_actual] = torch.sigmoid(
                    outputs['confidence'].squeeze(0).squeeze(-1)
                )

            if 'normals' in outputs:
                normals[i:i + batch_size_actual] = outputs['normals'].squeeze(0)

        # Reshape to output resolution
        result = {
            'depth': depth.view(H_out, W_out),
            'confidence': confidence.view(H_out, W_out),
        }

        if 'normals' in outputs:
            result['normals'] = normals.view(H_out, W_out, 3)

        return result

    @torch.no_grad()
    def reconstruct_depth_sequence(
        self,
        video: torch.Tensor,
        output_resolution: Tuple[int, int] = (512, 512),
        frames: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Generate depth maps for all or selected frames at high resolution.

        Args:
            video: [B, T, C, H, W] video tensor
            output_resolution: Target depth map resolution
            frames: List of frame indices (default: all frames)

        Returns:
            depth_maps: [T', H_out, W_out] depth maps
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)

        T = video.shape[1]
        if frames is None:
            frames = list(range(T))

        depth_maps = []
        for frame_idx in frames:
            result = self.reconstruct_depth(
                video, frame_idx, output_resolution=output_resolution
            )
            depth_maps.append(result['depth'])

        return torch.stack(depth_maps, dim=0)

    @torch.no_grad()
    def reconstruct_depth_multiresolution(
        self,
        video: torch.Tensor,
        frame_idx: int,
        resolutions: list = [(256, 256), (512, 512), (1024, 1024)],
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct depth at multiple resolutions for comparison.

        This demonstrates the sub-pixel capability by generating
        depth maps at progressively higher resolutions.

        Args:
            video: [B, T, C, H, W] video tensor
            frame_idx: Frame to reconstruct
            resolutions: List of (H, W) resolutions

        Returns:
            Dictionary mapping resolution to depth map
        """
        results = {}
        for res in resolutions:
            result = self.reconstruct_depth(
                video, frame_idx, output_resolution=res
            )
            key = f"{res[0]}x{res[1]}"
            results[key] = result['depth']

        return results

    def interpolate_depth(
        self,
        video: torch.Tensor,
        frame_idx: int,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query depth at specific sub-pixel locations.

        Args:
            video: [B, T, C, H, W] video tensor
            frame_idx: Frame to query
            query_points: [N, 2] (u, v) coordinates in [0, 1]

        Returns:
            depths: [N] depth values at query points
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)

        video = video.to(self.device)
        query_points = query_points.to(self.device)

        encoder_features = self.model.encode_video(video)

        N = query_points.shape[0]
        queries = {
            'u': query_points[:, 0].unsqueeze(0),
            'v': query_points[:, 1].unsqueeze(0),
            't_src': torch.full((1, N), frame_idx,
                                dtype=torch.long, device=self.device),
            't_tgt': torch.full((1, N), frame_idx,
                                dtype=torch.long, device=self.device),
            't_cam': torch.full((1, N), frame_idx,
                                dtype=torch.long, device=self.device),
        }

        outputs = self.model.predict_from_queries(
            encoder_features, queries, video
        )

        return outputs['xyz'].squeeze(0)[:, 2]  # [N]
