"""Long-term prediction for D4RT.

Predict positions beyond video length by varying t_tgt.
The model can extrapolate motion patterns learned from the video.
"""

import torch
from typing import Dict, Optional, Tuple, List
import numpy as np

from ..models.d4rt import D4RT


class LongTermPredictor:
    """
    Predict positions beyond video length by varying t_tgt.

    D4RT uses temporal embeddings that can extrapolate to unseen
    frame indices, enabling prediction of future positions.
    """

    def __init__(
        self,
        model: D4RT,
        device: torch.device = None,
        batch_size: int = 1024,
    ):
        """
        Initialize long-term predictor.

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
    def predict_trajectory(
        self,
        video: torch.Tensor,
        query_point: torch.Tensor,
        source_frame: int,
        target_frames: List[int],
        camera_frame: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict trajectory of a single point at specified target frames.

        Can extrapolate beyond video length if target_frames contains
        indices larger than video length.

        Args:
            video: [B, T, C, H, W] or [T, C, H, W] video tensor
            query_point: [2] or [1, 2] (u, v) query coordinates
            source_frame: Frame where point is defined
            target_frames: List of target frame indices (can exceed video length)
            camera_frame: Camera frame for viewing (default: last video frame)

        Returns:
            Dictionary with:
                - 'positions': [T', 3] predicted 3D positions
                - 'confidence': [T'] prediction confidence
                - 'visibility': [T'] visibility scores
        """
        # Ensure batch dimension
        if video.dim() == 4:
            video = video.unsqueeze(0)

        video = video.to(self.device)
        B, T, C, H, W = video.shape

        if query_point.dim() == 1:
            query_point = query_point.unsqueeze(0)
        query_point = query_point.to(self.device)

        if camera_frame is None:
            camera_frame = T - 1

        # Encode video
        encoder_features = self.model.encode_video(video)

        num_frames = len(target_frames)
        positions = torch.zeros(num_frames, 3, device=self.device)
        confidence = torch.zeros(num_frames, device=self.device)
        visibility = torch.zeros(num_frames, device=self.device)

        # Query each target frame
        for i, t_tgt in enumerate(target_frames):
            queries = {
                'u': query_point[:, 0].unsqueeze(0),  # [1, 1]
                'v': query_point[:, 1].unsqueeze(0),
                't_src': torch.full((1, 1), source_frame,
                                    dtype=torch.long, device=self.device),
                't_tgt': torch.full((1, 1), t_tgt,
                                    dtype=torch.long, device=self.device),
                't_cam': torch.full((1, 1), camera_frame,
                                    dtype=torch.long, device=self.device),
            }

            outputs = self.model.predict_from_queries(
                encoder_features, queries, video
            )

            positions[i] = outputs['xyz'].squeeze()
            visibility[i] = torch.sigmoid(outputs['visibility'].squeeze())

            if 'confidence' in outputs:
                confidence[i] = torch.sigmoid(outputs['confidence'].squeeze())

        return {
            'positions': positions,
            'confidence': confidence,
            'visibility': visibility,
        }

    @torch.no_grad()
    def predict_multiple_trajectories(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor,
        source_frame: int,
        target_frames: List[int],
        camera_frame: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict trajectories for multiple points.

        Args:
            video: [B, T, C, H, W] video tensor
            query_points: [N, 2] (u, v) coordinates for N points
            source_frame: Frame where points are defined
            target_frames: List of target frame indices
            camera_frame: Camera frame for viewing

        Returns:
            Dictionary with:
                - 'positions': [N, T', 3] predicted positions per point
                - 'confidence': [N, T'] confidence per point
                - 'visibility': [N, T'] visibility per point
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)

        video = video.to(self.device)
        query_points = query_points.to(self.device)
        B, T, C, H, W = video.shape

        if camera_frame is None:
            camera_frame = T - 1

        N = query_points.shape[0]
        num_frames = len(target_frames)

        # Encode video
        encoder_features = self.model.encode_video(video)

        positions = torch.zeros(N, num_frames, 3, device=self.device)
        confidence = torch.zeros(N, num_frames, device=self.device)
        visibility = torch.zeros(N, num_frames, device=self.device)

        # Process each target frame
        for t_idx, t_tgt in enumerate(target_frames):
            # Process points in batches
            for i in range(0, N, self.batch_size):
                batch_size_actual = min(self.batch_size, N - i)

                batch_u = query_points[i:i + batch_size_actual, 0]
                batch_v = query_points[i:i + batch_size_actual, 1]

                queries = {
                    'u': batch_u.unsqueeze(0),
                    'v': batch_v.unsqueeze(0),
                    't_src': torch.full((1, batch_size_actual), source_frame,
                                        dtype=torch.long, device=self.device),
                    't_tgt': torch.full((1, batch_size_actual), t_tgt,
                                        dtype=torch.long, device=self.device),
                    't_cam': torch.full((1, batch_size_actual), camera_frame,
                                        dtype=torch.long, device=self.device),
                }

                outputs = self.model.predict_from_queries(
                    encoder_features, queries, video
                )

                positions[i:i + batch_size_actual, t_idx] = outputs['xyz'].squeeze(0)
                visibility[i:i + batch_size_actual, t_idx] = torch.sigmoid(
                    outputs['visibility'].squeeze(0).squeeze(-1)
                )

                if 'confidence' in outputs:
                    confidence[i:i + batch_size_actual, t_idx] = torch.sigmoid(
                        outputs['confidence'].squeeze(0).squeeze(-1)
                    )

        return {
            'positions': positions,
            'confidence': confidence,
            'visibility': visibility,
        }

    @torch.no_grad()
    def predict_future(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor,
        source_frame: int,
        num_future_frames: int,
        step: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future positions beyond video length.

        Args:
            video: [B, T, C, H, W] video tensor
            query_points: [N, 2] query points
            source_frame: Frame where points are defined
            num_future_frames: Number of frames to predict beyond video
            step: Frame step size

        Returns:
            Dictionary with future trajectories
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)

        T = video.shape[1]

        # Create target frames extending beyond video
        target_frames = list(range(T, T + num_future_frames, step))

        return self.predict_multiple_trajectories(
            video,
            query_points,
            source_frame=source_frame,
            target_frames=target_frames,
            camera_frame=T - 1,  # Use last frame as camera
        )

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        video: torch.Tensor,
        query_point: torch.Tensor,
        source_frame: int,
        target_frames: List[int],
        num_samples: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict trajectory with uncertainty estimation via dropout.

        Uses Monte Carlo dropout to estimate prediction uncertainty.

        Args:
            video: [B, T, C, H, W] video tensor
            query_point: [2] query coordinates
            source_frame: Source frame
            target_frames: Target frame indices
            num_samples: Number of dropout samples

        Returns:
            Dictionary with:
                - 'positions_mean': [T', 3] mean predicted positions
                - 'positions_std': [T', 3] standard deviation
                - 'samples': [num_samples, T', 3] individual samples
        """
        # Enable dropout for uncertainty estimation
        self.model.train()

        if video.dim() == 4:
            video = video.unsqueeze(0)

        video = video.to(self.device)

        if query_point.dim() == 1:
            query_point = query_point.unsqueeze(0)
        query_point = query_point.to(self.device)

        encoder_features = self.model.encode_video(video)

        num_frames = len(target_frames)
        samples = torch.zeros(num_samples, num_frames, 3, device=self.device)

        for s in range(num_samples):
            for t_idx, t_tgt in enumerate(target_frames):
                queries = {
                    'u': query_point[:, 0].unsqueeze(0),
                    'v': query_point[:, 1].unsqueeze(0),
                    't_src': torch.full((1, 1), source_frame,
                                        dtype=torch.long, device=self.device),
                    't_tgt': torch.full((1, 1), t_tgt,
                                        dtype=torch.long, device=self.device),
                    't_cam': torch.full((1, 1), t_tgt,
                                        dtype=torch.long, device=self.device),
                }

                outputs = self.model.predict_from_queries(
                    encoder_features, queries, video
                )
                samples[s, t_idx] = outputs['xyz'].squeeze()

        # Return to eval mode
        self.model.eval()

        return {
            'positions_mean': samples.mean(dim=0),
            'positions_std': samples.std(dim=0),
            'samples': samples,
        }
