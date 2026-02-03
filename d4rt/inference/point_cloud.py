"""Point cloud reconstruction for D4RT.

Query all pixels at a fixed camera time to get 3D point cloud
reconstruction of the scene.
"""

import torch
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path

from ..models.d4rt import D4RT


class PointCloudReconstructor:
    """
    Reconstruct 3D point cloud from video using D4RT model.

    Query all pixels at a fixed t_cam to get 3D positions,
    colors, normals, and confidence values.
    """

    def __init__(
        self,
        model: D4RT,
        device: torch.device = None,
        batch_size: int = 4096,
    ):
        """
        Initialize point cloud reconstructor.

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
    def reconstruct(
        self,
        video: torch.Tensor,
        frame_idx: int,
        stride: int = 1,
        min_confidence: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct 3D point cloud for a single frame.

        Args:
            video: [B, T, C, H, W] or [T, C, H, W] video tensor
            frame_idx: Frame index to reconstruct
            stride: Spatial stride for pixel sampling
            min_confidence: Minimum confidence threshold for points

        Returns:
            Dictionary with:
                - 'points': [N, 3] 3D point positions
                - 'colors': [N, 3] RGB colors
                - 'normals': [N, 3] surface normals (if available)
                - 'confidence': [N] confidence values
                - 'visibility': [N] visibility scores
        """
        # Ensure batch dimension
        if video.dim() == 4:
            video = video.unsqueeze(0)

        video = video.to(self.device)
        B, T, C, H, W = video.shape

        # Compute output grid size
        H_out = H // stride
        W_out = W // stride

        # Create coordinate grid
        u_coords = torch.linspace(0, 1, W_out, device=self.device)
        v_coords = torch.linspace(0, 1, H_out, device=self.device)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')

        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
        num_points = len(u_flat)

        # Encode video once
        encoder_features = self.model.encode_video(video)

        # Initialize outputs
        points = torch.zeros(num_points, 3, device=self.device)
        colors = torch.zeros(num_points, 3, device=self.device)
        normals = torch.zeros(num_points, 3, device=self.device)
        confidence = torch.zeros(num_points, device=self.device)
        visibility = torch.zeros(num_points, device=self.device)

        # Extract colors from video frame
        frame = video[0, frame_idx]  # [C, H, W]
        for i in range(num_points):
            # Bilinear sample color at query position
            u, v = u_flat[i], v_flat[i]
            px = int(u * (W - 1))
            py = int(v * (H - 1))
            px = min(max(px, 0), W - 1)
            py = min(max(py, 0), H - 1)
            colors[i] = frame[:, py, px]

        # Process in batches
        for i in range(0, num_points, self.batch_size):
            batch_size_actual = min(self.batch_size, num_points - i)

            batch_u = u_flat[i:i + batch_size_actual]
            batch_v = v_flat[i:i + batch_size_actual]

            # Query: source, target, and camera all at same frame
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

            # Store outputs
            points[i:i + batch_size_actual] = outputs['xyz'].squeeze(0)
            visibility[i:i + batch_size_actual] = torch.sigmoid(
                outputs['visibility'].squeeze(0).squeeze(-1)
            )

            if 'confidence' in outputs:
                confidence[i:i + batch_size_actual] = torch.sigmoid(
                    outputs['confidence'].squeeze(0).squeeze(-1)
                )

            if 'normals' in outputs:
                normals[i:i + batch_size_actual] = outputs['normals'].squeeze(0)

        # Filter by confidence threshold
        if min_confidence > 0:
            mask = confidence >= min_confidence
            points = points[mask]
            colors = colors[mask]
            normals = normals[mask]
            confidence = confidence[mask]
            visibility = visibility[mask]

        return {
            'points': points,
            'colors': colors,
            'normals': normals,
            'confidence': confidence,
            'visibility': visibility,
        }

    @torch.no_grad()
    def reconstruct_sequence(
        self,
        video: torch.Tensor,
        frames: Optional[list] = None,
        stride: int = 1,
        min_confidence: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct point clouds for multiple frames.

        Args:
            video: [B, T, C, H, W] video tensor
            frames: List of frame indices (default: all frames)
            stride: Spatial stride
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary with:
                - 'points': [T', N, 3] points per frame
                - 'colors': [T', N, 3] colors per frame
                - 'normals': [T', N, 3] normals per frame
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)

        T = video.shape[1]
        if frames is None:
            frames = list(range(T))

        all_points = []
        all_colors = []
        all_normals = []

        for frame_idx in frames:
            result = self.reconstruct(
                video, frame_idx, stride=stride,
                min_confidence=min_confidence
            )
            all_points.append(result['points'])
            all_colors.append(result['colors'])
            all_normals.append(result['normals'])

        return {
            'points': torch.stack(all_points, dim=0),
            'colors': torch.stack(all_colors, dim=0),
            'normals': torch.stack(all_normals, dim=0),
        }

    def export_ply(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        output_path: str,
        normals: Optional[torch.Tensor] = None,
    ):
        """
        Export point cloud to PLY format.

        Args:
            points: [N, 3] point positions
            colors: [N, 3] RGB colors (0-1 range)
            output_path: Path to save PLY file
            normals: [N, 3] surface normals (optional)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        points_np = points.cpu().numpy()
        colors_np = (colors.cpu().numpy() * 255).astype(np.uint8)

        N = len(points_np)

        with open(output_path, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

            if normals is not None:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")

            f.write("end_header\n")

            # Data
            if normals is not None:
                normals_np = normals.cpu().numpy()
                for i in range(N):
                    f.write(f"{points_np[i, 0]} {points_np[i, 1]} {points_np[i, 2]} "
                            f"{colors_np[i, 0]} {colors_np[i, 1]} {colors_np[i, 2]} "
                            f"{normals_np[i, 0]} {normals_np[i, 1]} {normals_np[i, 2]}\n")
            else:
                for i in range(N):
                    f.write(f"{points_np[i, 0]} {points_np[i, 1]} {points_np[i, 2]} "
                            f"{colors_np[i, 0]} {colors_np[i, 1]} {colors_np[i, 2]}\n")

    def export_npz(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        output_path: str,
        normals: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ):
        """
        Export point cloud to NPZ format.

        Args:
            points: [N, 3] point positions
            colors: [N, 3] RGB colors
            output_path: Path to save NPZ file
            normals: [N, 3] surface normals (optional)
            confidence: [N] confidence values (optional)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'points': points.cpu().numpy(),
            'colors': colors.cpu().numpy(),
        }

        if normals is not None:
            data['normals'] = normals.cpu().numpy()
        if confidence is not None:
            data['confidence'] = confidence.cpu().numpy()

        np.savez(output_path, **data)
