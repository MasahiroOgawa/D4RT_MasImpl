"""Surface normal loss for D4RT."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalLoss(nn.Module):
    """
    Surface normal alignment loss using cosine similarity.

    Encourages predicted normals to align with ground truth normals.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_xyz: torch.Tensor,  # [B, N, 3]
        gt_normals: torch.Tensor,  # [B, N, 3]
    ) -> torch.Tensor:
        """
        Compute normal loss.

        Args:
            pred_xyz: Predicted 3D positions (used to estimate normals)
            gt_normals: Ground truth surface normals

        Returns:
            loss: Scalar loss value
        """
        # Estimate normals from predicted point cloud
        pred_normals = self._estimate_normals(pred_xyz)  # [B, N, 3]

        # Normalize both predictions and ground truth
        pred_normals = F.normalize(pred_normals, p=2, dim=-1)
        gt_normals = F.normalize(gt_normals, p=2, dim=-1)

        # Cosine similarity loss (1 - cosine similarity)
        cosine_sim = (pred_normals * gt_normals).sum(dim=-1)  # [B, N]
        loss = (1 - cosine_sim).mean()

        return loss

    def _estimate_normals(
        self,
        points: torch.Tensor,  # [B, N, 3]
    ) -> torch.Tensor:
        """
        Estimate normals from point cloud using local neighborhood.

        For simplicity, use central differences. In practice, you might want
        to use a more sophisticated method like PCA on k-nearest neighbors.

        Args:
            points: 3D points

        Returns:
            normals: [B, N, 3] estimated normals
        """
        B, N, _ = points.shape

        # Simple approximation: Use differences between consecutive points
        # This is a placeholder - a better method would use spatial neighbors
        if N < 3:
            # Not enough points to estimate normals, return zero
            return torch.zeros_like(points)

        # Compute differences
        diffs_forward = points[:, 1:] - points[:, :-1]  # [B, N-1, 3]
        diffs_backward = points[:, :-1] - points[:, 1:]  # [B, N-1, 3]

        # Pad to match original size
        diffs_forward = F.pad(diffs_forward, (0, 0, 0, 1), value=0)  # [B, N, 3]
        diffs_backward = F.pad(diffs_backward, (0, 0, 1, 0), value=0)  # [B, N, 3]

        # Estimate normal as cross product (perpendicular to both differences)
        normals = torch.cross(diffs_forward, diffs_backward, dim=-1)  # [B, N, 3]

        # Normalize
        normals = F.normalize(normals + 1e-8, p=2, dim=-1)

        return normals
