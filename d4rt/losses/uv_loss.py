"""UV (2D coordinate) loss for D4RT."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UVLoss(nn.Module):
    """
    L2 loss on 2D image coordinates (UV).

    This provides 2D supervision for predicted point positions,
    complementing the 3D supervision from L1_3D loss.
    """

    def __init__(self, loss_type: str = 'l2'):
        """
        Initialize UV loss.

        Args:
            loss_type: Type of loss ('l1', 'l2', or 'smooth_l1')
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        pred_uv: torch.Tensor,  # [B, N, 2] predicted UV coordinates
        gt_uv: torch.Tensor,    # [B, N, 2] ground truth UV coordinates
        visibility: torch.Tensor = None,  # [B, N] optional visibility mask
    ) -> torch.Tensor:
        """
        Compute UV loss.

        Args:
            pred_uv: Predicted UV coordinates in [0, 1]
            gt_uv: Ground truth UV coordinates in [0, 1]
            visibility: Optional visibility mask (loss only computed for visible points)

        Returns:
            loss: Scalar loss value
        """
        if self.loss_type == 'l1':
            loss = torch.abs(pred_uv - gt_uv)
        elif self.loss_type == 'l2':
            loss = (pred_uv - gt_uv) ** 2
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred_uv, gt_uv, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Sum over UV dimensions
        loss = loss.sum(dim=-1)  # [B, N]

        # Apply visibility mask if provided
        if visibility is not None:
            if visibility.dim() == 3:
                visibility = visibility.squeeze(-1)  # [B, N, 1] -> [B, N]
            loss = loss * visibility
            # Average over visible points only
            num_visible = visibility.sum().clamp(min=1.0)
            loss = loss.sum() / num_visible
        else:
            loss = loss.mean()

        return loss
