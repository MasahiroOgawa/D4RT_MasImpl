"""Visibility loss for D4RT."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisibilityLoss(nn.Module):
    """
    Binary cross-entropy loss for visibility prediction.

    Predicts whether a point is visible (not occluded) in the source frame.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_visibility: torch.Tensor,  # [B, N, 1] logits
        gt_visibility: torch.Tensor,  # [B, N] binary labels
    ) -> torch.Tensor:
        """
        Compute visibility loss.

        Args:
            pred_visibility: Predicted visibility logits
            gt_visibility: Ground truth visibility (0 or 1)

        Returns:
            loss: Scalar loss value
        """
        # Squeeze last dimension
        pred_vis = pred_visibility.squeeze(-1)  # [B, N]

        # Binary cross-entropy with logits
        loss = F.binary_cross_entropy_with_logits(
            pred_vis,
            gt_visibility.float(),
            reduction='mean',
        )

        return loss
