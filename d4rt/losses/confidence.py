"""Confidence loss for D4RT."""

import torch
import torch.nn as nn


class ConfidenceLoss(nn.Module):
    """
    Confidence-weighted loss with penalty term.

    Implements the confidence loss from D4RT paper:
    L_conf = error * confidence - log(confidence)

    The -log(confidence) term acts as a penalty that prevents the model
    from simply predicting low confidence for all points. This encourages
    the model to be confident when predictions are accurate and uncertain
    when they are not.

    The confidence weighting allows the model to down-weight uncertain
    predictions, effectively learning which predictions are reliable.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize confidence loss.

        Args:
            epsilon: Small constant to prevent log(0)
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        confidence: torch.Tensor,  # [B, N, 1] predicted confidence scores
        error: torch.Tensor,  # [B, N] or [B, N, 1] prediction error
    ) -> torch.Tensor:
        """
        Compute confidence loss.

        Args:
            confidence: Predicted confidence scores in range [0, 1]
            error: Prediction error (e.g., L1 distance between pred and GT)

        Returns:
            loss: Scalar confidence loss value
        """
        # Ensure shapes match
        if error.dim() == 2:
            error = error.unsqueeze(-1)  # [B, N] -> [B, N, 1]

        # Clip confidence to prevent log(0)
        confidence_clipped = torch.clamp(confidence, min=self.epsilon, max=1.0)

        # Confidence-weighted error term
        # High confidence → high weight on error
        # Low confidence → low weight on error
        weighted_error = error * confidence_clipped

        # Confidence penalty term: -log(c)
        # Low confidence → high penalty
        # High confidence → low penalty
        # This prevents the model from always predicting low confidence
        confidence_penalty = -torch.log(confidence_clipped)

        # Total loss: weighted error + penalty
        loss = (weighted_error + confidence_penalty).mean()

        return loss


class SeparateConfidenceLoss(nn.Module):
    """
    Alternative confidence loss that computes error and penalty separately.

    This variant allows for more explicit control over the trade-off
    between accuracy and confidence calibration via a weight parameter.
    """

    def __init__(self, penalty_weight: float = 1.0, epsilon: float = 1e-8):
        """
        Initialize separate confidence loss.

        Args:
            penalty_weight: Weight for confidence penalty term
            epsilon: Small constant to prevent log(0)
        """
        super().__init__()
        self.penalty_weight = penalty_weight
        self.epsilon = epsilon

    def forward(
        self,
        confidence: torch.Tensor,  # [B, N, 1]
        error: torch.Tensor,  # [B, N] or [B, N, 1]
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute confidence loss with separate components.

        Args:
            confidence: Predicted confidence scores
            error: Prediction error

        Returns:
            loss: Total confidence loss
            components: Dictionary with loss components for logging
        """
        # Ensure shapes match
        if error.dim() == 2:
            error = error.unsqueeze(-1)

        # Clip confidence
        confidence_clipped = torch.clamp(confidence, min=self.epsilon, max=1.0)

        # Weighted error
        weighted_error = (error * confidence_clipped).mean()

        # Penalty
        penalty = -torch.log(confidence_clipped).mean()

        # Total loss
        loss = weighted_error + self.penalty_weight * penalty

        components = {
            'weighted_error': weighted_error.item(),
            'confidence_penalty': penalty.item(),
        }

        return loss, components


def compute_prediction_error(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    error_type: str = 'l1',
) -> torch.Tensor:
    """
    Compute prediction error for confidence loss.

    Args:
        pred_xyz: [B, N, 3] predicted 3D positions
        gt_xyz: [B, N, 3] ground truth 3D positions
        error_type: Type of error ('l1', 'l2', or 'huber')

    Returns:
        error: [B, N] per-point error values
    """
    if error_type == 'l1':
        # L1 distance (Manhattan)
        error = torch.abs(pred_xyz - gt_xyz).sum(dim=-1)
    elif error_type == 'l2':
        # L2 distance (Euclidean)
        error = torch.norm(pred_xyz - gt_xyz, p=2, dim=-1)
    elif error_type == 'huber':
        # Huber loss (robust to outliers)
        delta = 1.0
        diff = torch.abs(pred_xyz - gt_xyz)
        huber = torch.where(
            diff < delta,
            0.5 * diff ** 2,
            delta * (diff - 0.5 * delta)
        )
        error = huber.sum(dim=-1)
    else:
        raise ValueError(f"Unknown error type: {error_type}")

    return error
