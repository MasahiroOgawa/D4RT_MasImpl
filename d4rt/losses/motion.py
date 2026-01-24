"""Motion consistency loss for D4RT."""

import torch
import torch.nn as nn


class MotionLoss(nn.Module):
    """
    Temporal motion consistency loss.

    Encourages smooth motion between consecutive frames.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_xyz: torch.Tensor,  # [B, N, 3]
        gt_motion: torch.Tensor,  # [B, N, 3] or None
        t_tgt: torch.Tensor,  # [B, N] target frame indices
    ) -> torch.Tensor:
        """
        Compute motion loss.

        Args:
            pred_xyz: Predicted 3D positions
            gt_motion: Ground truth motion vectors (optional)
            t_tgt: Target frame indices

        Returns:
            loss: Scalar loss value
        """
        if gt_motion is not None:
            # If ground truth motion is available, use it directly
            # Estimate motion from predictions
            pred_motion = self._estimate_motion_from_positions(
                pred_xyz, t_tgt
            )  # [B, N', 3]

            # Match with ground truth
            # This is simplified - in practice you'd need to match indices properly
            loss = torch.abs(pred_motion - gt_motion[:, :pred_motion.shape[1]]).mean()
        else:
            # Compute smoothness loss on predicted positions
            loss = self._compute_smoothness_loss(pred_xyz, t_tgt)

        return loss

    def _estimate_motion_from_positions(
        self,
        points: torch.Tensor,  # [B, N, 3]
        t_indices: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        """
        Estimate motion vectors from positions.

        Args:
            points: 3D positions
            t_indices: Frame indices

        Returns:
            motion: [B, N', 3] motion vectors
        """
        B, N, _ = points.shape

        motions = []

        for b in range(B):
            # Sort by time index
            t_b = t_indices[b]
            sorted_indices = torch.argsort(t_b)
            points_sorted = points[b, sorted_indices]
            t_sorted = t_b[sorted_indices]

            # Compute motion between consecutive unique frames
            motion_b = []
            for i in range(len(t_sorted) - 1):
                if t_sorted[i+1] == t_sorted[i] + 1:  # Consecutive frames
                    motion = points_sorted[i+1] - points_sorted[i]
                    motion_b.append(motion)

            if len(motion_b) > 0:
                motion_b = torch.stack(motion_b, dim=0)  # [M, 3]
            else:
                motion_b = torch.zeros(1, 3, device=points.device)

            motions.append(motion_b)

        # Pad to same length
        max_len = max([m.shape[0] for m in motions])
        motions_padded = []
        for m in motions:
            if m.shape[0] < max_len:
                padding = torch.zeros(max_len - m.shape[0], 3, device=m.device)
                m = torch.cat([m, padding], dim=0)
            motions_padded.append(m)

        motions = torch.stack(motions_padded, dim=0)  # [B, max_len, 3]

        return motions

    def _compute_smoothness_loss(
        self,
        points: torch.Tensor,  # [B, N, 3]
        t_indices: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        """
        Compute smoothness loss on motion.

        Args:
            points: 3D positions
            t_indices: Frame indices

        Returns:
            loss: Scalar smoothness loss
        """
        # Simple approximation: penalize large accelerations
        motions = self._estimate_motion_from_positions(points, t_indices)

        if motions.shape[1] < 2:
            return torch.tensor(0.0, device=points.device)

        # Compute acceleration (second derivative)
        accelerations = motions[:, 1:] - motions[:, :-1]  # [B, M-1, 3]

        # Penalize large accelerations
        loss = torch.abs(accelerations).mean()

        return loss
