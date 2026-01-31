"""L1 3D position loss for D4RT."""

import torch
import torch.nn as nn


class L1_3DLoss(nn.Module):
    """
    L1 loss on 3D positions with scene normalization.

    This is the primary supervision signal for D4RT.
    """

    def __init__(self, normalize_by_scene: bool = True, use_paper_formula: bool = False):
        """
        Initialize L1 3D loss.

        Args:
            normalize_by_scene: If True, normalize by scene scale for scale invariance (old method)
            use_paper_formula: If True, use paper's exact formula (mean depth + signed log)
        """
        super().__init__()
        self.normalize_by_scene = normalize_by_scene
        self.use_paper_formula = use_paper_formula

    def normalize_by_mean_depth(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Normalize coordinates by mean depth (Z value).

        Args:
            xyz: [B, N, 3] coordinates

        Returns:
            normalized: [B, N, 3] coordinates normalized by mean Z
        """
        # Get mean depth per batch
        mean_depth = xyz[..., 2:3].mean(dim=1, keepdim=True)  # [B, 1, 1]
        # Normalize all coordinates by mean depth
        normalized = xyz / (mean_depth + 1e-8)
        return normalized

    def signed_log_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply signed log transform: sign(x) * log(1 + |x|)

        This transform compresses the range while preserving sign and ordering.

        Args:
            x: Input tensor

        Returns:
            transformed: sign(x) * log(1 + |x|)
        """
        return torch.sign(x) * torch.log(1 + torch.abs(x))

    def forward(
        self,
        pred_xyz: torch.Tensor,  # [B, N, 3]
        gt_xyz: torch.Tensor,  # [B, N, 3]
        scene_bounds: torch.Tensor = None,  # [B, 6] (xmin, xmax, ymin, ymax, zmin, zmax)
    ) -> torch.Tensor:
        """
        Compute L1 3D loss.

        Args:
            pred_xyz: Predicted 3D positions
            gt_xyz: Ground truth 3D positions
            scene_bounds: Scene bounding box (optional, for normalization)

        Returns:
            loss: Scalar loss value
        """
        if self.use_paper_formula:
            # Paper's formula with FIXED normalization:
            # CRITICAL FIX: Use GT's mean depth for BOTH pred and gt
            # (Previously each was normalized by its own mean, making loss scale-invariant)

            # 1. Get GT mean depth as the normalization reference
            gt_mean_depth = gt_xyz[..., 2:3].mean(dim=1, keepdim=True)  # [B, 1, 1]

            # 2. Normalize BOTH by GT's mean depth
            pred_norm = pred_xyz / (gt_mean_depth + 1e-8)
            gt_norm = gt_xyz / (gt_mean_depth + 1e-8)

            # 3. Apply signed log transform
            pred_transformed = self.signed_log_transform(pred_norm)
            gt_transformed = self.signed_log_transform(gt_norm)

            # 4. Compute L1 loss
            loss = torch.abs(pred_transformed - gt_transformed).mean()

        elif self.normalize_by_scene and scene_bounds is not None:
            # Old method: Compute scene scale
            scene_scale = (scene_bounds[:, 1::2] - scene_bounds[:, 0::2]).max(dim=1)[0]  # [B]
            scene_scale = scene_scale.view(-1, 1, 1)  # [B, 1, 1]

            # Normalize positions
            pred_xyz_norm = pred_xyz / (scene_scale + 1e-8)
            gt_xyz_norm = gt_xyz / (scene_scale + 1e-8)

            loss = torch.abs(pred_xyz_norm - gt_xyz_norm).mean()
        else:
            loss = torch.abs(pred_xyz - gt_xyz).mean()

        return loss
