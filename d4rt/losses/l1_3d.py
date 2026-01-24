"""L1 3D position loss for D4RT."""

import torch
import torch.nn as nn


class L1_3DLoss(nn.Module):
    """
    L1 loss on 3D positions with scene normalization.

    This is the primary supervision signal for D4RT.
    """

    def __init__(self, normalize_by_scene: bool = True):
        """
        Initialize L1 3D loss.

        Args:
            normalize_by_scene: If True, normalize by scene scale for scale invariance
        """
        super().__init__()
        self.normalize_by_scene = normalize_by_scene

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
        if self.normalize_by_scene and scene_bounds is not None:
            # Compute scene scale
            scene_scale = (scene_bounds[:, 1::2] - scene_bounds[:, 0::2]).max(dim=1)[0]  # [B]
            scene_scale = scene_scale.view(-1, 1, 1)  # [B, 1, 1]

            # Normalize positions
            pred_xyz_norm = pred_xyz / (scene_scale + 1e-8)
            gt_xyz_norm = gt_xyz / (scene_scale + 1e-8)

            loss = torch.abs(pred_xyz_norm - gt_xyz_norm).mean()
        else:
            loss = torch.abs(pred_xyz - gt_xyz).mean()

        return loss
