"""2D projection loss for D4RT."""

import torch
import torch.nn as nn


class Projection2DLoss(nn.Module):
    """
    2D reprojection loss.

    Projects predicted 3D points to 2D and compares with ground truth 2D coordinates.
    """

    def __init__(self, loss_type: str = 'l2'):
        """
        Initialize 2D projection loss.

        Args:
            loss_type: 'l1' or 'l2'
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        pred_xyz: torch.Tensor,  # [B, N, 3]
        gt_uv: torch.Tensor,  # [B, N, 2]
        intrinsics: torch.Tensor,  # [B, T, 3, 3]
        extrinsics: torch.Tensor,  # [B, T, 4, 4]
        t_cam: torch.Tensor,  # [B, N] camera frame indices
    ) -> torch.Tensor:
        """
        Compute 2D projection loss.

        Args:
            pred_xyz: Predicted 3D positions in world/camera coordinates
            gt_uv: Ground truth 2D pixel coordinates
            intrinsics: Camera intrinsics for all frames
            extrinsics: Camera extrinsics for all frames
            t_cam: Camera frame index for each query

        Returns:
            loss: Scalar loss value
        """
        B, N, _ = pred_xyz.shape

        # Project 3D to 2D
        pred_uv = self._project_3d_to_2d(
            pred_xyz, intrinsics, extrinsics, t_cam
        )  # [B, N, 2]

        # Compute loss
        if self.loss_type == 'l1':
            loss = torch.abs(pred_uv - gt_uv).mean()
        elif self.loss_type == 'l2':
            loss = torch.sqrt(((pred_uv - gt_uv) ** 2).sum(dim=-1) + 1e-8).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def _project_3d_to_2d(
        self,
        points_3d: torch.Tensor,  # [B, N, 3]
        intrinsics: torch.Tensor,  # [B, T, 3, 3]
        extrinsics: torch.Tensor,  # [B, T, 4, 4]
        t_cam: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        """
        Project 3D points to 2D.

        Args:
            points_3d: 3D points
            intrinsics: Camera intrinsics
            extrinsics: Camera extrinsics
            t_cam: Camera frame indices

        Returns:
            points_2d: [B, N, 2] 2D pixel coordinates
        """
        B, N, _ = points_3d.shape

        points_2d_list = []

        for b in range(B):
            points_2d_b = []

            for n in range(N):
                # Get camera parameters for this query
                t = t_cam[b, n].long().item()
                t = max(0, min(t, intrinsics.shape[1] - 1))  # Clamp to valid range

                K = intrinsics[b, t]  # [3, 3]
                T = extrinsics[b, t]  # [4, 4]

                # Get 3D point
                xyz = points_3d[b, n]  # [3]

                # Transform to camera space
                xyz_hom = torch.cat([xyz, torch.ones(1, device=xyz.device)])  # [4]
                xyz_cam = (T @ xyz_hom.unsqueeze(1)).squeeze(1)[:3]  # [3]

                # Project to 2D
                if xyz_cam[2] > 0:  # Valid depth
                    uv_hom = (K @ xyz_cam.unsqueeze(1)).squeeze(1)  # [3]
                    uv = uv_hom[:2] / (uv_hom[2] + 1e-8)  # [2]
                else:
                    # Invalid projection, use zero
                    uv = torch.zeros(2, device=xyz.device)

                points_2d_b.append(uv)

            points_2d_b = torch.stack(points_2d_b, dim=0)  # [N, 2]
            points_2d_list.append(points_2d_b)

        points_2d = torch.stack(points_2d_list, dim=0)  # [B, N, 2]

        return points_2d
