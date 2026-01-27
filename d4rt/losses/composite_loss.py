"""Composite loss combining all D4RT loss functions."""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .l1_3d import L1_3DLoss
from .projection_2d import Projection2DLoss
from .visibility import VisibilityLoss
from .normal import NormalLoss
from .motion import MotionLoss
from .confidence import ConfidenceLoss, compute_prediction_error


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple supervision signals.

    Default weights from paper:
    - l1_3d: 1.0 (primary supervision)
    - l2_2d: 0.1 (reprojection)
    - normal: 0.5 (surface alignment) - Updated to paper value
    - motion: 0.1 (temporal consistency)
    - visibility: 0.1 (occlusion)
    - confidence: 0.2 (confidence weighting) - NEW from paper
    """

    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize composite loss.

        Args:
            loss_weights: Dictionary of loss weights
        """
        super().__init__()

        # Default weights from paper
        self.loss_weights = loss_weights or {
            'l1_3d': 1.0,
            'l2_2d': 0.1,
            'normal': 0.5,  # Paper value (was 0.05, 10× increase)
            'motion': 0.1,
            'visibility': 0.1,
            'confidence': 0.2,  # Paper value (NEW)
        }

        # Initialize loss functions
        # Use paper formula by default for 3D loss (can be overridden via config)
        use_paper_formula = loss_weights.get('use_paper_formula_3d', True) if loss_weights else True
        self.l1_3d_loss = L1_3DLoss(use_paper_formula=use_paper_formula)
        self.projection_2d_loss = Projection2DLoss(loss_type='l2')
        self.visibility_loss = VisibilityLoss()
        self.normal_loss = NormalLoss()
        self.motion_loss = MotionLoss()
        self.confidence_loss = ConfidenceLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        cameras: Dict[str, torch.Tensor],
        queries: Dict[str, torch.Tensor],
        scene_bounds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute composite loss.

        Args:
            predictions: Dictionary with model predictions:
                - 'xyz': [B, N, 3] predicted 3D positions
                - 'visibility': [B, N, 1] predicted visibility
            targets: Dictionary with ground truth:
                - 'xyz': [B, N, 3] ground truth 3D positions
                - 'uv': [B, N, 2] ground truth 2D coordinates
                - 'visibility': [B, N] ground truth visibility
                - 'normals': [B, N, 3] ground truth normals (optional)
                - 'motion': [B, N, 3] ground truth motion (optional)
            cameras: Dictionary with camera parameters:
                - 'intrinsics': [B, T, 3, 3]
                - 'extrinsics': [B, T, 4, 4]
            queries: Dictionary with query components (for t_cam indexing)
            scene_bounds: [B, 6] scene bounding boxes (optional)

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary with individual loss values
        """
        loss_dict = {}

        # 1. L1 3D Position Loss (primary supervision)
        if 'xyz' in targets:
            loss_3d = self.l1_3d_loss(
                predictions['xyz'],
                targets['xyz'],
                scene_bounds,
            )
            loss_dict['loss_3d'] = loss_3d.item()
        else:
            loss_3d = torch.zeros(1, device=predictions['xyz'].device, requires_grad=True)
            loss_dict['loss_3d'] = 0.0

        # 2. 2D Reprojection Loss
        if 'uv' in targets and 't_cam' in queries:
            loss_2d = self.projection_2d_loss(
                predictions['xyz'],
                targets['uv'],
                cameras['intrinsics'],
                cameras['extrinsics'],
                queries['t_cam'],
            )
            loss_dict['loss_2d'] = loss_2d.item()
        else:
            loss_2d = torch.zeros(1, device=predictions['xyz'].device, requires_grad=True)
            loss_dict['loss_2d'] = 0.0

        # 3. Visibility Loss
        if 'visibility' in predictions and 'visibility' in targets:
            loss_vis = self.visibility_loss(
                predictions['visibility'],
                targets['visibility'],
            )
            loss_dict['loss_visibility'] = loss_vis.item()
        else:
            loss_vis = torch.zeros(1, device=predictions['xyz'].device, requires_grad=True)
            loss_dict['loss_visibility'] = 0.0

        # 4. Normal Loss (optional)
        if 'normals' in targets:
            loss_normal = self.normal_loss(
                predictions['xyz'],
                targets['normals'],
            )
            loss_dict['loss_normal'] = loss_normal.item()
        else:
            loss_normal = torch.zeros(1, device=predictions['xyz'].device, requires_grad=True)
            loss_dict['loss_normal'] = 0.0

        # 5. Motion Loss (optional)
        if 'motion' in targets or 't_tgt' in queries:
            gt_motion = targets.get('motion', None)
            loss_motion = self.motion_loss(
                predictions['xyz'],
                gt_motion,
                queries.get('t_tgt', torch.zeros_like(queries['t_cam'])),
            )
            loss_dict['loss_motion'] = loss_motion.item()
        else:
            loss_motion = torch.zeros(1, device=predictions['xyz'].device, requires_grad=True)
            loss_dict['loss_motion'] = 0.0

        # 6. Confidence Loss (optional, NEW from paper)
        if 'confidence' in predictions and 'xyz' in targets:
            # Compute prediction error for confidence weighting
            error = compute_prediction_error(
                predictions['xyz'],
                targets['xyz'],
                error_type='l1'
            )
            loss_confidence = self.confidence_loss(
                predictions['confidence'],
                error,
            )
            loss_dict['loss_confidence'] = loss_confidence.item()
        else:
            loss_confidence = torch.zeros(1, device=predictions['xyz'].device, requires_grad=True)
            loss_dict['loss_confidence'] = 0.0

        # Compute weighted total loss
        total_loss = (
            self.loss_weights['l1_3d'] * loss_3d +
            self.loss_weights['l2_2d'] * loss_2d +
            self.loss_weights['normal'] * loss_normal +
            self.loss_weights['motion'] * loss_motion +
            self.loss_weights['visibility'] * loss_vis +
            self.loss_weights.get('confidence', 0.0) * loss_confidence
        )

        loss_dict['loss_total'] = total_loss.item()

        return total_loss, loss_dict


def build_composite_loss(config: Dict) -> CompositeLoss:
    """
    Build composite loss from config.

    Args:
        config: Configuration dictionary with 'loss_weights'

    Returns:
        loss_fn: CompositeLoss instance
    """
    loss_weights = config.get('loss_weights', {})
    return CompositeLoss(loss_weights=loss_weights)
