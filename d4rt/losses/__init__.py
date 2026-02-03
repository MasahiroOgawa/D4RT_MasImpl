"""Loss functions for D4RT."""

from .l1_3d import L1_3DLoss
from .projection_2d import Projection2DLoss
from .visibility import VisibilityLoss
from .normal import NormalLoss
from .motion import MotionLoss
from .confidence import ConfidenceLoss, compute_prediction_error
from .uv_loss import UVLoss
from .composite_loss import (
    CompositeLoss,
    D4RTCompositeLoss,
    build_composite_loss,
)

__all__ = [
    'L1_3DLoss',
    'Projection2DLoss',
    'VisibilityLoss',
    'NormalLoss',
    'MotionLoss',
    'ConfidenceLoss',
    'compute_prediction_error',
    'UVLoss',
    'CompositeLoss',
    'D4RTCompositeLoss',
    'build_composite_loss',
]
