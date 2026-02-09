"""Composite loss combining all D4RT loss functions.

Paper-exact loss formula:
    L = (1/N) * sum[ c*λ3D*L3D - λconf*log(c) + λ2D*L2D + λvis*Lvis + λdisp*Ldisp + λnormal*Lnormal ]

Key features:
- Confidence-weighted 3D loss: c*λ3D*L3D (3D error scaled by prediction confidence)
- Confidence penalty: -λconf*log(c) (encourages high confidence predictions)

Default weights from paper:
- λ3D = 1.0
- λ2D = 0.1
- λvis = 0.1
- λdisp = 0.1
- λconf = 0.2
- λnormal = 0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .l1_3d import L1_3DLoss
from .projection_2d import Projection2DLoss
from .visibility import VisibilityLoss
from .normal import NormalLoss
from .motion import MotionLoss
from .uv_loss import UVLoss


class D4RTCompositeLoss(nn.Module):
    """
    Paper-exact composite loss with confidence weighting.

    Implements the D4RT loss function:
        L = (1/N) * sum[ c*λ3D*L3D - λconf*log(c) + λ2D*L2D + λvis*Lvis + λdisp*Ldisp + λnormal*Lnormal ]

    Key features:
    - Confidence-weighted 3D loss: c*λ3D*L3D
    - Confidence penalty: -λconf*log(c)
    - Per-query averaging: (1/N) * sum
    """

    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        use_paper_formula: bool = True,
        confidence_warmup_steps: int = 0,
        z_gradient_scale: float = 1.0,
    ):
        """
        Initialize paper-exact composite loss.

        Args:
            loss_weights: Dictionary of loss weights
            use_paper_formula: If True, use paper's exact confidence-weighted formula
            confidence_warmup_steps: Number of steps to use c=1 for xyz loss weighting.
                This prevents the model from exploiting low confidence to minimize loss
                without learning proper 3D predictions. After warmup, the learned
                confidence is used. Set to 0 to disable warmup (original paper behavior).
                Recommended: 10000-25000 steps.
            z_gradient_scale: Scale factor for Z (depth) gradients. Higher values give
                stronger gradients for depth prediction, helping overcome variance collapse.
                Default 1.0. Recommended: 3.0-10.0 if depth variance is collapsing.
        """
        super().__init__()
        self.use_paper_formula = use_paper_formula
        self.confidence_warmup_steps = confidence_warmup_steps
        self.z_gradient_scale = z_gradient_scale
        self._current_step = 0

        # Default weights from paper
        default_weights = {
            "l1_3d": 1.0,  # λ3D
            "l2_2d": 0.1,  # λ2D (UV loss)
            "normal": 0.5,  # λnormal
            "motion": 0.1,  # λdisp
            "visibility": 0.1,  # λvis
            "confidence": 0.2,  # λconf (penalty weight)
            "depth_aux": 1.0,  # λdepth_aux (log-depth auxiliary loss for strong Z supervision)
        }

        # Merge custom weights with defaults
        if loss_weights:
            self.loss_weights = {**default_weights, **loss_weights}
        else:
            self.loss_weights = default_weights

        # Store individual lambda values
        self.lambda_3d = self.loss_weights["l1_3d"]
        self.lambda_2d = self.loss_weights["l2_2d"]
        self.lambda_vis = self.loss_weights["visibility"]
        self.lambda_disp = self.loss_weights["motion"]
        self.lambda_conf = self.loss_weights["confidence"]
        self.lambda_normal = self.loss_weights["normal"]
        self.lambda_depth_aux = self.loss_weights["depth_aux"]

        # Individual loss functions (used for non-paper mode or components)
        use_paper_3d = loss_weights.get("use_paper_formula_3d", True) if loss_weights else True
        self.l1_3d_loss = L1_3DLoss(use_paper_formula=use_paper_3d)
        self.uv_loss = UVLoss(loss_type="l2")
        self.visibility_loss = VisibilityLoss()
        self.normal_loss = NormalLoss()
        self.motion_loss = MotionLoss()
        self.projection_2d_loss = Projection2DLoss(loss_type="l2")

    def set_step(self, step: int):
        """Set the current training step for warmup scheduling."""
        self._current_step = step

    def get_confidence_weight(self) -> float:
        """Get the current confidence weight (0 to 1) based on warmup schedule."""
        if self.confidence_warmup_steps <= 0:
            return 1.0  # No warmup, use full confidence weighting
        if self._current_step >= self.confidence_warmup_steps:
            return 1.0  # Warmup complete
        # Linear warmup: start from 0 (c=1), increase to 1 (use learned c)
        return self._current_step / self.confidence_warmup_steps

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        cameras: Optional[Dict[str, torch.Tensor]] = None,
        queries: Optional[Dict[str, torch.Tensor]] = None,
        scene_bounds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute paper-exact composite loss.

        Args:
            predictions: Dictionary with model predictions:
                - 'xyz': [B, N, 3] predicted 3D positions
                - 'visibility': [B, N, 1] predicted visibility logits
                - 'confidence': [B, N, 1] predicted confidence logits
                - 'uv': [B, N, 2] predicted 2D coordinates (optional)
                - 'normals': [B, N, 3] predicted surface normals (optional)
                - 'motion': [B, N, 3] predicted motion (optional)
            targets: Dictionary with ground truth:
                - 'xyz': [B, N, 3] ground truth 3D positions
                - 'uv': [B, N, 2] ground truth 2D coordinates
                - 'visibility': [B, N] or [B, N, 1] ground truth visibility
                - 'normals': [B, N, 3] ground truth normals (optional)
                - 'motion': [B, N, 3] ground truth motion (optional)
            cameras: Dictionary with camera parameters (optional):
                - 'intrinsics': [B, T, 3, 3]
                - 'extrinsics': [B, T, 4, 4]
            queries: Dictionary with query components (optional)
            scene_bounds: [B, 6] scene bounding boxes (optional)

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary with individual loss values
        """
        loss_dict = {}

        if self.use_paper_formula:
            # Paper-exact formula: L = (1/N) * sum[ c*λ3D*L3D - λconf*log(c) + ... ]
            total_loss, loss_dict = self._compute_paper_loss(
                predictions, targets, cameras, queries, scene_bounds
            )
        else:
            # Legacy separate computation
            total_loss, loss_dict = self._compute_legacy_loss(
                predictions, targets, cameras, queries, scene_bounds
            )

        return total_loss, loss_dict

    def _compute_paper_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        cameras: Optional[Dict[str, torch.Tensor]],
        queries: Optional[Dict[str, torch.Tensor]],
        scene_bounds: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute paper-exact loss formula with confidence weighting.

        Paper formula (per query):
            L = c*λ3D*L3D - λconf*log(c) + λ2D*L2D + λvis*Lvis + λdisp*Ldisp + λnormal*Lnormal
        """
        loss_dict = {}
        device = predictions["xyz"].device
        B, N, _ = predictions["xyz"].shape

        # Get confidence (apply sigmoid and clamp to avoid log(0))
        conf_logits = predictions.get("confidence", torch.zeros(B, N, 1, device=device))
        c = torch.sigmoid(conf_logits).clamp(min=1e-6, max=1 - 1e-6)  # [B, N, 1]

        # ========== L3D: L1 loss on 3D positions (per-query) ==========
        if "xyz" in targets:
            # Compute per-query L1 loss
            pred_xyz = predictions["xyz"]
            gt_xyz = targets["xyz"]

            # Paper's normalization: each normalized by its OWN mean depth
            # "both the target and the estimated point sets are normalized by their respective mean depths"
            pred_mean_depth = pred_xyz[..., 2:3].mean(dim=1, keepdim=True)  # [B, 1, 1]
            gt_mean_depth = gt_xyz[..., 2:3].mean(dim=1, keepdim=True)  # [B, 1, 1]
            pred_norm = pred_xyz / (pred_mean_depth + 1e-8)
            gt_norm = gt_xyz / (gt_mean_depth + 1e-8)

            # Apply signed log transform
            pred_transformed = torch.sign(pred_norm) * torch.log(1 + torch.abs(pred_norm))
            gt_transformed = torch.sign(gt_norm) * torch.log(1 + torch.abs(gt_norm))

            # Per-query L1 loss with Z gradient scaling
            # Compute L1 for each axis separately to allow Z scaling
            L3D_per_axis = torch.abs(pred_transformed - gt_transformed)  # [B, N, 3]

            # Scale Z (depth) gradients to help overcome variance collapse
            if self.z_gradient_scale != 1.0:
                # Scale Z component (index 2) by z_gradient_scale
                scale_factors = torch.tensor(
                    [1.0, 1.0, self.z_gradient_scale], device=L3D_per_axis.device
                )
                L3D_per_axis = L3D_per_axis * scale_factors

            L3D = L3D_per_axis.sum(dim=-1, keepdim=True)  # [B, N, 1]
            loss_dict["loss_3d_raw"] = L3D.mean().item()
            loss_dict["z_gradient_scale"] = self.z_gradient_scale

            # ========== AUXILIARY: Absolute L1 depth loss ==========
            # Directly penalizes wrong depth values, encouraging correct variance
            # Unlike scale-invariant loss, this provides absolute depth supervision
            pred_z = pred_xyz[..., 2:3]  # [B, N, 1]
            gt_z = gt_xyz[..., 2:3]
            L_depth_aux = torch.abs(pred_z - gt_z)  # [B, N, 1]
            loss_dict["loss_depth_aux_raw"] = L_depth_aux.mean().item()

            # Log Z-specific metrics for debugging
            loss_dict["pred_z_mean"] = pred_z.mean().item()
            loss_dict["pred_z_std"] = pred_z.std().item()
            loss_dict["gt_z_mean"] = gt_z.mean().item()
            loss_dict["pred_z_negative_ratio"] = (pred_z < 0).float().mean().item()
        else:
            L3D = torch.zeros(B, N, 1, device=device)
            L_depth_aux = torch.zeros(B, N, 1, device=device)
            loss_dict["loss_3d_raw"] = 0.0
            loss_dict["loss_depth_aux_raw"] = 0.0

        # ========== L2D: L1 loss on 2D coordinates (per-query) ==========
        # Paper: "An L1 loss on 2D coordinates of the point positions in image space"
        if "uv" in predictions and "uv" in targets:
            pred_uv = predictions["uv"]
            gt_uv = targets["uv"]
            L2D = torch.abs(pred_uv - gt_uv).sum(dim=-1, keepdim=True)  # [B, N, 1]
            loss_dict["loss_2d_raw"] = L2D.mean().item()
        else:
            L2D = torch.zeros(B, N, 1, device=device)
            loss_dict["loss_2d_raw"] = 0.0

        # ========== Lvis: Binary cross-entropy on visibility (per-query) ==========
        if "visibility" in predictions and "visibility" in targets:
            pred_vis = predictions["visibility"]  # [B, N, 1] logits
            gt_vis = targets["visibility"]
            if gt_vis.dim() == 2:
                gt_vis = gt_vis.unsqueeze(-1)  # [B, N] -> [B, N, 1]
            Lvis = F.binary_cross_entropy_with_logits(
                pred_vis, gt_vis.float(), reduction="none"
            )  # [B, N, 1]
            loss_dict["loss_visibility_raw"] = Lvis.mean().item()
        else:
            Lvis = torch.zeros(B, N, 1, device=device)
            loss_dict["loss_visibility_raw"] = 0.0

        # ========== Ldisp: L1 loss on motion displacement (per-query) ==========
        if "motion" in predictions and "motion" in targets:
            pred_motion = predictions["motion"]
            gt_motion = targets["motion"]
            Ldisp = torch.abs(pred_motion - gt_motion).sum(dim=-1, keepdim=True)  # [B, N, 1]
            loss_dict["loss_motion_raw"] = Ldisp.mean().item()
        else:
            Ldisp = torch.zeros(B, N, 1, device=device)
            loss_dict["loss_motion_raw"] = 0.0

        # ========== Lnormal: Cosine loss on surface normals (per-query) ==========
        if "normals" in predictions and "normals" in targets:
            pred_normals = predictions["normals"]
            gt_normals = targets["normals"]
            # Cosine distance: 1 - cos_sim
            cos_sim = F.cosine_similarity(pred_normals, gt_normals, dim=-1, eps=1e-8)
            Lnormal = (1 - cos_sim).unsqueeze(-1)  # [B, N, 1]
            loss_dict["loss_normal_raw"] = Lnormal.mean().item()
        else:
            Lnormal = torch.zeros(B, N, 1, device=device)
            loss_dict["loss_normal_raw"] = 0.0

        # ========== Paper formula (per-query) ==========
        # L = c*λ3D*L3D - λconf*log(c) + λ2D*L2D + λvis*Lvis + λdisp*Ldisp + λnormal*Lnormal
        #
        # IMPORTANT: Confidence warmup schedule
        # Problem: The paper formula allows the model to minimize loss by outputting
        # low confidence, rather than learning accurate 3D predictions.
        # Analysis: If L3D=5, optimal c = λconf/(λ3D*L3D) = 0.2/(1*5) = 0.04
        #          The xyz gradient is multiplied by c ≈ 0.04, nearly vanishing!
        #
        # Solution: During warmup, use c_effective=1 for xyz loss weighting,
        # but still train the confidence head via the penalty term.
        # After warmup, gradually transition to using learned confidence.

        conf_weight = self.get_confidence_weight()
        if conf_weight < 1.0:
            # During warmup: blend between c=1 (no weighting) and learned c
            # c_effective = 1 * (1 - conf_weight) + c * conf_weight
            # When conf_weight=0: c_effective=1 (no confidence weighting)
            # When conf_weight=1: c_effective=c (full confidence weighting)
            c_effective = torch.ones_like(c) * (1 - conf_weight) + c * conf_weight
            loss_dict["confidence_warmup_weight"] = conf_weight
        else:
            c_effective = c
            loss_dict["confidence_warmup_weight"] = 1.0

        # Confidence-weighted 3D loss: c_effective * λ3D * L3D
        conf_weighted_3d = c_effective * self.lambda_3d * L3D

        # Confidence penalty: -λconf * log(c)
        # Note: Always use actual c here so the confidence head keeps learning
        conf_penalty = -self.lambda_conf * torch.log(c)

        # Other losses (not confidence-weighted)
        other_losses = (
            self.lambda_2d * L2D
            + self.lambda_vis * Lvis
            + self.lambda_disp * Ldisp
            + self.lambda_normal * Lnormal
            + self.lambda_depth_aux * L_depth_aux  # Auxiliary log-depth loss
        )

        # Per-query total loss
        per_query_loss = conf_weighted_3d + conf_penalty + other_losses  # [B, N, 1]

        # Average over all queries: (1/N) * sum
        total_loss = per_query_loss.mean()

        # Store individual weighted losses for logging
        loss_dict["loss_3d_weighted"] = conf_weighted_3d.mean().item()
        loss_dict["loss_confidence_penalty"] = conf_penalty.mean().item()
        loss_dict["c_effective_mean"] = c_effective.mean().item()
        loss_dict["loss_2d"] = (self.lambda_2d * L2D).mean().item()
        loss_dict["loss_visibility"] = (self.lambda_vis * Lvis).mean().item()
        loss_dict["loss_motion"] = (self.lambda_disp * Ldisp).mean().item()
        loss_dict["loss_normal"] = (self.lambda_normal * Lnormal).mean().item()
        loss_dict["loss_depth_aux"] = (self.lambda_depth_aux * L_depth_aux).mean().item()
        loss_dict["loss_total"] = total_loss.item()
        loss_dict["mean_confidence"] = c.mean().item()

        return total_loss, loss_dict

    def _compute_legacy_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        cameras: Optional[Dict[str, torch.Tensor]],
        queries: Optional[Dict[str, torch.Tensor]],
        scene_bounds: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss using legacy separate loss functions.

        This is for backward compatibility with existing training.
        """
        loss_dict = {}
        device = predictions["xyz"].device

        # 1. L1 3D Position Loss
        if "xyz" in targets:
            loss_3d = self.l1_3d_loss(
                predictions["xyz"],
                targets["xyz"],
                scene_bounds,
            )
            loss_dict["loss_3d"] = loss_3d.item()
        else:
            loss_3d = torch.zeros(1, device=device, requires_grad=True)
            loss_dict["loss_3d"] = 0.0

        # 2. UV / 2D Loss
        if "uv" in predictions and "uv" in targets:
            loss_2d = self.uv_loss(predictions["uv"], targets["uv"])
            loss_dict["loss_2d"] = loss_2d.item()
        elif cameras is not None and queries is not None and "t_cam" in queries:
            loss_2d = self.projection_2d_loss(
                predictions["xyz"],
                targets["uv"],
                cameras["intrinsics"],
                cameras["extrinsics"],
                queries["t_cam"],
            )
            loss_dict["loss_2d"] = loss_2d.item()
        else:
            loss_2d = torch.zeros(1, device=device, requires_grad=True)
            loss_dict["loss_2d"] = 0.0

        # 3. Visibility Loss
        if "visibility" in predictions and "visibility" in targets:
            loss_vis = self.visibility_loss(
                predictions["visibility"],
                targets["visibility"],
            )
            loss_dict["loss_visibility"] = loss_vis.item()
        else:
            loss_vis = torch.zeros(1, device=device, requires_grad=True)
            loss_dict["loss_visibility"] = 0.0

        # 4. Normal Loss
        if "normals" in predictions and "normals" in targets:
            loss_normal = self.normal_loss(
                predictions["normals"],
                targets["normals"],
            )
            loss_dict["loss_normal"] = loss_normal.item()
        else:
            loss_normal = torch.zeros(1, device=device, requires_grad=True)
            loss_dict["loss_normal"] = 0.0

        # 5. Motion Loss
        if "motion" in predictions and "motion" in targets:
            loss_motion = self.motion_loss(
                predictions["motion"],
                targets["motion"],
            )
            loss_dict["loss_motion"] = loss_motion.item()
        else:
            loss_motion = torch.zeros(1, device=device, requires_grad=True)
            loss_dict["loss_motion"] = 0.0

        # Compute weighted total loss
        total_loss = (
            self.lambda_3d * loss_3d
            + self.lambda_2d * loss_2d
            + self.lambda_normal * loss_normal
            + self.lambda_disp * loss_motion
            + self.lambda_vis * loss_vis
        )

        # Ensure total_loss is a scalar
        if total_loss.dim() > 0:
            total_loss = total_loss.mean()

        loss_dict["loss_total"] = total_loss.item()

        return total_loss, loss_dict


# Alias for backward compatibility
CompositeLoss = D4RTCompositeLoss


def build_composite_loss(config: Dict) -> D4RTCompositeLoss:
    """
    Build composite loss from config.

    Args:
        config: Configuration dictionary with 'loss_weights'

    Returns:
        loss_fn: D4RTCompositeLoss instance
    """
    loss_weights = config.get("loss_weights", {})
    use_paper_formula = config.get("use_paper_formula", True)
    confidence_warmup_steps = config.get("confidence_warmup_steps", 0)
    z_gradient_scale = config.get("z_gradient_scale", 1.0)
    return D4RTCompositeLoss(
        loss_weights=loss_weights,
        use_paper_formula=use_paper_formula,
        confidence_warmup_steps=confidence_warmup_steps,
        z_gradient_scale=z_gradient_scale,
    )
