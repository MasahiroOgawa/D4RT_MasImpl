"""TAP-Vid-3D evaluation metrics for D4RT."""

import numpy as np
from typing import Dict, Optional


def align_predictions(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    alignment: str = "scale_shift",
) -> np.ndarray:
    """
    Align predictions to ground truth using scale and/or shift.

    Following D4RT paper evaluation protocol:
    "determine a single global scale factor between the predicted and ground-truth depths"
    "first align the predicted and ground-truth point clouds via mean-shifting"

    Args:
        pred_xyz: [N, 3] predicted 3D positions
        gt_xyz: [N, 3] ground truth 3D positions
        alignment: Alignment method
            - "none": No alignment
            - "scale": Scale only (median depth ratio)
            - "shift": Shift only (mean centering)
            - "scale_shift": Scale and shift (Sim(3) without rotation)

    Returns:
        pred_aligned: [N, 3] aligned predictions
    """
    if alignment == "none":
        return pred_xyz.copy()

    pred_aligned = pred_xyz.copy()

    if alignment == "scale":
        # Scale by median depth ratio
        pred_depth = pred_aligned[:, 2]
        gt_depth = gt_xyz[:, 2]
        # Avoid division by zero
        valid = np.abs(pred_depth) > 1e-6
        if valid.sum() > 0:
            scale = np.median(gt_depth[valid] / pred_depth[valid])
            pred_aligned = pred_aligned * scale

    elif alignment == "shift":
        # Mean centering
        pred_mean = pred_aligned.mean(axis=0)
        gt_mean = gt_xyz.mean(axis=0)
        pred_aligned = pred_aligned - pred_mean + gt_mean

    elif alignment == "scale_shift":
        # Scale and shift alignment (standard protocol)
        # 1. Center both point clouds
        pred_mean = pred_aligned.mean(axis=0)
        gt_mean = gt_xyz.mean(axis=0)
        pred_centered = pred_aligned - pred_mean
        gt_centered = gt_xyz - gt_mean

        # 2. Compute scale factor
        pred_std = np.sqrt((pred_centered**2).sum())
        gt_std = np.sqrt((gt_centered**2).sum())

        if pred_std > 1e-6:
            scale = gt_std / pred_std
        else:
            scale = 1.0

        # 3. Apply scale and shift
        pred_aligned = pred_centered * scale + gt_mean

    return pred_aligned


def compute_tapvid_metrics(
    pred_xyz: np.ndarray,
    pred_visibility: np.ndarray,
    gt_xyz: np.ndarray,
    gt_visibility: np.ndarray,
    thresholds: list = [0.05, 0.1, 0.2, 0.5, 1.0],
    alignment: str = "scale_shift",
) -> Dict[str, float]:
    """
    Compute TAP-Vid-3D style metrics.

    Following D4RT paper evaluation protocol with scale-and-shift alignment.

    Args:
        pred_xyz: [B, N, 3] or [N, 3] predicted 3D positions
        pred_visibility: [B, N] or [N] predicted visibility scores (0-1)
        gt_xyz: [B, N, 3] or [N, 3] ground truth 3D positions
        gt_visibility: [B, N] or [N] ground truth visibility (binary)
        thresholds: Distance thresholds for APD3D computation
        alignment: Alignment method ("none", "scale", "shift", "scale_shift")

    Returns:
        metrics: Dictionary containing:
            - average_jaccard: Average Jaccard index
            - average_pts_within_thresh: Average fraction of points within threshold
            - occlusion_accuracy: Visibility prediction accuracy
    """
    # Flatten batch dimension if present
    if pred_xyz.ndim == 3:
        pred_xyz = pred_xyz.reshape(-1, 3)
        pred_visibility = pred_visibility.reshape(-1)
        gt_xyz = gt_xyz.reshape(-1, 3)
        gt_visibility = gt_visibility.reshape(-1)

    # Align predictions to ground truth (paper protocol)
    pred_xyz_aligned = align_predictions(pred_xyz, gt_xyz, alignment=alignment)

    # Compute 3D position error on aligned predictions
    position_error = np.linalg.norm(pred_xyz_aligned - gt_xyz, axis=-1)  # [N]

    # Get visibility masks
    visible_mask = gt_visibility > 0.5
    pred_visible_mask = pred_visibility > 0.5

    # 1. Average Points Within Threshold (APD3D)
    # For visible points, what fraction are within each threshold?
    if visible_mask.sum() > 0:
        visible_errors = position_error[visible_mask]
        pts_within_thresh = []
        for thresh in thresholds:
            frac = (visible_errors < thresh).mean()
            pts_within_thresh.append(frac)
        average_pts_within_thresh = np.mean(pts_within_thresh)
    else:
        average_pts_within_thresh = 0.0

    # 2. Occlusion Accuracy (OA)
    # Binary classification accuracy
    occlusion_accuracy = (pred_visible_mask == (gt_visibility > 0.5)).mean()

    # 3. Average Jaccard (AJ)
    # AJ = TP / (TP + FP + FN) for points within threshold
    # TP: predicted visible AND within threshold AND actually visible
    # FP: predicted visible but not within threshold OR not actually visible
    # FN: actually visible but not predicted visible
    threshold_for_aj = 0.5  # 50cm threshold for AJ
    within_thresh = position_error < threshold_for_aj

    tp = (pred_visible_mask & within_thresh & visible_mask).sum()
    fp = (pred_visible_mask & (~within_thresh | ~visible_mask)).sum()
    fn = (visible_mask & ~pred_visible_mask).sum()

    if tp + fp + fn > 0:
        average_jaccard = tp / (tp + fp + fn)
    else:
        average_jaccard = 0.0

    return {
        "average_jaccard": float(average_jaccard),
        "average_pts_within_thresh": float(average_pts_within_thresh),
        "occlusion_accuracy": float(occlusion_accuracy),
        "mean_position_error": (
            float(position_error[visible_mask].mean()) if visible_mask.sum() > 0 else float("nan")
        ),
        "median_position_error": (
            float(np.median(position_error[visible_mask]))
            if visible_mask.sum() > 0
            else float("nan")
        ),
    }
