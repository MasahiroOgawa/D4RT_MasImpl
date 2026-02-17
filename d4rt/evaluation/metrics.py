"""TAP-Vid-3D evaluation metrics for D4RT.

Official TAP-Vid-3D evaluation protocol from:
https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid3d/evaluation/metrics.py
"""

from typing import Dict, Optional

import numpy as np


def align_predictions(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    pred_occluded: Optional[np.ndarray] = None,
    gt_occluded: Optional[np.ndarray] = None,
    alignment: str = "median",
) -> np.ndarray:
    """
    Align predictions to ground truth using scale factor.

    Following official TAP-Vid-3D evaluation protocol:
    https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid3d/evaluation/metrics.py

    Args:
        pred_xyz: [N, 3] predicted 3D positions
        gt_xyz: [N, 3] ground truth 3D positions
        pred_occluded: [N] predicted occlusion (True = occluded), optional
        gt_occluded: [N] ground truth occlusion (True = occluded), optional
        alignment: Alignment method
            - "none": No alignment
            - "median": Global median rescaling (official TAP-Vid-3D default)
                       scale = median(||gt||) / median(||pred||)
            - "mean": Global mean rescaling
                      scale = mean(||gt||) / mean(||pred||)
            - "scale_shift": Legacy Procrustes-style (center + Frobenius scale)

    Returns:
        pred_aligned: [N, 3] aligned predictions
    """
    if alignment == "none":
        return pred_xyz.copy()

    pred_aligned = pred_xyz.copy()

    if alignment == "median" or alignment == "mean":
        # Official TAP-Vid-3D: Global median/mean rescaling
        # scale = median(||P_gt||) / median(||P_pred||)
        # where ||P|| = sqrt(x² + y² + z²) is 3D Euclidean norm from origin

        # Compute 3D Euclidean norms
        pred_norms = np.sqrt(np.maximum(1e-12, np.sum(pred_xyz**2, axis=-1)))
        gt_norms = np.sqrt(np.maximum(1e-12, np.sum(gt_xyz**2, axis=-1)))

        # Exclude occluded points if visibility is provided
        if pred_occluded is not None and gt_occluded is not None:
            either_occluded = np.logical_or(gt_occluded, pred_occluded)
            pred_norms = np.where(either_occluded, np.nan, pred_norms)
            gt_norms = np.where(either_occluded, np.nan, gt_norms)

        # Compute scale factor
        if alignment == "median":
            scale = np.nanmedian(gt_norms) / np.nanmedian(pred_norms)
        else:  # mean
            scale = np.nanmean(gt_norms) / np.nanmean(pred_norms)

        # Handle edge cases
        if not np.isfinite(scale):
            scale = 1.0

        pred_aligned = pred_xyz * scale

    elif alignment == "scale_shift":
        # Legacy: Procrustes-style alignment (center + Frobenius scale + shift)
        # This is NOT the official TAP-Vid-3D protocol but kept for compatibility

        # 1. Center both point clouds
        pred_mean = pred_aligned.mean(axis=0)
        gt_mean = gt_xyz.mean(axis=0)
        pred_centered = pred_aligned - pred_mean
        gt_centered = gt_xyz - gt_mean

        # 2. Compute scale factor from Frobenius norm
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
    alignment: str = "median",
) -> Dict[str, float]:
    """
    Compute TAP-Vid-3D style metrics.

    Following official TAP-Vid-3D evaluation protocol:
    https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid3d/evaluation/metrics.py

    Args:
        pred_xyz: [B, N, 3] or [N, 3] predicted 3D positions
        pred_visibility: [B, N] or [N] predicted visibility scores (0-1)
        gt_xyz: [B, N, 3] or [N, 3] ground truth 3D positions
        gt_visibility: [B, N] or [N] ground truth visibility (binary)
        thresholds: Distance thresholds for APD3D computation
        alignment: Alignment method ("none", "median", "mean", "scale_shift")
                   Default is "median" following official TAP-Vid-3D

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

    # Convert visibility to occlusion for alignment
    pred_occluded = pred_visibility < 0.5
    gt_occluded = gt_visibility < 0.5

    # Align predictions to ground truth (official TAP-Vid-3D protocol)
    pred_xyz_aligned = align_predictions(
        pred_xyz,
        gt_xyz,
        pred_occluded=pred_occluded,
        gt_occluded=gt_occluded,
        alignment=alignment,
    )

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
