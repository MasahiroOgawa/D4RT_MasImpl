"""TAP-Vid-3D evaluation metrics for D4RT."""

import numpy as np
from typing import Dict


def compute_tapvid_metrics(
    pred_xyz: np.ndarray,
    pred_visibility: np.ndarray,
    gt_xyz: np.ndarray,
    gt_visibility: np.ndarray,
    thresholds: list = [0.05, 0.1, 0.2, 0.5, 1.0],
) -> Dict[str, float]:
    """
    Compute TAP-Vid-3D style metrics.

    Args:
        pred_xyz: [B, N, 3] or [N, 3] predicted 3D positions
        pred_visibility: [B, N] or [N] predicted visibility scores (0-1)
        gt_xyz: [B, N, 3] or [N, 3] ground truth 3D positions
        gt_visibility: [B, N] or [N] ground truth visibility (binary)
        thresholds: Distance thresholds for APD3D computation

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

    # Compute 3D position error
    position_error = np.linalg.norm(pred_xyz - gt_xyz, axis=-1)  # [N]

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
