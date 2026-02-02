#!/usr/bin/env python3
"""Debug the mean depth normalization issue."""

import torch

def normalize_by_mean_depth(xyz):
    """Normalize by mean depth (current buggy implementation)."""
    mean_depth = xyz[..., 2:3].mean(dim=1, keepdim=True)
    normalized = xyz / (mean_depth + 1e-8)
    return normalized, mean_depth

def signed_log_transform(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def compute_loss_buggy(pred_xyz, gt_xyz):
    """Current buggy loss - normalizes separately."""
    pred_norm, pred_mean_d = normalize_by_mean_depth(pred_xyz)
    gt_norm, gt_mean_d = normalize_by_mean_depth(gt_xyz)

    pred_transformed = signed_log_transform(pred_norm)
    gt_transformed = signed_log_transform(gt_norm)

    loss = torch.abs(pred_transformed - gt_transformed).mean()
    return loss, pred_mean_d.item(), gt_mean_d.item()

def compute_loss_correct(pred_xyz, gt_xyz):
    """Correct loss - normalizes by GT mean depth only."""
    gt_mean_d = gt_xyz[..., 2:3].mean(dim=1, keepdim=True)

    # Normalize BOTH by GT mean depth
    pred_norm = pred_xyz / (gt_mean_d + 1e-8)
    gt_norm = gt_xyz / (gt_mean_d + 1e-8)

    pred_transformed = signed_log_transform(pred_norm)
    gt_transformed = signed_log_transform(gt_norm)

    loss = torch.abs(pred_transformed - gt_transformed).mean()
    return loss

def main():
    print("=" * 60)
    print("MEAN DEPTH NORMALIZATION BUG DEMONSTRATION")
    print("=" * 60)

    # Ground truth: varying 3D positions
    B, N = 1, 10
    gt_xyz = torch.tensor([[[
        [1.0, 0.5, 10.0],
        [2.0, -0.5, 15.0],
        [0.5, 1.0, 8.0],
        [-1.0, 0.0, 20.0],
        [1.5, -1.0, 12.0],
        [0.0, 0.5, 18.0],
        [-0.5, -0.5, 9.0],
        [2.0, 1.0, 25.0],
        [1.0, -1.0, 11.0],
        [-1.5, 0.0, 14.0],
    ]]])  # [1, 10, 3]
    gt_xyz = gt_xyz.squeeze(0)  # [1, 10, 3]

    gt_mean_d = gt_xyz[..., 2].mean().item()
    print(f"\nGT xyz (mean depth = {gt_mean_d:.2f}):")
    print(f"  Z values: {gt_xyz[0, :, 2].tolist()}")
    print(f"  Z range: [{gt_xyz[..., 2].min():.1f}, {gt_xyz[..., 2].max():.1f}]")

    print("\n" + "=" * 60)
    print("TEST 1: Perfect prediction (should have loss ≈ 0)")
    print("=" * 60)

    pred_perfect = gt_xyz.clone()
    loss_buggy, _, _ = compute_loss_buggy(pred_perfect, gt_xyz)
    loss_correct = compute_loss_correct(pred_perfect, gt_xyz)
    print(f"  Buggy loss:   {loss_buggy:.6f}")
    print(f"  Correct loss: {loss_correct:.6f}")

    print("\n" + "=" * 60)
    print("TEST 2: Constant depth prediction")
    print("=" * 60)

    # Prediction: same XY as GT but constant depth
    pred_const = gt_xyz.clone()
    pred_const[..., 2] = 10.0  # All same depth
    pred_mean_d = 10.0

    print(f"\nPrediction (constant depth = 10.0):")
    loss_buggy, _, _ = compute_loss_buggy(pred_const, gt_xyz)
    loss_correct = compute_loss_correct(pred_const, gt_xyz)
    print(f"  Buggy loss:   {loss_buggy:.6f}")
    print(f"  Correct loss: {loss_correct:.6f}")
    print(f"  (Buggy is lower because it normalizes by different means!)")

    print("\n" + "=" * 60)
    print("TEST 3: Scaled prediction (should be penalized)")
    print("=" * 60)

    # Prediction: correct relative positions but wrong scale
    pred_scaled = gt_xyz.clone() * 0.5  # Everything at half scale
    loss_buggy, _, _ = compute_loss_buggy(pred_scaled, gt_xyz)
    loss_correct = compute_loss_correct(pred_scaled, gt_xyz)
    print(f"\nPrediction: GT * 0.5 (half scale):")
    print(f"  Buggy loss:   {loss_buggy:.6f}")
    print(f"  Correct loss: {loss_correct:.6f}")

    pred_scaled = gt_xyz.clone() * 2.0  # Everything at double scale
    loss_buggy, _, _ = compute_loss_buggy(pred_scaled, gt_xyz)
    loss_correct = compute_loss_correct(pred_scaled, gt_xyz)
    print(f"\nPrediction: GT * 2.0 (double scale):")
    print(f"  Buggy loss:   {loss_buggy:.6f}")
    print(f"  Correct loss: {loss_correct:.6f}")

    print("\n" + "=" * 60)
    print("TEST 4: Random prediction")
    print("=" * 60)

    torch.manual_seed(42)
    pred_random = torch.randn(1, 10, 3) * 5 + 15
    pred_random[..., 2] = pred_random[..., 2].abs()  # Positive depth

    loss_buggy, _, _ = compute_loss_buggy(pred_random, gt_xyz)
    loss_correct = compute_loss_correct(pred_random, gt_xyz)
    print(f"\nPrediction: random values:")
    print(f"  Buggy loss:   {loss_buggy:.6f}")
    print(f"  Correct loss: {loss_correct:.6f}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The buggy normalization (normalizing pred and GT by their own means)
makes the loss INVARIANT to the absolute scale of predictions!

This allows the model to:
1. Predict any constant depth and get low loss
2. Predict wrong scale and still get low loss

The fix: normalize BOTH predictions and GT by GT's mean depth,
or better yet, don't use this normalization at all!
""")


if __name__ == "__main__":
    main()
