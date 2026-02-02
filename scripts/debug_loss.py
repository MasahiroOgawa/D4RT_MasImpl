#!/usr/bin/env python3
"""Debug script to investigate loss function behavior."""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from d4rt.data.datasets.kubric import KubricDataset
from d4rt.models.d4rt import build_d4rt_model
from d4rt.losses.composite_loss import CompositeLoss


def main():
    print("=" * 60)
    print("LOSS FUNCTION DEBUG")
    print("=" * 60)

    # Load model and checkpoint
    model_config = OmegaConf.load("configs/model/vit_b_movi.yaml")
    model = build_d4rt_model(model_config)

    ckpt = torch.load("checkpoints/checkpoint_step_0050000.pth", map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create dataset
    dataset = KubricDataset(
        data_dir="data/kubric",
        split="val",
        num_frames=24,
        resolution=(256, 256),
        num_queries=64,
    )

    sample = dataset[0]

    # Prepare batch
    video = sample['video'].unsqueeze(0).to(device)
    queries = {k: v.unsqueeze(0).to(device) for k, v in sample['queries'].items()}
    targets = {k: v.unsqueeze(0).to(device) for k, v in sample['targets'].items()}

    print(f"\nTarget 'xyz' shape: {targets['xyz'].shape}")
    print(f"Target 'xyz' sample values:\n{targets['xyz'][0, :5]}")

    # Run forward pass
    with torch.no_grad():
        outputs = model(video, queries)

    print(f"\nOutput 'xyz' shape: {outputs['xyz'].shape}")
    print(f"Output 'xyz' sample values:\n{outputs['xyz'][0, :5]}")

    # Create loss function with paper weights
    loss_config = {
        'l1_3d': 1.0,
        'l2_2d': 0.1,
        'normal': 0.5,
        'motion': 0.1,
        'visibility': 0.1,
        'confidence': 0.2,
        'use_paper_formula_3d': True,
    }
    loss_fn = CompositeLoss(loss_config)

    # Compute loss
    with torch.no_grad():
        loss_dict = loss_fn(outputs, targets)

    print("\n" + "=" * 60)
    print("LOSS BREAKDOWN")
    print("=" * 60)
    total = 0
    for name, value in loss_dict.items():
        if name != 'total':
            print(f"  {name}: {value.item():.4f}")
            total += value.item()
    print(f"  ---")
    print(f"  TOTAL: {loss_dict['total'].item():.4f}")

    # Check L1 3D loss manually
    print("\n" + "=" * 60)
    print("MANUAL L1 3D LOSS CHECK")
    print("=" * 60)

    pred_xyz = outputs['xyz'][0].cpu()
    gt_xyz = targets['xyz'][0].cpu()

    raw_diff = (pred_xyz - gt_xyz).abs()
    print(f"Raw L1 diff (no normalization):")
    print(f"  Mean: {raw_diff.mean():.4f}")
    print(f"  Per-dim: X={raw_diff[:,0].mean():.4f}, Y={raw_diff[:,1].mean():.4f}, Z={raw_diff[:,2].mean():.4f}")

    # With mean depth normalization (paper formula)
    mean_depth = gt_xyz[:, 2].mean()
    print(f"\nMean GT depth: {mean_depth:.4f}")

    normalized_pred = pred_xyz / mean_depth
    normalized_gt = gt_xyz / mean_depth
    normalized_diff = (normalized_pred - normalized_gt).abs()
    print(f"Normalized L1 diff (divide by mean depth):")
    print(f"  Mean: {normalized_diff.mean():.4f}")

    # Check if the problem is that GT xyz values are too large
    print("\n" + "=" * 60)
    print("GT VALUE ANALYSIS")
    print("=" * 60)

    print(f"GT xyz statistics:")
    print(f"  X: min={gt_xyz[:,0].min():.2f}, max={gt_xyz[:,0].max():.2f}, mean={gt_xyz[:,0].mean():.2f}")
    print(f"  Y: min={gt_xyz[:,1].min():.2f}, max={gt_xyz[:,1].max():.2f}, mean={gt_xyz[:,1].mean():.2f}")
    print(f"  Z: min={gt_xyz[:,2].min():.2f}, max={gt_xyz[:,2].max():.2f}, mean={gt_xyz[:,2].mean():.2f}")

    # Check for invalid values
    print(f"\n  Has NaN: {torch.isnan(gt_xyz).any()}")
    print(f"  Has Inf: {torch.isinf(gt_xyz).any()}")

    # Check extreme values
    large_vals = (gt_xyz.abs() > 30).sum()
    print(f"  Values > 30 (absolute): {large_vals} / {gt_xyz.numel()}")

    # The -33.749 value appears suspiciously - let's check if it's a sentinel
    print(f"\n  Exact -33.749 count: {(gt_xyz == -33.749).sum()}")
    print(f"  Exact 67.499 count: {(gt_xyz == 67.499).sum()}")

    # These might be invalid depth values!
    print("\n" + "=" * 60)
    print("VISIBILITY vs EXTREME VALUES")
    print("=" * 60)

    vis = targets['visibility'][0].cpu()
    print(f"Visibility: {vis.sum()}/{len(vis)} visible")

    # Check if extreme values correlate with invisible points
    for i in range(min(10, len(gt_xyz))):
        print(f"  Point {i}: xyz=[{gt_xyz[i,0]:.2f}, {gt_xyz[i,1]:.2f}, {gt_xyz[i,2]:.2f}], vis={vis[i].item()}")


if __name__ == "__main__":
    main()
