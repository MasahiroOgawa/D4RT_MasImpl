#!/usr/bin/env python3
"""Debug evaluation outputs to understand why AJ/APD3D are 0."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf

from d4rt.models import build_d4rt_model
from d4rt.data.datasets.kubric import KubricDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model_config = OmegaConf.load('configs/model/vit_b_d4rt.yaml')
model = build_d4rt_model(model_config).to(device)

# Load trained checkpoint
ckpt = torch.load('checkpoints/checkpoint_step_0050000.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f"Loaded checkpoint from step {ckpt.get('step', 'unknown')}")

model.eval()

# Load dataset
dataset = KubricDataset(
    data_dir='data/kubric',
    split='val',
    num_frames=24,
    resolution=(256, 256),
    num_queries=64,
)

sample = dataset[0]
video = sample['video'].unsqueeze(0).to(device)
queries = {k: v.unsqueeze(0).to(device) for k, v in sample['queries'].items()}
targets = {k: v.unsqueeze(0).to(device) for k, v in sample['targets'].items()}

with torch.no_grad():
    outputs = model(video, queries)

print("\n" + "=" * 60)
print("DEBUG: MODEL OUTPUTS vs GROUND TRUTH")
print("=" * 60)

# Check confidence
conf = torch.sigmoid(outputs['confidence'])
print(f"\n=== Confidence ===")
print(f"Mean: {conf.mean():.4f}")
print(f"Range: [{conf.min():.4f}, {conf.max():.4f}]")
print(f"% with conf > 0.5: {(conf > 0.5).float().mean() * 100:.1f}%")

# Check XYZ predictions
pred_xyz = outputs['xyz']
gt_xyz = targets['xyz']

print(f"\n=== XYZ Predictions ===")
print(f"Pred range: X=[{pred_xyz[...,0].min():.2f}, {pred_xyz[...,0].max():.2f}], "
      f"Y=[{pred_xyz[...,1].min():.2f}, {pred_xyz[...,1].max():.2f}], "
      f"Z=[{pred_xyz[...,2].min():.2f}, {pred_xyz[...,2].max():.2f}]")
print(f"GT range:   X=[{gt_xyz[...,0].min():.2f}, {gt_xyz[...,0].max():.2f}], "
      f"Y=[{gt_xyz[...,1].min():.2f}, {gt_xyz[...,1].max():.2f}], "
      f"Z=[{gt_xyz[...,2].min():.2f}, {gt_xyz[...,2].max():.2f}]")

print(f"\nPred mean: {pred_xyz.mean(dim=1)}")
print(f"GT mean:   {gt_xyz.mean(dim=1)}")

print(f"\nPred std: {pred_xyz.std():.4f}")
print(f"GT std:   {gt_xyz.std():.4f}")
print(f"Scale ratio (pred/GT): {pred_xyz.std() / gt_xyz.std():.4f}")

# Raw errors
raw_error = (pred_xyz - gt_xyz).abs()
print(f"\n=== Raw L1 Error ===")
print(f"Mean: {raw_error.mean():.4f}")
print(f"Per-axis: X={raw_error[...,0].mean():.4f}, Y={raw_error[...,1].mean():.4f}, Z={raw_error[...,2].mean():.4f}")

# Check visibility
pred_vis = torch.sigmoid(outputs['visibility'])
gt_vis = targets['visibility']
print(f"\n=== Visibility ===")
print(f"Pred vis mean: {pred_vis.mean():.4f}")
print(f"GT vis mean: {gt_vis.mean():.4f}")

# Check UV
pred_uv = outputs['uv']
gt_uv = targets['uv']
uv_error = (pred_uv - gt_uv).abs().mean()
print(f"\n=== UV (2D) ===")
print(f"UV error: {uv_error:.4f}")
print(f"Pred UV range: [{pred_uv.min():.4f}, {pred_uv.max():.4f}]")
print(f"GT UV range: [{gt_uv.min():.4f}, {gt_uv.max():.4f}]")

# Check if predictions are constant
print(f"\n=== Prediction Diversity Check ===")
pred_xyz_std_per_query = pred_xyz.std(dim=1)
print(f"Pred XYZ std (across queries): {pred_xyz_std_per_query.mean():.4f}")
if pred_xyz_std_per_query.mean() < 1.0:
    print("WARNING: Low prediction variance - model may be outputting near-constant values!")

# Compute what AJ would need
print(f"\n=== Tracking Metric Analysis ===")
# AJ requires predictions to be within some threshold of GT
# If scale is off, AJ will be 0
l2_errors = torch.sqrt(((pred_xyz - gt_xyz) ** 2).sum(dim=-1))
print(f"L2 distance: mean={l2_errors.mean():.4f}, median={l2_errors.median():.4f}")
print(f"% points within 1.0: {(l2_errors < 1.0).float().mean() * 100:.1f}%")
print(f"% points within 5.0: {(l2_errors < 5.0).float().mean() * 100:.1f}%")
print(f"% points within 10.0: {(l2_errors < 10.0).float().mean() * 100:.1f}%")

print("\n" + "=" * 60)
if pred_xyz.std() < gt_xyz.std() * 0.1:
    print("DIAGNOSIS: Model predictions have much smaller scale than GT")
    print("The model is NOT learning proper 3D geometry - predictions cluster near origin")
elif conf.mean() < 0.3:
    print("DIAGNOSIS: Model is exploiting confidence (outputting low confidence)")
else:
    print("DIAGNOSIS: Model scale looks OK, but predictions don't match GT")
print("=" * 60)
