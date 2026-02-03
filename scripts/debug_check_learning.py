#!/usr/bin/env python3
"""Check if model is learning real predictions or exploiting confidence."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from d4rt.models import build_d4rt_model
from d4rt.data.datasets.kubric import KubricDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model_config = OmegaConf.load('configs/model/vit_b_d4rt.yaml')
model = build_d4rt_model(model_config).to(device)

# Try to load latest checkpoint
ckpt_files = sorted(Path('checkpoints').glob('checkpoint_step_*.pth'),
                   key=lambda x: int(x.stem.split('_')[-1]))
# Filter for paper architecture (>2GB)
paper_ckpts = [f for f in ckpt_files if f.stat().st_size > 2_000_000_000]

if paper_ckpts:
    latest = paper_ckpts[-1]
    print(f"Loading checkpoint: {latest}")
    ckpt = torch.load(latest, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded step {ckpt.get('step', 'unknown')}")
else:
    print("No paper architecture checkpoint found, using random weights")

model.eval()

# Load data
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
print("MODEL LEARNING CHECK")
print("=" * 60)

# Check confidence
conf = torch.sigmoid(outputs['confidence'])
print(f"\nConfidence: mean={conf.mean():.4f}, range=[{conf.min():.4f}, {conf.max():.4f}]")

if conf.mean() < 0.3:
    print("WARNING: Low mean confidence - model may be exploiting confidence weighting!")
elif conf.mean() > 0.7:
    print("Good: Model has high confidence")
else:
    print("Moderate confidence")

# Check XYZ predictions
pred_xyz = outputs['xyz']
gt_xyz = targets['xyz']

print(f"\nXYZ Predictions:")
print(f"  Pred range: X=[{pred_xyz[...,0].min():.2f}, {pred_xyz[...,0].max():.2f}], "
      f"Y=[{pred_xyz[...,1].min():.2f}, {pred_xyz[...,1].max():.2f}], "
      f"Z=[{pred_xyz[...,2].min():.2f}, {pred_xyz[...,2].max():.2f}]")
print(f"  GT range:   X=[{gt_xyz[...,0].min():.2f}, {gt_xyz[...,0].max():.2f}], "
      f"Y=[{gt_xyz[...,1].min():.2f}, {gt_xyz[...,1].max():.2f}], "
      f"Z=[{gt_xyz[...,2].min():.2f}, {gt_xyz[...,2].max():.2f}]")

# Check if predictions match GT scale
pred_scale = pred_xyz.std()
gt_scale = gt_xyz.std()
print(f"\n  Pred std: {pred_scale:.4f}")
print(f"  GT std: {gt_scale:.4f}")
print(f"  Scale ratio: {pred_scale / gt_scale:.4f}")

if pred_scale / gt_scale < 0.1:
    print("WARNING: Predictions have much smaller scale than GT!")
    print("Model may not be learning proper 3D geometry")

# Raw L1 error
raw_error = (pred_xyz - gt_xyz).abs().mean()
print(f"\nRaw L1 error: {raw_error:.4f}")

# Check UV predictions
if 'uv' in outputs:
    pred_uv = outputs['uv']
    gt_uv = targets['uv']
    uv_error = (pred_uv - gt_uv).abs().mean()
    print(f"\nUV error: {uv_error:.4f}")
    print(f"  Pred UV range: [{pred_uv.min():.4f}, {pred_uv.max():.4f}]")
    print(f"  GT UV range: [{gt_uv.min():.4f}, {gt_uv.max():.4f}]")

# Check visibility
pred_vis = torch.sigmoid(outputs['visibility'])
gt_vis = targets['visibility']
print(f"\nVisibility:")
print(f"  Pred vis mean: {pred_vis.mean():.4f}")
print(f"  GT vis mean: {gt_vis.mean():.4f}")

print("\n" + "=" * 60)
if conf.mean() < 0.3 and pred_scale / gt_scale < 0.2:
    print("DIAGNOSIS: Model is likely exploiting confidence!")
    print("Fix: Increase λ_conf or disable confidence weighting during early training")
elif raw_error > 20:
    print("DIAGNOSIS: Model predictions are far from GT")
    print("Training may need more steps or hyperparameter tuning")
else:
    print("Model appears to be learning normally")
print("=" * 60)
