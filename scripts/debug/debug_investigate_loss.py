#!/usr/bin/env python3
"""Investigate why initial loss is so low."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from d4rt.models import build_d4rt_model
from d4rt.losses import D4RTCompositeLoss
from d4rt.data.datasets.kubric import KubricDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model (random init)
model_config = OmegaConf.load('configs/model/vit_b_d4rt.yaml')
model = build_d4rt_model(model_config).to(device)
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

print("=" * 60)
print("INVESTIGATING LOW INITIAL LOSS")
print("=" * 60)

# Check GT statistics
print("\n=== Ground Truth Statistics ===")
gt_xyz = targets['xyz']
print(f"GT XYZ shape: {gt_xyz.shape}")
print(f"GT XYZ range: X=[{gt_xyz[...,0].min():.2f}, {gt_xyz[...,0].max():.2f}]")
print(f"             Y=[{gt_xyz[...,1].min():.2f}, {gt_xyz[...,1].max():.2f}]")
print(f"             Z=[{gt_xyz[...,2].min():.2f}, {gt_xyz[...,2].max():.2f}]")
print(f"GT XYZ mean: {gt_xyz.mean(dim=1)}")
print(f"GT XYZ std: {gt_xyz.std(dim=1)}")

gt_mean_depth = gt_xyz[..., 2:3].mean(dim=1, keepdim=True)
print(f"\nGT mean depth (for normalization): {gt_mean_depth.item():.4f}")

# Check if GT points are diverse or clustered
print(f"\n=== GT Point Diversity ===")
pairwise_dist = torch.cdist(gt_xyz, gt_xyz)
mask = ~torch.eye(64, dtype=bool, device=device)
mean_pairwise_dist = pairwise_dist[0][mask].mean()
print(f"Mean pairwise distance between GT points: {mean_pairwise_dist:.4f}")

# Forward pass
with torch.no_grad():
    outputs = model(video, queries)

pred_xyz = outputs['xyz']
print(f"\n=== Prediction Statistics ===")
print(f"Pred XYZ range: X=[{pred_xyz[...,0].min():.2f}, {pred_xyz[...,0].max():.2f}]")
print(f"              Y=[{pred_xyz[...,1].min():.2f}, {pred_xyz[...,1].max():.2f}]")
print(f"              Z=[{pred_xyz[...,2].min():.2f}, {pred_xyz[...,2].max():.2f}]")
print(f"Pred XYZ mean: {pred_xyz.mean(dim=1)}")
print(f"Pred XYZ std: {pred_xyz.std(dim=1)}")

# Raw L1 error (no normalization)
raw_l1_error = torch.abs(pred_xyz - gt_xyz).sum(dim=-1).mean()
print(f"\n=== Raw L1 Error (no normalization) ===")
print(f"Raw L1 error: {raw_l1_error.item():.4f}")

# Check the paper's normalization effect
print(f"\n=== Paper Normalization Effect ===")

# Step 1: Normalize by GT mean depth
pred_norm = pred_xyz / (gt_mean_depth + 1e-8)
gt_norm = gt_xyz / (gt_mean_depth + 1e-8)
print(f"After depth normalization:")
print(f"  Pred norm range: [{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
print(f"  GT norm range: [{gt_norm.min():.4f}, {gt_norm.max():.4f}]")

# Step 2: Apply signed-log transform
pred_transformed = torch.sign(pred_norm) * torch.log(1 + torch.abs(pred_norm))
gt_transformed = torch.sign(gt_norm) * torch.log(1 + torch.abs(gt_norm))
print(f"\nAfter signed-log transform:")
print(f"  Pred transformed range: [{pred_transformed.min():.4f}, {pred_transformed.max():.4f}]")
print(f"  GT transformed range: [{gt_transformed.min():.4f}, {gt_transformed.max():.4f}]")

# Step 3: L1 loss
L3D = torch.abs(pred_transformed - gt_transformed).sum(dim=-1)
print(f"\nPer-query L3D (after all transforms): mean={L3D.mean():.4f}, max={L3D.max():.4f}")

# Confidence effect
conf_logits = outputs['confidence']
c = torch.sigmoid(conf_logits).clamp(min=1e-6, max=1-1e-6)
print(f"\n=== Confidence Effect ===")
print(f"Confidence (sigmoid): mean={c.mean():.4f}, range=[{c.min():.4f}, {c.max():.4f}]")

# Confidence-weighted loss
weighted_3d = c.squeeze(-1) * L3D
print(f"Weighted 3D loss: mean={weighted_3d.mean():.4f}")

# Confidence penalty
conf_penalty = -0.2 * torch.log(c.squeeze(-1))
print(f"Confidence penalty: mean={conf_penalty.mean():.4f}")

print(f"\n=== Loss Breakdown ===")
print(f"L3D (raw, after transforms): {L3D.mean():.4f}")
print(f"c * L3D (confidence weighted): {weighted_3d.mean():.4f}")
print(f"-λ*log(c) (penalty): {conf_penalty.mean():.4f}")

# What would loss be WITHOUT normalization?
print(f"\n=== Comparison: With vs Without Normalization ===")
raw_error = torch.abs(pred_xyz - gt_xyz).sum(dim=-1)
print(f"Raw L1 (no normalization): {raw_error.mean():.4f}")
print(f"Paper L3D (with normalization): {L3D.mean():.4f}")
print(f"Compression ratio: {raw_error.mean() / L3D.mean():.1f}x")

# Check if loss is dominated by easy predictions
print(f"\n=== Are Some Predictions Trivially Easy? ===")
# Check visibility - maybe all points are at same location?
gt_vis = targets['visibility']
print(f"GT visibility: {gt_vis.sum().item()}/{gt_vis.numel()} visible")

# Check query temporal structure
t_src = queries['t_src']
t_tgt = queries['t_tgt']
same_frame = (t_src == t_tgt).sum().item()
print(f"Queries where t_src == t_tgt: {same_frame}/{len(t_src[0])}")
print(f"  (same-frame queries are trivially easy)")
