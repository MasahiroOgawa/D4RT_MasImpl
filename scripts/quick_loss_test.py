#!/usr/bin/env python3
"""Quick test to verify loss is reasonable after UV fix."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from d4rt.models import build_d4rt_model
from d4rt.losses import D4RTCompositeLoss
from d4rt.data.datasets.kubric import KubricDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model (random init, no checkpoint)
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

print(f"\nGT UV range: [{targets['uv'].min():.4f}, {targets['uv'].max():.4f}]")

# Forward
with torch.no_grad():
    outputs = model(video, queries)

print(f"Pred UV range: [{outputs['uv'].min():.4f}, {outputs['uv'].max():.4f}]")

# Compute loss
loss_fn = D4RTCompositeLoss(use_paper_formula=True)
total_loss, loss_dict = loss_fn(outputs, targets)

print(f"\n=== Loss Breakdown ===")
print(f"Total loss: {total_loss.item():.4f}")
for k, v in sorted(loss_dict.items()):
    print(f"  {k}: {v:.4f}")

# Check if loss is reasonable
if loss_dict['loss_2d'] < 10:
    print(f"\n✓ 2D loss is reasonable: {loss_dict['loss_2d']:.4f}")
else:
    print(f"\n✗ 2D loss still too high: {loss_dict['loss_2d']:.4f}")
