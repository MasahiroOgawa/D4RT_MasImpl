#!/usr/bin/env python3
"""Check ground truth UV range in dataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from d4rt.data.datasets.kubric import KubricDataset

dataset = KubricDataset(
    data_dir='data/kubric',
    split='val',
    num_frames=24,
    resolution=(256, 256),
    num_queries=64,
)

sample = dataset[0]
gt_uv = sample['targets']['uv']

print(f"GT UV shape: {gt_uv.shape}")
print(f"GT UV range: [{gt_uv.min():.4f}, {gt_uv.max():.4f}]")
print(f"GT UV mean: {gt_uv.mean():.4f}")
print(f"GT UV first 5:\n{gt_uv[:5]}")

# Check if UV is in [0,1] or pixel coordinates
H, W = 256, 256
if gt_uv.max() > 1.5:
    print(f"\nUV appears to be in PIXEL coordinates (not normalized)")
    print(f"Expected range for normalized: [0, 1]")
    print(f"Actual max: {gt_uv.max():.1f}")
else:
    print(f"\nUV appears to be normalized [0, 1]")
