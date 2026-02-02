#!/usr/bin/env python3
"""Debug script to check if decoder responds to different queries."""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from d4rt.data.datasets.kubric import KubricDataset
from d4rt.models.d4rt import build_d4rt_model


def main():
    print("=" * 60)
    print("DECODER RESPONSE TEST")
    print("=" * 60)

    # Load model
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
    video = sample['video'].unsqueeze(0).to(device)

    print("\n1. Test: Same query, different t_tgt")
    print("-" * 40)

    # Create query at same (u,v) but different target times
    B, N = 1, 24
    u = torch.full((B, N), 0.5, device=device)
    v = torch.full((B, N), 0.5, device=device)
    t_src = torch.zeros(B, N, dtype=torch.long, device=device)
    t_tgt = torch.arange(N, device=device).unsqueeze(0)  # Different target times
    t_cam = t_tgt.clone()

    queries = {'u': u, 'v': v, 't_src': t_src, 't_tgt': t_tgt, 't_cam': t_cam}

    with torch.no_grad():
        outputs = model(video, queries)

    pred_xyz = outputs['xyz'][0].cpu()  # [24, 3]
    print(f"Prediction xyz for t_tgt 0-23 at (0.5, 0.5):")
    print(f"  X range: [{pred_xyz[:, 0].min():.3f}, {pred_xyz[:, 0].max():.3f}]")
    print(f"  Y range: [{pred_xyz[:, 1].min():.3f}, {pred_xyz[:, 1].max():.3f}]")
    print(f"  Z range: [{pred_xyz[:, 2].min():.3f}, {pred_xyz[:, 2].max():.3f}]")
    print(f"  Std(X): {pred_xyz[:, 0].std():.4f}")
    print(f"  Std(Y): {pred_xyz[:, 1].std():.4f}")
    print(f"  Std(Z): {pred_xyz[:, 2].std():.4f}")

    print("\n2. Test: Different (u,v), same t_tgt")
    print("-" * 40)

    # Create queries at different (u,v) but same target time
    N = 16
    u_grid = torch.linspace(0.1, 0.9, 4, device=device)
    v_grid = torch.linspace(0.1, 0.9, 4, device=device)
    u_coords, v_coords = torch.meshgrid(u_grid, v_grid, indexing='xy')
    u = u_coords.flatten().unsqueeze(0)  # [1, 16]
    v = v_coords.flatten().unsqueeze(0)  # [1, 16]

    t_src = torch.zeros(1, N, dtype=torch.long, device=device)
    t_tgt = torch.full((1, N), 12, dtype=torch.long, device=device)  # All at frame 12
    t_cam = t_tgt.clone()

    queries = {'u': u, 'v': v, 't_src': t_src, 't_tgt': t_tgt, 't_cam': t_cam}

    with torch.no_grad():
        outputs = model(video, queries)

    pred_xyz = outputs['xyz'][0].cpu()  # [16, 3]
    print(f"Prediction xyz for 4x4 grid at t_tgt=12:")
    print(f"  X range: [{pred_xyz[:, 0].min():.3f}, {pred_xyz[:, 0].max():.3f}]")
    print(f"  Y range: [{pred_xyz[:, 1].min():.3f}, {pred_xyz[:, 1].max():.3f}]")
    print(f"  Z range: [{pred_xyz[:, 2].min():.3f}, {pred_xyz[:, 2].max():.3f}]")
    print(f"  Std(X): {pred_xyz[:, 0].std():.4f}")
    print(f"  Std(Y): {pred_xyz[:, 1].std():.4f}")
    print(f"  Std(Z): {pred_xyz[:, 2].std():.4f}")

    # Show grid pattern
    print("\n  Predictions at each grid point (X, Y, Z):")
    pred_xyz_grid = pred_xyz.reshape(4, 4, 3)
    for i in range(4):
        row = []
        for j in range(4):
            xyz = pred_xyz_grid[i, j]
            row.append(f"({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f})")
        print(f"    {' '.join(row)}")

    print("\n3. Test: Random encoder features (ablation)")
    print("-" * 40)

    # Test if decoder uses encoder features at all
    with torch.no_grad():
        # Get real encoder features
        encoder_features = model.encode_video(video)

        # Create random encoder features
        random_features = torch.randn_like(encoder_features)

        # Get predictions with real vs random features
        u = torch.full((1, 8), 0.5, device=device)
        v = torch.full((1, 8), 0.5, device=device)
        t_src = torch.zeros(1, 8, dtype=torch.long, device=device)
        t_tgt = torch.arange(8, device=device).unsqueeze(0)
        t_cam = t_tgt.clone()
        queries = {'u': u, 'v': v, 't_src': t_src, 't_tgt': t_tgt, 't_cam': t_cam}

        # With real features
        out_real = model.predict_from_queries(encoder_features, queries, video)
        xyz_real = out_real['xyz'][0].cpu()

        # With random features
        out_random = model.predict_from_queries(random_features, queries, video)
        xyz_random = out_random['xyz'][0].cpu()

    print(f"Real encoder features - XYZ mean: [{xyz_real[:, 0].mean():.3f}, {xyz_real[:, 1].mean():.3f}, {xyz_real[:, 2].mean():.3f}]")
    print(f"Random encoder features - XYZ mean: [{xyz_random[:, 0].mean():.3f}, {xyz_random[:, 1].mean():.3f}, {xyz_random[:, 2].mean():.3f}]")
    diff = (xyz_real - xyz_random).abs().mean()
    print(f"Difference: {diff:.4f}")
    if diff < 0.5:
        print("  WARNING: Decoder barely uses encoder features!")
    else:
        print("  OK: Decoder uses encoder features")

    print("\n4. Check encoder feature statistics")
    print("-" * 40)
    print(f"Encoder features shape: {encoder_features.shape}")
    print(f"Encoder features mean: {encoder_features.mean():.4f}")
    print(f"Encoder features std: {encoder_features.std():.4f}")
    print(f"Encoder features range: [{encoder_features.min():.4f}, {encoder_features.max():.4f}]")


if __name__ == "__main__":
    main()
