#!/usr/bin/env python3
"""Compare trained vs untrained model behavior."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from d4rt.data.datasets.kubric import KubricDataset
from d4rt.models.d4rt import build_d4rt_model


def test_model(model, video, name):
    """Test model response to different queries."""
    device = next(model.parameters()).device

    # Create queries at a 4x4 grid
    N = 16
    u_grid = torch.linspace(0.1, 0.9, 4, device=device)
    v_grid = torch.linspace(0.1, 0.9, 4, device=device)
    u_coords, v_coords = torch.meshgrid(u_grid, v_grid, indexing='xy')
    u = u_coords.flatten().unsqueeze(0)
    v = v_coords.flatten().unsqueeze(0)

    t_src = torch.zeros(1, N, dtype=torch.long, device=device)
    t_tgt = torch.full((1, N), 12, dtype=torch.long, device=device)
    t_cam = t_tgt.clone()

    queries = {'u': u, 'v': v, 't_src': t_src, 't_tgt': t_tgt, 't_cam': t_cam}

    with torch.no_grad():
        outputs = model(video, queries)

    pred_xyz = outputs['xyz'][0].cpu()

    print(f"\n{name}:")
    print(f"  X: range=[{pred_xyz[:,0].min():.3f}, {pred_xyz[:,0].max():.3f}], std={pred_xyz[:,0].std():.4f}")
    print(f"  Y: range=[{pred_xyz[:,1].min():.3f}, {pred_xyz[:,1].max():.3f}], std={pred_xyz[:,1].std():.4f}")
    print(f"  Z: range=[{pred_xyz[:,2].min():.3f}, {pred_xyz[:,2].max():.3f}], std={pred_xyz[:,2].std():.4f}")

    return pred_xyz


def main():
    print("=" * 60)
    print("TRAINED VS UNTRAINED MODEL COMPARISON")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model config
    model_config = OmegaConf.load("configs/model/vit_b_movi.yaml")

    # Build untrained model
    print("\nBuilding untrained model...")
    model_untrained = build_d4rt_model(model_config)
    model_untrained = model_untrained.to(device)
    model_untrained.eval()

    # Build trained model
    print("Loading trained model...")
    model_trained = build_d4rt_model(model_config)
    ckpt = torch.load("checkpoints/checkpoint_step_0050000.pth", map_location='cpu')
    model_trained.load_state_dict(ckpt['model_state_dict'])
    model_trained = model_trained.to(device)
    model_trained.eval()

    # Load test video
    dataset = KubricDataset(
        data_dir="data/kubric",
        split="val",
        num_frames=24,
        resolution=(256, 256),
        num_queries=64,
    )
    sample = dataset[0]
    video = sample['video'].unsqueeze(0).to(device)

    print("\n" + "=" * 60)
    print("TEST: 4x4 grid of queries at frame 12")
    print("=" * 60)

    pred_untrained = test_model(model_untrained, video, "UNTRAINED MODEL")
    pred_trained = test_model(model_trained, video, "TRAINED MODEL")

    # Check GT for comparison
    targets = sample['targets']
    gt_xyz = targets['xyz']
    print(f"\nGROUND TRUTH (for reference):")
    print(f"  X: range=[{gt_xyz[:,0].min():.3f}, {gt_xyz[:,0].max():.3f}], std={gt_xyz[:,0].std():.4f}")
    print(f"  Y: range=[{gt_xyz[:,1].min():.3f}, {gt_xyz[:,1].max():.3f}], std={gt_xyz[:,1].std():.4f}")
    print(f"  Z: range=[{gt_xyz[:,2].min():.3f}, {gt_xyz[:,2].max():.3f}], std={gt_xyz[:,2].std():.4f}")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    untrained_std = pred_untrained.std()
    trained_std = pred_trained.std()

    print(f"\nOverall prediction std:")
    print(f"  Untrained: {untrained_std:.4f}")
    print(f"  Trained:   {trained_std:.4f}")

    if trained_std < untrained_std * 0.5:
        print("\n  WARNING: Training REDUCED output variance!")
        print("  This suggests the model collapsed to a constant during training.")
    elif trained_std > untrained_std * 1.5:
        print("\n  OK: Training INCREASED output variance (expected)")
    else:
        print("\n  SIMILAR variance before/after training")

    # Check if trained model outputs are close to GT mean
    gt_mean = gt_xyz.mean(dim=0)
    pred_mean = pred_trained.mean(dim=0)
    print(f"\nGT mean: [{gt_mean[0]:.3f}, {gt_mean[1]:.3f}, {gt_mean[2]:.3f}]")
    print(f"Trained pred mean: [{pred_mean[0]:.3f}, {pred_mean[1]:.3f}, {pred_mean[2]:.3f}]")


if __name__ == "__main__":
    main()
