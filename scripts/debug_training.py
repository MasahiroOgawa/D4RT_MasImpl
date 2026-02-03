#!/usr/bin/env python3
"""Debug training issues - check loss components and model behavior."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf

from d4rt.models import build_d4rt_model
from d4rt.losses import D4RTCompositeLoss
from d4rt.data.datasets.kubric import KubricDataset


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model config
    model_config = OmegaConf.load('configs/model/vit_b_d4rt.yaml')
    print("\n=== Model Config ===")
    print(OmegaConf.to_yaml(model_config))

    # Build model
    print("\n=== Building Model ===")
    model = build_d4rt_model(model_config)
    model = model.to(device)
    model.eval()

    # Load a checkpoint to check trained state
    ckpt_path = 'checkpoints/checkpoint_latest.pth'
    if Path(ckpt_path).exists():
        print(f"\nLoading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from step {ckpt.get('step', 'unknown')}")
    else:
        print(f"\nNo checkpoint found at {ckpt_path}, using random weights")

    # Load a sample from dataset
    print("\n=== Loading Sample Data ===")
    dataset = KubricDataset(
        data_dir='data/kubric',
        split='val',
        num_frames=24,
        resolution=(256, 256),
        num_queries=64,
    )

    sample = dataset[0]

    # Move to device
    video = sample['video'].unsqueeze(0).to(device)  # [1, T, C, H, W]
    queries = {k: v.unsqueeze(0).to(device) for k, v in sample['queries'].items()}
    targets = {k: v.unsqueeze(0).to(device) for k, v in sample['targets'].items()}

    print(f"Video shape: {video.shape}")
    print(f"Query keys: {list(queries.keys())}")
    print(f"Target keys: {list(targets.keys())}")

    # Forward pass
    print("\n=== Forward Pass ===")
    with torch.no_grad():
        outputs = model(video, queries)

    print("\nOutput keys and shapes:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}, range=[{v.min():.4f}, {v.max():.4f}], mean={v.mean():.4f}")

    # Check confidence values
    if 'confidence' in outputs:
        conf_logits = outputs['confidence']
        conf_probs = torch.sigmoid(conf_logits)
        print(f"\nConfidence logits: range=[{conf_logits.min():.4f}, {conf_logits.max():.4f}]")
        print(f"Confidence probs:  range=[{conf_probs.min():.4f}, {conf_probs.max():.4f}], mean={conf_probs.mean():.4f}")

    # Check XYZ predictions vs GT
    if 'xyz' in outputs and 'xyz' in targets:
        pred_xyz = outputs['xyz']
        gt_xyz = targets['xyz']
        xyz_diff = (pred_xyz - gt_xyz).abs()
        print(f"\nXYZ prediction error:")
        print(f"  Mean abs error: {xyz_diff.mean():.4f}")
        print(f"  Max abs error:  {xyz_diff.max():.4f}")
        print(f"  Pred range: X=[{pred_xyz[...,0].min():.2f}, {pred_xyz[...,0].max():.2f}], "
              f"Y=[{pred_xyz[...,1].min():.2f}, {pred_xyz[...,1].max():.2f}], "
              f"Z=[{pred_xyz[...,2].min():.2f}, {pred_xyz[...,2].max():.2f}]")
        print(f"  GT range:   X=[{gt_xyz[...,0].min():.2f}, {gt_xyz[...,0].max():.2f}], "
              f"Y=[{gt_xyz[...,1].min():.2f}, {gt_xyz[...,1].max():.2f}], "
              f"Z=[{gt_xyz[...,2].min():.2f}, {gt_xyz[...,2].max():.2f}]")

        # Check if predictions are constant (token homogenization symptom)
        pred_std = pred_xyz.std(dim=1)  # std across queries
        print(f"  Pred std (across queries): X={pred_std[0,0]:.4f}, Y={pred_std[0,1]:.4f}, Z={pred_std[0,2]:.4f}")

        if pred_std.mean() < 0.1:
            print("  WARNING: Predictions have very low variance - possible constant output!")

    # Compute loss with components
    print("\n=== Loss Breakdown ===")
    loss_fn = D4RTCompositeLoss(use_paper_formula=True)

    # Need to enable gradients for loss computation
    model.train()
    outputs = model(video, queries)

    total_loss, loss_dict = loss_fn(outputs, targets)

    print(f"Total loss: {total_loss.item():.4f}")
    print("\nLoss components:")
    for k, v in sorted(loss_dict.items()):
        print(f"  {k}: {v:.4f}")

    # Check gradient flow
    print("\n=== Gradient Check ===")
    total_loss.backward()

    # Check encoder gradients
    encoder_grads = []
    for name, param in model.encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            encoder_grads.append(grad_norm)
            if 'blocks.0' in name and 'weight' in name:
                print(f"  Encoder {name}: grad_norm={grad_norm:.6f}")

    print(f"\nEncoder gradient stats:")
    print(f"  Mean: {np.mean(encoder_grads):.6f}")
    print(f"  Max:  {np.max(encoder_grads):.6f}")
    print(f"  Min:  {np.min(encoder_grads):.6f}")

    # Check decoder gradients
    decoder_grads = []
    for name, param in model.decoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            decoder_grads.append(grad_norm)

    print(f"\nDecoder gradient stats:")
    print(f"  Mean: {np.mean(decoder_grads):.6f}")
    print(f"  Max:  {np.max(decoder_grads):.6f}")
    print(f"  Min:  {np.min(decoder_grads):.6f}")

    # Check token diversity in encoder output
    print("\n=== Token Diversity Check ===")
    model.eval()
    with torch.no_grad():
        encoder_features = model.encoder(video)

    # Compute cosine similarity between tokens
    features_norm = encoder_features / encoder_features.norm(dim=-1, keepdim=True)
    sim_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))

    # Exclude diagonal
    num_tokens = encoder_features.shape[1]
    mask = ~torch.eye(num_tokens, dtype=bool, device=device)
    mean_sim = sim_matrix[0][mask].mean().item()

    print(f"Mean cosine similarity between encoder tokens: {mean_sim:.4f}")
    if mean_sim > 0.5:
        print("WARNING: High token similarity detected - spatial information may be lost!")
    else:
        print("Token diversity looks OK")

    print("\n=== Summary ===")
    issues = []
    if mean_sim > 0.5:
        issues.append("Token homogenization (high similarity)")
    if 'xyz' in outputs and outputs['xyz'].std(dim=1).mean() < 0.1:
        issues.append("Constant XYZ predictions")
    if loss_dict.get('mean_confidence', 0.5) < 0.1 or loss_dict.get('mean_confidence', 0.5) > 0.9:
        issues.append(f"Extreme confidence values ({loss_dict.get('mean_confidence', 'N/A')})")
    if np.mean(encoder_grads) < 1e-7:
        issues.append("Vanishing encoder gradients")

    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No obvious issues detected")


if __name__ == '__main__':
    main()
