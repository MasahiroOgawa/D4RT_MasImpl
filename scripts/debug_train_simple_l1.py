#!/usr/bin/env python3
"""Debug training with simple L1 3D loss only (no confidence weighting, no transforms)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from omegaconf import OmegaConf

from d4rt.models import build_d4rt_model
from d4rt.data.datasets.kubric import KubricDataset

def simple_l1_loss(pred_xyz, gt_xyz):
    """Simple L1 loss - no normalization, no confidence weighting."""
    return F.l1_loss(pred_xyz, gt_xyz)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model_config = OmegaConf.load('configs/model/vit_b_d4rt.yaml')
    model = build_d4rt_model(model_config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Dataset
    dataset = KubricDataset(
        data_dir='data/kubric',
        split='train',
        num_frames=24,
        resolution=(256, 256),
        num_queries=64,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer - simple setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = GradScaler()

    # Training loop
    model.train()
    num_steps = 500  # ~10 minutes at ~1 it/s
    step = 0
    losses = []

    print(f"\n{'='*60}")
    print("DEBUG TRAINING: Simple L1 3D Loss Only")
    print(f"{'='*60}")
    print("No confidence weighting, no signed-log transform")
    print(f"Training for {num_steps} steps...")
    print()

    pbar = tqdm(total=num_steps, desc="Training")

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            video = batch['video'].to(device)
            queries = {k: v.to(device) for k, v in batch['queries'].items()}
            targets = {k: v.to(device) for k, v in batch['targets'].items()}

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.float16):
                outputs = model(video, queries)

                # Simple L1 loss on XYZ only
                loss = simple_l1_loss(outputs['xyz'], targets['xyz'])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

            # Log every 50 steps
            if step % 50 == 0:
                avg_loss = sum(losses[-50:]) / len(losses[-50:])

                # Also check prediction scale
                with torch.no_grad():
                    pred_std = outputs['xyz'].std().item()
                    gt_std = targets['xyz'].std().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{avg_loss:.4f}',
                    'pred_std': f'{pred_std:.2f}',
                    'gt_std': f'{gt_std:.2f}',
                })

            pbar.update(1)
            step += 1

    pbar.close()

    # Final analysis
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")

    print(f"\nLoss progression:")
    print(f"  Step 0:   {losses[0]:.4f}")
    print(f"  Step 100: {sum(losses[95:105])/10:.4f}" if len(losses) > 105 else "  N/A")
    print(f"  Step 250: {sum(losses[245:255])/10:.4f}" if len(losses) > 255 else "  N/A")
    print(f"  Step 500: {sum(losses[-10:])/10:.4f}" if len(losses) >= 500 else f"  Final: {sum(losses[-10:])/10:.4f}")

    # Check if loss decreased
    initial_loss = sum(losses[:10]) / 10
    final_loss = sum(losses[-10:]) / 10

    print(f"\nInitial loss (first 10): {initial_loss:.4f}")
    print(f"Final loss (last 10):    {final_loss:.4f}")
    print(f"Reduction: {(1 - final_loss/initial_loss)*100:.1f}%")

    if final_loss < initial_loss * 0.8:
        print("\n✓ Loss decreased significantly - model CAN learn with simple L1 loss!")
    elif final_loss < initial_loss:
        print("\n~ Loss decreased slightly - learning is happening but slow")
    else:
        print("\n✗ Loss did not decrease - there may be other issues")

    # Test on validation sample
    print(f"\n{'='*60}")
    print("VALIDATION CHECK")
    print(f"{'='*60}")

    model.eval()
    val_dataset = KubricDataset(
        data_dir='data/kubric',
        split='val',
        num_frames=24,
        resolution=(256, 256),
        num_queries=64,
    )

    sample = val_dataset[0]
    video = sample['video'].unsqueeze(0).to(device)
    queries = {k: v.unsqueeze(0).to(device) for k, v in sample['queries'].items()}
    targets = {k: v.unsqueeze(0).to(device) for k, v in sample['targets'].items()}

    with torch.no_grad():
        outputs = model(video, queries)

    pred_xyz = outputs['xyz']
    gt_xyz = targets['xyz']

    print(f"\nPred XYZ range: X=[{pred_xyz[...,0].min():.2f}, {pred_xyz[...,0].max():.2f}], "
          f"Y=[{pred_xyz[...,1].min():.2f}, {pred_xyz[...,1].max():.2f}], "
          f"Z=[{pred_xyz[...,2].min():.2f}, {pred_xyz[...,2].max():.2f}]")
    print(f"GT XYZ range:   X=[{gt_xyz[...,0].min():.2f}, {gt_xyz[...,0].max():.2f}], "
          f"Y=[{gt_xyz[...,1].min():.2f}, {gt_xyz[...,1].max():.2f}], "
          f"Z=[{gt_xyz[...,2].min():.2f}, {gt_xyz[...,2].max():.2f}]")

    print(f"\nPred std: {pred_xyz.std():.4f}")
    print(f"GT std:   {gt_xyz.std():.4f}")
    print(f"Scale ratio: {pred_xyz.std() / gt_xyz.std():.4f}")

    val_loss = simple_l1_loss(pred_xyz, gt_xyz)
    print(f"\nValidation L1 loss: {val_loss.item():.4f}")

if __name__ == '__main__':
    main()
