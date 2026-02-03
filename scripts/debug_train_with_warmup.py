#!/usr/bin/env python3
"""Debug training with confidence warmup to verify fix works.

This script tests the confidence warmup fix that prevents the model from
exploiting low confidence to minimize loss without learning proper 3D predictions.

Key change: During warmup (first N steps), c_effective=1 for 3D loss weighting,
so full gradients flow to xyz predictions. After warmup, learned confidence is used.
"""

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
from d4rt.losses.composite_loss import D4RTCompositeLoss


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

    # Loss function WITH confidence warmup (25000 steps = half of total)
    # This prevents the model from exploiting low confidence
    loss_fn = D4RTCompositeLoss(
        loss_weights={
            'l1_3d': 1.0,
            'l2_2d': 0.1,
            'normal': 0.5,
            'motion': 0.1,
            'visibility': 0.1,
            'confidence': 0.2,
            'use_paper_formula_3d': True,
        },
        use_paper_formula=True,
        confidence_warmup_steps=250,  # Short warmup for this debug test
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = GradScaler()

    # Training loop
    model.train()
    num_steps = 500
    step = 0
    losses = []
    confidences = []

    print(f"\n{'='*60}")
    print("DEBUG TRAINING: Paper Loss WITH Confidence Warmup")
    print(f"{'='*60}")
    print("Warmup: 250 steps (c_effective=1 → learned c)")
    print("This should prevent the model from exploiting low confidence!")
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

            # Set current step for warmup schedule
            loss_fn.set_step(step)

            with autocast('cuda', dtype=torch.float16):
                outputs = model(video, queries)

                # Compute loss with confidence warmup
                loss, loss_dict = loss_fn(
                    predictions=outputs,
                    targets=targets,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            confidences.append(loss_dict.get('mean_confidence', 0))

            # Log every 50 steps
            if step % 50 == 0:
                avg_loss = sum(losses[-50:]) / len(losses[-50:])
                avg_conf = sum(confidences[-50:]) / len(confidences[-50:])

                # Also check prediction scale
                with torch.no_grad():
                    pred_std = outputs['xyz'].std().item()
                    gt_std = targets['xyz'].std().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{avg_loss:.4f}',
                    'conf': f'{avg_conf:.2f}',
                    'warmup': f'{loss_dict.get("confidence_warmup_weight", 1.0):.2f}',
                    'pred/gt': f'{pred_std/gt_std:.2f}',
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

    print(f"\nConfidence progression:")
    print(f"  Step 0:   {confidences[0]:.4f}")
    print(f"  Step 100: {sum(confidences[95:105])/10:.4f}" if len(confidences) > 105 else "  N/A")
    print(f"  Step 250: {sum(confidences[245:255])/10:.4f}" if len(confidences) > 255 else "  N/A")
    print(f"  Step 500: {sum(confidences[-10:])/10:.4f}" if len(confidences) >= 500 else f"  Final: {sum(confidences[-10:])/10:.4f}")

    # Check if loss decreased
    initial_loss = sum(losses[:10]) / 10
    final_loss = sum(losses[-10:]) / 10

    print(f"\nInitial loss (first 10): {initial_loss:.4f}")
    print(f"Final loss (last 10):    {final_loss:.4f}")
    print(f"Reduction: {(1 - final_loss/initial_loss)*100:.1f}%")

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
    conf = torch.sigmoid(outputs['confidence'])

    print(f"\nPred XYZ range: X=[{pred_xyz[...,0].min():.2f}, {pred_xyz[...,0].max():.2f}], "
          f"Y=[{pred_xyz[...,1].min():.2f}, {pred_xyz[...,1].max():.2f}], "
          f"Z=[{pred_xyz[...,2].min():.2f}, {pred_xyz[...,2].max():.2f}]")
    print(f"GT XYZ range:   X=[{gt_xyz[...,0].min():.2f}, {gt_xyz[...,0].max():.2f}], "
          f"Y=[{gt_xyz[...,1].min():.2f}, {gt_xyz[...,1].max():.2f}], "
          f"Z=[{gt_xyz[...,2].min():.2f}, {gt_xyz[...,2].max():.2f}]")

    print(f"\nPred std: {pred_xyz.std():.4f}")
    print(f"GT std:   {gt_xyz.std():.4f}")
    print(f"Scale ratio: {pred_xyz.std() / gt_xyz.std():.4f}")

    print(f"\nMean confidence: {conf.mean():.4f}")

    val_loss = F.l1_loss(pred_xyz, gt_xyz)
    print(f"Validation L1 loss: {val_loss.item():.4f}")

    print(f"\n{'='*60}")
    print("COMPARISON WITH SIMPLE L1 TRAINING (500 steps)")
    print(f"{'='*60}")
    print("Simple L1:   Scale ratio = 0.42, conf = N/A")
    print(f"With warmup: Scale ratio = {pred_xyz.std() / gt_xyz.std():.4f}, conf = {conf.mean():.4f}")

    if pred_xyz.std() / gt_xyz.std() > 0.2:
        print("\n✓ Scale ratio improved! Warmup is working.")
    else:
        print("\n✗ Scale ratio still low. May need longer warmup or other fixes.")


if __name__ == '__main__':
    main()
