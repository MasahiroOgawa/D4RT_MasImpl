#!/usr/bin/env python3
"""
Debug script for analyzing gradient flow in cross-attention layers.

This script diagnoses why the decoder ignores encoder features by tracking:
1. Gradient norms per layer
2. Weight changes per layer
3. Cross-attention attention weight statistics

Diagnosis table:
| Symptom                                    | Diagnosis                     |
|--------------------------------------------|-------------------------------|
| Cross-attn gradients = 0                   | Gradient disconnected         |
| Cross-attn gradients small, weights static | LR too low or grad clipping   |
| Cross-attn gradients exist, outputs same   | Output head dominates         |
| Encoder gradients = 0                      | Encoder frozen (expected)     |
| All gradients normal but model collapses   | Loss function issue           |
"""

import sys
import json
import copy
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from d4rt.models.d4rt import build_d4rt_model
from d4rt.data.datasets.kubric import KubricDataset
from d4rt.losses import CompositeLoss


class GradientFlowDebugger:
    """Debug gradient flow through the model."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.gradient_norms = defaultdict(list)
        self.weight_snapshots = {}
        self.attention_stats = defaultdict(list)
        self.hooks = []

        # Define layers to monitor
        self.layers_to_monitor = self._get_layers_to_monitor()

    def _get_layers_to_monitor(self):
        """Get dictionary of layer name -> module pairs to monitor."""
        layers = {}

        # Encoder (frozen - expect zero gradients)
        if hasattr(self.model.encoder, 'blocks') and len(self.model.encoder.blocks) > 0:
            layers['encoder.block0.self_attn.q'] = self.model.encoder.blocks[0].self_attn.q_proj

        # Context projection (768 -> 512)
        layers['decoder.context_proj'] = self.model.decoder.context_proj

        # Query projection
        layers['decoder.query_proj'] = self.model.decoder.query_proj

        # Cross-attention layers (first, middle, last)
        num_layers = len(self.model.decoder.layers)
        layer_indices = [0, num_layers // 2, num_layers - 1]

        for i in layer_indices:
            layer = self.model.decoder.layers[i]

            # Cross-attention Q, K, V
            layers[f'decoder.layer{i}.cross_attn.q'] = layer.cross_attn_block.cross_attn.q_proj
            layers[f'decoder.layer{i}.cross_attn.k'] = layer.cross_attn_block.cross_attn.k_proj
            layers[f'decoder.layer{i}.cross_attn.v'] = layer.cross_attn_block.cross_attn.v_proj
            layers[f'decoder.layer{i}.cross_attn.out'] = layer.cross_attn_block.cross_attn.out_proj

            # Self-attention (for comparison)
            layers[f'decoder.layer{i}.self_attn.q'] = layer.self_attn_block.self_attn.q_proj

        # Output heads
        layers['decoder.xyz_head'] = self.model.decoder.xyz_head
        layers['decoder.vis_head'] = self.model.decoder.vis_head
        layers['decoder.confidence_head'] = self.model.decoder.confidence_head

        return layers

    def snapshot_weights(self):
        """Take a snapshot of current weights."""
        self.weight_snapshots = {}
        for name, module in self.layers_to_monitor.items():
            if hasattr(module, 'weight') and module.weight is not None:
                self.weight_snapshots[name] = module.weight.data.clone()

    def compute_weight_changes(self):
        """Compute L2 norm of weight changes since last snapshot."""
        changes = {}
        for name, module in self.layers_to_monitor.items():
            if name in self.weight_snapshots and hasattr(module, 'weight'):
                delta = module.weight.data - self.weight_snapshots[name]
                changes[name] = delta.norm().item()
        return changes

    def register_gradient_hooks(self):
        """Register hooks to capture gradients."""
        self._clear_hooks()

        for name, module in self.layers_to_monitor.items():
            if hasattr(module, 'weight') and module.weight is not None:
                # Skip frozen weights (no gradient)
                if not module.weight.requires_grad:
                    # Record zero gradients for frozen layers
                    self.gradient_norms[name] = []  # Will be filled with zeros
                    continue

                # Create closure to capture name
                def make_hook(layer_name):
                    def hook(grad):
                        if grad is not None:
                            self.gradient_norms[layer_name].append(grad.norm().item())
                        else:
                            self.gradient_norms[layer_name].append(0.0)
                    return hook

                handle = module.weight.register_hook(make_hook(name))
                self.hooks.append(handle)

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def get_gradient_stats(self, step):
        """Get gradient statistics for current step."""
        stats = {}
        for name, module in self.layers_to_monitor.items():
            # Check if frozen
            if hasattr(module, 'weight') and module.weight is not None:
                if not module.weight.requires_grad:
                    stats[name] = 0.0  # Frozen layer
                    continue

            if name in self.gradient_norms and len(self.gradient_norms[name]) > step:
                stats[name] = self.gradient_norms[name][step]
            else:
                stats[name] = 0.0
        return stats


def create_mini_dataset(data_dir, num_samples=3):
    """Create a mini dataset with only a few samples for overfitting test."""
    dataset = KubricDataset(
        data_dir=data_dir,
        split="train",
        num_frames=24,
        resolution=(256, 256),
        num_queries=128,
    )

    # Limit to first N samples
    dataset.samples = dataset.samples[:num_samples]
    print(f"Limited dataset to {len(dataset.samples)} samples")

    return dataset


def run_training_steps(
    model,
    dataset,
    loss_fn,
    lr=1e-4,
    num_steps=50,
    device='cuda',
    drop_path_override=None,
):
    """Run training steps and collect gradient statistics."""
    model = model.to(device)
    model.train()

    # Override drop_path if specified
    if drop_path_override is not None:
        _set_drop_path(model, drop_path_override)

    # Create debugger
    debugger = GradientFlowDebugger(model, device)
    debugger.register_gradient_hooks()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.03,
    )

    # Track results
    results = {
        'lr': lr,
        'drop_path': drop_path_override,
        'steps': [],
    }

    # Take initial weight snapshot
    debugger.snapshot_weights()
    initial_weights = {k: v.clone() for k, v in debugger.weight_snapshots.items()}

    print(f"\n{'='*60}")
    print(f"Training with LR={lr}, drop_path={drop_path_override}")
    print(f"{'='*60}")

    for step in range(num_steps):
        # Get batch (cycle through mini dataset)
        sample_idx = step % len(dataset)
        sample = dataset[sample_idx]

        # Prepare batch
        video = sample['video'].unsqueeze(0).to(device)
        queries = {k: v.unsqueeze(0).to(device) for k, v in sample['queries'].items()}
        targets = {k: v.unsqueeze(0).to(device) for k, v in sample['targets'].items()}
        cameras = {k: v.unsqueeze(0).to(device) for k, v in sample['cameras'].items()}

        # Forward pass
        optimizer.zero_grad()
        outputs = model(video, queries)

        # Compute loss
        loss, loss_dict = loss_fn(
            predictions=outputs,
            targets=targets,
            cameras=cameras,
            queries=queries,
        )

        # Backward pass
        loss.backward()

        # Collect gradient stats before optimizer step
        grad_stats = debugger.get_gradient_stats(step)

        # Optimizer step
        optimizer.step()

        # Compute weight changes
        weight_changes = debugger.compute_weight_changes()
        debugger.snapshot_weights()

        # Compute cumulative weight changes from initial
        cumulative_changes = {}
        for name in initial_weights.keys():
            if hasattr(debugger.layers_to_monitor[name], 'weight'):
                delta = debugger.layers_to_monitor[name].weight.data - initial_weights[name]
                cumulative_changes[name] = delta.norm().item()

        # Store step results
        step_result = {
            'step': step,
            'loss_total': loss.item(),
            'loss_breakdown': {k: v if isinstance(v, float) else v for k, v in loss_dict.items()},
            'gradient_norms': grad_stats,
            'weight_changes': weight_changes,
            'cumulative_weight_changes': cumulative_changes,
        }
        results['steps'].append(step_result)

        # Print progress every 10 steps
        if step % 10 == 0 or step == num_steps - 1:
            print(f"\nStep {step}:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Gradient norms (sample):")
            for name in ['decoder.layer0.cross_attn.q', 'decoder.layer0.self_attn.q', 'decoder.xyz_head']:
                if name in grad_stats:
                    print(f"    {name}: {grad_stats[name]:.6e}")
            print(f"  Weight changes (sample):")
            for name in ['decoder.layer0.cross_attn.q', 'decoder.xyz_head']:
                if name in weight_changes:
                    print(f"    {name}: {weight_changes[name]:.6e}")

    # Cleanup
    debugger._clear_hooks()

    return results


def _set_drop_path(model, drop_prob):
    """Set drop_path probability for all DropPath modules."""
    from d4rt.models.components.attention import DropPath

    for module in model.modules():
        if isinstance(module, DropPath):
            module.drop_prob = drop_prob


def analyze_results(results_list, output_dir):
    """Analyze and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    for i, results in enumerate(results_list):
        filename = f"results_lr{results['lr']}_dp{results['drop_path']}.json"
        # Convert numpy types to Python types for JSON serialization
        clean_results = _clean_for_json(results)
        with open(output_dir / filename, 'w') as f:
            json.dump(clean_results, f, indent=2)

    # Generate summary report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("GRADIENT FLOW ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    for results in results_list:
        lr = results['lr']
        dp = results['drop_path']
        steps = results['steps']

        report_lines.append(f"\n{'='*50}")
        report_lines.append(f"Configuration: LR={lr}, DropPath={dp}")
        report_lines.append(f"{'='*50}")

        # Loss trajectory
        losses = [s['loss_total'] for s in steps]
        report_lines.append(f"\nLoss: {losses[0]:.4f} -> {losses[-1]:.4f} (delta: {losses[-1]-losses[0]:.4f})")

        # Final gradient norms
        final_grads = steps[-1]['gradient_norms']
        report_lines.append("\nFinal Gradient Norms:")

        # Group by type
        encoder_grads = {k: v for k, v in final_grads.items() if k.startswith('encoder')}
        cross_attn_grads = {k: v for k, v in final_grads.items() if 'cross_attn' in k}
        self_attn_grads = {k: v for k, v in final_grads.items() if 'self_attn' in k and 'decoder' in k}
        head_grads = {k: v for k, v in final_grads.items() if 'head' in k}
        proj_grads = {k: v for k, v in final_grads.items() if 'proj' in k and 'attn' not in k}

        for group_name, group_grads in [
            ("Encoder (should be 0 if frozen)", encoder_grads),
            ("Cross-Attention", cross_attn_grads),
            ("Self-Attention", self_attn_grads),
            ("Output Heads", head_grads),
            ("Projections", proj_grads),
        ]:
            if group_grads:
                report_lines.append(f"\n  {group_name}:")
                for k, v in sorted(group_grads.items()):
                    report_lines.append(f"    {k}: {v:.6e}")

        # Cumulative weight changes
        final_changes = steps[-1]['cumulative_weight_changes']
        report_lines.append("\nCumulative Weight Changes (from initial):")

        cross_attn_changes = {k: v for k, v in final_changes.items() if 'cross_attn' in k}
        self_attn_changes = {k: v for k, v in final_changes.items() if 'self_attn' in k}
        head_changes = {k: v for k, v in final_changes.items() if 'head' in k}

        for group_name, group_changes in [
            ("Cross-Attention", cross_attn_changes),
            ("Self-Attention", self_attn_changes),
            ("Output Heads", head_changes),
        ]:
            if group_changes:
                report_lines.append(f"\n  {group_name}:")
                for k, v in sorted(group_changes.items()):
                    report_lines.append(f"    {k}: {v:.6e}")

        # Diagnosis
        report_lines.append("\n" + "-" * 40)
        report_lines.append("DIAGNOSIS:")

        # Check cross-attention gradients
        cross_attn_grad_vals = list(cross_attn_grads.values())
        if cross_attn_grad_vals:
            mean_cross_grad = np.mean(cross_attn_grad_vals)
            if mean_cross_grad == 0:
                report_lines.append("  [CRITICAL] Cross-attention gradients are ZERO - gradient disconnected!")
            elif mean_cross_grad < 1e-8:
                report_lines.append("  [WARNING] Cross-attention gradients very small (<1e-8)")
            else:
                report_lines.append(f"  [OK] Cross-attention gradients exist: mean={mean_cross_grad:.6e}")

        # Check if cross-attn vs self-attn gradient ratio
        self_attn_grad_vals = list(self_attn_grads.values())
        if cross_attn_grad_vals and self_attn_grad_vals:
            ratio = np.mean(cross_attn_grad_vals) / (np.mean(self_attn_grad_vals) + 1e-12)
            report_lines.append(f"  Cross/Self attention gradient ratio: {ratio:.4f}")
            if ratio < 0.01:
                report_lines.append("  [WARNING] Cross-attention receives much weaker gradients than self-attention")

        # Check weight changes
        cross_attn_change_vals = list(cross_attn_changes.values())
        if cross_attn_change_vals:
            mean_change = np.mean(cross_attn_change_vals)
            if mean_change < 1e-6:
                report_lines.append(f"  [WARNING] Cross-attention weights barely changed: {mean_change:.6e}")
            else:
                report_lines.append(f"  [OK] Cross-attention weights changed: {mean_change:.6e}")

    # Save report
    report = "\n".join(report_lines)
    with open(output_dir / "analysis_report.txt", 'w') as f:
        f.write(report)

    print("\n" + report)

    return report


def _clean_for_json(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def main():
    print("=" * 70)
    print("GRADIENT FLOW DEBUG SCRIPT")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model config
    model_config = OmegaConf.load("configs/model/vit_b_movi.yaml")

    # Create loss function
    loss_weights = {
        'l1_3d': 1.0,
        'l2_2d': 0.1,
        'visibility': 0.1,
        'normal': 0.0,  # No ground truth normals in MOVi
        'motion': 0.1,
        'confidence': 0.2,
        'use_paper_formula_3d': True,
    }
    loss_fn = CompositeLoss(loss_weights=loss_weights).to(device)

    # Create mini dataset
    print("\nCreating mini dataset (3 samples for overfitting test)...")
    dataset = create_mini_dataset("data/kubric", num_samples=3)

    # Output directory
    output_dir = Path("logs/gradient_debug") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    results_list = []

    # Experiment 1: Different learning rates
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Learning Rate Comparison")
    print("=" * 70)

    for lr in [1e-3, 1e-4, 1e-5]:
        print(f"\n--- Testing LR={lr} ---")
        model = build_d4rt_model(model_config).to(device)

        # Freeze encoder (as in normal training)
        for param in model.encoder.parameters():
            param.requires_grad = False

        results = run_training_steps(
            model=model,
            dataset=dataset,
            loss_fn=loss_fn,
            lr=lr,
            num_steps=50,
            device=device,
            drop_path_override=None,  # Use default (0.1)
        )
        results_list.append(results)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Experiment 2: DropPath disabled
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: DropPath Disabled")
    print("=" * 70)

    model = build_d4rt_model(model_config).to(device)
    for param in model.encoder.parameters():
        param.requires_grad = False

    results = run_training_steps(
        model=model,
        dataset=dataset,
        loss_fn=loss_fn,
        lr=1e-4,  # Use middle LR
        num_steps=50,
        device=device,
        drop_path_override=0.0,  # Disable DropPath
    )
    results_list.append(results)

    del model
    torch.cuda.empty_cache()

    # Experiment 3: Encoder unfrozen (to verify gradient flow path)
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Encoder Unfrozen (verify gradient path)")
    print("=" * 70)

    model = build_d4rt_model(model_config).to(device)
    # Don't freeze encoder
    results = run_training_steps(
        model=model,
        dataset=dataset,
        loss_fn=loss_fn,
        lr=1e-4,
        num_steps=50,
        device=device,
        drop_path_override=None,
    )
    results['drop_path'] = 'unfrozen_encoder'
    results_list.append(results)

    del model
    torch.cuda.empty_cache()

    # Analyze and save results
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    analyze_results(results_list, output_dir)

    print(f"\nResults saved to: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
