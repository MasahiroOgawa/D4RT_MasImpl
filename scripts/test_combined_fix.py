#!/usr/bin/env python3
"""
Test combined fixes: Context Pooling + Higher Q/K LR

Based on Phase 2 results:
- Context Pooling (3072→256): Best overall, 2.3× Q/K gradients
- Q/K LR 10×: Best entropy reduction (1.0→0.94)

This script tests combinations with longer training.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from d4rt.models.d4rt import build_d4rt_model
from d4rt.data.datasets.kubric import KubricDataset
from d4rt.losses import CompositeLoss


class ContextPooling(nn.Module):
    """Pool context tokens to reduce count."""

    def __init__(self, input_tokens=3072, output_tokens=256, embed_dim=512):
        super().__init__()
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

        # Learnable pooling via linear projection
        self.pool = nn.Linear(input_tokens, output_tokens)
        nn.init.xavier_uniform_(self.pool.weight)
        nn.init.zeros_(self.pool.bias)

    def forward(self, x):
        # x: [B, input_tokens, embed_dim]
        x = x.transpose(1, 2)  # [B, embed_dim, input_tokens]
        x = self.pool(x)       # [B, embed_dim, output_tokens]
        x = x.transpose(1, 2)  # [B, output_tokens, embed_dim]
        return x


def apply_context_pooling(model, output_tokens=256):
    """Add context pooling before decoder."""
    device = next(model.parameters()).device
    original_context_proj = model.decoder.context_proj

    pooling = ContextPooling(3072, output_tokens, 512).to(device)

    class CombinedContextProj(nn.Module):
        def __init__(self, proj, pool):
            super().__init__()
            self.proj = proj
            self.pool = pool

        def forward(self, x):
            x = self.proj(x)
            x = self.pool(x)
            return x

    model.decoder.context_proj = CombinedContextProj(original_context_proj, pooling)
    return model


def compute_attention_stats(model, video, queries):
    """Compute attention statistics."""
    model.eval()

    with torch.no_grad():
        encoder_features = model.encoder(video)

        from d4rt.utils.patch_utils import extract_patches
        patches = extract_patches(video, queries['u'], queries['v'], queries['t_src'], patch_size=9)
        query_embeddings = model.query_encoder(
            queries['u'], queries['v'], queries['t_src'], queries['t_tgt'], queries['t_cam'], patches
        )

        x = model.decoder.query_proj(query_embeddings)
        ctx = model.decoder.context_proj(encoder_features)

        # Get layer 0 attention stats
        layer = model.decoder.layers[0]
        x = layer.self_attn_block(x)

        cross_attn = layer.cross_attn_block.cross_attn
        query_for_attn = layer.cross_attn_block.norm2(x)

        B, N, C = query_for_attn.shape
        M = ctx.shape[1]

        q = cross_attn.q_proj(query_for_attn).reshape(B, N, 8, 64).transpose(1, 2)
        k = cross_attn.k_proj(ctx).reshape(B, M, 8, 64).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) * cross_attn.scale
        attn_weights = F.softmax(attn_logits, dim=-1)

        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)
        max_entropy = np.log(M)
        normalized_entropy = (entropy / max_entropy).mean().item()

        effective_tokens = (1.0 / (attn_weights ** 2).sum(dim=-1)).mean().item()
        max_attn = attn_weights.max(dim=-1)[0].mean().item()

        return {
            'normalized_entropy': normalized_entropy,
            'effective_tokens': effective_tokens,
            'num_context_tokens': M,
            'max_attention': max_attn,
        }


def train_model(model, dataset, loss_fn, config, device='cuda'):
    """Train model with given configuration."""
    model = model.to(device)
    model.train()

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Setup optimizer
    qk_lr_mult = config.get('qk_lr_multiplier', 1.0)
    base_lr = config.get('lr', 1e-4)

    if qk_lr_mult != 1.0:
        qk_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'q_proj' in name or 'k_proj' in name:
                qk_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': qk_params, 'lr': base_lr * qk_lr_mult},
            {'params': other_params, 'lr': base_lr},
        ], weight_decay=0.03)

        print(f"  Q/K params: {len(qk_params)}, LR={base_lr * qk_lr_mult}")
        print(f"  Other params: {len(other_params)}, LR={base_lr}")
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr, weight_decay=0.03
        )

    num_steps = config.get('num_steps', 200)
    log_interval = config.get('log_interval', 20)

    # Tracking
    history = {
        'loss': [],
        'entropy': [],
        'effective_tokens': [],
        'qk_grad_norm': [],
    }

    print(f"\n  Training for {num_steps} steps...")

    for step in range(num_steps):
        sample = dataset[step % len(dataset)]
        video = sample['video'].unsqueeze(0).to(device)
        queries = {k: v.unsqueeze(0).to(device) for k, v in sample['queries'].items()}
        targets = {k: v.unsqueeze(0).to(device) for k, v in sample['targets'].items()}
        cameras = {k: v.unsqueeze(0).to(device) for k, v in sample['cameras'].items()}

        optimizer.zero_grad()
        outputs = model(video, queries)

        loss, loss_dict = loss_fn(
            predictions=outputs,
            targets=targets,
            cameras=cameras,
            queries=queries,
        )

        loss.backward()

        # Track Q/K gradient norms
        qk_grad = 0
        count = 0
        for name, param in model.named_parameters():
            if param.grad is not None and 'cross_attn' in name and ('q_proj' in name or 'k_proj' in name):
                qk_grad += param.grad.norm().item()
                count += 1

        optimizer.step()

        history['loss'].append(loss.item())
        if count > 0:
            history['qk_grad_norm'].append(qk_grad / count)

        # Periodic evaluation
        if step % log_interval == 0 or step == num_steps - 1:
            model.eval()
            stats = compute_attention_stats(model, video, queries)
            history['entropy'].append(stats['normalized_entropy'])
            history['effective_tokens'].append(stats['effective_tokens'])
            model.train()

            print(f"    Step {step:3d}: loss={loss.item():.4f}, "
                  f"entropy={stats['normalized_entropy']:.4f}, "
                  f"eff_tokens={stats['effective_tokens']:.1f}/{stats['num_context_tokens']}")

    # Final evaluation
    model.eval()
    sample = dataset[0]
    video = sample['video'].unsqueeze(0).to(device)
    queries = {k: v.unsqueeze(0).to(device) for k, v in sample['queries'].items()}
    final_stats = compute_attention_stats(model, video, queries)

    return {
        'history': history,
        'final_loss': history['loss'][-1],
        'final_entropy': final_stats['normalized_entropy'],
        'final_effective_tokens': final_stats['effective_tokens'],
        'num_context_tokens': final_stats['num_context_tokens'],
        'final_max_attention': final_stats['max_attention'],
        'mean_qk_grad': np.mean(history['qk_grad_norm']) if history['qk_grad_norm'] else 0,
    }


def main():
    print("=" * 70)
    print("COMBINED FIX TEST: Context Pooling + Higher Q/K LR")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model_config = OmegaConf.load("configs/model/vit_b_movi.yaml")

    dataset = KubricDataset(
        data_dir="data/kubric",
        split="train",
        num_frames=24,
        resolution=(256, 256),
        num_queries=128,
    )
    dataset.samples = dataset.samples[:5]  # 5 samples for better coverage
    print(f"Using {len(dataset.samples)} samples")

    loss_weights = {
        'l1_3d': 1.0,
        'l2_2d': 0.1,
        'visibility': 0.1,
        'normal': 0.0,
        'motion': 0.1,
        'confidence': 0.2,
    }
    loss_fn = CompositeLoss(loss_weights=loss_weights).to(device)

    results = []
    num_steps = 200

    # Test configurations
    configs = [
        {
            'name': 'Baseline',
            'pool_tokens': None,
            'qk_lr_multiplier': 1.0,
            'num_steps': num_steps,
        },
        {
            'name': 'Pool256',
            'pool_tokens': 256,
            'qk_lr_multiplier': 1.0,
            'num_steps': num_steps,
        },
        {
            'name': 'Pool256 + QK_LR_10x',
            'pool_tokens': 256,
            'qk_lr_multiplier': 10.0,
            'num_steps': num_steps,
        },
        {
            'name': 'Pool128',
            'pool_tokens': 128,
            'qk_lr_multiplier': 1.0,
            'num_steps': num_steps,
        },
        {
            'name': 'Pool128 + QK_LR_10x',
            'pool_tokens': 128,
            'qk_lr_multiplier': 10.0,
            'num_steps': num_steps,
        },
        {
            'name': 'Pool64 + QK_LR_10x',
            'pool_tokens': 64,
            'qk_lr_multiplier': 10.0,
            'num_steps': num_steps,
        },
    ]

    for config in configs:
        print("\n" + "=" * 60)
        print(f"TEST: {config['name']}")
        print("=" * 60)

        model = build_d4rt_model(model_config)

        if config['pool_tokens']:
            model = apply_context_pooling(model, output_tokens=config['pool_tokens'])
            print(f"  Context pooling: 3072 → {config['pool_tokens']}")

        if config['qk_lr_multiplier'] != 1.0:
            print(f"  Q/K LR multiplier: {config['qk_lr_multiplier']}×")

        result = train_model(model, dataset, loss_fn, config, device=device)
        result['name'] = config['name']
        result['config'] = config
        results.append(result)

        del model
        torch.cuda.empty_cache()

    # Print comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Configuration':<25} {'Loss':>8} {'Entropy':>8} {'Eff.Tok':>10} {'MaxAttn':>10} {'Q/K Grad':>10}")
    print("-" * 75)

    for r in results:
        eff_tok_str = f"{r['final_effective_tokens']:.0f}/{r['num_context_tokens']}"
        print(f"{r['name']:<25} {r['final_loss']:>8.4f} {r['final_entropy']:>8.4f} "
              f"{eff_tok_str:>10} {r['final_max_attention']:>10.6f} {r['mean_qk_grad']:>10.2e}")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS:")

    baseline = results[0]

    for r in results[1:]:
        print(f"\n{r['name']}:")

        loss_pct = (r['final_loss'] - baseline['final_loss']) / baseline['final_loss'] * 100
        entropy_delta = r['final_entropy'] - baseline['final_entropy']
        eff_tok_ratio = r['final_effective_tokens'] / baseline['final_effective_tokens']
        max_attn_ratio = r['final_max_attention'] / baseline['final_max_attention']

        print(f"  Loss: {loss_pct:+.1f}% vs baseline")
        print(f"  Entropy: {entropy_delta:+.4f} ({r['final_entropy']:.4f})")
        print(f"  Effective tokens: {eff_tok_ratio:.2f}× baseline ({r['final_effective_tokens']:.0f})")
        print(f"  Max attention: {max_attn_ratio:.1f}× baseline ({r['final_max_attention']:.6f})")

        # Score the result
        score = 0
        reasons = []

        if r['final_entropy'] < 0.95:
            score += 2
            reasons.append("entropy < 0.95")
        if r['final_effective_tokens'] < baseline['final_effective_tokens'] * 0.1:
            score += 2
            reasons.append("eff_tokens < 10% baseline")
        if r['final_max_attention'] > baseline['final_max_attention'] * 5:
            score += 1
            reasons.append("max_attn > 5× baseline")
        if r['final_loss'] < baseline['final_loss']:
            score += 1
            reasons.append("lower loss")

        if score >= 3:
            print(f"  ✅ EXCELLENT ({', '.join(reasons)})")
        elif score >= 2:
            print(f"  ✅ GOOD ({', '.join(reasons)})")
        elif score >= 1:
            print(f"  🔶 OK ({', '.join(reasons)})")
        else:
            print(f"  ❌ No significant improvement")

    # Find best
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")

    best_entropy = min(results, key=lambda r: r['final_entropy'])
    best_loss = min(results, key=lambda r: r['final_loss'])
    best_focus = min(results, key=lambda r: r['final_effective_tokens'])

    print(f"  Best entropy: {best_entropy['name']} ({best_entropy['final_entropy']:.4f})")
    print(f"  Best loss: {best_loss['name']} ({best_loss['final_loss']:.4f})")
    print(f"  Best focus: {best_focus['name']} ({best_focus['final_effective_tokens']:.0f} tokens)")

    # Overall recommendation
    print("\n  SUGGESTED CONFIGURATION:")
    # Pick the one with best combination of low entropy and good loss
    scored_results = []
    for r in results:
        score = (1 - r['final_entropy']) * 100 - r['final_loss']
        scored_results.append((score, r))

    best = max(scored_results, key=lambda x: x[0])[1]
    print(f"    → {best['name']}")
    print(f"      Pool tokens: {best['config'].get('pool_tokens', 'None')}")
    print(f"      Q/K LR multiplier: {best['config'].get('qk_lr_multiplier', 1.0)}×")


if __name__ == "__main__":
    main()
