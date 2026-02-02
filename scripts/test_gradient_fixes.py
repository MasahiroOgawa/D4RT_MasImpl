#!/usr/bin/env python3
"""
Test potential fixes for the cross-attention gradient vanishing problem.

Fixes to test:
1. Temperature scaling (τ < 1 to sharpen attention)
2. Auxiliary entropy loss (encourage peaked attention)
3. Context pooling (reduce 3072 → 256 tokens)
4. Higher LR for Q/K projections
"""

import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from d4rt.models.d4rt import build_d4rt_model
from d4rt.data.datasets.kubric import KubricDataset
from d4rt.losses import CompositeLoss


def compute_attention_entropy(model, video, queries):
    """Compute attention entropy for layer 0."""
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

        # Get layer 0 attention
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

        return normalized_entropy, effective_tokens, M


class TemperatureScaledAttention(nn.Module):
    """Attention with temperature scaling."""

    def __init__(self, original_attn, temperature=0.1):
        super().__init__()
        self.original = original_attn
        self.temperature = temperature

        # Copy all attributes
        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.scale = original_attn.scale
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.out_proj = original_attn.out_proj
        self.dropout = original_attn.dropout

    def forward(self, query, key=None, value=None, attn_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key

        B, N, C = query.shape
        M = key.shape[1]

        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Scale by temperature (τ < 1 sharpens attention)
        attn = (q @ k.transpose(-2, -1)) * self.scale / self.temperature
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        return out


class ContextPooling(nn.Module):
    """Pool context tokens to reduce count."""

    def __init__(self, input_tokens=3072, output_tokens=256, embed_dim=512):
        super().__init__()
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

        # Learnable pooling via linear projection
        self.pool = nn.Linear(input_tokens, output_tokens)

    def forward(self, x):
        # x: [B, input_tokens, embed_dim]
        # Transpose, apply linear, transpose back
        x = x.transpose(1, 2)  # [B, embed_dim, input_tokens]
        x = self.pool(x)       # [B, embed_dim, output_tokens]
        x = x.transpose(1, 2)  # [B, output_tokens, embed_dim]
        return x


def apply_temperature_scaling(model, temperature=0.1):
    """Apply temperature scaling to all cross-attention layers."""
    for layer in model.decoder.layers:
        original_attn = layer.cross_attn_block.cross_attn
        layer.cross_attn_block.cross_attn = TemperatureScaledAttention(original_attn, temperature)
    return model


def apply_context_pooling(model, output_tokens=256):
    """Add context pooling before decoder."""
    device = next(model.parameters()).device

    # Store original context_proj
    original_context_proj = model.decoder.context_proj

    # Create pooling layer
    pooling = ContextPooling(3072, output_tokens, 512).to(device)

    # Create combined projection
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


def compute_entropy_loss(model, video, queries):
    """Compute entropy of attention weights as auxiliary loss."""
    encoder_features = model.encoder(video)

    from d4rt.utils.patch_utils import extract_patches
    patches = extract_patches(video, queries['u'], queries['v'], queries['t_src'], patch_size=9)
    query_embeddings = model.query_encoder(
        queries['u'], queries['v'], queries['t_src'], queries['t_tgt'], queries['t_cam'], patches
    )

    x = model.decoder.query_proj(query_embeddings)
    ctx = model.decoder.context_proj(encoder_features)

    total_entropy = 0

    for layer in model.decoder.layers:
        x = layer.self_attn_block(x)

        cross_attn = layer.cross_attn_block.cross_attn
        query_for_attn = layer.cross_attn_block.norm2(x)

        B, N, C = query_for_attn.shape
        M = ctx.shape[1]

        q = cross_attn.q_proj(query_for_attn).reshape(B, N, 8, 64).transpose(1, 2)
        k = cross_attn.k_proj(ctx).reshape(B, M, 8, 64).transpose(1, 2)

        scale = getattr(cross_attn, 'scale', 64 ** -0.5)
        temp = getattr(cross_attn, 'temperature', 1.0)

        attn_logits = (q @ k.transpose(-2, -1)) * scale / temp
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Entropy loss: minimize entropy = sharpen attention
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)
        total_entropy = total_entropy + entropy.mean()

        # Continue forward pass
        x = layer.cross_attn_block(x, context=ctx)
        x = x + layer.drop_path(layer.mlp(layer.norm_ffn(x)))

    return total_entropy / len(model.decoder.layers)


def train_with_fix(model, dataset, loss_fn, fix_name, num_steps=50, lr=1e-4,
                   entropy_weight=0.0, qk_lr_multiplier=1.0, device='cuda'):
    """Train model with a specific fix and measure results."""
    model = model.to(device)
    model.train()

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Setup optimizer with optional separate LR for Q/K
    if qk_lr_multiplier != 1.0:
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
            {'params': qk_params, 'lr': lr * qk_lr_multiplier},
            {'params': other_params, 'lr': lr},
        ], weight_decay=0.03)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=0.03
        )

    # Track metrics
    losses = []
    entropies = []
    effective_tokens_list = []
    qk_grad_norms = []

    for step in range(num_steps):
        sample = dataset[step % len(dataset)]
        video = sample['video'].unsqueeze(0).to(device)
        queries = {k: v.unsqueeze(0).to(device) for k, v in sample['queries'].items()}
        targets = {k: v.unsqueeze(0).to(device) for k, v in sample['targets'].items()}
        cameras = {k: v.unsqueeze(0).to(device) for k, v in sample['cameras'].items()}

        optimizer.zero_grad()

        # Forward
        outputs = model(video, queries)

        # Main loss
        loss, loss_dict = loss_fn(
            predictions=outputs,
            targets=targets,
            cameras=cameras,
            queries=queries,
        )

        # Add entropy loss if specified
        if entropy_weight > 0:
            entropy_loss = compute_entropy_loss(model, video, queries)
            loss = loss + entropy_weight * entropy_loss

        # Backward
        loss.backward()

        # Capture Q/K gradient norms before optimizer step
        qk_grad = 0
        count = 0
        for name, param in model.named_parameters():
            if param.grad is not None and ('cross_attn' in name) and ('q_proj' in name or 'k_proj' in name):
                qk_grad += param.grad.norm().item()
                count += 1
        if count > 0:
            qk_grad_norms.append(qk_grad / count)

        optimizer.step()

        losses.append(loss.item())

        # Measure attention entropy periodically
        if step % 10 == 0:
            model.eval()
            with torch.no_grad():
                norm_entropy, eff_tokens, num_ctx = compute_attention_entropy(model, video, queries)
            entropies.append(norm_entropy)
            effective_tokens_list.append(eff_tokens)
            model.train()

    # Final evaluation
    model.eval()
    sample = dataset[0]
    video = sample['video'].unsqueeze(0).to(device)
    queries = {k: v.unsqueeze(0).to(device) for k, v in sample['queries'].items()}

    with torch.no_grad():
        final_entropy, final_eff_tokens, num_ctx = compute_attention_entropy(model, video, queries)

    return {
        'fix_name': fix_name,
        'final_loss': losses[-1],
        'loss_trajectory': losses,
        'final_entropy': final_entropy,
        'entropy_trajectory': entropies,
        'final_effective_tokens': final_eff_tokens,
        'effective_tokens_trajectory': effective_tokens_list,
        'num_context_tokens': num_ctx,
        'mean_qk_grad': np.mean(qk_grad_norms) if qk_grad_norms else 0,
    }


def main():
    print("=" * 70)
    print("GRADIENT FIX COMPARISON TEST")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config and dataset
    model_config = OmegaConf.load("configs/model/vit_b_movi.yaml")

    dataset = KubricDataset(
        data_dir="data/kubric",
        split="train",
        num_frames=24,
        resolution=(256, 256),
        num_queries=128,
    )
    dataset.samples = dataset.samples[:3]  # Mini dataset
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

    # Baseline (no fix)
    print("\n" + "=" * 50)
    print("TEST 1: BASELINE (no fix)")
    print("=" * 50)
    model = build_d4rt_model(model_config)
    result = train_with_fix(model, dataset, loss_fn, "Baseline", num_steps=50, device=device)
    results.append(result)
    del model
    torch.cuda.empty_cache()

    # Fix 1: Temperature scaling (τ = 0.1)
    print("\n" + "=" * 50)
    print("TEST 2: TEMPERATURE SCALING (τ = 0.1)")
    print("=" * 50)
    model = build_d4rt_model(model_config)
    model = apply_temperature_scaling(model, temperature=0.1)
    result = train_with_fix(model, dataset, loss_fn, "Temperature τ=0.1", num_steps=50, device=device)
    results.append(result)
    del model
    torch.cuda.empty_cache()

    # Fix 2: Temperature scaling (τ = 0.5)
    print("\n" + "=" * 50)
    print("TEST 3: TEMPERATURE SCALING (τ = 0.5)")
    print("=" * 50)
    model = build_d4rt_model(model_config)
    model = apply_temperature_scaling(model, temperature=0.5)
    result = train_with_fix(model, dataset, loss_fn, "Temperature τ=0.5", num_steps=50, device=device)
    results.append(result)
    del model
    torch.cuda.empty_cache()

    # Fix 3: Entropy loss
    print("\n" + "=" * 50)
    print("TEST 4: ENTROPY LOSS (weight=0.1)")
    print("=" * 50)
    model = build_d4rt_model(model_config)
    result = train_with_fix(model, dataset, loss_fn, "Entropy Loss λ=0.1",
                           num_steps=50, entropy_weight=0.1, device=device)
    results.append(result)
    del model
    torch.cuda.empty_cache()

    # Fix 4: Context pooling
    print("\n" + "=" * 50)
    print("TEST 5: CONTEXT POOLING (3072 → 256)")
    print("=" * 50)
    model = build_d4rt_model(model_config)
    model = apply_context_pooling(model, output_tokens=256)
    result = train_with_fix(model, dataset, loss_fn, "Context Pool 256", num_steps=50, device=device)
    results.append(result)
    del model
    torch.cuda.empty_cache()

    # Fix 5: Higher LR for Q/K
    print("\n" + "=" * 50)
    print("TEST 6: HIGHER LR FOR Q/K (10×)")
    print("=" * 50)
    model = build_d4rt_model(model_config)
    result = train_with_fix(model, dataset, loss_fn, "Q/K LR 10×",
                           num_steps=50, qk_lr_multiplier=10.0, device=device)
    results.append(result)
    del model
    torch.cuda.empty_cache()

    # Fix 6: Combined (Temperature + Entropy)
    print("\n" + "=" * 50)
    print("TEST 7: COMBINED (Temperature τ=0.5 + Entropy λ=0.1)")
    print("=" * 50)
    model = build_d4rt_model(model_config)
    model = apply_temperature_scaling(model, temperature=0.5)
    result = train_with_fix(model, dataset, loss_fn, "Temp+Entropy",
                           num_steps=50, entropy_weight=0.1, device=device)
    results.append(result)
    del model
    torch.cuda.empty_cache()

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Fix':<25} {'Loss':>10} {'Entropy':>10} {'Eff.Tokens':>12} {'Q/K Grad':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['fix_name']:<25} {r['final_loss']:>10.4f} {r['final_entropy']:>10.4f} "
              f"{r['final_effective_tokens']:>10.1f}/{r['num_context_tokens']} {r['mean_qk_grad']:>12.2e}")

    # Find best
    print("\n" + "-" * 70)
    print("ANALYSIS:")

    baseline = results[0]

    for r in results[1:]:
        entropy_change = r['final_entropy'] - baseline['final_entropy']
        eff_tokens_change = r['final_effective_tokens'] - baseline['final_effective_tokens']
        grad_change = r['mean_qk_grad'] / (baseline['mean_qk_grad'] + 1e-12)

        print(f"\n{r['fix_name']}:")
        print(f"  Entropy: {baseline['final_entropy']:.4f} → {r['final_entropy']:.4f} ({entropy_change:+.4f})")
        print(f"  Eff.Tokens: {baseline['final_effective_tokens']:.0f} → {r['final_effective_tokens']:.0f} ({eff_tokens_change:+.0f})")
        print(f"  Q/K Grad: {grad_change:.1f}× vs baseline")

        if r['final_entropy'] < 0.95 and r['final_effective_tokens'] < baseline['final_effective_tokens'] * 0.5:
            print(f"  ✅ PROMISING - Attention is sharper!")
        elif r['mean_qk_grad'] > baseline['mean_qk_grad'] * 2:
            print(f"  ✅ PROMISING - Q/K gradients increased!")

    print("\n" + "=" * 70)
    print("RECOMMENDATION:")

    best_entropy = min(results, key=lambda r: r['final_entropy'])
    best_eff_tokens = min(results, key=lambda r: r['final_effective_tokens'])
    best_grad = max(results, key=lambda r: r['mean_qk_grad'])

    print(f"  Best entropy reduction: {best_entropy['fix_name']}")
    print(f"  Best effective tokens reduction: {best_eff_tokens['fix_name']}")
    print(f"  Best Q/K gradient increase: {best_grad['fix_name']}")


if __name__ == "__main__":
    main()
