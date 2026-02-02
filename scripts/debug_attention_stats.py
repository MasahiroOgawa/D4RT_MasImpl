#!/usr/bin/env python3
"""
Debug script to analyze cross-attention statistics.

This script verifies the hypothesis that softmax over many tokens
causes gradient vanishing in Q and K projections.

Key metrics:
1. Attention entropy (high = uniform, low = peaked)
2. Max attention weight per query
3. Softmax gradient magnitude: mean(attn * (1 - attn))
4. Effective number of tokens attended to
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from d4rt.models.d4rt import build_d4rt_model
from d4rt.data.datasets.kubric import KubricDataset


def compute_attention_stats(query, key, module):
    """Compute attention statistics for a single attention layer."""
    B, N, C = query.shape
    M = key.shape[1]

    num_heads = module.num_heads
    head_dim = module.head_dim
    scale = module.scale

    # Project Q and K
    with torch.no_grad():
        q = module.q_proj(query).reshape(B, N, num_heads, head_dim).transpose(1, 2)
        k = module.k_proj(key).reshape(B, M, num_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        attn_logits = (q @ k.transpose(-2, -1)) * scale  # [B, H, N, M]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, H, N, M]

    # 1. Entropy
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)
    max_entropy = np.log(M)
    normalized_entropy = entropy / max_entropy

    # 2. Max attention weight
    max_attn = attn_weights.max(dim=-1)[0]

    # 3. Softmax gradient magnitude
    softmax_grad = attn_weights * (1 - attn_weights)

    # 4. Effective tokens
    effective_tokens = 1.0 / (attn_weights ** 2).sum(dim=-1)

    # 5. Logit statistics
    logit_range = attn_logits.max(dim=-1)[0] - attn_logits.min(dim=-1)[0]

    return {
        'num_context_tokens': M,
        'num_query_tokens': N,
        'num_heads': num_heads,

        'entropy_mean': entropy.mean().item(),
        'normalized_entropy_mean': normalized_entropy.mean().item(),
        'max_entropy': max_entropy,

        'max_attn_mean': max_attn.mean().item(),
        'max_attn_max': max_attn.max().item(),

        'softmax_grad_mean': softmax_grad.mean().item(),
        'softmax_grad_max': softmax_grad.max().item(),

        'effective_tokens_mean': effective_tokens.mean().item(),

        'logit_range_mean': logit_range.mean().item(),

        'uniform_attn_value': 1.0 / M,
    }


def analyze_model_attention(model, video, queries, title):
    """Analyze attention for all cross-attention layers."""
    print(f"\n{'=' * 60}")
    print(title)
    print('=' * 60)

    model.eval()

    with torch.no_grad():
        # Get encoder features
        encoder_features = model.encoder(video)
        print(f"\nEncoder output shape: {encoder_features.shape}")

        # Get query embeddings
        from d4rt.utils.patch_utils import extract_patches
        patches = extract_patches(
            video,
            queries['u'],
            queries['v'],
            queries['t_src'],
            patch_size=9,
        )
        query_embeddings = model.query_encoder(
            queries['u'],
            queries['v'],
            queries['t_src'],
            queries['t_tgt'],
            queries['t_cam'],
            patches,
        )
        print(f"Query embeddings shape: {query_embeddings.shape}")

        # Project to decoder dimensions
        x = model.decoder.query_proj(query_embeddings)
        ctx = model.decoder.context_proj(encoder_features)
        print(f"Projected queries shape: {x.shape}")
        print(f"Projected context shape: {ctx.shape}")

        # Analyze each decoder layer's cross-attention
        all_stats = {}

        for i, layer in enumerate(model.decoder.layers):
            # Self-attention first
            x = layer.self_attn_block(x)

            # Get cross-attention module
            cross_attn_block = layer.cross_attn_block
            cross_attn = cross_attn_block.cross_attn

            # Compute attention stats before cross-attention
            query_for_attn = cross_attn_block.norm2(x)
            stats = compute_attention_stats(query_for_attn, ctx, cross_attn)
            all_stats[f'layer_{i}'] = stats

            # Apply cross-attention and FFN
            x = cross_attn_block(x, context=ctx)
            x = x + layer.drop_path(layer.mlp(layer.norm_ffn(x)))

        return all_stats


def print_stats(all_stats):
    """Print attention statistics."""
    print("\n" + "-" * 60)

    for layer_name, stats in sorted(all_stats.items()):
        print(f"\n{layer_name}:")
        print(f"  Context: {stats['num_context_tokens']} tokens, Query: {stats['num_query_tokens']} tokens")
        print(f"  Normalized entropy: {stats['normalized_entropy_mean']:.4f} (1.0 = uniform)")
        print(f"  Max attention: {stats['max_attn_mean']:.6f} (uniform = {stats['uniform_attn_value']:.6f})")
        print(f"  Softmax gradient: {stats['softmax_grad_mean']:.6e}")
        print(f"  Effective tokens: {stats['effective_tokens_mean']:.1f} / {stats['num_context_tokens']}")
        print(f"  Logit range: {stats['logit_range_mean']:.4f}")


def print_diagnosis(untrained_stats, trained_stats=None):
    """Print diagnosis."""
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    stats = trained_stats if trained_stats else untrained_stats
    label = "Trained" if trained_stats else "Untrained"

    layer_0 = stats.get('layer_0', {})
    layer_7 = stats.get('layer_7', {})

    num_tokens = layer_0.get('num_context_tokens', 0)
    norm_entropy = layer_0.get('normalized_entropy_mean', 1.0)
    softmax_grad = layer_0.get('softmax_grad_mean', 0)
    effective = layer_0.get('effective_tokens_mean', 0)

    print(f"\n{label} Model - Layer 0:")
    print(f"  Context tokens: {num_tokens}")
    print(f"  Normalized entropy: {norm_entropy:.4f}")
    print(f"  Softmax gradient: {softmax_grad:.6e}")
    print(f"  Effective tokens: {effective:.1f}")

    # Theoretical analysis
    print("\n" + "-" * 60)
    print("THEORETICAL GRADIENT ANALYSIS:")

    if num_tokens > 0:
        # If attention is uniform: each weight = 1/M
        uniform_weight = 1.0 / num_tokens
        uniform_grad = uniform_weight * (1 - uniform_weight)
        print(f"\n  If attention is UNIFORM over {num_tokens} tokens:")
        print(f"    Each attention weight = 1/{num_tokens} = {uniform_weight:.6f}")
        print(f"    Softmax gradient = w*(1-w) = {uniform_grad:.6e}")

        # If attention is peaked (one-hot)
        print(f"\n  If attention is ONE-HOT (peaked):")
        print(f"    Winner weight = 1.0, gradient = 1*(1-1) = 0")
        print(f"    Loser weights ≈ 0, gradient ≈ 0")
        print(f"    -> Both extremes have vanishing gradients!")

        # Optimal case
        optimal_tokens = 10
        optimal_weight = 1.0 / optimal_tokens
        optimal_grad = optimal_weight * (1 - optimal_weight)
        print(f"\n  OPTIMAL case (attend to ~{optimal_tokens} tokens):")
        print(f"    Each attention weight ≈ {optimal_weight:.4f}")
        print(f"    Softmax gradient ≈ {optimal_grad:.4e}")
        print(f"    -> {optimal_grad / uniform_grad:.1f}× larger gradient than uniform!")

    print("\n" + "-" * 60)
    print("CONCLUSION:")

    if norm_entropy > 0.95:
        print(f"  [CRITICAL] Attention nearly UNIFORM (entropy={norm_entropy:.4f})")
        print(f"  -> Q and K cannot learn because gradients are ~{softmax_grad:.2e}")
        print(f"  -> Need to encourage sharper attention patterns")
    elif norm_entropy > 0.8:
        print(f"  [WARNING] Attention too diffuse (entropy={norm_entropy:.4f})")
    else:
        print(f"  [OK] Attention shows meaningful patterns (entropy={norm_entropy:.4f})")

    if trained_stats and untrained_stats:
        u_entropy = untrained_stats.get('layer_0', {}).get('normalized_entropy_mean', 1.0)
        t_entropy = trained_stats.get('layer_0', {}).get('normalized_entropy_mean', 1.0)

        if t_entropy >= u_entropy:
            print(f"\n  [CRITICAL] Training made attention MORE uniform!")
            print(f"    Untrained: {u_entropy:.4f} -> Trained: {t_entropy:.4f}")
            print(f"    -> Cross-attention is collapsing, not learning")

    print("\n" + "-" * 60)
    print("RECOMMENDED FIXES:")
    print("  1. Temperature scaling: attn = softmax(QK/√d / τ) with τ < 1")
    print("  2. Auxiliary loss: minimize attention entropy")
    print("  3. Reduce context: pool 3072 → 256 tokens")
    print("  4. 10× higher LR for q_proj and k_proj")


def main():
    print("=" * 60)
    print("ATTENTION STATISTICS DEBUG SCRIPT")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model_config = OmegaConf.load("configs/model/vit_b_movi.yaml")
    model = build_d4rt_model(model_config).to(device)

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Load dataset
    dataset = KubricDataset(
        data_dir="data/kubric",
        split="val",
        num_frames=24,
        resolution=(256, 256),
        num_queries=128,
    )

    # Get sample
    sample = dataset[0]
    video = sample['video'].unsqueeze(0).to(device)
    queries = {k: v.unsqueeze(0).to(device) for k, v in sample['queries'].items()}

    # Analyze untrained model
    untrained_stats = analyze_model_attention(
        model, video, queries, "UNTRAINED MODEL"
    )
    print_stats(untrained_stats)

    # Check for trained checkpoint
    checkpoint_path = Path("checkpoints/checkpoint_step_0050000.pth")
    trained_stats = None

    if checkpoint_path.exists():
        # Load trained model
        model_trained = build_d4rt_model(model_config).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_trained.load_state_dict(ckpt['model_state_dict'])

        trained_stats = analyze_model_attention(
            model_trained, video, queries, "TRAINED MODEL (50k steps)"
        )
        print_stats(trained_stats)
    else:
        print(f"\nNo checkpoint at {checkpoint_path}")

    # Diagnosis
    print_diagnosis(untrained_stats, trained_stats)


if __name__ == "__main__":
    main()
