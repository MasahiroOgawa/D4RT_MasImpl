#!/usr/bin/env python3
"""Debug cross-attention mechanism."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from d4rt.models.d4rt import build_d4rt_model


def main():
    print("=" * 60)
    print("CROSS-ATTENTION DEBUG")
    print("=" * 60)

    # Load model
    model_config = OmegaConf.load("configs/model/vit_b_movi.yaml")
    model = build_d4rt_model(model_config)

    ckpt = torch.load("checkpoints/checkpoint_step_0050000.pth", map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print("\n1. Decoder architecture check")
    print("-" * 40)

    decoder = model.decoder
    print(f"Decoder num_layers: {decoder.num_layers}")
    print(f"Decoder hidden_dim: {decoder.hidden_dim}")

    # Check first decoder layer
    layer0 = decoder.layers[0]
    print(f"\nDecoderLayer 0:")
    print(f"  self_attn_block mlp_ratio: {layer0.self_attn_block.mlp}")
    print(f"  cross_attn_block mlp_ratio: {layer0.cross_attn_block.mlp}")

    # Check if cross_attn exists
    if hasattr(layer0.cross_attn_block, 'cross_attn'):
        print(f"  cross_attn exists: YES")
        cross_attn = layer0.cross_attn_block.cross_attn
        print(f"    q_proj weight shape: {cross_attn.q_proj.weight.shape}")
        print(f"    k_proj weight shape: {cross_attn.k_proj.weight.shape}")
        print(f"    v_proj weight shape: {cross_attn.v_proj.weight.shape}")
    else:
        print(f"  cross_attn exists: NO (!!!)")

    print("\n2. Test cross-attention directly")
    print("-" * 40)

    B, N, M = 1, 10, 100
    hidden_dim = decoder.hidden_dim
    device = 'cpu'

    # Create test inputs
    queries = torch.randn(B, N, hidden_dim, device=device)
    context1 = torch.randn(B, M, hidden_dim, device=device)
    context2 = torch.randn(B, M, hidden_dim, device=device) * 10  # Very different

    # Run through first decoder layer
    with torch.no_grad():
        out1 = layer0(queries, context1)
        out2 = layer0(queries, context2)

    diff = (out1 - out2).abs().mean()
    print(f"Same queries, different contexts:")
    print(f"  Output difference: {diff:.6f}")
    if diff < 0.01:
        print("  WARNING: Outputs nearly identical - cross-attention not working!")
    else:
        print("  OK: Cross-attention affects output")

    print("\n3. Check attention weights")
    print("-" * 40)

    # Test with hook to see attention patterns
    attn_weights = []

    def hook_fn(module, input, output):
        # Get attention weights before softmax
        q = input[0]
        k = input[1] if len(input) > 1 and input[1] is not None else q
        B, N, C = q.shape
        head_dim = C // module.num_heads
        q = module.q_proj(q).reshape(B, N, module.num_heads, head_dim).transpose(1, 2)
        k = module.k_proj(k).reshape(B, k.shape[1], module.num_heads, head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn_softmax = torch.softmax(attn, dim=-1)
        attn_weights.append(attn_softmax.detach().cpu())

    # Register hooks on cross-attention
    hooks = []
    for i, layer in enumerate(decoder.layers):
        if hasattr(layer.cross_attn_block, 'cross_attn'):
            h = layer.cross_attn_block.cross_attn.register_forward_hook(hook_fn)
            hooks.append(h)

    # Run forward pass through decoder
    with torch.no_grad():
        decoder_out = decoder(queries, context1)

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"Captured {len(attn_weights)} attention maps")
    if len(attn_weights) > 0:
        for i, attn in enumerate(attn_weights[:3]):  # Show first 3 layers
            print(f"\nLayer {i} cross-attention:")
            print(f"  Shape: {attn.shape}")
            print(f"  Mean attn weight: {attn.mean():.6f}")
            print(f"  Std attn weight: {attn.std():.6f}")
            print(f"  Max attn weight: {attn.max():.6f}")
            print(f"  Min attn weight: {attn.min():.6f}")

            # Check if attention is collapsed (all same weights)
            attn_2d = attn[0, 0]  # [N, M] for first head
            if attn_2d.std() < 0.001:
                print(f"  WARNING: Attention collapsed to uniform!")
            else:
                print(f"  OK: Attention has variation")

    print("\n4. Check query encoder output variation")
    print("-" * 40)

    query_encoder = model.query_encoder

    # Create different query inputs
    u1 = torch.tensor([[0.1, 0.5, 0.9]])
    v1 = torch.tensor([[0.5, 0.5, 0.5]])
    u2 = torch.tensor([[0.5, 0.5, 0.5]])
    v2 = torch.tensor([[0.1, 0.5, 0.9]])

    t_src = torch.zeros(1, 3, dtype=torch.long)
    t_tgt = torch.tensor([[0, 12, 23]])
    t_cam = t_tgt.clone()

    # Need dummy patches
    patches = torch.randn(1, 3, 3, 9, 9)

    with torch.no_grad():
        emb1 = query_encoder(u1, v1, t_src, t_tgt, t_cam, patches)
        emb2 = query_encoder(u2, v2, t_src, t_tgt, t_cam, patches)

    print(f"Query embedding shape: {emb1.shape}")
    print(f"Different u values - embedding diff: {(emb1 - emb2).abs().mean():.4f}")

    # Also check within same embedding
    within_var = emb1[0].std(dim=0).mean()
    print(f"Within-batch std (should be high if queries different): {within_var:.4f}")


if __name__ == "__main__":
    main()
