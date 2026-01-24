"""Display detailed information about D4RT models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf
import argparse

from d4rt.models import build_d4rt_model, print_model_info, count_parameters


def display_model_architecture(model, config_name):
    """Display detailed model architecture."""
    print("=" * 80)
    print(f"D4RT MODEL: {config_name.upper()}")
    print("=" * 80)

    # Basic info
    print_model_info(model)

    # Detailed breakdown
    print("\n" + "=" * 80)
    print("DETAILED ARCHITECTURE")
    print("=" * 80)

    print("\nENCODER (Spatio-Temporal ViT):")
    print(f"  - Architecture: {model.encoder.num_layers} layer transformer")
    print(f"  - Hidden dimension: {model.encoder.embed_dim}")
    print(f"  - Number of heads: {model.encoder.blocks[0].self_attn.num_heads}")
    print(f"  - Patch size: {model.encoder.patch_size}")
    print(f"  - Input resolution: {model.encoder.input_resolution}")
    print(f"  - Number of patches: {model.encoder.num_patches}")
    print(f"  - Gradient checkpointing: {model.encoder.use_checkpoint}")

    print("\nQUERY ENCODER:")
    print(f"  - Fourier encoding: {model.query_encoder.fourier.output_dim} dims")
    print(f"  - Temporal embeddings: 3 × {model.query_encoder.temporal_src.embedding_dim} dims")
    print(f"  - Patch CNN output: {model.query_encoder.patch_cnn.fc.out_features} dims")
    print(f"  - Total input: {40 + 3*256 + 256} dims")
    print(f"  - Output dimension: {model.query_encoder.output_dim}")

    print("\nDECODER (Cross-Attention):")
    print(f"  - Architecture: {model.decoder.num_layers} layer transformer")
    print(f"  - Hidden dimension: {model.decoder.hidden_dim}")
    print(f"  - Number of heads: {model.decoder.layers[0].self_attn_block.self_attn.num_heads}")
    print(f"  - Context dimension: {model.decoder.context_dim}")

    print("\nOUTPUT HEADS:")
    print(f"  - XYZ prediction: Linear({model.decoder.hidden_dim} → 3)")
    print(f"  - Visibility prediction: Linear({model.decoder.hidden_dim} → 1)")


def compare_model_sizes():
    """Compare all model configurations."""
    print("\n" + "=" * 80)
    print("MODEL SIZE COMPARISON")
    print("=" * 80)

    configs = ['vit_b', 'vit_l', 'vit_g']
    results = []

    for config_name in configs:
        config_path = Path(__file__).parent.parent / 'configs' / 'model' / f'{config_name}.yaml'
        config = OmegaConf.load(config_path)

        model = build_d4rt_model(config)
        params = count_parameters(model)

        results.append({
            'name': config_name.upper(),
            'encoder_layers': config.encoder.num_layers,
            'hidden_dim': config.encoder.hidden_dim,
            'encoder_params': params['encoder'],
            'decoder_params': params['decoder'],
            'query_encoder_params': params['query_encoder'],
            'total_params': params['total'],
        })

    # Print table
    print("\n{:<10} {:<8} {:<12} {:<15} {:<15} {:<20} {:<15}".format(
        "Model", "Layers", "Hidden Dim", "Encoder (M)", "Decoder (M)", "Query Encoder (M)", "Total (M)"
    ))
    print("-" * 115)

    for r in results:
        print("{:<10} {:<8} {:<12} {:<15.1f} {:<15.1f} {:<20.1f} {:<15.1f}".format(
            r['name'],
            r['encoder_layers'],
            r['hidden_dim'],
            r['encoder_params'] / 1e6,
            r['decoder_params'] / 1e6,
            r['query_encoder_params'] / 1e6,
            r['total_params'] / 1e6,
        ))

    print("\nNote: M = Million parameters")


def estimate_memory_usage(config_name):
    """Estimate GPU memory usage."""
    print("\n" + "=" * 80)
    print(f"ESTIMATED MEMORY USAGE: {config_name.upper()}")
    print("=" * 80)

    config_path = Path(__file__).parent.parent / 'configs' / 'model' / f'{config_name}.yaml'
    config = OmegaConf.load(config_path)

    model = build_d4rt_model(config)
    params = count_parameters(model)

    # Model parameters (fp32)
    model_memory_gb = (params['total'] * 4) / (1024**3)

    # Optimizer state (AdamW: 2x parameters for momentum + variance)
    optimizer_memory_gb = model_memory_gb * 2

    # Activations (rough estimate for batch_size=4, 48 frames, 256x256)
    batch_size = 4
    num_frames = 48
    resolution = 256

    # Encoder activations
    num_patches = model.encoder.num_patches
    hidden_dim = model.encoder.embed_dim
    num_layers = model.encoder.num_layers

    # Rough estimate: each layer stores activations
    activation_memory_gb = (
        batch_size * num_patches * hidden_dim * num_layers * 4
    ) / (1024**3)

    # Gradients (same as parameters)
    gradient_memory_gb = model_memory_gb

    total_memory_gb = model_memory_gb + optimizer_memory_gb + activation_memory_gb + gradient_memory_gb

    print(f"\nMemory breakdown (batch_size={batch_size}, fp32):")
    print(f"  - Model parameters: {model_memory_gb:.2f} GB")
    print(f"  - Optimizer state: {optimizer_memory_gb:.2f} GB")
    print(f"  - Activations (est.): {activation_memory_gb:.2f} GB")
    print(f"  - Gradients: {gradient_memory_gb:.2f} GB")
    print(f"  - Total (estimated): {total_memory_gb:.2f} GB")

    print(f"\nWith mixed precision (bf16):")
    total_memory_bf16 = total_memory_gb * 0.6  # Rough estimate
    print(f"  - Total (estimated): {total_memory_bf16:.2f} GB")

    print(f"\nRecommended GPU:")
    if total_memory_bf16 < 24:
        print(f"  - RTX 4090 (24GB) or A5000 (24GB)")
    elif total_memory_bf16 < 40:
        print(f"  - A100 40GB")
    else:
        print(f"  - A100 80GB or H100 80GB")


def main():
    parser = argparse.ArgumentParser(description='Display D4RT model information')
    parser.add_argument(
        '--model',
        type=str,
        choices=['vit_b', 'vit_l', 'vit_g', 'all'],
        default='all',
        help='Model configuration to display'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all model sizes'
    )
    parser.add_argument(
        '--memory',
        action='store_true',
        help='Show memory usage estimates'
    )

    args = parser.parse_args()

    if args.model == 'all':
        # Show all models
        for config_name in ['vit_b', 'vit_l', 'vit_g']:
            config_path = Path(__file__).parent.parent / 'configs' / 'model' / f'{config_name}.yaml'
            config = OmegaConf.load(config_path)
            model = build_d4rt_model(config)
            display_model_architecture(model, config_name)
            if args.memory:
                estimate_memory_usage(config_name)
            print("\n")

        if args.compare:
            compare_model_sizes()
    else:
        # Show specific model
        config_path = Path(__file__).parent.parent / 'configs' / 'model' / f'{args.model}.yaml'
        config = OmegaConf.load(config_path)
        model = build_d4rt_model(config)
        display_model_architecture(model, args.model)

        if args.memory:
            estimate_memory_usage(args.model)

        if args.compare:
            compare_model_sizes()


if __name__ == '__main__':
    main()
