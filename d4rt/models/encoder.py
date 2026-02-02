"""Spatio-Temporal ViT Encoder for D4RT."""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Tuple
import torch.utils.checkpoint as checkpoint

from .components.attention import TransformerBlock


class PatchEmbed3D(nn.Module):
    """3D patch embedding for video."""

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        """
        Initialize 3D patch embedding.

        Args:
            patch_size: (temporal, height, width) patch dimensions
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 3D convolution for patch embedding
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, C, T, H, W] video tensor

        Returns:
            embeddings: [B, num_patches, embed_dim]
        """
        # Project patches
        x = self.proj(x)  # [B, embed_dim, T', H', W']

        # Flatten spatial dimensions
        x = rearrange(x, 'b c t h w -> b (t h w) c')  # [B, num_patches, embed_dim]

        return x


class SpatioTemporalViT(nn.Module):
    """
    Spatio-Temporal Vision Transformer for video encoding.

    Processes video frames using 3D patch embedding followed by
    standard transformer blocks with self-attention.

    Note: This implementation adds LayerNorm after patch embedding,
    which differs from the original D4RT paper. The original paper uses
    VideoMAE pretrained weights where the patch/positional embedding scales
    are already balanced from pretraining. We add LayerNorm to maintain
    this balance during fine-tuning from scratch or with pretrained weights.
    See README.md for details.
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int, int] = (48, 256, 256),
        patch_size: Tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        use_patch_norm: bool = True,  # LayerNorm after patch embedding
    ):
        """
        Initialize Spatio-Temporal ViT.

        Args:
            input_resolution: (T, H, W) input video resolution
            patch_size: (t, h, w) patch dimensions
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self.use_patch_norm = use_patch_norm

        # Calculate number of patches
        T, H, W = input_resolution
        t_patch, h_patch, w_patch = patch_size
        self.num_patches = (T // t_patch) * (H // h_patch) * (W // w_patch)

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # LayerNorm after patch embedding (not in original D4RT, but helps maintain
        # scale balance between patch features and positional embeddings during
        # fine-tuning. See README.md for architectural differences.)
        if use_patch_norm:
            self.patch_norm = nn.LayerNorm(embed_dim)
            # When using patch norm, features have norm ≈ sqrt(embed_dim) ≈ 27.7
            # Scale up positional embedding to be comparable (std=0.5 → norm ≈ 13.9)
            pos_embed_std = 0.5
        else:
            self.patch_norm = nn.Identity()
            # Standard ViT initialization (assumes pretrained patch embed scale)
            pos_embed_std = 0.02

        # Positional embedding (learned)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=pos_embed_std)

        self.pos_drop = nn.Dropout(p=dropout)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=dpr[i],
                use_cross_attention=False,  # Encoder uses only self-attention
            )
            for i in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))

        # Initialize transformer blocks
        for block in self.blocks:
            # Initialize attention weights
            nn.init.xavier_uniform_(block.self_attn.q_proj.weight)
            nn.init.xavier_uniform_(block.self_attn.k_proj.weight)
            nn.init.xavier_uniform_(block.self_attn.v_proj.weight)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)

            # Initialize MLP weights
            for layer in block.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, C, H, W] video tensor

        Returns:
            features: [B, num_patches, embed_dim] encoded features
        """
        B = x.shape[0]

        # Rearrange to [B, C, T, H, W] for Conv3d
        x = rearrange(x, 'b t c h w -> b c t h w')

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Normalize patch embeddings (maintains scale balance with positional embeddings)
        x = self.patch_norm(x)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(block, x, None, None)
            else:
                x = block(x)

        # Final layer norm
        x = self.norm(x)

        return x

    @property
    def output_dim(self) -> int:
        """Output dimension of encoder."""
        return self.embed_dim

    @property
    def num_patches_out(self) -> int:
        """Number of output patches."""
        return self.num_patches


def build_vit_encoder(config: dict) -> SpatioTemporalViT:
    """
    Build encoder from config dictionary.

    Args:
        config: Configuration dictionary with encoder parameters

    Returns:
        encoder: SpatioTemporalViT instance
    """
    return SpatioTemporalViT(
        input_resolution=tuple(config.get('input_resolution', [48, 256, 256])),
        patch_size=tuple(config.get('patch_size', [2, 16, 16])),
        in_channels=3,
        embed_dim=config.get('hidden_dim', 768),
        num_layers=config.get('num_layers', 12),
        num_heads=config.get('num_heads', 12),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config.get('dropout', 0.0),
        attention_dropout=config.get('attention_dropout', 0.0),
        drop_path_rate=config.get('drop_path_rate', 0.1),
        use_checkpoint=config.get('use_checkpoint', False),
        use_patch_norm=config.get('use_patch_norm', True),  # Default True for stability
    )


def load_videomae_weights(
    encoder: SpatioTemporalViT,
    checkpoint_path: str,
    strict: bool = False,
) -> Tuple[list, list]:
    """
    Load VideoMAE pretrained weights into the encoder.

    The original D4RT paper uses VideoMAE pretrained weights for initialization.
    This function maps VideoMAE state dict keys to our encoder architecture.

    Args:
        encoder: SpatioTemporalViT instance
        checkpoint_path: Path to VideoMAE checkpoint (.pth file)
        strict: Whether to require exact key matching

    Returns:
        Tuple of (missing_keys, unexpected_keys)

    Note:
        VideoMAE checkpoints can be downloaded from:
        https://github.com/MCG-NJU/VideoMAE

        Key differences from our encoder:
        - VideoMAE keys have 'encoder.' prefix
        - VideoMAE uses fused QKV weights (attn.qkv.weight)
        - VideoMAE has separate q_bias/v_bias (no k_bias)
        - VideoMAE uses sinusoidal pos_embed (not in checkpoint)
    """
    import logging
    logger = logging.getLogger(__name__)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Build new state dict with mapped keys
    new_state_dict = {}
    unexpected_keys = []

    for old_key, value in state_dict.items():
        # Only process encoder keys (VideoMAE has encoder. prefix)
        if not old_key.startswith('encoder.'):
            unexpected_keys.append(old_key)
            continue

        # Remove 'encoder.' prefix
        key = old_key[8:]  # len('encoder.') = 8

        # Patch embedding
        if key == 'patch_embed.proj.weight':
            new_state_dict['patch_embed.proj.weight'] = value
        elif key == 'patch_embed.proj.bias':
            new_state_dict['patch_embed.proj.bias'] = value

        # Final norm
        elif key == 'norm.weight':
            new_state_dict['norm.weight'] = value
        elif key == 'norm.bias':
            new_state_dict['norm.bias'] = value

        # Transformer blocks
        elif key.startswith('blocks.'):
            parts = key.split('.')
            block_idx = parts[1]

            # LayerNorms
            if parts[2] == 'norm1':
                new_key = f'blocks.{block_idx}.norm1.{parts[3]}'
                new_state_dict[new_key] = value
            elif parts[2] == 'norm2':
                new_key = f'blocks.{block_idx}.norm2.{parts[3]}'
                new_state_dict[new_key] = value

            # Attention
            elif parts[2] == 'attn':
                if parts[3] == 'qkv' and parts[4] == 'weight':
                    # Split fused QKV weight
                    q, k, v = value.chunk(3, dim=0)
                    new_state_dict[f'blocks.{block_idx}.self_attn.q_proj.weight'] = q
                    new_state_dict[f'blocks.{block_idx}.self_attn.k_proj.weight'] = k
                    new_state_dict[f'blocks.{block_idx}.self_attn.v_proj.weight'] = v
                elif parts[3] == 'q_bias':
                    new_state_dict[f'blocks.{block_idx}.self_attn.q_proj.bias'] = value
                elif parts[3] == 'v_bias':
                    new_state_dict[f'blocks.{block_idx}.self_attn.v_proj.bias'] = value
                elif parts[3] == 'proj' and parts[4] == 'weight':
                    new_state_dict[f'blocks.{block_idx}.self_attn.out_proj.weight'] = value
                elif parts[3] == 'proj' and parts[4] == 'bias':
                    new_state_dict[f'blocks.{block_idx}.self_attn.out_proj.bias'] = value

            # MLP
            elif parts[2] == 'mlp':
                if parts[3] == 'fc1':
                    new_key = f'blocks.{block_idx}.mlp.0.{parts[4]}'
                    new_state_dict[new_key] = value
                elif parts[3] == 'fc2':
                    new_key = f'blocks.{block_idx}.mlp.3.{parts[4]}'
                    new_state_dict[new_key] = value
        else:
            unexpected_keys.append(old_key)

    # Note: VideoMAE uses sinusoidal positional embeddings generated on-the-fly,
    # not stored in checkpoint. We keep our learned positional embeddings.
    # Also, patch_norm is not in VideoMAE (our addition).

    # Load the mapped state dict
    load_result = encoder.load_state_dict(new_state_dict, strict=False)
    missing_keys = load_result.missing_keys

    logger.info(f"Loaded VideoMAE weights from {checkpoint_path}")
    logger.info(f"Loaded {len(new_state_dict)} parameters")
    logger.info(f"Missing keys (expected): {missing_keys}")

    if unexpected_keys:
        logger.debug(f"Skipped keys (decoder/other): {len(unexpected_keys)}")

    return missing_keys, unexpected_keys


def _interpolate_pos_embed(
    pos_embed: torch.Tensor,
    target_num_patches: int,
    encoder: SpatioTemporalViT,
) -> torch.Tensor:
    """
    Interpolate positional embeddings to match target number of patches.

    Args:
        pos_embed: Pretrained positional embedding [1, N, D]
        target_num_patches: Target number of patches
        encoder: Encoder instance (for getting grid dimensions)

    Returns:
        Interpolated positional embedding [1, target_num_patches, D]
    """
    N, D = pos_embed.shape[1], pos_embed.shape[2]

    if N == target_num_patches:
        return pos_embed

    # Calculate source and target grid sizes
    # Assuming the positional embedding is for a spatio-temporal grid
    T, H, W = encoder.input_resolution
    t_patch, h_patch, w_patch = encoder.patch_size
    target_t = T // t_patch
    target_h = H // h_patch
    target_w = W // w_patch

    # Reshape to 3D grid, interpolate, reshape back
    # This is a simplified version - full implementation would need
    # to know the source grid dimensions
    pos_embed = pos_embed.reshape(1, -1, D)

    # Use linear interpolation along the sequence dimension
    pos_embed = pos_embed.permute(0, 2, 1)  # [1, D, N]
    pos_embed = torch.nn.functional.interpolate(
        pos_embed, size=target_num_patches, mode='linear', align_corners=False
    )
    pos_embed = pos_embed.permute(0, 2, 1)  # [1, target_num_patches, D]

    return pos_embed
