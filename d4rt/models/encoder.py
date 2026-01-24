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

        # Positional embedding (learned)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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
    )
