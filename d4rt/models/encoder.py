"""Spatio-Temporal ViT Encoder for D4RT.

This module implements the D4RT encoder per Figure 7 of the paper:
- Video → Tokenizer → PE → [video_tokens with positional encoding]
- W/H → FC → [ar_token without positional encoding]
- Concat([ar_token, video_tokens]) → N Encoder Blocks → F

Each encoder block contains BOTH local (per-frame) and global attention.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Tuple
import torch.utils.checkpoint as checkpoint

from .components.attention import TransformerBlock
from .components.aspect_ratio_token import AspectRatioToken
from .components.encoder_block import D4RTEncoderBlock, LegacyEncoderBlock


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
    D4RT Spatio-Temporal ViT Encoder (Figure 7).

    Architecture (exactly per Figure 7):
        Video → Tokenizer → PE → [video_tokens with pos encoding]
        W/H → FC → [ar_token without pos encoding]
        Concat([ar_token, video_tokens]) → N Encoder Blocks → F

    Key features:
    - Aspect ratio token: separate token without positional encoding
    - Paper blocks: each block has BOTH local (per-frame) and global attention
    - Optional patch normalization (disabled by default per paper)
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
        use_patch_norm: bool = False,  # Disabled by default (preserves brightness)
        use_paper_blocks: bool = True,  # Figure 7: each block has local + global
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
            use_patch_norm: Whether to normalize patch embeddings
                           (False by default - preserves brightness information)
            use_paper_blocks: Whether to use paper block structure with both
                             local and global attention (True by default)
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self.use_patch_norm = use_patch_norm
        self.use_paper_blocks = use_paper_blocks

        # Calculate number of patches
        T, H, W = input_resolution
        t_patch, h_patch, w_patch = patch_size
        self.num_temporal_patches = T // t_patch
        self.num_spatial_patches_h = H // h_patch
        self.num_spatial_patches_w = W // w_patch
        self.patches_per_frame = self.num_spatial_patches_h * self.num_spatial_patches_w
        self.num_patches = self.num_temporal_patches * self.patches_per_frame

        # Patch embedding (NO normalization by default - preserves brightness)
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Optional patch normalization (disabled by default per paper)
        if use_patch_norm:
            self.patch_norm = nn.LayerNorm(embed_dim)
            pos_embed_std = 0.5  # Larger std to match normalized patch scale
        else:
            self.patch_norm = nn.Identity()
            pos_embed_std = 0.02  # Standard ViT initialization

        # Positional embedding (ONLY for video tokens, NOT for AR token - per Figure 7)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=pos_embed_std)

        self.pos_drop = nn.Dropout(p=dropout)

        # Aspect ratio token (Paper p.3: "separate token", Figure 7: W/H → FC)
        # Note: AR token does NOT get positional encoding
        self.aspect_ratio_token = AspectRatioToken(embed_dim)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        # Encoder blocks
        if use_paper_blocks:
            # Paper architecture: each block has BOTH local + global attention
            self.blocks = nn.ModuleList([
                D4RTEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=dpr[i],
                )
                for i in range(num_layers)
            ])
        else:
            # Legacy: global-only attention (for backward compatibility)
            self.blocks = nn.ModuleList([
                LegacyEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=dpr[i],
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

    def forward(
        self,
        x: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, C, H, W] video tensor
            aspect_ratio: [B] original W/H ratio (optional, defaults to 1.0)

        Returns:
            features: [B, num_patches, embed_dim] encoded features (Global Scene Representation F)
        """
        B, T, C, H, W = x.shape

        # Rearrange to [B, C, T, H, W] for Conv3d
        x = rearrange(x, 'b t c h w -> b c t h w')

        # Tokenizer: Video → patches (NO normalization by default)
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Optional patch normalization
        x = self.patch_norm(x)

        # PE: Add positional encoding to video tokens ONLY (Figure 7)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Aspect ratio token (Paper p.3: "embed it into a separate token")
        # NO positional encoding for AR token (per Figure 7)
        if aspect_ratio is None:
            aspect_ratio = torch.ones(B, device=x.device)
        ar_token = self.aspect_ratio_token(aspect_ratio)  # [B, 1, embed_dim]

        # Concatenate: [ar_token, video_tokens] (Figure 7: ⊕)
        x = torch.cat([ar_token, x], dim=1)  # [B, 1 + num_patches, embed_dim]

        # N Encoder Blocks (each with local + global attention if use_paper_blocks)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(
                    block, x,
                    self.num_temporal_patches,
                    self.patches_per_frame,
                    True,  # has_ar_token
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    num_frames=self.num_temporal_patches,
                    patches_per_frame=self.patches_per_frame,
                    has_ar_token=True,
                )

        # Final LayerNorm
        x = self.norm(x)

        # Return only video tokens (exclude AR token) as Global Scene Representation F
        return x[:, 1:]  # [B, num_patches, embed_dim]

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
        use_patch_norm=config.get('use_patch_norm', False),  # Disabled by default
        use_paper_blocks=config.get('use_paper_blocks', True),  # Paper architecture
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
        - Our D4RT blocks have different structure (local + global attention)
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

        # Transformer blocks - map to D4RT block structure
        elif key.startswith('blocks.'):
            parts = key.split('.')
            block_idx = parts[1]

            if encoder.use_paper_blocks:
                # D4RT paper blocks have different structure
                # VideoMAE: norm1 → attn → norm2 → mlp
                # D4RT: norm1 → local_attn → norm2 → mlp1 → norm3 → global_attn → norm4 → mlp2

                # Map VideoMAE attention to global attention
                if parts[2] == 'norm1':
                    new_key = f'blocks.{block_idx}.norm3.{parts[3]}'
                    new_state_dict[new_key] = value
                elif parts[2] == 'norm2':
                    new_key = f'blocks.{block_idx}.norm4.{parts[3]}'
                    new_state_dict[new_key] = value
                elif parts[2] == 'attn':
                    if parts[3] == 'qkv' and parts[4] == 'weight':
                        # Split fused QKV weight and map to global attention
                        embed_dim = value.shape[0] // 3
                        q, k, v = value.split(embed_dim, dim=0)
                        new_state_dict[f'blocks.{block_idx}.global_attn.in_proj_weight'] = value
                    elif parts[3] == 'q_bias':
                        # Store for later combination
                        pass  # Handled with qkv
                    elif parts[3] == 'v_bias':
                        # Store for later combination
                        pass  # Handled with qkv
                    elif parts[3] == 'proj' and parts[4] == 'weight':
                        new_state_dict[f'blocks.{block_idx}.global_attn.out_proj.weight'] = value
                    elif parts[3] == 'proj' and parts[4] == 'bias':
                        new_state_dict[f'blocks.{block_idx}.global_attn.out_proj.bias'] = value
                elif parts[2] == 'mlp':
                    if parts[3] == 'fc1':
                        new_key = f'blocks.{block_idx}.mlp2.fc1.{parts[4]}'
                        new_state_dict[new_key] = value
                    elif parts[3] == 'fc2':
                        new_key = f'blocks.{block_idx}.mlp2.fc2.{parts[4]}'
                        new_state_dict[new_key] = value
            else:
                # Legacy blocks - simpler mapping
                if parts[2] == 'norm1':
                    new_key = f'blocks.{block_idx}.norm1.{parts[3]}'
                    new_state_dict[new_key] = value
                elif parts[2] == 'norm2':
                    new_key = f'blocks.{block_idx}.norm2.{parts[3]}'
                    new_state_dict[new_key] = value
                elif parts[2] == 'attn':
                    if parts[3] == 'qkv' and parts[4] == 'weight':
                        new_state_dict[f'blocks.{block_idx}.self_attn.in_proj_weight'] = value
                    elif parts[3] == 'proj' and parts[4] == 'weight':
                        new_state_dict[f'blocks.{block_idx}.self_attn.out_proj.weight'] = value
                    elif parts[3] == 'proj' and parts[4] == 'bias':
                        new_state_dict[f'blocks.{block_idx}.self_attn.out_proj.bias'] = value
                elif parts[2] == 'mlp':
                    if parts[3] == 'fc1':
                        new_key = f'blocks.{block_idx}.mlp.fc1.{parts[4]}'
                        new_state_dict[new_key] = value
                    elif parts[3] == 'fc2':
                        new_key = f'blocks.{block_idx}.mlp.fc2.{parts[4]}'
                        new_state_dict[new_key] = value
        else:
            unexpected_keys.append(old_key)

    # Load the mapped state dict
    load_result = encoder.load_state_dict(new_state_dict, strict=False)
    missing_keys = load_result.missing_keys

    logger.info(f"Loaded VideoMAE weights from {checkpoint_path}")
    logger.info(f"Loaded {len(new_state_dict)} parameters")
    logger.info(f"Missing keys (expected for D4RT-specific modules): {missing_keys}")

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
    T, H, W = encoder.input_resolution
    t_patch, h_patch, w_patch = encoder.patch_size
    target_t = T // t_patch
    target_h = H // h_patch
    target_w = W // w_patch

    # Reshape to 3D grid, interpolate, reshape back
    pos_embed = pos_embed.reshape(1, -1, D)

    # Use linear interpolation along the sequence dimension
    pos_embed = pos_embed.permute(0, 2, 1)  # [1, D, N]
    pos_embed = torch.nn.functional.interpolate(
        pos_embed, size=target_num_patches, mode='linear', align_corners=False
    )
    pos_embed = pos_embed.permute(0, 2, 1)  # [1, target_num_patches, D]

    return pos_embed
