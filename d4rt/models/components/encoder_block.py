"""D4RT Encoder Block with Local + Global Attention (Figure 7).

Per Figure 7, EACH encoder block contains BOTH attention types in sequence:
1. Per-Frame Self-Attention (local) → MLP - operates on patches within each frame
2. Global Self-Attention → MLP - operates on all tokens across all frames

This design preserves spatial diversity within frames (local attention)
while enabling temporal/cross-frame information flow (global attention).
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import DropPath


class MLP(nn.Module):
    """Simple MLP with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class LocalFrameAttention(nn.Module):
    """
    Per-frame self-attention that operates within each frame independently.

    This preserves spatial diversity within frames by preventing tokens
    from different frames from attending to each other.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Use PyTorch's MultiheadAttention for efficiency
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        patches_per_frame: int,
    ) -> torch.Tensor:
        """
        Apply per-frame self-attention.

        Args:
            x: [B, T*H*W, D] input tokens (video tokens only, no AR token)
            num_frames: T (number of frames)
            patches_per_frame: H*W (patches per frame)

        Returns:
            out: [B, T*H*W, D] output tokens after per-frame attention
        """
        B, N, D = x.shape
        assert N == num_frames * patches_per_frame, (
            f"Expected {num_frames * patches_per_frame} tokens, got {N}"
        )

        # Reshape to [B*T, H*W, D] for per-frame attention
        x = x.view(B * num_frames, patches_per_frame, D)

        # Self-attention within each frame
        attn_out, _ = self.attn(x, x, x)

        # Reshape back to [B, T*H*W, D]
        # Use reshape instead of view to handle non-contiguous tensors
        attn_out = attn_out.reshape(B, N, D)

        return attn_out


class D4RTEncoderBlock(nn.Module):
    """
    Single encoder block with BOTH local and global attention (Figure 7).

    Structure per block:
        1. Per-Frame Self-Attention (local) → MLP
        2. Global Self-Attention → MLP

    The local attention preserves spatial diversity within frames,
    while global attention enables cross-frame temporal aggregation.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        """
        Initialize D4RT encoder block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            drop_path: Stochastic depth rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        # ========== Per-Frame Self-Attention (local) ==========
        self.norm1 = nn.LayerNorm(embed_dim)
        self.local_attn = LocalFrameAttention(
            embed_dim, num_heads, dropout=attention_dropout
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # MLP after local attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp1 = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # ========== Global Self-Attention ==========
        self.norm3 = nn.LayerNorm(embed_dim)
        self.global_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.drop_path3 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # MLP after global attention
        self.norm4 = nn.LayerNorm(embed_dim)
        self.mlp2 = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)
        self.drop_path4 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        patches_per_frame: int,
        has_ar_token: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through encoder block.

        Args:
            x: [B, N, D] input tokens (N = 1 + T*H*W if has_ar_token, else T*H*W)
            num_frames: T (number of frames)
            patches_per_frame: H*W (patches per frame)
            has_ar_token: Whether the input includes aspect ratio token at position 0

        Returns:
            x: [B, N, D] output tokens
        """
        B = x.shape[0]

        if has_ar_token:
            # Separate AR token from video tokens
            ar_token = x[:, :1, :]      # [B, 1, D]
            video_tokens = x[:, 1:, :]  # [B, T*H*W, D]
        else:
            ar_token = None
            video_tokens = x

        # ========== 1. Per-Frame Self-Attention (local) ==========
        # Only applied to video tokens (AR token doesn't belong to any frame)
        video_norm = self.norm1(video_tokens)
        local_out = self.local_attn(video_norm, num_frames, patches_per_frame)
        video_tokens = video_tokens + self.drop_path1(local_out)

        # ========== 2. MLP after local attention ==========
        video_tokens = video_tokens + self.drop_path2(self.mlp1(self.norm2(video_tokens)))

        # Recombine with AR token for global attention
        if has_ar_token:
            x = torch.cat([ar_token, video_tokens], dim=1)  # [B, 1+T*H*W, D]
        else:
            x = video_tokens

        # ========== 3. Global Self-Attention ==========
        # AR token participates in global attention (can attend to all video tokens)
        x_norm = self.norm3(x)
        global_out, _ = self.global_attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path3(global_out)

        # ========== 4. MLP after global attention ==========
        x = x + self.drop_path4(self.mlp2(self.norm4(x)))

        return x


class LegacyEncoderBlock(nn.Module):
    """
    Legacy encoder block with global-only self-attention.

    Preserved for backward compatibility with existing checkpoints.
    This block only uses global self-attention (no local per-frame attention).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int = None,
        patches_per_frame: int = None,
        has_ar_token: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass (global-only attention).

        Args:
            x: [B, N, D] input tokens
            num_frames: Unused (for API compatibility)
            patches_per_frame: Unused (for API compatibility)
            has_ar_token: Unused (for API compatibility)

        Returns:
            x: [B, N, D] output tokens
        """
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(attn_out)

        # MLP
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x
