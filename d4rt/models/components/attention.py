"""Attention modules for D4RT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Multi-head attention module with optional cross-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            query: [B, N, C] query tensor
            key: [B, M, C] key tensor (optional, defaults to query for self-attention)
            value: [B, M, C] value tensor (optional, defaults to key)
            attn_mask: [B, N, M] or [N, M] attention mask (optional)

        Returns:
            output: [B, N, C] attention output
        """
        B, N, C = query.shape

        # Default to self-attention
        if key is None:
            key = query
        if value is None:
            value = key

        M = key.shape[1]

        # Project and reshape for multi-head attention
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [B, num_heads, N/M, head_dim]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, M]

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, M]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, N, M]
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        # Output projection
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with self-attention, optional cross-attention, and FFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        use_cross_attention: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cross_attention = use_cross_attention

        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(
            embed_dim, num_heads, dropout=attention_dropout
        )

        # Cross-attention (optional)
        if use_cross_attention:
            self.norm2 = nn.LayerNorm(embed_dim)
            self.cross_attn = MultiHeadAttention(
                embed_dim, num_heads, dropout=attention_dropout
            )

        # FFN
        self.norm_ffn = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of transformer block.

        Args:
            x: [B, N, C] input tensor
            context: [B, M, C] context tensor for cross-attention (optional)
            attn_mask: attention mask (optional)

        Returns:
            output: [B, N, C] transformed tensor
        """
        # Self-attention
        x = x + self.drop_path(self.self_attn(self.norm1(x), attn_mask=attn_mask))

        # Cross-attention (if enabled and context provided)
        if self.use_cross_attention and context is not None:
            x = x + self.drop_path(
                self.cross_attn(self.norm2(x), key=context, value=context)
            )

        # FFN
        x = x + self.drop_path(self.mlp(self.norm_ffn(x)))

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for residual blocks."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with diff dim tensors
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output
