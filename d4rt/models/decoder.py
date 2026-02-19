"""Cross-Attention Decoder for D4RT.

The decoder takes query embeddings and cross-attends to the encoder's
global scene representation to output:
- xyz: 3D position [B, N, 3]
- uv: 2D image coordinates [B, N, 2]
- normals: surface normals [B, N, 3] (unit normalized)
- motion: motion displacement [B, N, 3]
- visibility: occlusion logit [B, N, 1]
- confidence: prediction quality [B, N, 1]
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.attention import DropPath, MultiHeadAttention


class ContextPooling(nn.Module):
    """
    Pool context tokens to reduce count for better attention gradient flow.

    When attention is computed over many tokens (e.g., 3072), the softmax
    produces near-uniform weights with very small gradients. Pooling to
    fewer tokens (e.g., 128) allows sharper attention patterns.
    """

    def __init__(self, input_tokens: int, output_tokens: int, embed_dim: int):
        """
        Initialize context pooling.

        Args:
            input_tokens: Number of input context tokens (e.g., 3072)
            output_tokens: Number of output tokens after pooling (e.g., 128)
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.embed_dim = embed_dim

        # Learnable pooling via linear projection
        self.pool = nn.Linear(input_tokens, output_tokens)
        nn.init.xavier_uniform_(self.pool.weight)
        nn.init.zeros_(self.pool.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool context tokens.

        Args:
            x: [B, input_tokens, embed_dim] context features

        Returns:
            pooled: [B, output_tokens, embed_dim] pooled features
        """
        # x: [B, input_tokens, embed_dim]
        x = x.transpose(1, 2)  # [B, embed_dim, input_tokens]
        x = self.pool(x)  # [B, embed_dim, output_tokens]
        x = x.transpose(1, 2)  # [B, output_tokens, embed_dim]
        return x


class CrossAttentionDecoder(nn.Module):
    """
    Cross-attention decoder that queries encoder features to predict outputs.

    The decoder takes query embeddings and cross-attends to the encoder's
    global scene representation to output 3D positions, 2D coordinates,
    surface normals, motion, visibility, and confidence.
    """

    def __init__(
        self,
        query_dim: int = 512,
        context_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        context_pool_tokens: Optional[int] = None,
        context_input_tokens: int = 3072,
        output_uv: bool = True,
        output_normals: bool = True,
        output_motion: bool = True,
    ):
        """
        Initialize cross-attention decoder.

        Args:
            query_dim: Input query dimension
            context_dim: Context (encoder) feature dimension
            hidden_dim: Hidden dimension of decoder
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            context_pool_tokens: Number of tokens after pooling (None = no pooling)
            context_input_tokens: Number of input context tokens (default: 3072 for ViT-B)
            output_uv: Whether to output 2D coordinates
            output_normals: Whether to output surface normals
            output_motion: Whether to output motion displacement
        """
        super().__init__()

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_pool_tokens = context_pool_tokens
        self.output_uv = output_uv
        self.output_normals = output_normals
        self.output_motion = output_motion

        # Project query to hidden dimension if needed
        if query_dim != hidden_dim:
            self.query_proj = nn.Linear(query_dim, hidden_dim)
        else:
            self.query_proj = nn.Identity()

        # Project context to hidden dimension if needed
        if context_dim != hidden_dim:
            self.context_proj = nn.Linear(context_dim, hidden_dim)
        else:
            self.context_proj = nn.Identity()

        # Optional context pooling for better attention gradient flow
        if context_pool_tokens is not None:
            self.context_pool = ContextPooling(
                input_tokens=context_input_tokens,
                output_tokens=context_pool_tokens,
                embed_dim=hidden_dim,
            )
        else:
            self.context_pool = None

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        # Decoder layers with cross-attention
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=dpr[i],
                )
                for i in range(num_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # ========== Output Heads ==========
        # Primary outputs (always present)
        self.xyz_head = nn.Linear(hidden_dim, 3)  # 3D position (x, y, z)
        self.vis_head = nn.Linear(hidden_dim, 1)  # Visibility logit
        self.confidence_head = nn.Linear(hidden_dim, 1)  # Confidence score

        # Optional output heads
        if output_uv:
            self.uv_head = nn.Linear(hidden_dim, 2)  # 2D coordinates (u, v)
        else:
            self.uv_head = None

        if output_normals:
            self.normals_head = nn.Linear(hidden_dim, 3)  # Surface normals
        else:
            self.normals_head = None

        if output_motion:
            self.motion_head = nn.Linear(hidden_dim, 3)  # Motion displacement
        else:
            self.motion_head = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize output heads with small values
        nn.init.xavier_uniform_(self.xyz_head.weight)
        nn.init.zeros_(self.xyz_head.bias)

        nn.init.xavier_uniform_(self.vis_head.weight)
        nn.init.zeros_(self.vis_head.bias)

        nn.init.xavier_uniform_(self.confidence_head.weight)
        nn.init.zeros_(self.confidence_head.bias)

        if self.uv_head is not None:
            nn.init.xavier_uniform_(self.uv_head.weight)
            nn.init.zeros_(self.uv_head.bias)

        if self.normals_head is not None:
            nn.init.xavier_uniform_(self.normals_head.weight)
            nn.init.zeros_(self.normals_head.bias)

        if self.motion_head is not None:
            nn.init.xavier_uniform_(self.motion_head.weight)
            nn.init.zeros_(self.motion_head.bias)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            queries: [B, N, query_dim] query embeddings
            context: [B, M, context_dim] encoder features
            query_mask: [B, N] query mask (optional)
            context_mask: [B, M] context mask (optional)

        Returns:
            outputs: Dictionary with keys:
                - 'xyz': [B, N, 3] predicted 3D positions
                - 'visibility': [B, N, 1] visibility logits
                - 'confidence': [B, N, 1] confidence scores (raw logits)
                - 'uv': [B, N, 2] 2D coordinates (if output_uv=True)
                - 'normals': [B, N, 3] surface normals, unit normalized (if output_normals=True)
                - 'motion': [B, N, 3] motion displacement (if output_motion=True)
        """
        # Project inputs
        x = self.query_proj(queries)  # [B, N, hidden_dim]
        ctx = self.context_proj(context)  # [B, M, hidden_dim]

        # Apply context pooling if enabled
        if self.context_pool is not None:
            ctx = self.context_pool(ctx)  # [B, pool_tokens, hidden_dim]

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, ctx)

        # Final layer norm
        x = self.norm(x)

        # ========== Output Heads ==========
        outputs = {}

        # Primary outputs
        # Add +1 to depth (Z) as per D4RT author:
        # "we add 1 to the estimated depth values since the initialization
        # would otherwise start at 0, hindering training dynamics"
        xyz = self.xyz_head(x)  # [B, N, 3]
        xyz = torch.cat([xyz[..., :2], xyz[..., 2:3] + 1.0], dim=-1)
        outputs["xyz"] = xyz
        outputs["visibility"] = self.vis_head(x)  # [B, N, 1]
        outputs["confidence"] = self.confidence_head(x)  # [B, N, 1] raw logits

        # Optional outputs
        if self.uv_head is not None:
            # UV coordinates in [0, 1] range
            outputs["uv"] = torch.sigmoid(self.uv_head(x))  # [B, N, 2]

        if self.normals_head is not None:
            # Surface normals, unit normalized
            normals = self.normals_head(x)  # [B, N, 3]
            outputs["normals"] = F.normalize(normals, dim=-1)  # [B, N, 3]

        if self.motion_head is not None:
            outputs["motion"] = self.motion_head(x)  # [B, N, 3]

        return outputs


class DecoderLayer(nn.Module):
    """Single decoder layer with cross-attention and FFN (NO self-attention per paper).

    The D4RT paper specifies that "queries do not interact" - they should only
    cross-attend to encoder features. Self-attention would allow queries to
    mix information, potentially diluting depth-specific features.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        """
        Initialize decoder layer.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            drop_path: Stochastic depth rate
        """
        super().__init__()

        # Cross-attention only (NO self-attention per paper: "queries do not interact")
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads, dropout=attention_dropout)

        # FFN block
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, N, hidden_dim] query features
            context: [B, M, hidden_dim] context features

        Returns:
            output: [B, N, hidden_dim] transformed features
        """
        # Cross-attention to encoder features ONLY (no self-attention per paper)
        x = x + self.drop_path(self.cross_attn(self.cross_norm(x), key=context, value=context))

        # FFN
        x = x + self.drop_path(self.mlp(self.norm_ffn(x)))

        return x


def build_decoder(config: dict) -> CrossAttentionDecoder:
    """
    Build decoder from config dictionary.

    Args:
        config: Configuration dictionary with decoder parameters

    Returns:
        decoder: CrossAttentionDecoder instance
    """
    return CrossAttentionDecoder(
        query_dim=config.get("query_dim", 512),
        context_dim=config.get("context_dim", 768),
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 8),
        num_heads=config.get("num_heads", 8),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        dropout=config.get("dropout", 0.0),
        attention_dropout=config.get("attention_dropout", 0.0),
        drop_path_rate=config.get("drop_path_rate", 0.1),
        context_pool_tokens=config.get("context_pool_tokens", None),
        context_input_tokens=config.get("context_input_tokens", 3072),
        output_uv=config.get("output_uv", True),
        output_normals=config.get("output_normals", True),
        output_motion=config.get("output_motion", True),
    )
