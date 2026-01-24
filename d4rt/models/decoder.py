"""Cross-Attention Decoder for D4RT."""

import torch
import torch.nn as nn
from typing import Optional
from .components.attention import TransformerBlock


class CrossAttentionDecoder(nn.Module):
    """
    Cross-attention decoder that queries encoder features to predict 3D positions.

    The decoder takes query embeddings and cross-attends to the encoder's
    global scene representation to output 3D positions and visibility.
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
        """
        super().__init__()

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        # Decoder layers with cross-attention
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=dpr[i],
            )
            for i in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Output heads
        self.xyz_head = nn.Linear(hidden_dim, 3)  # 3D position (x, y, z)
        self.vis_head = nn.Linear(hidden_dim, 1)  # Visibility logit

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize output heads with small values
        nn.init.xavier_uniform_(self.xyz_head.weight)
        nn.init.zeros_(self.xyz_head.bias)

        nn.init.xavier_uniform_(self.vis_head.weight)
        nn.init.zeros_(self.vis_head.bias)

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
        """
        # Project inputs
        x = self.query_proj(queries)  # [B, N, hidden_dim]
        ctx = self.context_proj(context)  # [B, M, hidden_dim]

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, ctx)

        # Final layer norm
        x = self.norm(x)

        # Output heads
        xyz = self.xyz_head(x)  # [B, N, 3]
        visibility = self.vis_head(x)  # [B, N, 1]

        return {
            'xyz': xyz,
            'visibility': visibility,
        }


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention, cross-attention, and FFN."""

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

        # Self-attention block
        self.self_attn_block = TransformerBlock(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=0,  # No MLP in self-attention block
            dropout=dropout,
            attention_dropout=attention_dropout,
            drop_path=drop_path,
            use_cross_attention=False,
        )

        # Cross-attention block
        self.cross_attn_block = TransformerBlock(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=0,  # No MLP in cross-attention block
            dropout=dropout,
            attention_dropout=attention_dropout,
            drop_path=drop_path,
            use_cross_attention=True,
        )

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
        from .components.attention import DropPath
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
        # Self-attention
        x = self.self_attn_block(x)

        # Cross-attention to encoder features
        x = self.cross_attn_block(x, context=context)

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
        query_dim=config.get('query_dim', 512),
        context_dim=config.get('context_dim', 768),
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 8),
        num_heads=config.get('num_heads', 8),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config.get('dropout', 0.0),
        attention_dropout=config.get('attention_dropout', 0.0),
        drop_path_rate=config.get('drop_path_rate', 0.1),
    )
