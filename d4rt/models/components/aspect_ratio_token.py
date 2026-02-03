"""Aspect Ratio Token for D4RT (Paper p.3, Figure 7).

Per paper: "we embed it into a separate token and pass it to the transformer
along with the main video tokens."

The aspect ratio W/H is passed through an FC layer to create a single token
that is concatenated with the video patch tokens before the transformer encoder.
Note: This token does NOT receive positional encoding (per Figure 7).
"""

import torch
import torch.nn as nn


class AspectRatioToken(nn.Module):
    """
    Embed aspect ratio as a separate token (Paper p.3, Figure 7).

    The aspect ratio W/H is passed through an FC layer to create a single
    token that is concatenated with the video patch tokens before the
    transformer encoder.
    """

    def __init__(self, embed_dim: int = 768):
        """
        Initialize aspect ratio token embedder.

        Args:
            embed_dim: Output embedding dimension (should match encoder embed_dim)
        """
        super().__init__()
        self.embed_dim = embed_dim

        # FC layer: scalar → embed_dim (Figure 7: W/H → FC)
        self.fc = nn.Linear(1, embed_dim)

        # Initialize with Xavier for stable gradients
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, aspect_ratio: torch.Tensor) -> torch.Tensor:
        """
        Convert aspect ratio scalar to token embedding.

        Args:
            aspect_ratio: [B] or [B, 1] scalar W/H ratio

        Returns:
            ar_token: [B, 1, embed_dim] single token to concatenate with video tokens
        """
        # Ensure shape is [B, 1]
        if aspect_ratio.dim() == 1:
            aspect_ratio = aspect_ratio.unsqueeze(-1)  # [B, 1]

        # FC layer: [B, 1] → [B, embed_dim]
        ar_token = self.fc(aspect_ratio)  # [B, embed_dim]

        # Return as [B, 1, embed_dim] for concatenation
        return ar_token.unsqueeze(1)  # [B, 1, embed_dim]
