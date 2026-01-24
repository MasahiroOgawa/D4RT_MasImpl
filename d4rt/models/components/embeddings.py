"""Embedding modules for D4RT query encoding."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FourierPositionalEncoding(nn.Module):
    """Fourier positional encoding for (u, v) coordinates."""

    def __init__(self, num_frequencies: int = 10, max_frequency: int = 9):
        """
        Initialize Fourier encoding.

        Args:
            num_frequencies: Number of frequency bands (default: 10)
            max_frequency: Maximum frequency power (2^max_frequency)
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        # Frequency bands: [2^0, 2^1, ..., 2^max_frequency]
        frequencies = 2.0 ** torch.linspace(0, max_frequency, num_frequencies)
        self.register_buffer('frequencies', frequencies)

    @property
    def output_dim(self) -> int:
        """Output dimension: 2 coords * num_freq * 2 (sin + cos) = 40 for default."""
        return 2 * self.num_frequencies * 2

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates with Fourier features.

        Args:
            coords: [B, N, 2] normalized coordinates in [0, 1]

        Returns:
            features: [B, N, 40] Fourier features
        """
        B, N, _ = coords.shape

        # Expand coordinates for all frequencies
        # coords: [B, N, 2] -> [B, N, 2, 1]
        # frequencies: [num_freq] -> [1, 1, 1, num_freq]
        coords = coords.unsqueeze(-1)  # [B, N, 2, 1]
        freqs = self.frequencies.view(1, 1, 1, -1)  # [1, 1, 1, num_freq]

        # Compute angles
        angles = coords * freqs * math.pi  # [B, N, 2, num_freq]

        # Compute sin and cos
        sin_features = torch.sin(angles)  # [B, N, 2, num_freq]
        cos_features = torch.cos(angles)  # [B, N, 2, num_freq]

        # Interleave sin and cos, flatten last two dimensions
        features = torch.stack([sin_features, cos_features], dim=-1)  # [B, N, 2, num_freq, 2]
        features = features.flatten(-3)  # [B, N, 2*num_freq*2]

        return features


class TemporalEmbedding(nn.Module):
    """Learned temporal embeddings for frame indices."""

    def __init__(self, max_frames: int = 128, embedding_dim: int = 256):
        """
        Initialize temporal embedding.

        Args:
            max_frames: Maximum number of frames
            embedding_dim: Dimension of embeddings
        """
        super().__init__()
        self.max_frames = max_frames
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(max_frames, embedding_dim)

        # Initialize with small values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for frame indices.

        Args:
            frame_indices: [B, N] frame indices (integers)

        Returns:
            embeddings: [B, N, embedding_dim]
        """
        return self.embedding(frame_indices)


class PatchCNN(nn.Module):
    """Small CNN for encoding local RGB patches."""

    def __init__(
        self,
        patch_size: int = 9,
        in_channels: int = 3,
        hidden_channels: Tuple[int, ...] = (64, 128),
        output_dim: int = 256,
    ):
        """
        Initialize patch CNN.

        Args:
            patch_size: Size of input patch (default: 9x9)
            in_channels: Number of input channels (default: 3 for RGB)
            hidden_channels: Hidden channel dimensions
            output_dim: Output feature dimension
        """
        super().__init__()
        self.patch_size = patch_size

        layers = []
        prev_channels = in_channels

        for hidden_dim in hidden_channels:
            layers.extend([
                nn.Conv2d(prev_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_channels = hidden_dim

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling and projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(prev_channels, output_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Encode patches to feature vectors.

        Args:
            patches: [B, N, 3, patch_size, patch_size] RGB patches

        Returns:
            features: [B, N, output_dim] patch features
        """
        B, N, C, H, W = patches.shape

        # Reshape for batch processing
        patches = patches.reshape(B * N, C, H, W)

        # Apply conv layers
        features = self.conv_layers(patches)  # [B*N, hidden_dim, H, W]

        # Global pooling
        features = self.pool(features)  # [B*N, hidden_dim, 1, 1]
        features = features.flatten(1)  # [B*N, hidden_dim]

        # Project to output dimension
        features = self.fc(features)  # [B*N, output_dim]

        # Reshape back
        features = features.reshape(B, N, -1)  # [B, N, output_dim]

        return features


class QueryEncoder(nn.Module):
    """
    Complete query encoder that combines Fourier encoding, temporal embeddings,
    and patch features into a single query representation.
    """

    def __init__(
        self,
        num_frequencies: int = 10,
        max_frequency: int = 9,
        max_frames: int = 128,
        temporal_dim: int = 256,
        patch_size: int = 9,
        patch_channels: Tuple[int, ...] = (64, 128),
        patch_dim: int = 256,
        output_dim: int = 512,
    ):
        """
        Initialize query encoder.

        Args:
            num_frequencies: Number of Fourier frequency bands
            max_frequency: Maximum Fourier frequency power
            max_frames: Maximum number of video frames
            temporal_dim: Dimension of temporal embeddings
            patch_size: Size of RGB patches
            patch_channels: Hidden channels for patch CNN
            patch_dim: Output dimension of patch CNN
            output_dim: Final output dimension for decoder
        """
        super().__init__()

        # Fourier encoding for (u, v)
        self.fourier = FourierPositionalEncoding(num_frequencies, max_frequency)
        fourier_dim = self.fourier.output_dim

        # Temporal embeddings (one for each temporal component)
        self.temporal_src = TemporalEmbedding(max_frames, temporal_dim)
        self.temporal_tgt = TemporalEmbedding(max_frames, temporal_dim)
        self.temporal_cam = TemporalEmbedding(max_frames, temporal_dim)

        # Patch CNN
        self.patch_cnn = PatchCNN(
            patch_size=patch_size,
            hidden_channels=patch_channels,
            output_dim=patch_dim,
        )

        # Total input dimension
        total_dim = fourier_dim + 3 * temporal_dim + patch_dim

        # Final projection to decoder dimension
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.output_dim = output_dim

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
        patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode queries from components.

        Args:
            u: [B, N] normalized u coordinates
            v: [B, N] normalized v coordinates
            t_src: [B, N] source frame indices
            t_tgt: [B, N] target frame indices
            t_cam: [B, N] camera frame indices
            patches: [B, N, 3, patch_size, patch_size] RGB patches

        Returns:
            query_embeddings: [B, N, output_dim] encoded queries
        """
        # Stack (u, v) coordinates
        coords = torch.stack([u, v], dim=-1)  # [B, N, 2]

        # Fourier encoding
        fourier_features = self.fourier(coords)  # [B, N, 40]

        # Temporal embeddings
        t_src_emb = self.temporal_src(t_src)  # [B, N, temporal_dim]
        t_tgt_emb = self.temporal_tgt(t_tgt)  # [B, N, temporal_dim]
        t_cam_emb = self.temporal_cam(t_cam)  # [B, N, temporal_dim]

        # Patch features
        patch_features = self.patch_cnn(patches)  # [B, N, patch_dim]

        # Concatenate all features
        query_features = torch.cat([
            fourier_features,
            t_src_emb,
            t_tgt_emb,
            t_cam_emb,
            patch_features,
        ], dim=-1)  # [B, N, total_dim]

        # Project to output dimension
        query_embeddings = self.projection(query_features)  # [B, N, output_dim]

        return query_embeddings
