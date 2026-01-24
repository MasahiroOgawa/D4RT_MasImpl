"""Main D4RT model that combines encoder, query encoder, and decoder."""

import torch
import torch.nn as nn
from typing import Dict, Optional
from omegaconf import DictConfig

from .encoder import SpatioTemporalViT, build_vit_encoder
from .decoder import CrossAttentionDecoder, build_decoder
from .components.embeddings import QueryEncoder
from ..utils.patch_utils import extract_patches


class D4RT(nn.Module):
    """
    D4RT: Dynamic 4D Reconstruction and Tracking.

    Complete model that combines:
    1. Spatio-temporal ViT encoder for video
    2. Query encoder for 5-tuple queries
    3. Cross-attention decoder for 3D prediction
    """

    def __init__(
        self,
        encoder: SpatioTemporalViT,
        decoder: CrossAttentionDecoder,
        query_encoder: QueryEncoder,
    ):
        """
        Initialize D4RT model.

        Args:
            encoder: Spatio-temporal ViT encoder
            decoder: Cross-attention decoder
            query_encoder: Query encoder
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.query_encoder = query_encoder

    def forward(
        self,
        video: torch.Tensor,
        queries: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            video: [B, T, C, H, W] video tensor
            queries: Dictionary with keys:
                - 'u': [B, N] normalized u coordinates
                - 'v': [B, N] normalized v coordinates
                - 't_src': [B, N] source frame indices
                - 't_tgt': [B, N] target frame indices
                - 't_cam': [B, N] camera frame indices

        Returns:
            outputs: Dictionary with keys:
                - 'xyz': [B, N, 3] predicted 3D positions
                - 'visibility': [B, N, 1] visibility logits
                - 'encoder_features': [B, num_patches, embed_dim] (optional)
        """
        # Extract query components
        u = queries['u']
        v = queries['v']
        t_src = queries['t_src']
        t_tgt = queries['t_tgt']
        t_cam = queries['t_cam']

        # Extract RGB patches from video
        patches = extract_patches(
            video,
            u,
            v,
            t_src,
            patch_size=9,
        )  # [B, N, 3, 9, 9]

        # Encode video
        encoder_features = self.encoder(video)  # [B, num_patches, embed_dim]

        # Encode queries
        query_embeddings = self.query_encoder(
            u, v, t_src, t_tgt, t_cam, patches
        )  # [B, N, query_dim]

        # Decode to 3D predictions
        outputs = self.decoder(query_embeddings, encoder_features)

        # Optionally include encoder features in output
        outputs['encoder_features'] = encoder_features

        return outputs

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to global representation (for inference).

        Args:
            video: [B, T, C, H, W] video tensor

        Returns:
            features: [B, num_patches, embed_dim] encoded features
        """
        return self.encoder(video)

    def predict_from_queries(
        self,
        encoder_features: torch.Tensor,
        queries: Dict[str, torch.Tensor],
        video: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict 3D positions from pre-computed encoder features (for inference).

        Args:
            encoder_features: [B, num_patches, embed_dim] pre-computed features
            queries: Dictionary with query components
            video: [B, T, C, H, W] video (needed for patch extraction)

        Returns:
            outputs: Dictionary with predictions
        """
        # Extract query components
        u = queries['u']
        v = queries['v']
        t_src = queries['t_src']
        t_tgt = queries['t_tgt']
        t_cam = queries['t_cam']

        # Extract RGB patches
        patches = extract_patches(video, u, v, t_src, patch_size=9)

        # Encode queries
        query_embeddings = self.query_encoder(
            u, v, t_src, t_tgt, t_cam, patches
        )

        # Decode
        outputs = self.decoder(query_embeddings, encoder_features)

        return outputs


def build_d4rt_model(config: DictConfig) -> D4RT:
    """
    Build complete D4RT model from config.

    Args:
        config: OmegaConf configuration

    Returns:
        model: D4RT model instance
    """
    # Build encoder
    encoder_config = config.encoder
    encoder = build_vit_encoder(encoder_config)

    # Build query encoder
    query_config = config.query_encoder
    query_encoder = QueryEncoder(
        num_frequencies=query_config.fourier.num_frequencies,
        max_frequency=query_config.fourier.max_frequency,
        max_frames=query_config.temporal.max_frames,
        temporal_dim=query_config.temporal.embedding_dim,
        patch_size=query_config.patch_cnn.patch_size,
        patch_channels=tuple(query_config.patch_cnn.channels),
        patch_dim=query_config.patch_cnn.output_dim,
        output_dim=query_config.output_dim,
    )

    # Build decoder
    decoder_config = config.decoder
    decoder = CrossAttentionDecoder(
        query_dim=query_config.output_dim,
        context_dim=encoder.output_dim,
        hidden_dim=decoder_config.hidden_dim,
        num_layers=decoder_config.num_layers,
        num_heads=decoder_config.num_heads,
        mlp_ratio=decoder_config.mlp_ratio,
        dropout=decoder_config.dropout,
        attention_dropout=decoder_config.attention_dropout,
        drop_path_rate=decoder_config.drop_path_rate,
    )

    # Create full model
    model = D4RT(encoder, decoder, query_encoder)

    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in model.

    Args:
        model: PyTorch model

    Returns:
        param_dict: Dictionary with parameter counts
    """
    if isinstance(model, D4RT):
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        query_encoder_params = sum(p.numel() for p in model.query_encoder.parameters())
        total_params = sum(p.numel() for p in model.parameters())

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'query_encoder': query_encoder_params,
            'total': total_params,
        }
    else:
        total_params = sum(p.numel() for p in model.parameters())
        return {'total': total_params}


def print_model_info(model: D4RT):
    """Print model architecture information."""
    param_counts = count_parameters(model)

    print("=" * 60)
    print("D4RT Model Architecture")
    print("=" * 60)
    print(f"Encoder:       {param_counts['encoder']:,} parameters")
    print(f"Query Encoder: {param_counts['query_encoder']:,} parameters")
    print(f"Decoder:       {param_counts['decoder']:,} parameters")
    print("-" * 60)
    print(f"Total:         {param_counts['total']:,} parameters")
    print("=" * 60)
    print(f"\nEncoder: {model.encoder.num_layers} layers, "
          f"{model.encoder.embed_dim} hidden dim")
    print(f"Decoder: {model.decoder.num_layers} layers, "
          f"{model.decoder.hidden_dim} hidden dim")
    print(f"Number of patches: {model.encoder.num_patches}")
    print("=" * 60)
