"""Unit tests for AspectRatioToken (Paper p.3, Figure 7: W/H → FC → separate token)."""
import pytest
import torch
from d4rt.models.components.aspect_ratio_token import AspectRatioToken


class TestAspectRatioToken:
    """Tests for AspectRatioToken module."""

    def test_output_shape_single_batch(self, device):
        """FC layer produces single token with correct shape."""
        ar_token = AspectRatioToken(embed_dim=768).to(device)

        ar = torch.tensor([1.5], device=device)  # W/H = 1.5
        out = ar_token(ar)
        assert out.shape == (1, 1, 768), f"Expected (1, 1, 768), got {out.shape}"

    def test_output_shape_batch(self, device):
        """Handles batched input correctly."""
        ar_token = AspectRatioToken(embed_dim=768).to(device)

        ar_batch = torch.tensor([1.0, 1.5, 0.75, 2.0], device=device)
        out_batch = ar_token(ar_batch)
        assert out_batch.shape == (4, 1, 768), f"Expected (4, 1, 768), got {out_batch.shape}"

    def test_different_ratios_different_tokens(self, device):
        """Different aspect ratios should produce different tokens."""
        ar_token = AspectRatioToken(embed_dim=768).to(device)

        ar_wide = torch.tensor([2.0], device=device)    # Wide video
        ar_square = torch.tensor([1.0], device=device)  # Square
        ar_tall = torch.tensor([0.5], device=device)    # Tall video

        token_wide = ar_token(ar_wide)
        token_square = ar_token(ar_square)
        token_tall = ar_token(ar_tall)

        # All tokens should be different
        assert not torch.allclose(token_wide, token_square, atol=1e-6), \
            "Wide and square tokens should differ"
        assert not torch.allclose(token_square, token_tall, atol=1e-6), \
            "Square and tall tokens should differ"
        assert not torch.allclose(token_wide, token_tall, atol=1e-6), \
            "Wide and tall tokens should differ"

    def test_gradient_flow(self, device):
        """Gradients flow through FC layer."""
        ar_token = AspectRatioToken(embed_dim=768).to(device)

        ar = torch.tensor([1.5], device=device, requires_grad=True)
        out = ar_token(ar)
        loss = out.sum()
        loss.backward()

        assert ar.grad is not None, "No gradient for input"
        assert not torch.isnan(ar.grad).any(), "NaN in gradient"

    def test_accepts_2d_input(self, device):
        """Handles [B, 1] input shape."""
        ar_token = AspectRatioToken(embed_dim=768).to(device)

        ar_2d = torch.tensor([[1.5], [2.0]], device=device)
        out = ar_token(ar_2d)
        assert out.shape == (2, 1, 768)

    def test_concatenation_with_video_tokens(self, device):
        """AR token can be concatenated with video tokens."""
        ar_token = AspectRatioToken(embed_dim=768).to(device)

        ar = torch.tensor([1.5], device=device)
        video_tokens = torch.randn(1, 3072, 768, device=device)

        ar_tok = ar_token(ar)  # [1, 1, 768]
        combined = torch.cat([ar_tok, video_tokens], dim=1)  # [1, 3073, 768]

        assert combined.shape == (1, 3073, 768)

    def test_different_embed_dims(self, device):
        """Works with different embedding dimensions."""
        for embed_dim in [256, 512, 768, 1024]:
            ar_token = AspectRatioToken(embed_dim=embed_dim).to(device)
            ar = torch.tensor([1.0], device=device)
            out = ar_token(ar)
            assert out.shape == (1, 1, embed_dim)
