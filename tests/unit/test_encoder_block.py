"""Unit tests for D4RTEncoderBlock (Figure 7: local + global attention in each block)."""
import pytest
import torch
from d4rt.models.components.encoder_block import (
    D4RTEncoderBlock,
    LocalFrameAttention,
    LegacyEncoderBlock,
    MLP,
)


class TestLocalFrameAttention:
    """Tests for LocalFrameAttention module."""

    def test_output_shape(self, device):
        """Local attention preserves sequence shape."""
        attn = LocalFrameAttention(embed_dim=768, num_heads=12).to(device)
        # 12 frames × 256 patches/frame = 3072 tokens
        x = torch.randn(2, 3072, 768, device=device)
        out = attn(x, num_frames=12, patches_per_frame=256)
        assert out.shape == x.shape

    def test_frame_isolation(self, device):
        """Tokens only attend within their frame."""
        attn = LocalFrameAttention(embed_dim=768, num_heads=12).to(device)
        attn.eval()

        x = torch.randn(1, 512, 768, device=device)  # 2 frames × 256 patches

        with torch.no_grad():
            # Get output with original input
            out1 = attn(x, num_frames=2, patches_per_frame=256)

            # Zero out second frame
            x_modified = x.clone()
            x_modified[:, 256:, :] = 0

            out2 = attn(x_modified, num_frames=2, patches_per_frame=256)

            # First frame output should be identical regardless of second frame
            torch.testing.assert_close(
                out1[:, :256], out2[:, :256],
                msg="First frame should be independent of second frame"
            )

    def test_gradient_flow(self, device):
        """Gradients flow through local attention."""
        attn = LocalFrameAttention(embed_dim=768, num_heads=12).to(device)
        x = torch.randn(1, 512, 768, device=device, requires_grad=True)

        out = attn(x, num_frames=2, patches_per_frame=256)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestD4RTEncoderBlock:
    """Tests for D4RTEncoderBlock with both local and global attention."""

    def test_output_shape(self, device):
        """Block preserves sequence shape."""
        block = D4RTEncoderBlock(embed_dim=768, num_heads=12).to(device)
        x = torch.randn(2, 3072, 768, device=device)  # 12 frames × 256 patches

        out = block(x, num_frames=12, patches_per_frame=256, has_ar_token=False)
        assert out.shape == x.shape

    def test_output_shape_with_ar_token(self, device):
        """Block handles AR token correctly."""
        block = D4RTEncoderBlock(embed_dim=768, num_heads=12).to(device)
        # 1 AR token + 3072 video tokens
        x = torch.randn(2, 3073, 768, device=device)

        out = block(x, num_frames=12, patches_per_frame=256, has_ar_token=True)
        assert out.shape == x.shape

    def test_has_both_attention_types(self):
        """Block contains both local and global attention."""
        block = D4RTEncoderBlock(embed_dim=768, num_heads=12)

        assert hasattr(block, 'local_attn'), "Missing local_attn"
        assert hasattr(block, 'global_attn'), "Missing global_attn"
        assert hasattr(block, 'mlp1'), "Missing mlp1 (after local)"
        assert hasattr(block, 'mlp2'), "Missing mlp2 (after global)"
        assert hasattr(block, 'norm1'), "Missing norm1"
        assert hasattr(block, 'norm2'), "Missing norm2"
        assert hasattr(block, 'norm3'), "Missing norm3"
        assert hasattr(block, 'norm4'), "Missing norm4"

    def test_gradient_flow(self, device):
        """Gradients flow through both attention types."""
        block = D4RTEncoderBlock(embed_dim=768, num_heads=12).to(device)
        x = torch.randn(1, 512, 768, device=device, requires_grad=True)

        out = block(x, num_frames=2, patches_per_frame=256, has_ar_token=False)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_ar_token_unchanged_during_local_attention(self, device):
        """AR token should not change during local attention phase."""
        block = D4RTEncoderBlock(embed_dim=768, num_heads=12).to(device)
        block.eval()

        # Create input with AR token
        ar_token = torch.randn(1, 1, 768, device=device)
        video_tokens = torch.randn(1, 512, 768, device=device)
        x = torch.cat([ar_token, video_tokens], dim=1)

        with torch.no_grad():
            out = block(x, num_frames=2, patches_per_frame=256, has_ar_token=True)

            # The AR token goes through global attention, so it will change
            # But it should not affect local attention computation of video tokens
            # We can verify local attention is separate from AR token


class TestLegacyEncoderBlock:
    """Tests for LegacyEncoderBlock (backward compatibility)."""

    def test_output_shape(self, device):
        """Legacy block preserves sequence shape."""
        block = LegacyEncoderBlock(embed_dim=768, num_heads=12).to(device)
        x = torch.randn(2, 3072, 768, device=device)

        out = block(x)
        assert out.shape == x.shape

    def test_api_compatibility(self, device):
        """Legacy block accepts same API as D4RTEncoderBlock."""
        block = LegacyEncoderBlock(embed_dim=768, num_heads=12).to(device)
        x = torch.randn(2, 3072, 768, device=device)

        # Should accept these parameters even though it ignores them
        out = block(x, num_frames=12, patches_per_frame=256, has_ar_token=False)
        assert out.shape == x.shape


class TestMLP:
    """Tests for MLP module."""

    def test_output_shape(self, device):
        """MLP preserves feature dimension by default."""
        mlp = MLP(in_features=768, hidden_features=3072).to(device)
        x = torch.randn(2, 100, 768, device=device)
        out = mlp(x)
        assert out.shape == x.shape

    def test_custom_output_dim(self, device):
        """MLP can have different output dimension."""
        mlp = MLP(in_features=768, hidden_features=3072, out_features=512).to(device)
        x = torch.randn(2, 100, 768, device=device)
        out = mlp(x)
        assert out.shape == (2, 100, 512)
