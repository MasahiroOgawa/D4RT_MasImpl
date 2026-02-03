"""Unit tests for D4RT encoder (Figure 7)."""
import pytest
import torch
from d4rt.models.encoder import SpatioTemporalViT, build_vit_encoder


class TestSpatioTemporalViT:
    """Tests for SpatioTemporalViT encoder."""

    @pytest.fixture
    def small_encoder_config(self):
        """Config for small encoder (faster tests)."""
        return {
            'input_resolution': [8, 128, 128],
            'patch_size': [2, 16, 16],
            'hidden_dim': 256,
            'num_layers': 2,
            'num_heads': 4,
            'use_paper_blocks': True,
            'use_patch_norm': False,
        }

    @pytest.fixture
    def small_video(self, device):
        """Small video for testing."""
        return torch.randn(1, 8, 3, 128, 128, device=device)

    def test_output_shape(self, small_encoder_config, small_video, device):
        """Encoder produces correct output shape."""
        encoder = build_vit_encoder(small_encoder_config).to(device)
        out = encoder(small_video)

        # 8/2 × 128/16 × 128/16 = 4 × 8 × 8 = 256 patches
        expected_patches = 4 * 8 * 8
        assert out.shape == (1, expected_patches, 256), f"Got {out.shape}"

    def test_aspect_ratio_affects_output(self, small_encoder_config, small_video, device):
        """Different aspect ratios should produce different outputs."""
        encoder = build_vit_encoder(small_encoder_config).to(device)
        encoder.eval()

        with torch.no_grad():
            out_wide = encoder(small_video, aspect_ratio=torch.tensor([2.0], device=device))
            out_square = encoder(small_video, aspect_ratio=torch.tensor([1.0], device=device))
            out_tall = encoder(small_video, aspect_ratio=torch.tensor([0.5], device=device))

        # Outputs should differ based on aspect ratio
        assert not torch.allclose(out_wide, out_square, atol=1e-3), \
            "Wide and square outputs should differ"
        assert not torch.allclose(out_square, out_tall, atol=1e-3), \
            "Square and tall outputs should differ"

    def test_default_aspect_ratio(self, small_encoder_config, small_video, device):
        """Encoder works without explicit aspect ratio (defaults to 1.0)."""
        encoder = build_vit_encoder(small_encoder_config).to(device)
        encoder.eval()

        with torch.no_grad():
            out_default = encoder(small_video)  # No aspect ratio
            out_explicit = encoder(small_video, aspect_ratio=torch.tensor([1.0], device=device))

        # Should be the same
        torch.testing.assert_close(out_default, out_explicit)

    def test_paper_blocks_structure(self, small_encoder_config):
        """Each block has BOTH local and global attention (Figure 7)."""
        encoder = build_vit_encoder(small_encoder_config)

        for i, block in enumerate(encoder.blocks):
            assert hasattr(block, 'local_attn'), f"Block {i} missing local_attn"
            assert hasattr(block, 'global_attn'), f"Block {i} missing global_attn"
            assert hasattr(block, 'mlp1'), f"Block {i} missing mlp1"
            assert hasattr(block, 'mlp2'), f"Block {i} missing mlp2"

    def test_legacy_blocks_available(self, device):
        """Global-only mode still works for backward compatibility."""
        config = {
            'input_resolution': [8, 128, 128],
            'patch_size': [2, 16, 16],
            'hidden_dim': 256,
            'num_layers': 2,
            'num_heads': 4,
            'use_paper_blocks': False,  # Legacy mode
            'use_patch_norm': False,
        }
        encoder = build_vit_encoder(config).to(device)
        video = torch.randn(1, 8, 3, 128, 128, device=device)

        out = encoder(video)
        assert out.shape[0] == 1
        assert out.shape[2] == 256

    def test_no_patch_norm_default(self, small_encoder_config):
        """Patch normalization is disabled by default."""
        encoder = build_vit_encoder(small_encoder_config)
        # patch_norm should be Identity when disabled
        assert isinstance(encoder.patch_norm, torch.nn.Identity)

    def test_with_patch_norm(self, device):
        """Patch normalization can be enabled."""
        config = {
            'input_resolution': [8, 128, 128],
            'patch_size': [2, 16, 16],
            'hidden_dim': 256,
            'num_layers': 2,
            'num_heads': 4,
            'use_paper_blocks': True,
            'use_patch_norm': True,  # Enabled
        }
        encoder = build_vit_encoder(config).to(device)
        assert isinstance(encoder.patch_norm, torch.nn.LayerNorm)

        # Should still work
        video = torch.randn(1, 8, 3, 128, 128, device=device)
        out = encoder(video)
        assert out.shape[0] == 1

    def test_gradient_flow(self, small_encoder_config, small_video, device):
        """Gradients flow from output to input."""
        encoder = build_vit_encoder(small_encoder_config).to(device)

        video = small_video.clone().requires_grad_(True)
        out = encoder(video)
        loss = out.sum()
        loss.backward()

        assert video.grad is not None
        assert not torch.isnan(video.grad).any()

    def test_batched_input(self, small_encoder_config, device):
        """Encoder handles batched input."""
        encoder = build_vit_encoder(small_encoder_config).to(device)
        video = torch.randn(4, 8, 3, 128, 128, device=device)
        aspect_ratio = torch.tensor([1.0, 1.5, 0.75, 2.0], device=device)

        out = encoder(video, aspect_ratio=aspect_ratio)
        expected_patches = 4 * 8 * 8
        assert out.shape == (4, expected_patches, 256)

    def test_ar_token_excluded_from_output(self, small_encoder_config, small_video, device):
        """AR token is removed from final output."""
        encoder = build_vit_encoder(small_encoder_config).to(device)

        # Calculate expected number of video patches only
        T, H, W = small_encoder_config['input_resolution']
        t_patch, h_patch, w_patch = small_encoder_config['patch_size']
        expected_video_patches = (T // t_patch) * (H // h_patch) * (W // w_patch)

        out = encoder(small_video)

        # Output should NOT include AR token
        assert out.shape[1] == expected_video_patches, \
            f"Expected {expected_video_patches} patches (no AR token), got {out.shape[1]}"


class TestTokenDiversity:
    """Tests for token diversity (regression test for homogenization)."""

    @pytest.fixture
    def encoder(self, device):
        """Encoder for diversity tests."""
        config = {
            'input_resolution': [8, 128, 128],
            'patch_size': [2, 16, 16],
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 4,
            'use_paper_blocks': True,
            'use_patch_norm': False,
        }
        return build_vit_encoder(config).to(device)

    def test_token_diversity_maintained(self, encoder, device):
        """CRITICAL: Tokens must remain diverse after encoding."""
        video = torch.randn(1, 8, 3, 128, 128, device=device)
        encoder.eval()

        with torch.no_grad():
            features = encoder(video)

        # Compute pairwise cosine similarity
        features_norm = features / features.norm(dim=-1, keepdim=True)
        sim = torch.bmm(features_norm, features_norm.transpose(1, 2))

        # Exclude diagonal
        num_patches = features.shape[1]
        mask = ~torch.eye(num_patches, dtype=bool, device=device)
        mean_sim = sim[0][mask].mean()

        # REGRESSION TEST: Must be < 0.5 (was 0.98 with global-only attention)
        assert mean_sim < 0.5, f"Token homogenization detected: similarity={mean_sim:.3f}"
