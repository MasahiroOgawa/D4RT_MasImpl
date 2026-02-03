"""Unit tests for decoder output heads."""
import pytest
import torch
import torch.nn.functional as F
from d4rt.models.decoder import CrossAttentionDecoder, build_decoder


class TestCrossAttentionDecoder:
    """Tests for CrossAttentionDecoder with all output heads."""

    @pytest.fixture
    def decoder_config(self):
        """Config for decoder."""
        return {
            'query_dim': 256,
            'context_dim': 256,
            'hidden_dim': 256,
            'num_layers': 2,
            'num_heads': 4,
            'output_uv': True,
            'output_normals': True,
            'output_motion': True,
        }

    @pytest.fixture
    def decoder(self, decoder_config, device):
        """Decoder instance."""
        return build_decoder(decoder_config).to(device)

    @pytest.fixture
    def sample_inputs(self, device):
        """Sample inputs for decoder."""
        B, N = 2, 32
        queries = torch.randn(B, N, 256, device=device)
        context = torch.randn(B, 128, 256, device=device)
        return queries, context

    def test_all_output_heads_present(self, decoder, sample_inputs):
        """All paper output heads are present."""
        queries, context = sample_inputs
        outputs = decoder(queries, context)

        required_outputs = ['xyz', 'visibility', 'confidence']
        for key in required_outputs:
            assert key in outputs, f"Missing required output: {key}"

        # Optional but enabled by config
        assert 'uv' in outputs, "Missing uv output"
        assert 'normals' in outputs, "Missing normals output"
        assert 'motion' in outputs, "Missing motion output"

    def test_output_shapes(self, decoder, sample_inputs):
        """Output shapes match paper spec."""
        queries, context = sample_inputs
        B, N = queries.shape[:2]
        outputs = decoder(queries, context)

        assert outputs['xyz'].shape == (B, N, 3), f"xyz: expected {(B, N, 3)}, got {outputs['xyz'].shape}"
        assert outputs['uv'].shape == (B, N, 2), f"uv: expected {(B, N, 2)}, got {outputs['uv'].shape}"
        assert outputs['normals'].shape == (B, N, 3), f"normals: expected {(B, N, 3)}, got {outputs['normals'].shape}"
        assert outputs['motion'].shape == (B, N, 3), f"motion: expected {(B, N, 3)}, got {outputs['motion'].shape}"
        assert outputs['visibility'].shape == (B, N, 1), f"visibility: expected {(B, N, 1)}, got {outputs['visibility'].shape}"
        assert outputs['confidence'].shape == (B, N, 1), f"confidence: expected {(B, N, 1)}, got {outputs['confidence'].shape}"

    def test_normals_unit_length(self, decoder, sample_inputs):
        """Surface normals are unit normalized."""
        queries, context = sample_inputs
        outputs = decoder(queries, context)

        norms = outputs['normals'].norm(dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5, rtol=1e-5,
            msg="Normals should be unit vectors"
        )

    def test_uv_in_valid_range(self, decoder, sample_inputs):
        """UV coordinates are in [0, 1] range."""
        queries, context = sample_inputs
        outputs = decoder(queries, context)

        assert outputs['uv'].min() >= 0, "UV coordinates should be >= 0"
        assert outputs['uv'].max() <= 1, "UV coordinates should be <= 1"

    def test_gradient_flow(self, decoder, sample_inputs):
        """Gradients flow to all output heads."""
        queries, context = sample_inputs
        queries.requires_grad_(True)
        context.requires_grad_(True)

        outputs = decoder(queries, context)

        # Compute loss from all outputs
        loss = (
            outputs['xyz'].sum() +
            outputs['uv'].sum() +
            outputs['normals'].sum() +
            outputs['motion'].sum() +
            outputs['visibility'].sum() +
            outputs['confidence'].sum()
        )
        loss.backward()

        assert queries.grad is not None, "No gradient for queries"
        assert context.grad is not None, "No gradient for context"
        assert not torch.isnan(queries.grad).any(), "NaN in queries gradient"
        assert not torch.isnan(context.grad).any(), "NaN in context gradient"

    def test_optional_outputs_disabled(self, device):
        """Optional outputs can be disabled."""
        config = {
            'query_dim': 256,
            'context_dim': 256,
            'hidden_dim': 256,
            'num_layers': 2,
            'num_heads': 4,
            'output_uv': False,
            'output_normals': False,
            'output_motion': False,
        }
        decoder = build_decoder(config).to(device)

        queries = torch.randn(1, 32, 256, device=device)
        context = torch.randn(1, 128, 256, device=device)
        outputs = decoder(queries, context)

        # Required outputs still present
        assert 'xyz' in outputs
        assert 'visibility' in outputs
        assert 'confidence' in outputs

        # Optional outputs not present
        assert 'uv' not in outputs
        assert 'normals' not in outputs
        assert 'motion' not in outputs

    def test_context_pooling(self, device):
        """Context pooling reduces token count."""
        config = {
            'query_dim': 256,
            'context_dim': 256,
            'hidden_dim': 256,
            'num_layers': 2,
            'num_heads': 4,
            'context_pool_tokens': 64,
            'context_input_tokens': 256,
        }
        decoder = build_decoder(config).to(device)

        assert decoder.context_pool is not None
        assert decoder.context_pool_tokens == 64

        queries = torch.randn(1, 32, 256, device=device)
        context = torch.randn(1, 256, 256, device=device)
        outputs = decoder(queries, context)

        # Should still work
        assert outputs['xyz'].shape == (1, 32, 3)
