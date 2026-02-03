"""Integration tests for full D4RT model."""
import pytest
import torch


class TestFullModel:
    """Integration tests for complete D4RT model forward pass."""

    @pytest.fixture
    def small_config(self):
        """Small config for faster tests."""
        from omegaconf import OmegaConf
        return OmegaConf.create({
            'encoder': {
                'input_resolution': [8, 128, 128],
                'patch_size': [2, 16, 16],
                'hidden_dim': 256,
                'num_layers': 2,
                'num_heads': 4,
                'use_paper_blocks': True,
                'use_patch_norm': False,
            },
            'decoder': {
                'query_dim': 256,
                'context_dim': 256,
                'hidden_dim': 256,
                'num_layers': 2,
                'num_heads': 4,
                'mlp_ratio': 4.0,
                'dropout': 0.0,
                'attention_dropout': 0.0,
                'drop_path_rate': 0.0,
                'output_uv': True,
                'output_normals': True,
                'output_motion': True,
            },
            'query_encoder': {
                'fourier': {
                    'num_frequencies': 10,
                    'max_frequency': 9,
                },
                'temporal': {
                    'max_frames': 128,
                    'embedding_dim': 64,
                },
                'patch_cnn': {
                    'patch_size': 9,
                    'channels': [32, 64],
                    'output_dim': 64,
                },
                'output_dim': 256,
            },
        })

    @pytest.fixture
    def model(self, small_config, device):
        """Build model for tests."""
        from d4rt.models import build_d4rt_model
        return build_d4rt_model(small_config).to(device)

    @pytest.fixture
    def sample_data(self, device):
        """Sample video and queries."""
        B, T, C, H, W = 1, 8, 3, 128, 128
        N = 16

        video = torch.randn(B, T, C, H, W, device=device)
        queries = {
            'u': torch.rand(B, N, device=device),
            'v': torch.rand(B, N, device=device),
            't_src': torch.randint(0, T, (B, N), device=device),
            't_tgt': torch.randint(0, T, (B, N), device=device),
            't_cam': torch.randint(0, T, (B, N), device=device),
        }
        return video, queries

    def test_forward_pass_produces_all_outputs(self, model, sample_data):
        """Full forward pass produces all outputs."""
        video, queries = sample_data
        model.eval()

        with torch.no_grad():
            outputs = model(video, queries)

        # Check all required outputs present
        required = ['xyz', 'visibility', 'confidence']
        for key in required:
            assert key in outputs, f"Missing output: {key}"

        # Check optional outputs (enabled in config)
        optional = ['uv', 'normals', 'motion']
        for key in optional:
            assert key in outputs, f"Missing optional output: {key}"

        # Check encoder features
        assert 'encoder_features' in outputs

    def test_output_shapes(self, model, sample_data):
        """Output shapes are correct."""
        video, queries = sample_data
        B, N = 1, 16

        outputs = model(video, queries)

        assert outputs['xyz'].shape == (B, N, 3)
        assert outputs['uv'].shape == (B, N, 2)
        assert outputs['normals'].shape == (B, N, 3)
        assert outputs['motion'].shape == (B, N, 3)
        assert outputs['visibility'].shape == (B, N, 1)
        assert outputs['confidence'].shape == (B, N, 1)

    def test_gradient_flow_encoder_to_output(self, model, sample_data):
        """Gradients flow from loss to encoder."""
        video, queries = sample_data

        outputs = model(video, queries)

        # Simple loss
        loss = outputs['xyz'].sum()
        loss.backward()

        # Check encoder has gradients
        encoder_param = next(model.encoder.parameters())
        assert encoder_param.grad is not None
        assert not torch.isnan(encoder_param.grad).any()

    def test_gradient_flow_decoder_to_output(self, model, sample_data):
        """Gradients flow through decoder."""
        video, queries = sample_data

        outputs = model(video, queries)
        loss = outputs['xyz'].sum() + outputs['visibility'].sum()
        loss.backward()

        # Check decoder has gradients
        decoder_param = next(model.decoder.parameters())
        assert decoder_param.grad is not None
        assert not torch.isnan(decoder_param.grad).any()

    def test_encode_video_method(self, model, sample_data):
        """encode_video method works correctly."""
        video, _ = sample_data

        features = model.encode_video(video)

        # Expected patches: (8/2) * (128/16) * (128/16) = 4 * 8 * 8 = 256
        expected_patches = 4 * 8 * 8
        assert features.shape == (1, expected_patches, 256)

    def test_predict_from_queries_method(self, model, sample_data, device):
        """predict_from_queries works with pre-computed features."""
        video, queries = sample_data

        # Encode video
        with torch.no_grad():
            features = model.encode_video(video)

        # Predict from queries
        outputs = model.predict_from_queries(features, queries, video)

        assert 'xyz' in outputs
        assert outputs['xyz'].shape == (1, 16, 3)

    def test_batched_inference(self, model, device):
        """Model handles batched input."""
        B, T, C, H, W = 2, 8, 3, 128, 128
        N = 16

        video = torch.randn(B, T, C, H, W, device=device)
        queries = {
            'u': torch.rand(B, N, device=device),
            'v': torch.rand(B, N, device=device),
            't_src': torch.randint(0, T, (B, N), device=device),
            't_tgt': torch.randint(0, T, (B, N), device=device),
            't_cam': torch.randint(0, T, (B, N), device=device),
        }

        outputs = model(video, queries)
        assert outputs['xyz'].shape == (B, N, 3)

    def test_eval_mode(self, model, sample_data):
        """Model produces consistent output in eval mode."""
        video, queries = sample_data
        model.eval()

        with torch.no_grad():
            out1 = model(video, queries)
            out2 = model(video, queries)

        torch.testing.assert_close(out1['xyz'], out2['xyz'])
