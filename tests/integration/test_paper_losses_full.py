"""Full integration tests for all paper losses working together."""

import torch
import pytest
from pathlib import Path
from d4rt.losses.composite_loss import CompositeLoss
from d4rt.losses.confidence import compute_prediction_error
from d4rt.models.decoder import CrossAttentionDecoder


class TestFullPaperLosses:
    """Test all paper losses integrated together."""

    def test_all_loss_components_active(self):
        """Test that all loss components are computed and weighted correctly."""
        # Create loss with paper weights
        loss_weights = {
            'l1_3d': 1.0,
            'l2_2d': 0.1,
            'normal': 0.5,
            'motion': 0.1,
            'visibility': 0.1,
            'confidence': 0.2,
            'use_paper_formula_3d': True,
        }
        loss_fn = CompositeLoss(loss_weights=loss_weights)

        # Create realistic predictions
        batch_size = 2
        num_points = 64

        predictions = {
            'xyz': torch.randn(batch_size, num_points, 3) * 2 + 10,
            'visibility': torch.sigmoid(torch.randn(batch_size, num_points, 1)),
            'confidence': torch.sigmoid(torch.randn(batch_size, num_points, 1)),
        }

        targets = {
            'xyz': predictions['xyz'] + torch.randn_like(predictions['xyz']) * 0.5,
            'uv': torch.rand(batch_size, num_points, 2) * 256,
            'visibility': torch.rand(batch_size, num_points),
            'normals': torch.randn(batch_size, num_points, 3),
        }

        cameras = {
            'intrinsics': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, 24, 1, 1),
            'extrinsics': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, 24, 1, 1),
        }

        queries = {
            't_cam': torch.randint(0, 24, (batch_size, num_points)),
        }

        # Compute loss
        total_loss, loss_dict = loss_fn(predictions, targets, cameras, queries)

        # Verify all components are present
        assert 'loss_3d' in loss_dict
        assert 'loss_2d' in loss_dict
        assert 'loss_visibility' in loss_dict
        assert 'loss_normal' in loss_dict
        assert 'loss_motion' in loss_dict
        assert 'loss_confidence' in loss_dict
        assert 'loss_total' in loss_dict

        # Verify loss is finite and positive
        assert torch.isfinite(total_loss)
        assert total_loss > 0

        # Verify weights are correct
        assert loss_fn.loss_weights['l1_3d'] == 1.0
        assert loss_fn.loss_weights['normal'] == 0.5  # Paper value!
        assert loss_fn.loss_weights['confidence'] == 0.2

    def test_paper_3d_loss_active(self):
        """Test that paper's 3D loss formula is actually used."""
        loss_weights = {'use_paper_formula_3d': True}
        loss_fn = CompositeLoss(loss_weights=loss_weights)

        # Verify paper formula is enabled
        assert loss_fn.l1_3d_loss.use_paper_formula is True

        # Test with actual data
        pred = torch.randn(2, 32, 3) * 2 + 10
        gt = pred + torch.randn_like(pred) * 0.1

        predictions = {'xyz': pred, 'visibility': torch.ones(2, 32, 1)}
        targets = {'xyz': gt, 'visibility': torch.ones(2, 32)}
        cameras = {
            'intrinsics': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(2, 24, 1, 1),
            'extrinsics': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(2, 24, 1, 1),
        }
        queries = {'t_cam': torch.zeros(2, 32).long()}

        loss, loss_dict = loss_fn(predictions, targets, cameras, queries)

        # Loss should be reasonable with paper formula
        assert torch.isfinite(loss)
        assert loss > 0
        assert loss < 10.0

    def test_confidence_loss_integration(self):
        """Test that confidence loss works with other losses."""
        loss_fn = CompositeLoss()

        batch_size = 4
        num_points = 32

        # Predictions with confidence
        predictions = {
            'xyz': torch.randn(batch_size, num_points, 3) * 2 + 10,
            'visibility': torch.sigmoid(torch.randn(batch_size, num_points, 1)),
            'confidence': torch.sigmoid(torch.randn(batch_size, num_points, 1)),
        }

        targets = {
            'xyz': predictions['xyz'] + torch.randn_like(predictions['xyz']) * 0.3,
            'visibility': torch.rand(batch_size, num_points),
        }

        cameras = {
            'intrinsics': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, 24, 1, 1),
            'extrinsics': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, 24, 1, 1),
        }

        queries = {'t_cam': torch.zeros(batch_size, num_points).long()}

        loss, loss_dict = loss_fn(predictions, targets, cameras, queries)

        # Confidence loss should be active
        assert loss_dict['loss_confidence'] > 0
        assert torch.isfinite(loss)

    def test_gradient_flow_all_losses(self):
        """Test that gradients flow through all loss components."""
        loss_fn = CompositeLoss()

        batch_size = 2
        num_points = 16

        # Create predictions that require gradients
        xyz_pred = torch.randn(batch_size, num_points, 3, requires_grad=True)
        vis_pred = torch.randn(batch_size, num_points, 1, requires_grad=True)
        conf_pred = torch.randn(batch_size, num_points, 1, requires_grad=True)

        predictions = {
            'xyz': xyz_pred * 2 + 10,
            'visibility': torch.sigmoid(vis_pred),
            'confidence': torch.sigmoid(conf_pred),
        }

        targets = {
            'xyz': torch.randn(batch_size, num_points, 3) * 2 + 10,
            'visibility': torch.rand(batch_size, num_points),
        }

        cameras = {
            'intrinsics': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, 24, 1, 1),
            'extrinsics': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, 24, 1, 1),
        }

        queries = {'t_cam': torch.zeros(batch_size, num_points).long()}

        # Forward + backward
        loss, _ = loss_fn(predictions, targets, cameras, queries)
        loss.backward()

        # Check gradients exist and are finite
        assert xyz_pred.grad is not None
        assert vis_pred.grad is not None
        assert conf_pred.grad is not None

        assert torch.all(torch.isfinite(xyz_pred.grad))
        assert torch.all(torch.isfinite(vis_pred.grad))
        assert torch.all(torch.isfinite(conf_pred.grad))

    def test_decoder_with_confidence(self):
        """Test that decoder outputs confidence correctly."""
        decoder = CrossAttentionDecoder(
            query_dim=256,
            context_dim=768,
            hidden_dim=256,
            num_layers=2,
        )

        batch_size = 2
        num_queries = 32
        num_context = 196

        queries = torch.randn(batch_size, num_queries, 256)
        context = torch.randn(batch_size, num_context, 768)

        outputs = decoder(queries, context)

        # Check all outputs
        assert 'xyz' in outputs
        assert 'visibility' in outputs
        assert 'confidence' in outputs

        # Check shapes
        assert outputs['xyz'].shape == (batch_size, num_queries, 3)
        assert outputs['visibility'].shape == (batch_size, num_queries, 1)
        assert outputs['confidence'].shape == (batch_size, num_queries, 1)

        # Check confidence range [0, 1]
        assert torch.all(outputs['confidence'] >= 0)
        assert torch.all(outputs['confidence'] <= 1)


class TestLossRatios:
    """Test that loss component ratios are reasonable."""

    def test_loss_component_magnitudes(self):
        """Test that loss components have expected relative magnitudes."""
        loss_fn = CompositeLoss()

        batch_size = 4
        num_points = 64

        # Create data with moderate errors
        predictions = {
            'xyz': torch.randn(batch_size, num_points, 3) * 2 + 10,
            'visibility': torch.sigmoid(torch.randn(batch_size, num_points, 1)),
            'confidence': torch.ones(batch_size, num_points, 1) * 0.7,
        }

        targets = {
            'xyz': predictions['xyz'] + torch.randn_like(predictions['xyz']) * 0.5,
            'visibility': torch.rand(batch_size, num_points),
            'normals': torch.randn(batch_size, num_points, 3),
        }

        cameras = {
            'intrinsics': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, 24, 1, 1),
            'extrinsics': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, 24, 1, 1),
        }

        queries = {'t_cam': torch.zeros(batch_size, num_points).long()}

        _, loss_dict = loss_fn(predictions, targets, cameras, queries)

        # 3D loss should typically be largest (weight=1.0)
        assert loss_dict['loss_3d'] > 0

        # Normal loss should be significant (weight=0.5)
        if loss_dict['loss_normal'] > 0:
            # Normal loss should contribute meaningfully
            assert loss_dict['loss_normal'] > 0.001

        # All losses should be finite
        for key, value in loss_dict.items():
            if isinstance(value, float):
                assert value >= 0 or key == 'loss_total'  # Total can be zero if all disabled
                assert not (value != value)  # Check not NaN


class TestPaperConfig:
    """Test the paper config file."""

    def test_paper_config_loads(self):
        """Test that paper config can be loaded."""
        import yaml

        config_path = Path("configs/training/train_5k_movi_paper.yaml")
        assert config_path.exists(), "Paper config should exist"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check all paper weights are present
        assert 'loss_weights' in config
        weights = config['loss_weights']

        assert weights['l1_3d'] == 1.0
        assert weights['l2_2d'] == 0.1
        assert weights['normal'] == 0.5  # Paper value!
        assert weights['motion'] == 0.1
        assert weights['visibility'] == 0.1
        assert weights['confidence'] == 0.2  # New!
        assert weights['use_paper_formula_3d'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
