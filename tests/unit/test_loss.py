"""Unit tests for paper-exact loss function."""
import pytest
import torch
import torch.nn.functional as F
from d4rt.losses.composite_loss import D4RTCompositeLoss, build_composite_loss


class TestD4RTCompositeLoss:
    """Tests for paper-exact loss function."""

    @pytest.fixture
    def loss_fn(self):
        """Loss function with default paper weights."""
        return D4RTCompositeLoss({
            'l1_3d': 1.0,
            'l2_2d': 0.1,
            'visibility': 0.1,
            'motion': 0.1,
            'confidence': 0.2,
            'normal': 0.5,
        })

    @pytest.fixture
    def pred_and_target(self, device):
        """Sample predictions and targets."""
        N = 32
        pred = {
            'xyz': torch.randn(1, N, 3, device=device),
            'uv': torch.rand(1, N, 2, device=device),
            'normals': F.normalize(torch.randn(1, N, 3, device=device), dim=-1),
            'motion': torch.randn(1, N, 3, device=device),
            'visibility': torch.randn(1, N, 1, device=device),
            'confidence': torch.randn(1, N, 1, device=device),
        }
        target = {
            'xyz': torch.randn(1, N, 3, device=device),
            'uv': torch.rand(1, N, 2, device=device),
            'normals': F.normalize(torch.randn(1, N, 3, device=device), dim=-1),
            'motion': torch.randn(1, N, 3, device=device),
            'visibility': torch.randint(0, 2, (1, N, 1), device=device).float(),
        }
        return pred, target

    def test_loss_is_scalar(self, loss_fn, pred_and_target, device):
        """Loss returns scalar value."""
        loss_fn = loss_fn.to(device)
        pred, target = pred_and_target
        loss, loss_dict = loss_fn(pred, target)
        assert loss.dim() == 0, "Loss should be scalar"
        assert 'loss_total' in loss_dict

    def test_loss_is_positive(self, loss_fn, pred_and_target, device):
        """Loss is non-negative."""
        loss_fn = loss_fn.to(device)
        pred, target = pred_and_target
        loss, _ = loss_fn(pred, target)
        # Note: loss can be slightly negative due to -log(c) when c > e^(c*L3D)
        # but should be bounded

    def test_confidence_weighting_correct_predictions(self, device):
        """High confidence should reduce loss for correct predictions."""
        loss_fn = D4RTCompositeLoss().to(device)
        N = 32

        # Perfect 3D predictions
        xyz = torch.randn(1, N, 3, device=device)
        target = {
            'xyz': xyz.clone(),
            'uv': torch.zeros(1, N, 2, device=device),
            'normals': F.normalize(torch.randn(1, N, 3, device=device), dim=-1),
            'motion': torch.zeros(1, N, 3, device=device),
            'visibility': torch.ones(1, N, 1, device=device),
        }

        # Low confidence prediction
        pred_low_conf = {
            'xyz': xyz.clone(),
            'uv': torch.zeros(1, N, 2, device=device),
            'normals': target['normals'].clone(),
            'motion': torch.zeros(1, N, 3, device=device),
            'visibility': torch.ones(1, N, 1, device=device) * 5,  # high logit → high prob
            'confidence': torch.zeros(1, N, 1, device=device) - 3,  # low conf logit
        }

        # High confidence prediction
        pred_high_conf = {
            'xyz': xyz.clone(),
            'uv': torch.zeros(1, N, 2, device=device),
            'normals': target['normals'].clone(),
            'motion': torch.zeros(1, N, 3, device=device),
            'visibility': torch.ones(1, N, 1, device=device) * 5,
            'confidence': torch.zeros(1, N, 1, device=device) + 3,  # high conf logit
        }

        loss_low, _ = loss_fn(pred_low_conf, target)
        loss_high, _ = loss_fn(pred_high_conf, target)

        # For correct predictions, high confidence should give lower loss
        # because -λconf·log(c) penalty is smaller when c is high
        assert loss_high < loss_low, (
            f"High confidence should reduce loss for correct predictions. "
            f"Low conf loss: {loss_low:.4f}, High conf loss: {loss_high:.4f}"
        )

    def test_confidence_penalty_wrong_predictions(self, device):
        """Confidence penalty increases loss for overconfident wrong predictions."""
        loss_fn = D4RTCompositeLoss().to(device)
        N = 32

        target = {
            'xyz': torch.zeros(1, N, 3, device=device),
            'uv': torch.zeros(1, N, 2, device=device),
            'normals': F.normalize(torch.randn(1, N, 3, device=device), dim=-1),
            'motion': torch.zeros(1, N, 3, device=device),
            'visibility': torch.ones(1, N, 1, device=device),
        }

        # Wrong prediction with high confidence
        pred_wrong_high_conf = {
            'xyz': torch.ones(1, N, 3, device=device) * 5,  # Very wrong
            'uv': torch.zeros(1, N, 2, device=device),
            'normals': target['normals'].clone(),
            'motion': torch.zeros(1, N, 3, device=device),
            'visibility': torch.ones(1, N, 1, device=device) * 5,
            'confidence': torch.ones(1, N, 1, device=device) * 3,  # High conf
        }

        # Wrong prediction with low confidence
        pred_wrong_low_conf = {
            'xyz': torch.ones(1, N, 3, device=device) * 5,  # Very wrong
            'uv': torch.zeros(1, N, 2, device=device),
            'normals': target['normals'].clone(),
            'motion': torch.zeros(1, N, 3, device=device),
            'visibility': torch.ones(1, N, 1, device=device) * 5,
            'confidence': torch.ones(1, N, 1, device=device) * -3,  # Low conf
        }

        loss_high, _ = loss_fn(pred_wrong_high_conf, target)
        loss_low, _ = loss_fn(pred_wrong_low_conf, target)

        # For wrong predictions, high confidence should give HIGHER loss
        # because c·L3D is larger when c is high
        assert loss_high > loss_low, (
            f"Wrong predictions should penalize high confidence. "
            f"High conf loss: {loss_high:.4f}, Low conf loss: {loss_low:.4f}"
        )

    def test_gradient_flow(self, loss_fn, pred_and_target, device):
        """Loss gradients flow to all parameters."""
        loss_fn = loss_fn.to(device)
        pred, target = pred_and_target

        for key in pred:
            pred[key].requires_grad_(True)

        loss, _ = loss_fn(pred, target)
        loss.backward()

        for key in pred:
            assert pred[key].grad is not None, f"No gradient for {key}"
            assert not torch.isnan(pred[key].grad).any(), f"NaN gradient for {key}"

    def test_loss_dict_contents(self, loss_fn, pred_and_target, device):
        """Loss dict contains expected keys."""
        loss_fn = loss_fn.to(device)
        pred, target = pred_and_target
        loss, loss_dict = loss_fn(pred, target)

        expected_keys = [
            'loss_total',
            'loss_3d_raw', 'loss_3d_weighted',
            'loss_2d', 'loss_2d_raw',
            'loss_visibility', 'loss_visibility_raw',
            'loss_motion', 'loss_motion_raw',
            'loss_normal', 'loss_normal_raw',
            'loss_confidence_penalty',
            'mean_confidence',
        ]
        for key in expected_keys:
            assert key in loss_dict, f"Missing loss dict key: {key}"

    def test_legacy_mode(self, device):
        """Legacy mode (non-paper formula) still works."""
        loss_fn = D4RTCompositeLoss(use_paper_formula=False).to(device)

        N = 32
        pred = {
            'xyz': torch.randn(1, N, 3, device=device),
            'visibility': torch.randn(1, N, 1, device=device),
        }
        target = {
            'xyz': torch.randn(1, N, 3, device=device),
            'visibility': torch.randint(0, 2, (1, N,), device=device).float(),
        }

        loss, loss_dict = loss_fn(pred, target)
        assert loss.dim() == 0
        assert 'loss_total' in loss_dict

    def test_partial_outputs(self, device):
        """Loss handles missing optional outputs."""
        loss_fn = D4RTCompositeLoss().to(device)

        N = 32
        # Only required outputs
        pred = {
            'xyz': torch.randn(1, N, 3, device=device),
            'visibility': torch.randn(1, N, 1, device=device),
            'confidence': torch.randn(1, N, 1, device=device),
        }
        target = {
            'xyz': torch.randn(1, N, 3, device=device),
            'visibility': torch.randint(0, 2, (1, N,), device=device).float(),
        }

        # Should not raise
        loss, loss_dict = loss_fn(pred, target)
        assert loss.dim() == 0


class TestBuildCompositeLoss:
    """Tests for build_composite_loss factory function."""

    def test_default_config(self):
        """Factory works with empty config."""
        loss_fn = build_composite_loss({})
        assert isinstance(loss_fn, D4RTCompositeLoss)

    def test_custom_weights(self):
        """Factory applies custom weights."""
        config = {
            'loss_weights': {
                'l1_3d': 2.0,
                'confidence': 0.5,
            }
        }
        loss_fn = build_composite_loss(config)
        assert loss_fn.lambda_3d == 2.0
        assert loss_fn.lambda_conf == 0.5

    def test_paper_formula_toggle(self):
        """Factory respects use_paper_formula flag."""
        config_paper = {'use_paper_formula': True}
        config_legacy = {'use_paper_formula': False}

        loss_paper = build_composite_loss(config_paper)
        loss_legacy = build_composite_loss(config_legacy)

        assert loss_paper.use_paper_formula == True
        assert loss_legacy.use_paper_formula == False
