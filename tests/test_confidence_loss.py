"""Unit tests for confidence loss."""

import torch
import pytest
from d4rt.losses.confidence import (
    ConfidenceLoss,
    SeparateConfidenceLoss,
    compute_prediction_error,
)


class TestConfidenceLoss:
    """Tests for basic confidence loss."""

    def test_high_confidence_high_error(self):
        """Test that high confidence + high error results in high loss."""
        loss_fn = ConfidenceLoss()

        confidence = torch.tensor([[[0.9]]])  # High confidence
        error = torch.tensor([[[10.0]]])  # High error

        loss = loss_fn(confidence, error)

        # Loss should be substantial (confident but wrong)
        assert loss > 1.0

    def test_low_confidence_high_error(self):
        """Test that low confidence + high error results in moderate loss."""
        loss_fn = ConfidenceLoss()

        confidence = torch.tensor([[[0.1]]])  # Low confidence
        error = torch.tensor([[[10.0]]])  # High error

        loss = loss_fn(confidence, error)

        # Loss should be moderate (low confidence penalty dominates)
        assert loss > 0.5

    def test_high_confidence_low_error(self):
        """Test that high confidence + low error results in low loss."""
        loss_fn = ConfidenceLoss()

        confidence = torch.tensor([[[0.9]]])  # High confidence
        error = torch.tensor([[[0.1]]])  # Low error

        loss = loss_fn(confidence, error)

        # Loss should be small (correct and confident)
        assert loss < 1.0

    def test_confidence_penalty_behavior(self):
        """Test that penalty term encourages high confidence when error is low."""
        loss_fn = ConfidenceLoss()

        # Same low error, different confidence
        error = torch.tensor([[[0.1]]])

        conf_high = torch.tensor([[[0.9]]])
        conf_low = torch.tensor([[[0.2]]])

        loss_high_conf = loss_fn(conf_high, error)
        loss_low_conf = loss_fn(conf_low, error)

        # Higher confidence should result in lower loss (when error is low)
        assert loss_high_conf < loss_low_conf

    def test_batch_processing(self):
        """Test that loss works with batches."""
        loss_fn = ConfidenceLoss()

        batch_size = 4
        num_points = 32

        confidence = torch.rand(batch_size, num_points, 1) * 0.5 + 0.3  # Range [0.3, 0.8]
        error = torch.rand(batch_size, num_points, 1) * 2.0  # Range [0, 2]

        loss = loss_fn(confidence, error)

        assert torch.isfinite(loss)
        assert loss > 0

    def test_error_shape_broadcasting(self):
        """Test that loss handles different error shapes."""
        loss_fn = ConfidenceLoss()

        confidence = torch.tensor([[[0.5], [0.7]]])  # [B=1, N=2, 1]

        # Test with [B, N] shape
        error_2d = torch.tensor([[1.0, 2.0]])  # [B=1, N=2]
        loss_2d = loss_fn(confidence, error_2d)

        # Test with [B, N, 1] shape
        error_3d = torch.tensor([[[1.0], [2.0]]])  # [B=1, N=2, 1]
        loss_3d = loss_fn(confidence, error_3d)

        # Should be equal (broadcasting works)
        assert torch.allclose(loss_2d, loss_3d, atol=1e-6)

    def test_no_nan_or_inf(self):
        """Test that loss doesn't produce nan or inf."""
        loss_fn = ConfidenceLoss()

        # Edge case: very high confidence, very high error
        confidence = torch.tensor([[[0.99]]])
        error = torch.tensor([[[100.0]]])

        loss = loss_fn(confidence, error)
        assert torch.isfinite(loss)

        # Edge case: very low confidence (but not zero due to epsilon)
        confidence = torch.tensor([[[0.01]]])
        error = torch.tensor([[[1.0]]])

        loss = loss_fn(confidence, error)
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        """Test that gradients flow through confidence loss."""
        loss_fn = ConfidenceLoss()

        confidence = torch.rand(2, 16, 1, requires_grad=True)
        error = torch.rand(2, 16, 1)

        loss = loss_fn(confidence, error)
        loss.backward()

        assert confidence.grad is not None
        assert torch.all(torch.isfinite(confidence.grad))
        assert torch.any(confidence.grad != 0)


class TestSeparateConfidenceLoss:
    """Tests for separate confidence loss with explicit penalty weight."""

    def test_separate_components(self):
        """Test that separate loss returns components."""
        loss_fn = SeparateConfidenceLoss(penalty_weight=0.5)

        confidence = torch.tensor([[[0.8], [0.6]]])
        error = torch.tensor([[[1.0], [2.0]]])

        loss, components = loss_fn(confidence, error)

        assert 'weighted_error' in components
        assert 'confidence_penalty' in components
        assert torch.isfinite(loss)

    def test_penalty_weight_effect(self):
        """Test that penalty weight controls trade-off."""
        confidence = torch.tensor([[[0.5]]])
        error = torch.tensor([[[1.0]]])

        # High penalty weight
        loss_fn_high = SeparateConfidenceLoss(penalty_weight=2.0)
        loss_high, _ = loss_fn_high(confidence, error)

        # Low penalty weight
        loss_fn_low = SeparateConfidenceLoss(penalty_weight=0.1)
        loss_low, _ = loss_fn_low(confidence, error)

        # Higher penalty weight → higher total loss (if confidence is low)
        assert loss_high > loss_low

    def test_components_sum_to_total(self):
        """Test that weighted components sum to total loss."""
        penalty_weight = 0.7
        loss_fn = SeparateConfidenceLoss(penalty_weight=penalty_weight)

        confidence = torch.tensor([[[0.6], [0.8]]])
        error = torch.tensor([[[1.5], [0.5]]])

        loss, components = loss_fn(confidence, error)

        # Manually compute expected loss
        expected_loss = components['weighted_error'] + penalty_weight * components['confidence_penalty']

        assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-5)


class TestComputePredictionError:
    """Tests for prediction error computation."""

    def test_l1_error(self):
        """Test L1 (Manhattan) distance."""
        pred = torch.tensor([[[1.0, 2.0, 3.0]]])
        gt = torch.tensor([[[2.0, 3.0, 4.0]]])

        error = compute_prediction_error(pred, gt, error_type='l1')

        # L1 distance = |1-2| + |2-3| + |3-4| = 3.0
        expected = torch.tensor([[3.0]])
        assert torch.allclose(error, expected, atol=1e-6)

    def test_l2_error(self):
        """Test L2 (Euclidean) distance."""
        pred = torch.tensor([[[0.0, 0.0, 0.0]]])
        gt = torch.tensor([[[3.0, 4.0, 0.0]]])

        error = compute_prediction_error(pred, gt, error_type='l2')

        # L2 distance = sqrt(9 + 16) = 5.0
        expected = torch.tensor([[5.0]])
        assert torch.allclose(error, expected, atol=1e-6)

    def test_huber_error(self):
        """Test Huber loss (robust to outliers)."""
        pred = torch.tensor([[[0.0, 0.0, 0.0]]])
        gt = torch.tensor([[[0.5, 0.5, 0.5]]])

        error = compute_prediction_error(pred, gt, error_type='huber')

        # Small errors: use quadratic
        assert torch.all(torch.isfinite(error))
        assert error > 0

    def test_batch_error_computation(self):
        """Test error computation with batches."""
        batch_size = 4
        num_points = 16

        pred = torch.randn(batch_size, num_points, 3)
        gt = pred + torch.randn_like(pred) * 0.1

        error_l1 = compute_prediction_error(pred, gt, error_type='l1')
        error_l2 = compute_prediction_error(pred, gt, error_type='l2')

        assert error_l1.shape == (batch_size, num_points)
        assert error_l2.shape == (batch_size, num_points)
        assert torch.all(error_l1 > 0)
        assert torch.all(error_l2 > 0)

    def test_zero_error(self):
        """Test that identical predictions give zero error."""
        pred = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        gt = pred.clone()

        for error_type in ['l1', 'l2', 'huber']:
            error = compute_prediction_error(pred, gt, error_type=error_type)
            assert torch.allclose(error, torch.zeros_like(error), atol=1e-6)


class TestConfidenceLossIntegration:
    """Integration tests with realistic scenarios."""

    def test_realistic_confidence_weighting(self):
        """Test that confidence loss behaves correctly with realistic data."""
        loss_fn = ConfidenceLoss()

        batch_size = 2
        num_points = 32

        # Simulate realistic predictions
        pred_xyz = torch.randn(batch_size, num_points, 3)
        gt_xyz = pred_xyz + torch.randn_like(pred_xyz) * 0.5

        # Compute errors
        error = compute_prediction_error(pred_xyz, gt_xyz, error_type='l1')

        # Simulate confidence (should be correlated with inverse error)
        # Good model: high confidence when error is low
        confidence = torch.sigmoid(-error.unsqueeze(-1) + 2.0)

        loss = loss_fn(confidence, error)

        assert torch.isfinite(loss)
        assert loss > 0
        assert loss < 10.0  # Reasonable magnitude

    def test_confidence_learning_signal(self):
        """Test that loss provides learning signal for confidence."""
        loss_fn = ConfidenceLoss()

        pred_xyz = torch.randn(2, 16, 3)
        gt_xyz = torch.randn(2, 16, 3)

        error = compute_prediction_error(pred_xyz, gt_xyz, error_type='l2')

        # Two scenarios: good confidence vs bad confidence
        confidence_good = torch.tensor(0.8).expand(2, 16, 1).requires_grad_(True)
        confidence_bad = torch.tensor(0.2).expand(2, 16, 1).requires_grad_(True)

        loss_good = loss_fn(confidence_good, error)
        loss_bad = loss_fn(confidence_bad, error)

        # With large errors, high confidence should have higher loss
        # (we're confident but wrong)
        if error.mean() > 1.0:
            assert loss_good > loss_bad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
