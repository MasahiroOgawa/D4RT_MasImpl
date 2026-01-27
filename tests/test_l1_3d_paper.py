"""Unit tests for paper's 3D loss implementation."""

import torch
import pytest
from d4rt.losses.l1_3d import L1_3DLoss


class TestMeanDepthNormalization:
    """Tests for mean depth normalization."""

    def test_mean_depth_normalization_simple(self):
        """Test that points are normalized by mean Z value."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        # Create test data with mean Z = 10
        pred = torch.tensor([[[1.0, 2.0, 10.0], [2.0, 3.0, 10.0]]])  # [B=1, N=2, 3]

        # After normalization: divide by mean(Z) = 10
        # Expected: [[0.1, 0.2, 1.0], [0.2, 0.3, 1.0]]
        pred_norm = loss_fn.normalize_by_mean_depth(pred)

        expected = torch.tensor([[[0.1, 0.2, 1.0], [0.2, 0.3, 1.0]]])
        assert torch.allclose(pred_norm, expected, atol=1e-6)

    def test_mean_depth_normalization_batch(self):
        """Test mean depth normalization with batch size > 1."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        # Batch of 2, each with different mean depths
        pred = torch.tensor([
            [[2.0, 4.0, 8.0], [4.0, 6.0, 12.0]],   # mean Z = 10
            [[1.0, 2.0, 5.0], [3.0, 4.0, 15.0]]     # mean Z = 10
        ])  # [B=2, N=2, 3]

        pred_norm = loss_fn.normalize_by_mean_depth(pred)

        # First batch: mean Z = (8 + 12) / 2 = 10
        # Second batch: mean Z = (5 + 15) / 2 = 10
        expected = torch.tensor([
            [[0.2, 0.4, 0.8], [0.4, 0.6, 1.2]],
            [[0.1, 0.2, 0.5], [0.3, 0.4, 1.5]]
        ])

        assert torch.allclose(pred_norm, expected, atol=1e-6)

    def test_mean_depth_no_divide_by_zero(self):
        """Test that epsilon prevents division by zero."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        # Edge case: all Z values are zero
        pred = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]])

        # Should not raise error due to epsilon
        pred_norm = loss_fn.normalize_by_mean_depth(pred)

        # With epsilon=1e-8, result should be huge but not inf
        assert torch.all(torch.isfinite(pred_norm))


class TestSignedLogTransform:
    """Tests for signed log transform: sign(x) * log(1 + |x|)."""

    def test_signed_log_positive_values(self):
        """Test signed log on positive values."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        x = torch.tensor([1.0, 2.0, 10.0])
        result = loss_fn.signed_log_transform(x)

        # Expected: log(1+1)=log(2)≈0.693, log(3)≈1.099, log(11)≈2.398
        expected = torch.tensor([
            torch.log(torch.tensor(2.0)),
            torch.log(torch.tensor(3.0)),
            torch.log(torch.tensor(11.0))
        ])

        assert torch.allclose(result, expected, atol=1e-6)

    def test_signed_log_negative_values(self):
        """Test signed log on negative values."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        x = torch.tensor([-1.0, -2.0, -10.0])
        result = loss_fn.signed_log_transform(x)

        # Expected: -log(2), -log(3), -log(11)
        expected = torch.tensor([
            -torch.log(torch.tensor(2.0)),
            -torch.log(torch.tensor(3.0)),
            -torch.log(torch.tensor(11.0))
        ])

        assert torch.allclose(result, expected, atol=1e-6)

    def test_signed_log_zero(self):
        """Test signed log at zero."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        x = torch.tensor([0.0])
        result = loss_fn.signed_log_transform(x)

        # sign(0) * log(1 + 0) = 0 * 0 = 0
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    def test_signed_log_symmetry(self):
        """Test that f(-x) = -f(x)."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        x = torch.tensor([1.0, 5.0, 10.0])
        result_pos = loss_fn.signed_log_transform(x)
        result_neg = loss_fn.signed_log_transform(-x)

        assert torch.allclose(result_pos, -result_neg, atol=1e-6)

    def test_signed_log_monotonic(self):
        """Test that transform preserves ordering."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        x = torch.tensor([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
        result = loss_fn.signed_log_transform(x)

        # Result should be monotonically increasing
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]


class TestPaperLossForward:
    """Tests for full forward pass with paper formula."""

    def test_paper_loss_identical_predictions(self):
        """Test loss is zero when predictions match ground truth."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        pred = torch.tensor([[[1.0, 2.0, 10.0], [2.0, 3.0, 15.0]]])
        gt = pred.clone()

        loss = loss_fn(pred, gt)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_paper_loss_different_predictions(self):
        """Test loss is positive when predictions differ."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        pred = torch.tensor([[[1.0, 2.0, 10.0], [2.0, 3.0, 10.0]]])
        gt = torch.tensor([[[2.0, 3.0, 10.0], [3.0, 4.0, 10.0]]])

        loss = loss_fn(pred, gt)

        # Loss should be positive
        assert loss > 0

    def test_paper_loss_scale_invariance(self):
        """Test that loss is scale-invariant (scaling all coords doesn't change loss much)."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        # Original data
        pred1 = torch.tensor([[[1.0, 2.0, 10.0], [2.0, 3.0, 10.0]]])
        gt1 = torch.tensor([[[1.5, 2.5, 10.0], [2.5, 3.5, 10.0]]])

        # Scaled by 2
        pred2 = pred1 * 2
        gt2 = gt1 * 2

        loss1 = loss_fn(pred1, gt1)
        loss2 = loss_fn(pred2, gt2)

        # Losses should be similar (not exactly equal due to log transform)
        # but should be close due to normalization
        ratio = loss2 / loss1
        assert 0.5 < ratio < 2.0, f"Loss ratio {ratio} is too different"

    def test_paper_loss_reasonable_magnitude(self):
        """Test that loss values are in reasonable range."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        # Realistic 3D coordinates (in meters)
        pred = torch.randn(2, 64, 3) * 5 + torch.tensor([0.0, 0.0, 10.0])
        gt = pred + torch.randn(2, 64, 3) * 0.1  # Small perturbation

        loss = loss_fn(pred, gt)

        # Loss should be finite and reasonable
        assert torch.isfinite(loss)
        assert 0 < loss < 10.0

    def test_backward_gradients(self):
        """Test that gradients flow properly through the loss."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        pred = torch.randn(2, 32, 3, requires_grad=True)
        gt = torch.randn(2, 32, 3)

        loss = loss_fn(pred, gt)
        loss.backward()

        # Gradients should exist and be non-zero
        assert pred.grad is not None
        assert torch.any(pred.grad != 0)
        assert torch.all(torch.isfinite(pred.grad))


class TestLegacyMode:
    """Tests for legacy scene-bounds normalization mode."""

    def test_legacy_mode_with_scene_bounds(self):
        """Test that old normalization still works."""
        loss_fn = L1_3DLoss(normalize_by_scene=True, use_paper_formula=False)

        pred = torch.tensor([[[1.0, 2.0, 10.0]]])
        gt = torch.tensor([[[2.0, 3.0, 10.0]]])
        scene_bounds = torch.tensor([[0.0, 10.0, 0.0, 10.0, 0.0, 20.0]])  # max extent = 20

        loss = loss_fn(pred, gt, scene_bounds)

        # Loss should be positive and finite
        assert torch.isfinite(loss)
        assert loss > 0

    def test_legacy_mode_without_scene_bounds(self):
        """Test fallback to no normalization."""
        loss_fn = L1_3DLoss(normalize_by_scene=True, use_paper_formula=False)

        pred = torch.tensor([[[1.0, 2.0, 10.0]]])
        gt = torch.tensor([[[2.0, 3.0, 10.0]]])

        loss = loss_fn(pred, gt, scene_bounds=None)

        # Should compute simple L1 loss
        expected = torch.abs(pred - gt).mean()
        assert torch.allclose(loss, expected, atol=1e-6)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_point(self):
        """Test with single point."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        pred = torch.tensor([[[1.0, 2.0, 10.0]]])
        gt = torch.tensor([[[1.5, 2.5, 10.0]]])

        loss = loss_fn(pred, gt)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_large_batch(self):
        """Test with large batch size."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        pred = torch.randn(32, 256, 3) * 5 + torch.tensor([0.0, 0.0, 10.0])
        gt = pred + torch.randn(32, 256, 3) * 0.5

        loss = loss_fn(pred, gt)

        assert torch.isfinite(loss)
        assert loss > 0

    def test_very_large_depths(self):
        """Test with very large depth values."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        pred = torch.tensor([[[1.0, 2.0, 1000.0]]])
        gt = torch.tensor([[[1.5, 2.5, 1000.0]]])

        loss = loss_fn(pred, gt)

        # Log transform should keep this reasonable
        assert torch.isfinite(loss)
        assert loss < 100.0

    def test_negative_depths(self):
        """Test with negative depth values (behind camera)."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        pred = torch.tensor([[[1.0, 2.0, -5.0]]])
        gt = torch.tensor([[[1.5, 2.5, -5.0]]])

        loss = loss_fn(pred, gt)

        # Should handle negative depths gracefully
        assert torch.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
