"""Integration tests for paper losses with realistic data."""

import torch
import pytest
from d4rt.losses.l1_3d import L1_3DLoss


class TestPaperLossIntegration:
    """Integration tests with realistic MOVi-like data."""

    def test_realistic_batch_paper_formula(self):
        """Test paper formula with realistic batch size and data."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        # Simulate realistic MOVi data:
        # Batch size: 4
        # Number of query points: 256
        # 3D coordinates in world space (meters)
        batch_size = 4
        num_points = 256

        # Generate realistic 3D points
        # Typically: X, Y in [-5, 5], Z (depth) in [5, 20]
        pred_xyz = torch.randn(batch_size, num_points, 3)
        pred_xyz[..., 0] *= 2.5  # X range
        pred_xyz[..., 1] *= 2.5  # Y range
        pred_xyz[..., 2] = pred_xyz[..., 2] * 3 + 12  # Z centered at 12m

        # Ground truth with small perturbation (simulating prediction error)
        gt_xyz = pred_xyz + torch.randn_like(pred_xyz) * 0.1

        # Compute loss
        loss = loss_fn(pred_xyz, gt_xyz)

        # Verify loss properties
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss > 0, "Loss should be positive for different pred/gt"
        assert loss < 10.0, f"Loss {loss.item():.4f} should be reasonable (<10)"

        print(f"✓ Paper formula loss: {loss.item():.4f}")

    def test_old_vs_new_formula_comparison(self):
        """Compare old (scene bounds) vs new (paper) formula."""
        loss_old = L1_3DLoss(normalize_by_scene=True, use_paper_formula=False)
        loss_new = L1_3DLoss(use_paper_formula=True)

        # Realistic data
        batch_size = 4
        num_points = 64

        pred_xyz = torch.randn(batch_size, num_points, 3)
        pred_xyz[..., 2] = pred_xyz[..., 2] * 3 + 10  # Depth around 10m

        gt_xyz = pred_xyz + torch.randn_like(pred_xyz) * 0.2

        # Scene bounds for old formula: [-5, 5] x [-5, 5] x [5, 15]
        scene_bounds = torch.tensor([[-5.0, 5.0, -5.0, 5.0, 5.0, 15.0]] * batch_size)

        # Compute both losses
        loss_old_value = loss_old(pred_xyz, gt_xyz, scene_bounds)
        loss_new_value = loss_new(pred_xyz, gt_xyz)

        # Both should be finite and positive
        assert torch.isfinite(loss_old_value)
        assert torch.isfinite(loss_new_value)
        assert loss_old_value > 0
        assert loss_new_value > 0

        print(f"✓ Old formula loss: {loss_old_value.item():.4f}")
        print(f"✓ New formula loss: {loss_new_value.item():.4f}")
        print(f"✓ Ratio (new/old): {(loss_new_value / loss_old_value).item():.4f}")

    def test_gradient_flow_realistic_data(self):
        """Test that gradients flow properly with realistic data."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        batch_size = 8
        num_points = 128

        # Create predictions that require gradients
        pred_xyz = torch.randn(batch_size, num_points, 3, requires_grad=True)
        pred_xyz.data[..., 2] = pred_xyz.data[..., 2] * 3 + 10

        gt_xyz = torch.randn(batch_size, num_points, 3)
        gt_xyz[..., 2] = gt_xyz[..., 2] * 3 + 10

        # Forward pass
        loss = loss_fn(pred_xyz, gt_xyz)

        # Backward pass
        loss.backward()

        # Check gradients
        assert pred_xyz.grad is not None, "Gradients should exist"
        assert torch.all(torch.isfinite(pred_xyz.grad)), "Gradients should be finite"
        assert torch.any(pred_xyz.grad != 0), "Gradients should be non-zero"

        # Check gradient magnitude is reasonable (small but non-zero)
        grad_norm = pred_xyz.grad.norm()
        assert 0.0001 < grad_norm < 100.0, f"Gradient norm {grad_norm:.4f} should be reasonable"

        print(f"✓ Gradient norm: {grad_norm.item():.6f}")

    def test_loss_decreases_with_better_predictions(self):
        """Test that loss decreases as predictions improve."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        batch_size = 4
        num_points = 64

        gt_xyz = torch.randn(batch_size, num_points, 3)
        gt_xyz[..., 2] = gt_xyz[..., 2] * 3 + 10

        # Three levels of prediction quality
        pred_bad = gt_xyz + torch.randn_like(gt_xyz) * 1.0  # Large error
        pred_medium = gt_xyz + torch.randn_like(gt_xyz) * 0.1  # Medium error
        pred_good = gt_xyz + torch.randn_like(gt_xyz) * 0.01  # Small error

        loss_bad = loss_fn(pred_bad, gt_xyz)
        loss_medium = loss_fn(pred_medium, gt_xyz)
        loss_good = loss_fn(pred_good, gt_xyz)

        # Loss should decrease as predictions improve
        assert loss_bad > loss_medium, "Bad predictions should have higher loss"
        assert loss_medium > loss_good, "Medium predictions should have higher loss than good"

        print(f"✓ Bad prediction loss: {loss_bad.item():.4f}")
        print(f"✓ Medium prediction loss: {loss_medium.item():.4f}")
        print(f"✓ Good prediction loss: {loss_good.item():.4f}")

    def test_batch_consistency(self):
        """Test that loss computation is consistent across batches."""
        loss_fn = L1_3DLoss(use_paper_formula=True)

        # Single sample
        pred_single = torch.randn(1, 64, 3)
        pred_single[..., 2] = pred_single[..., 2] * 3 + 10
        gt_single = pred_single + torch.randn_like(pred_single) * 0.1

        # Same sample repeated in batch
        pred_batch = pred_single.repeat(4, 1, 1)
        gt_batch = gt_single.repeat(4, 1, 1)

        loss_single = loss_fn(pred_single, gt_single)
        loss_batch = loss_fn(pred_batch, gt_batch)

        # Losses should be equal (since all batch items are identical)
        assert torch.allclose(loss_single, loss_batch, atol=1e-6)

        print(f"✓ Single sample loss: {loss_single.item():.4f}")
        print(f"✓ Batch loss: {loss_batch.item():.4f}")


def test_quick_integration():
    """Quick smoke test for CI/CD."""
    loss_fn = L1_3DLoss(use_paper_formula=True)

    pred = torch.randn(2, 32, 3)
    pred[..., 2] = pred[..., 2].abs() + 5  # Ensure positive depth

    gt = pred + torch.randn_like(pred) * 0.1

    loss = loss_fn(pred, gt)

    assert torch.isfinite(loss)
    assert loss > 0
    assert loss < 10.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
