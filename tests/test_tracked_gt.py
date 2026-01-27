"""Tests for tracked ground truth extraction (Phase 4 critical fix)."""

import torch
import pytest
import numpy as np
from d4rt.data.query_sampling import extract_ground_truth_at_queries


class TestTrackedGroundTruth:
    """Tests for tracked vs fixed pixel ground truth extraction."""

    def test_fixed_vs_tracked_difference(self):
        """Test that fixed and tracked GT are different for moving objects."""
        # Create simple test data
        T, H, W = 5, 10, 10
        num_queries = 4

        # Create 3D points that "move" over time
        # Object moves from left to right
        points_3d = torch.zeros(T, H, W, 3)
        for t in range(T):
            # Object at x = t+2 (moves right)
            points_3d[t, 5, t+2, :] = torch.tensor([float(t+2), 5.0, 10.0])

        visibility = torch.ones(T, H, W)

        # Dummy cameras
        K = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
        T_mat = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)

        # Query at fixed position (5, 2) at t=0, but check target t=2
        queries = {
            'u': torch.tensor([0.2, 0.3, 0.4, 0.5]),  # u = [2, 3, 4, 5] in pixels
            'v': torch.tensor([0.5, 0.5, 0.5, 0.5]),  # v = 5
            't_src': torch.tensor([0, 0, 0, 0]),
            't_tgt': torch.tensor([2, 2, 2, 2]),  # Target time = 2
            't_cam': torch.tensor([2, 2, 2, 2]),
        }

        # Create tracked positions that follow the object
        # At t=2, object is at x=4, so tracked position should be (4, 5)
        tracked_positions = torch.zeros(num_queries, T, 2)
        for i in range(num_queries):
            for t in range(T):
                # Track moving object: x position increases with t
                tracked_positions[i, t, 0] = t + 2 + i  # u coordinate
                tracked_positions[i, t, 1] = 5  # v coordinate (constant)

        # Extract GT without tracks (fixed pixels - BUG)
        targets_fixed = extract_ground_truth_at_queries(
            queries, points_3d, visibility, K, T_mat,
            tracked_positions=None
        )

        # Extract GT with tracks (correct tracking)
        targets_tracked = extract_ground_truth_at_queries(
            queries, points_3d, visibility, K, T_mat,
            tracked_positions=tracked_positions
        )

        # They should be DIFFERENT (tracked follows moving object)
        assert not torch.allclose(targets_fixed['xyz'], targets_tracked['xyz'], atol=1e-6), \
            "Fixed and tracked GT should differ for moving objects!"

    def test_tracked_follows_motion(self):
        """Test that tracked GT correctly follows moving objects."""
        T, H, W = 3, 8, 8
        num_queries = 1

        # Create object that moves diagonally
        points_3d = torch.zeros(T, H, W, 3)
        for t in range(T):
            x, y = t + 2, t + 2
            if x < W and y < H:
                points_3d[t, y, x, :] = torch.tensor([float(x), float(y), 5.0 + t])

        visibility = torch.ones(T, H, W)
        K = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
        T_mat = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)

        queries = {
            'u': torch.tensor([0.25]),  # x=2 at start
            'v': torch.tensor([0.25]),  # y=2 at start
            't_src': torch.tensor([0]),
            't_tgt': torch.tensor([2]),  # Check at t=2
            't_cam': torch.tensor([2]),
        }

        # Tracked positions follow diagonal motion
        tracked_positions = torch.zeros(num_queries, T, 2)
        for t in range(T):
            tracked_positions[0, t, 0] = t + 2  # x
            tracked_positions[0, t, 1] = t + 2  # y

        targets = extract_ground_truth_at_queries(
            queries, points_3d, visibility, K, T_mat,
            tracked_positions=tracked_positions
        )

        # At t=2, tracked position is (4, 4), depth should be 7.0
        expected_z = 7.0  # 5.0 + 2
        assert torch.allclose(targets['xyz'][0, 2], torch.tensor(expected_z), atol=0.1), \
            f"Tracked depth should be {expected_z}, got {targets['xyz'][0, 2]}"

    def test_backward_compatibility_no_tracks(self):
        """Test that function works without tracked_positions (backward compatible)."""
        T, H, W = 3, 10, 10
        num_queries = 8

        points_3d = torch.randn(T, H, W, 3) * 2 + 10
        visibility = torch.ones(T, H, W)
        K = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
        T_mat = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)

        queries = {
            'u': torch.rand(num_queries),
            'v': torch.rand(num_queries),
            't_src': torch.randint(0, T, (num_queries,)),
            't_tgt': torch.randint(0, T, (num_queries,)),
            't_cam': torch.randint(0, T, (num_queries,)),
        }

        # Should work without tracked_positions (old behavior)
        targets = extract_ground_truth_at_queries(
            queries, points_3d, visibility, K, T_mat,
            tracked_positions=None
        )

        assert 'xyz' in targets
        assert targets['xyz'].shape == (num_queries, 3)

    def test_tracked_positions_shape_validation(self):
        """Test that tracked positions have correct shape."""
        T, H, W = 4, 12, 12
        num_queries = 16

        points_3d = torch.randn(T, H, W, 3) * 2 + 10
        visibility = torch.ones(T, H, W)
        K = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
        T_mat = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)

        queries = {
            'u': torch.rand(num_queries),
            'v': torch.rand(num_queries),
            't_src': torch.randint(0, T, (num_queries,)),
            't_tgt': torch.randint(0, T, (num_queries,)),
            't_cam': torch.randint(0, T, (num_queries,)),
        }

        # Correct shape: [num_queries, T, 2]
        tracked_positions = torch.rand(num_queries, T, 2) * torch.tensor([W, H])

        targets = extract_ground_truth_at_queries(
            queries, points_3d, visibility, K, T_mat,
            tracked_positions=tracked_positions
        )

        assert targets['xyz'].shape == (num_queries, 3)

    def test_normals_use_tracked_positions(self):
        """Test that normals also use tracked positions."""
        T, H, W = 3, 10, 10
        num_queries = 4

        points_3d = torch.randn(T, H, W, 3) * 2 + 10
        visibility = torch.ones(T, H, W)

        # Create normals that vary by position
        normals = torch.zeros(T, H, W, 3)
        for t in range(T):
            for y in range(H):
                for x in range(W):
                    # Normal encodes position
                    normals[t, y, x, :] = torch.tensor([float(x)/W, float(y)/H, 1.0])

        K = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
        T_mat = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)

        queries = {
            'u': torch.tensor([0.3, 0.5, 0.7, 0.9]),
            'v': torch.tensor([0.3, 0.5, 0.7, 0.9]),
            't_src': torch.tensor([0, 0, 0, 0]),
            't_tgt': torch.tensor([1, 1, 1, 1]),
            't_cam': torch.tensor([1, 1, 1, 1]),
        }

        # Different tracked positions at t=1
        tracked_positions = torch.zeros(num_queries, T, 2)
        tracked_positions[:, 1, 0] = torch.tensor([5.0, 6.0, 7.0, 8.0])  # Different x
        tracked_positions[:, 1, 1] = 5.0  # Same y

        # Extract with tracks
        targets = extract_ground_truth_at_queries(
            queries, points_3d, visibility, K, T_mat,
            normals=normals,
            tracked_positions=tracked_positions
        )

        assert 'normals' in targets
        assert targets['normals'].shape == (num_queries, 3)

        # Normals should correspond to tracked positions, not fixed pixels
        # This is verified by the different x values resulting in different normal.x
        for i in range(num_queries - 1):
            assert targets['normals'][i, 0] != targets['normals'][i+1, 0], \
                "Normals should differ based on tracked positions"

    def test_clamping_out_of_bounds_tracks(self):
        """Test that out-of-bounds tracked positions are clamped."""
        T, H, W = 3, 10, 10
        num_queries = 4

        points_3d = torch.randn(T, H, W, 3) * 2 + 10
        visibility = torch.ones(T, H, W)
        K = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
        T_mat = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)

        queries = {
            'u': torch.tensor([0.5, 0.5, 0.5, 0.5]),
            'v': torch.tensor([0.5, 0.5, 0.5, 0.5]),
            't_src': torch.tensor([0, 0, 0, 0]),
            't_tgt': torch.tensor([1, 1, 1, 1]),
            't_cam': torch.tensor([1, 1, 1, 1]),
        }

        # Out-of-bounds tracked positions
        tracked_positions = torch.zeros(num_queries, T, 2)
        tracked_positions[:, 1, 0] = torch.tensor([-5.0, 15.0, 5.0, 5.0])  # Some out of bounds
        tracked_positions[:, 1, 1] = torch.tensor([5.0, 5.0, -3.0, 20.0])  # Some out of bounds

        # Should not crash - positions should be clamped
        targets = extract_ground_truth_at_queries(
            queries, points_3d, visibility, K, T_mat,
            tracked_positions=tracked_positions
        )

        assert torch.all(torch.isfinite(targets['xyz']))
        assert targets['xyz'].shape == (num_queries, 3)


class TestTrackedGTIntegration:
    """Integration tests with realistic scenarios."""

    def test_moving_object_scenario(self):
        """Full scenario: object moves, tracked GT should follow."""
        # Simulate a simple scene with a moving object
        T, H, W = 10, 64, 64
        num_queries = 32

        # Create a scene where depth changes as object moves
        points_3d = torch.ones(T, H, W, 3) * 10.0  # Background at 10m
        for t in range(T):
            # Object moves from left (x=10) to right (x=50)
            x_pos = 10 + int(t * 4.0)
            if x_pos < W:
                # Object closer to camera (5m) moving across scene
                points_3d[t, 32, x_pos, :] = torch.tensor([float(x_pos), 32.0, 5.0])

        visibility = torch.ones(T, H, W)
        K = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
        T_mat = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)

        # Sample queries at start position
        queries = {
            'u': torch.ones(num_queries) * 10 / W,  # Start at x=10
            'v': torch.ones(num_queries) * 32 / H,  # y=32
            't_src': torch.zeros(num_queries).long(),
            't_tgt': torch.randint(0, T, (num_queries,)),
            't_cam': torch.randint(0, T, (num_queries,)),
        }

        # Create tracks that follow the moving object
        tracked_positions = torch.zeros(num_queries, T, 2)
        for t in range(T):
            x_pos = min(10 + t * 4, W - 1)
            tracked_positions[:, t, 0] = x_pos  # Follow x movement
            tracked_positions[:, t, 1] = 32  # Constant y

        # Extract GT with tracks
        targets = extract_ground_truth_at_queries(
            queries, points_3d, visibility, K, T_mat,
            tracked_positions=tracked_positions
        )

        # Depth should be around 5m (object depth), not 10m (background)
        mean_depth = targets['xyz'][:, 2].mean()
        assert mean_depth < 7.0, \
            f"Mean depth should be closer to object (5m) than background (10m), got {mean_depth}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
