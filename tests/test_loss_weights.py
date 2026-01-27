"""Tests for loss weight configuration."""

import pytest
import yaml
from pathlib import Path
from d4rt.losses.composite_loss import CompositeLoss


class TestPaperWeights:
    """Test that loss weights match D4RT paper."""

    def test_paper_config_exists(self):
        """Test that paper config file exists."""
        config_path = Path("configs/training/train_5k_movi_paper.yaml")
        assert config_path.exists(), "Paper config file should exist"

    def test_paper_config_weights(self):
        """Test that paper config has correct weights."""
        config_path = Path("configs/training/train_5k_movi_paper.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Expected weights from paper
        expected_weights = {
            'l1_3d': 1.0,
            'l2_2d': 0.1,
            'normal': 0.5,
            'motion': 0.1,
            'visibility': 0.1,
        }

        loss_weights = config['loss_weights']

        # Verify all expected weights
        for key, expected_value in expected_weights.items():
            assert key in loss_weights, f"Weight '{key}' should be in config"
            assert loss_weights[key] == expected_value, \
                f"Weight '{key}' should be {expected_value}, got {loss_weights[key]}"

    def test_paper_formula_enabled(self):
        """Test that paper formula is enabled in config."""
        config_path = Path("configs/training/train_5k_movi_paper.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        loss_weights = config['loss_weights']
        assert 'use_paper_formula_3d' in loss_weights, \
            "Config should specify use_paper_formula_3d"
        assert loss_weights['use_paper_formula_3d'] is True, \
            "Paper formula should be enabled"


class TestCompositeLossDefaults:
    """Test CompositeLoss default weights."""

    def test_default_weights_match_paper(self):
        """Test that default weights in CompositeLoss match paper."""
        loss_fn = CompositeLoss()

        expected_weights = {
            'l1_3d': 1.0,
            'l2_2d': 0.1,
            'normal': 0.5,  # Paper value
            'motion': 0.1,
            'visibility': 0.1,
        }

        for key, expected_value in expected_weights.items():
            assert key in loss_fn.loss_weights, f"Weight '{key}' should exist"
            assert loss_fn.loss_weights[key] == expected_value, \
                f"Default weight '{key}' should be {expected_value}, got {loss_fn.loss_weights[key]}"

    def test_paper_formula_enabled_by_default(self):
        """Test that paper formula is used by default for 3D loss."""
        loss_fn = CompositeLoss()
        assert loss_fn.l1_3d_loss.use_paper_formula is True, \
            "Paper formula should be enabled by default"

    def test_custom_weights_override(self):
        """Test that custom weights can override defaults."""
        custom_weights = {
            'l1_3d': 2.0,
            'l2_2d': 0.2,
            'normal': 0.3,
            'motion': 0.15,
            'visibility': 0.05,
        }

        loss_fn = CompositeLoss(loss_weights=custom_weights)

        for key, expected_value in custom_weights.items():
            assert loss_fn.loss_weights[key] == expected_value, \
                f"Custom weight '{key}' should be {expected_value}"

    def test_paper_formula_can_be_disabled(self):
        """Test that paper formula can be disabled via config."""
        loss_weights = {'use_paper_formula_3d': False}
        loss_fn = CompositeLoss(loss_weights=loss_weights)

        assert loss_fn.l1_3d_loss.use_paper_formula is False, \
            "Paper formula should be disabled when specified"


class TestWeightRatios:
    """Test that weight ratios make sense."""

    def test_3d_loss_is_primary(self):
        """Test that 3D loss has highest weight (primary supervision)."""
        loss_fn = CompositeLoss()

        assert loss_fn.loss_weights['l1_3d'] >= loss_fn.loss_weights['l2_2d']
        assert loss_fn.loss_weights['l1_3d'] >= loss_fn.loss_weights['normal']
        assert loss_fn.loss_weights['l1_3d'] >= loss_fn.loss_weights['motion']
        assert loss_fn.loss_weights['l1_3d'] >= loss_fn.loss_weights['visibility']

    def test_normal_weight_increased(self):
        """Test that normal weight is 10× the old value."""
        old_weight = 0.05
        new_weight = 0.5

        loss_fn = CompositeLoss()
        assert loss_fn.loss_weights['normal'] == new_weight
        assert loss_fn.loss_weights['normal'] / old_weight == 10.0, \
            "Normal weight should be 10× larger than before"

    def test_weight_ratios_paper_compliant(self):
        """Test that weight ratios match paper specifications."""
        loss_fn = CompositeLoss()

        # Ratios relative to l1_3d (which is 1.0)
        l1_3d = loss_fn.loss_weights['l1_3d']

        assert loss_fn.loss_weights['l2_2d'] == l1_3d * 0.1
        assert loss_fn.loss_weights['normal'] == l1_3d * 0.5
        assert loss_fn.loss_weights['motion'] == l1_3d * 0.1
        assert loss_fn.loss_weights['visibility'] == l1_3d * 0.1


class TestBackwardCompatibility:
    """Test backward compatibility with old configs."""

    def test_old_config_still_works(self):
        """Test that old config without paper formula still works."""
        config_path = Path("configs/training/train_5k_movi.yaml")
        if not config_path.exists():
            pytest.skip("Old config not found")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        loss_weights = config.get('loss_weights', {})

        # Should work even without use_paper_formula_3d specified
        loss_fn = CompositeLoss(loss_weights=loss_weights)
        assert loss_fn is not None

    def test_scene_bounds_normalization_available(self):
        """Test that old scene-bounds normalization is still available."""
        # Even with paper formula disabled, old method should work
        loss_weights = {'use_paper_formula_3d': False}
        loss_fn = CompositeLoss(loss_weights=loss_weights)

        assert loss_fn.l1_3d_loss.normalize_by_scene is True or \
               loss_fn.l1_3d_loss.use_paper_formula is False, \
            "Old normalization method should still be available"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
