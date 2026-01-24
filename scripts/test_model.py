"""Test script for D4RT model components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from d4rt.models import build_d4rt_model, print_model_info, count_parameters
from d4rt.losses import build_composite_loss


def test_model_initialization():
    """Test model initialization with different configs."""
    print("=" * 80)
    print("TEST 1: Model Initialization")
    print("=" * 80)

    configs = ['vit_b', 'vit_l', 'vit_g']

    for config_name in configs:
        print(f"\nTesting {config_name.upper()}...")
        config_path = Path(__file__).parent.parent / 'configs' / 'model' / f'{config_name}.yaml'
        config = OmegaConf.load(config_path)

        try:
            model = build_d4rt_model(config)
            params = count_parameters(model)

            print(f"✓ {config_name.upper()} initialized successfully")
            print(f"  - Encoder: {params['encoder']:,} parameters")
            print(f"  - Decoder: {params['decoder']:,} parameters")
            print(f"  - Query Encoder: {params['query_encoder']:,} parameters")
            print(f"  - Total: {params['total']:,} parameters")

        except Exception as e:
            print(f"✗ {config_name.upper()} failed: {e}")
            return False

    return True


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\n" + "=" * 80)
    print("TEST 2: Forward Pass")
    print("=" * 80)

    # Load ViT-B config (smallest model)
    config_path = Path(__file__).parent.parent / 'configs' / 'model' / 'vit_b.yaml'
    config = OmegaConf.load(config_path)

    # Build model
    model = build_d4rt_model(config)
    model.eval()

    # Create dummy input
    batch_size = 2
    num_frames = 48
    height, width = 256, 256
    num_queries = 128

    print(f"\nInput shapes:")
    print(f"  - Video: [{batch_size}, {num_frames}, 3, {height}, {width}]")
    print(f"  - Queries: {num_queries} per sample")

    # Create dummy video
    video = torch.randn(batch_size, num_frames, 3, height, width)

    # Create dummy queries
    queries = {
        'u': torch.rand(batch_size, num_queries),
        'v': torch.rand(batch_size, num_queries),
        't_src': torch.randint(0, num_frames, (batch_size, num_queries)),
        't_tgt': torch.randint(0, num_frames, (batch_size, num_queries)),
        't_cam': torch.randint(0, num_frames, (batch_size, num_queries)),
    }

    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            outputs = model(video, queries)

        print("✓ Forward pass successful")
        print(f"\nOutput shapes:")
        print(f"  - xyz: {list(outputs['xyz'].shape)}")
        print(f"  - visibility: {list(outputs['visibility'].shape)}")
        print(f"  - encoder_features: {list(outputs['encoder_features'].shape)}")

        # Verify shapes
        assert outputs['xyz'].shape == (batch_size, num_queries, 3), "Wrong xyz shape"
        assert outputs['visibility'].shape == (batch_size, num_queries, 1), "Wrong visibility shape"

        print("\n✓ All output shapes correct")

        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoder_decoder_separately():
    """Test encoder and decoder separately."""
    print("\n" + "=" * 80)
    print("TEST 3: Encoder and Decoder Separately")
    print("=" * 80)

    config_path = Path(__file__).parent.parent / 'configs' / 'model' / 'vit_b.yaml'
    config = OmegaConf.load(config_path)

    model = build_d4rt_model(config)
    model.eval()

    batch_size = 1
    num_frames = 48
    height, width = 256, 256
    num_queries = 64

    video = torch.randn(batch_size, num_frames, 3, height, width)

    print("\n1. Testing encoder...")
    try:
        with torch.no_grad():
            encoder_features = model.encode_video(video)
        print(f"✓ Encoder output: {list(encoder_features.shape)}")
    except Exception as e:
        print(f"✗ Encoder failed: {e}")
        return False

    print("\n2. Testing query encoder + decoder...")
    queries = {
        'u': torch.rand(batch_size, num_queries),
        'v': torch.rand(batch_size, num_queries),
        't_src': torch.randint(0, num_frames, (batch_size, num_queries)),
        't_tgt': torch.randint(0, num_frames, (batch_size, num_queries)),
        't_cam': torch.randint(0, num_frames, (batch_size, num_queries)),
    }

    try:
        with torch.no_grad():
            outputs = model.predict_from_queries(encoder_features, queries, video)
        print(f"✓ Decoder output xyz: {list(outputs['xyz'].shape)}")
        print(f"✓ Decoder output visibility: {list(outputs['visibility'].shape)}")
        return True
    except Exception as e:
        print(f"✗ Decoder failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test loss computation."""
    print("\n" + "=" * 80)
    print("TEST 4: Loss Functions")
    print("=" * 80)

    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'training' / 'debug.yaml'
    config = OmegaConf.load(config_path)

    # Build loss function
    loss_fn = build_composite_loss(config)

    # Create dummy predictions and targets
    batch_size = 2
    num_queries = 128
    num_frames = 48

    predictions = {
        'xyz': torch.randn(batch_size, num_queries, 3),
        'visibility': torch.randn(batch_size, num_queries, 1),
    }

    targets = {
        'xyz': torch.randn(batch_size, num_queries, 3),
        'uv': torch.rand(batch_size, num_queries, 2) * 256,
        'visibility': torch.randint(0, 2, (batch_size, num_queries)).float(),
    }

    cameras = {
        'intrinsics': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_frames, 1, 1),
        'extrinsics': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_frames, 1, 1),
    }

    queries = {
        't_cam': torch.randint(0, num_frames, (batch_size, num_queries)),
        't_tgt': torch.randint(0, num_frames, (batch_size, num_queries)),
    }

    print("\nComputing losses...")
    try:
        total_loss, loss_dict = loss_fn(predictions, targets, cameras, queries)

        print("✓ Loss computation successful")
        print("\nLoss values:")
        for key, value in loss_dict.items():
            print(f"  - {key}: {value:.6f}")

        assert total_loss.requires_grad, "Loss should require gradients"
        print("\n✓ Loss requires gradients (can backpropagate)")

        return True

    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test gradient flow through model."""
    print("\n" + "=" * 80)
    print("TEST 5: Gradient Flow")
    print("=" * 80)

    config_path = Path(__file__).parent.parent / 'configs' / 'model' / 'vit_b.yaml'
    config = OmegaConf.load(config_path)

    model = build_d4rt_model(config)
    model.train()

    # Small batch for gradient test
    batch_size = 1
    num_frames = 48
    num_queries = 32

    video = torch.randn(batch_size, num_frames, 3, 256, 256, requires_grad=True)
    queries = {
        'u': torch.rand(batch_size, num_queries),
        'v': torch.rand(batch_size, num_queries),
        't_src': torch.randint(0, num_frames, (batch_size, num_queries)),
        't_tgt': torch.randint(0, num_frames, (batch_size, num_queries)),
        't_cam': torch.randint(0, num_frames, (batch_size, num_queries)),
    }

    print("\nRunning forward pass...")
    try:
        outputs = model(video, queries)
        loss = outputs['xyz'].mean() + outputs['visibility'].mean()

        print("✓ Forward pass successful")
        print(f"  Loss value: {loss.item():.6f}")

        print("\nRunning backward pass...")
        loss.backward()

        print("✓ Backward pass successful")

        # Check if gradients exist
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        if has_grad:
            print("✓ Gradients computed successfully")
            return True
        else:
            print("✗ No gradients found")
            return False

    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_configs():
    """Test all model configurations."""
    print("\n" + "=" * 80)
    print("TEST 6: Model Configurations")
    print("=" * 80)

    configs = {
        'vit_b': {'expected_params': 230_000_000, 'tolerance': 0.1},
        'vit_l': {'expected_params': 451_000_000, 'tolerance': 0.1},
        'vit_g': {'expected_params': 1_144_000_000, 'tolerance': 0.1},
    }

    all_passed = True

    for config_name, expected in configs.items():
        print(f"\nTesting {config_name.upper()} configuration...")
        config_path = Path(__file__).parent.parent / 'configs' / 'model' / f'{config_name}.yaml'
        config = OmegaConf.load(config_path)

        try:
            model = build_d4rt_model(config)
            params = count_parameters(model)
            total_params = params['total']

            expected_params = expected['expected_params']
            tolerance = expected['tolerance']

            diff = abs(total_params - expected_params) / expected_params

            if diff <= tolerance:
                print(f"✓ Parameter count within tolerance: {total_params:,} "
                      f"(expected ~{expected_params:,})")
            else:
                print(f"⚠ Parameter count differs: {total_params:,} "
                      f"(expected ~{expected_params:,}, diff: {diff*100:.1f}%)")
                all_passed = False

        except Exception as e:
            print(f"✗ Failed: {e}")
            all_passed = False

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("D4RT MODEL VALIDATION TESTS")
    print("=" * 80)

    tests = [
        ("Model Initialization", test_model_initialization),
        ("Forward Pass", test_forward_pass),
        ("Encoder/Decoder Separation", test_encoder_decoder_separately),
        ("Loss Functions", test_loss_functions),
        ("Gradient Flow", test_gradient_flow),
        ("Model Configurations", test_model_configs),
    ]

    results = []

    for test_name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
