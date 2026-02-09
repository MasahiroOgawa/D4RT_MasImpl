"""Test script for D4RT data pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf

from d4rt.data import (
    QuerySampler,
    get_train_transforms,
    get_val_transforms,
    CameraParameters,
)


def test_query_sampler():
    """Test query sampling strategy."""
    print("=" * 80)
    print("TEST 1: Query Sampler")
    print("=" * 80)

    sampler = QuerySampler(
        num_queries=2048,
        visible_ratio=0.5,
        occluded_ratio=0.25,
        random_ratio=0.25,
    )

    print(f"\nQuery distribution:")
    print(f"  - Visible points: {sampler.num_visible}")
    print(f"  - Occluded points: {sampler.num_occluded}")
    print(f"  - Random points: {sampler.num_random}")
    print(f"  - Total: {sampler.num_queries}")

    # Create dummy data
    T, H, W = 48, 256, 256
    visibility_mask = torch.rand(T, H, W) > 0.3  # ~70% visible
    depth_map = torch.rand(T, H, W) * 10.0  # 0-10m depth
    points_3d = torch.randn(T, H, W, 3)

    print(f"\nInput data:")
    print(f"  - Video shape: ({T}, 3, {H}, {W})")
    print(f"  - Visibility: {visibility_mask.float().mean()*100:.1f}% visible")

    try:
        queries = sampler.sample_queries(
            video_shape=(T, 3, H, W),
            visibility_mask=visibility_mask,
            depth_map=depth_map,
            points_3d=points_3d,
        )

        print("\n✓ Query sampling successful")
        print(f"\nQuery shapes:")
        for key, value in queries.items():
            print(f"  - {key}: {list(value.shape)}")

        # Verify query coordinates are in valid range
        assert queries['u'].min() >= 0 and queries['u'].max() <= 1, "u out of range"
        assert queries['v'].min() >= 0 and queries['v'].max() <= 1, "v out of range"
        assert queries['t_src'].min() >= 0 and queries['t_src'].max() < T, "t_src out of range"
        assert queries['t_tgt'].min() >= 0 and queries['t_tgt'].max() < T, "t_tgt out of range"
        assert queries['t_cam'].min() >= 0 and queries['t_cam'].max() < T, "t_cam out of range"

        print("\n✓ All query coordinates in valid range")

        return True

    except Exception as e:
        print(f"✗ Query sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transforms():
    """Test data augmentation transforms."""
    print("\n" + "=" * 80)
    print("TEST 2: Data Transforms")
    print("=" * 80)

    # Create dummy config
    config = {
        'augmentation': {
            'random_crop': True,
            'crop_size': [224, 224],
            'horizontal_flip': True,
            'flip_prob': 0.5,
            'color_jitter': True,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
        }
    }

    print("\nTesting training transforms...")
    try:
        train_transforms = get_train_transforms(config)
        print("✓ Training transforms created")

        # Create dummy data
        T, H, W = 48, 256, 256
        video_data = {
            'frames': torch.rand(T, 3, H, W),
            'cameras': {
                'intrinsics': torch.eye(3).unsqueeze(0).repeat(T, 1, 1),
                'extrinsics': torch.eye(4).unsqueeze(0).repeat(T, 1, 1),
            },
        }
        ground_truth = {
            'queries': {},
            'targets': {},
        }

        print("\nApplying transforms...")
        video_data_t, ground_truth_t = train_transforms(video_data, ground_truth)

        print(f"✓ Transforms applied successfully")
        print(f"  - Input shape: [{T}, 3, {H}, {W}]")
        print(f"  - Output shape: {list(video_data_t['frames'].shape)}")

    except Exception as e:
        print(f"✗ Training transforms failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\nTesting validation transforms...")
    try:
        val_transforms = get_val_transforms()
        print("✓ Validation transforms created")

        video_data = {
            'frames': torch.rand(T, 3, H, W),
        }
        ground_truth = {}

        video_data_t, ground_truth_t = val_transforms(video_data, ground_truth)
        print("✓ Validation transforms applied successfully")

        return True

    except Exception as e:
        print(f"✗ Validation transforms failed: {e}")
        return False


def test_camera_utilities():
    """Test camera parameter utilities."""
    print("\n" + "=" * 80)
    print("TEST 3: Camera Utilities")
    print("=" * 80)

    print("\n1. Testing intrinsics creation...")
    try:
        K = CameraParameters.create_intrinsics(
            focal_length=256.0,
            cx=128.0,
            cy=128.0,
        )
        print(f"✓ Intrinsics matrix:\n{K}")
        assert K.shape == (3, 3), "Wrong shape"
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    print("\n2. Testing extrinsics creation...")
    try:
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        T = CameraParameters.create_extrinsics(R, t)
        print(f"✓ Extrinsics matrix:\n{T}")
        assert T.shape == (4, 4), "Wrong shape"
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    print("\n3. Testing 3D to 2D projection...")
    try:
        # Create test points
        points_3d = np.array([
            [0, 0, 5],
            [1, 1, 5],
            [-1, -1, 5],
        ])

        # Project
        points_2d = CameraParameters.project_3d_to_2d(
            points_3d, K, T
        )

        print(f"✓ Projection successful")
        print(f"  - 3D points shape: {points_3d.shape}")
        print(f"  - 2D points shape: {points_2d.shape}")
        print(f"  - Sample 2D point: [{points_2d[0, 0]:.1f}, {points_2d[0, 1]:.1f}]")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_collation():
    """Test batch collation."""
    print("\n" + "=" * 80)
    print("TEST 4: Batch Collation")
    print("=" * 80)

    from d4rt.data import BaseVideoDataset

    # Create dummy samples
    batch_size = 4
    T, H, W = 48, 256, 256
    N = 128

    samples = []
    for _ in range(batch_size):
        sample = {
            'video': torch.rand(T, 3, H, W),
            'queries': {
                'u': torch.rand(N),
                'v': torch.rand(N),
                't_src': torch.randint(0, T, (N,)),
                't_tgt': torch.randint(0, T, (N,)),
                't_cam': torch.randint(0, T, (N,)),
            },
            'targets': {
                'xyz': torch.randn(N, 3),
                'uv': torch.rand(N, 2) * 256,
                'visibility': torch.randint(0, 2, (N,)).float(),
            },
            'cameras': {
                'intrinsics': torch.eye(3).unsqueeze(0).repeat(T, 1, 1),
                'extrinsics': torch.eye(4).unsqueeze(0).repeat(T, 1, 1),
            },
            'metadata': {'scene_id': f'scene_{_}'},
        }
        samples.append(sample)

    print(f"Collating batch of {batch_size} samples...")
    try:
        batched = BaseVideoDataset.collate_fn(samples)

        print("✓ Batch collation successful")
        print(f"\nBatched shapes:")
        print(f"  - video: {list(batched['video'].shape)}")
        print(f"  - queries['u']: {list(batched['queries']['u'].shape)}")
        print(f"  - targets['xyz']: {list(batched['targets']['xyz'].shape)}")
        print(f"  - cameras['intrinsics']: {list(batched['cameras']['intrinsics'].shape)}")
        print(f"  - metadata: {len(batched['metadata'])} items")

        # Verify shapes
        assert batched['video'].shape == (batch_size, T, 3, H, W), "Wrong video shape"
        assert batched['queries']['u'].shape == (batch_size, N), "Wrong query shape"
        assert batched['targets']['xyz'].shape == (batch_size, N, 3), "Wrong target shape"

        print("\n✓ All shapes correct")

        return True

    except Exception as e:
        print(f"✗ Batch collation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patch_extraction():
    """Test patch extraction from video."""
    print("\n" + "=" * 80)
    print("TEST 5: Patch Extraction")
    print("=" * 80)

    from d4rt.utils import extract_patches

    # Create dummy video
    B, T, C, H, W = 2, 48, 3, 256, 256
    video = torch.rand(B, T, C, H, W)

    # Create query coordinates
    N = 10
    u = torch.rand(B, N)
    v = torch.rand(B, N)
    t_src = torch.randint(0, T, (B, N))

    patch_size = 9

    print(f"Extracting {N} patches of size {patch_size}×{patch_size}...")
    try:
        patches = extract_patches(video, u, v, t_src, patch_size=patch_size)

        print("✓ Patch extraction successful")
        print(f"  - Input video: [{B}, {T}, {C}, {H}, {W}]")
        print(f"  - Output patches: {list(patches.shape)}")

        assert patches.shape == (B, N, C, patch_size, patch_size), "Wrong patch shape"

        print("✓ Patch shape correct")

        return True

    except Exception as e:
        print(f"✗ Patch extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all data pipeline tests."""
    print("\n" + "=" * 80)
    print("D4RT DATA PIPELINE VALIDATION TESTS")
    print("=" * 80)

    tests = [
        ("Query Sampler", test_query_sampler),
        ("Data Transforms", test_transforms),
        ("Camera Utilities", test_camera_utilities),
        ("Batch Collation", test_batch_collation),
        ("Patch Extraction", test_patch_extraction),
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
