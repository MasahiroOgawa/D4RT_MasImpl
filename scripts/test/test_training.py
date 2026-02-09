"""Test training setup with dummy data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

from d4rt.models import build_d4rt_model
from d4rt.losses import build_composite_loss
from d4rt.training import D4RTTrainer


class DummyDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, num_samples=10, num_frames=24, resolution=(128, 128), num_queries=64):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.resolution = resolution
        self.num_queries = num_queries

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T, H, W = self.num_frames, self.resolution[0], self.resolution[1]
        N = self.num_queries

        return {
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
                'uv': torch.rand(N, 2) * H,
                'visibility': torch.randint(0, 2, (N,)).float(),
            },
            'cameras': {
                'intrinsics': torch.eye(3).unsqueeze(0).repeat(T, 1, 1),
                'extrinsics': torch.eye(4).unsqueeze(0).repeat(T, 1, 1),
            },
            'metadata': {'scene_id': f'dummy_{idx}'},
        }


def collate_fn(batch):
    """Custom collate function."""
    return {
        'video': torch.stack([s['video'] for s in batch]),
        'queries': {
            'u': torch.stack([s['queries']['u'] for s in batch]),
            'v': torch.stack([s['queries']['v'] for s in batch]),
            't_src': torch.stack([s['queries']['t_src'] for s in batch]),
            't_tgt': torch.stack([s['queries']['t_tgt'] for s in batch]),
            't_cam': torch.stack([s['queries']['t_cam'] for s in batch]),
        },
        'targets': {
            'xyz': torch.stack([s['targets']['xyz'] for s in batch]),
            'uv': torch.stack([s['targets']['uv'] for s in batch]),
            'visibility': torch.stack([s['targets']['visibility'] for s in batch]),
        },
        'cameras': {
            'intrinsics': torch.stack([s['cameras']['intrinsics'] for s in batch]),
            'extrinsics': torch.stack([s['cameras']['extrinsics'] for s in batch]),
        },
        'metadata': [s['metadata'] for s in batch],
    }


def test_training_setup():
    """Test training setup."""
    print("=" * 80)
    print("Testing Training Setup with Dummy Data")
    print("=" * 80)

    # Load configs
    print("\n1. Loading configurations...")
    model_config_path = Path(__file__).parent.parent / 'configs' / 'model' / 'vit_b_tiny.yaml'
    model_config = OmegaConf.load(model_config_path)

    training_config_path = Path(__file__).parent.parent / 'configs' / 'training' / 'debug.yaml'
    training_config = OmegaConf.load(training_config_path)
    print("✓ Configs loaded")

    # Build model
    print("\n2. Building model...")
    model = build_d4rt_model(model_config)
    print("✓ Model built")

    # Build loss
    print("\n3. Building loss function...")
    loss_fn = build_composite_loss(training_config.training)
    print("✓ Loss function built")

    # Create dummy dataloaders
    print("\n4. Creating dummy dataloaders...")
    train_dataset = DummyDataset(num_samples=20, num_frames=24, resolution=(128, 128), num_queries=64)
    val_dataset = DummyDataset(num_samples=5, num_frames=24, resolution=(128, 128), num_queries=64)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n5. Using device: {device}")

    # Modify training config for quick test
    training_config.training.max_steps = 10
    training_config.training.log_every_n_steps = 2
    training_config.training.val_every_n_steps = 5
    training_config.training.save_every_n_steps = 100  # Don't save during test
    training_config.training.save_dir = 'test_checkpoints'

    # Build trainer
    print("\n6. Initializing trainer...")
    trainer = D4RTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=OmegaConf.to_container(training_config.training, resolve=True),
        device=device,
        logger=None,  # No logging for test
        distributed=False,
        local_rank=0,
    )
    print("✓ Trainer initialized")

    # Run training for a few steps
    print("\n7. Running training for 10 steps...")
    print("=" * 80)

    try:
        trainer.train()
        print("\n✓ Training completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("\n" + "=" * 80)
    print("Testing Checkpoint Save/Load")
    print("=" * 80)

    from d4rt.training import CheckpointManager

    # Create temporary checkpoint directory
    import tempfile
    temp_dir = tempfile.mkdtemp()

    print(f"\nUsing temp directory: {temp_dir}")

    # Build simple model
    model_config_path = Path(__file__).parent.parent / 'configs' / 'model' / 'vit_b_tiny.yaml'
    model_config = OmegaConf.load(model_config_path)
    model = build_d4rt_model(model_config)

    # Build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        save_dir=temp_dir,
        keep_last_n=2,
    )

    # Save checkpoint
    print("\n1. Saving checkpoint...")
    metrics = {'loss': 1.5, 'val/loss_total': 1.8}
    checkpoint_path = checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=100,
        metrics=metrics,
    )
    print(f"✓ Checkpoint saved: {checkpoint_path}")

    # Load checkpoint
    print("\n2. Loading checkpoint...")
    checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
    )
    print(f"✓ Checkpoint loaded, step: {checkpoint['step']}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"✓ Cleaned up temp directory")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("D4RT TRAINING INFRASTRUCTURE TESTS")
    print("=" * 80)

    tests = [
        ("Training Setup", test_training_setup),
        ("Checkpoint Save/Load", test_checkpoint_save_load),
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
