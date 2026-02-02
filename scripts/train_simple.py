"""Simplified training script for D4RT (without Hydra complexity)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
import os
import argparse

from d4rt.models import build_d4rt_model
from d4rt.losses import build_composite_loss
from d4rt.training import D4RTTrainer
from d4rt.data.datasets.kubric import create_kubric_dataloaders


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
        )

        torch.cuda.set_device(local_rank)

        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_logger(use_wandb, wandb_project, wandb_entity, experiment_name, config, local_rank):
    """Setup logger (WandB or TensorBoard)."""
    if not use_wandb or local_rank != 0:
        return None

    try:
        import wandb

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
        )

        return wandb

    except ImportError:
        print("WandB not installed, skipping logging")
        return None


def main():
    parser = argparse.ArgumentParser(description='Train D4RT model')
    parser.add_argument('--model-config', type=str, default='configs/model/vit_b_tiny.yaml',
                       help='Path to model config')
    parser.add_argument('--training-config', type=str, default='configs/training/debug.yaml',
                       help='Path to training config')
    parser.add_argument('--data-dir', type=str, default='data/kubric',
                       help='Path to dataset')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--pretrained-encoder', type=str, default=None,
                       help='Path to pretrained encoder weights (e.g., VideoMAE)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    print("=" * 80)
    print("D4RT Training")
    print("=" * 80)

    # Setup distributed training
    distributed, rank, local_rank, world_size = setup_distributed()

    if distributed:
        print(f"Distributed training: rank {rank}/{world_size}, local_rank {local_rank}")

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}' if distributed else 'cuda:0')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load configs
    print(f"\nLoading model config from: {args.model_config}")
    model_config = OmegaConf.load(args.model_config)

    print(f"Loading training config from: {args.training_config}")
    training_config = OmegaConf.load(args.training_config)

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(model_config))
    print(OmegaConf.to_yaml(training_config))
    print("=" * 80)

    # Build model
    print("\nBuilding model...")
    model = build_d4rt_model(model_config)

    # Load pretrained encoder weights if specified
    if args.pretrained_encoder:
        print(f"\nLoading pretrained encoder weights from: {args.pretrained_encoder}")
        from d4rt.models.encoder import load_videomae_weights
        missing, unexpected = load_videomae_weights(model.encoder, args.pretrained_encoder)
        print(f"  Loaded pretrained weights (missing: {len(missing)}, skipped: {len(unexpected)})")

    # Print model info
    if local_rank == 0:
        from d4rt.models import print_model_info
        print_model_info(model)

    # Build loss function
    print("\nBuilding loss function...")
    loss_fn = build_composite_loss(training_config.training)

    # Build dataloaders
    print("\nBuilding dataloaders...")
    # Use model's expected input resolution, not training config's
    model_num_frames = model_config.encoder.input_resolution[0]
    model_resolution = model_config.encoder.input_resolution[1:3]

    dataset_config = {
        'data_dir': args.data_dir,
        'num_frames': model_num_frames,
        'resolution': model_resolution,
        'num_queries': training_config.training.get('num_queries_per_step', 128),
        'batch_size': training_config.training.get('batch_size', 1),
        'augmentation': {},  # Disable augmentation for testing
    }

    train_loader, val_loader = create_kubric_dataloaders(
        config=dataset_config,
        num_workers=training_config.training.get('num_workers', 0),
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Setup logger
    logger = setup_logger(
        use_wandb=training_config.logging.get('use_wandb', False),
        wandb_project=training_config.logging.get('wandb_project', 'd4rt'),
        wandb_entity=training_config.logging.get('wandb_entity', None),
        experiment_name=training_config.get('experiment_name', 'd4rt_experiment'),
        config={'model': model_config, 'training': training_config},
        local_rank=local_rank,
    )

    # Build trainer
    print("\nInitializing trainer...")
    # Merge training, optimizer, and scheduler configs for the Trainer
    trainer_config = OmegaConf.to_container(training_config.training, resolve=True)
    trainer_config['optimizer'] = OmegaConf.to_container(training_config.get('optimizer', {}), resolve=True)
    trainer_config['scheduler'] = OmegaConf.to_container(training_config.get('scheduler', {}), resolve=True)
    trainer = D4RTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=trainer_config,
        device=device,
        logger=logger,
        distributed=distributed,
        local_rank=local_rank,
    )

    # Start training
    print("\nStarting training...")
    print("=" * 80)

    try:
        trainer.train(resume_from=args.resume_from)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if logger is not None and hasattr(logger, 'finish'):
            logger.finish()

        if distributed:
            cleanup_distributed()

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
