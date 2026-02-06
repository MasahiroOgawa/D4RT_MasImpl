"""Training script for D4RT."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from d4rt.models import build_d4rt_model
from d4rt.losses import build_composite_loss
from d4rt.training import D4RTTrainer
from d4rt.data.datasets.kubric import create_kubric_dataloaders
from d4rt.data.datasets.multi_dataset import create_multi_dataloaders


def setup_distributed():
    """Setup distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
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


def setup_logger(config: DictConfig, local_rank: int):
    """Setup logger (WandB or TensorBoard)."""
    if not config.logging.get("use_wandb", False) or local_rank != 0:
        return None

    try:
        import wandb

        wandb.init(
            project=config.logging.get("wandb_project", "d4rt"),
            entity=config.logging.get("wandb_entity", None),
            name=config.get("experiment_name", "d4rt_experiment"),
            config=OmegaConf.to_container(config, resolve=True),
        )

        return wandb

    except ImportError:
        print("WandB not installed, skipping logging")
        return None


@hydra.main(version_base=None, config_path="../configs/training", config_name="debug")
def main(config: DictConfig):
    """
    Main training function.

    Args:
        config: Hydra configuration
    """
    print("=" * 80)
    print("D4RT Training")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    # Setup distributed training
    distributed, rank, local_rank, world_size = setup_distributed()

    if distributed:
        print(f"Distributed training: rank {rank}/{world_size}, local_rank {local_rank}")

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda:0")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Set seed for reproducibility
    if config.get("seed", None):
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    # Build model
    print("\nBuilding model...")
    model_config_path = Path(__file__).parent.parent / "configs" / "model" / f"{config.model}.yaml"
    model_config = OmegaConf.load(model_config_path)

    model = build_d4rt_model(model_config)

    # Print model info
    if local_rank == 0:
        from d4rt.models import print_model_info

        print_model_info(model)

    # Build loss function
    print("\nBuilding loss function...")
    loss_fn = build_composite_loss(config.training)

    # Build dataloaders
    print("\nBuilding dataloaders...")

    # Check if multi-dataset is enabled
    multi_dataset_config = config.get("multi_dataset", {})
    if multi_dataset_config.get("enabled", False):
        print("Using multi-dataset training")
        md_config = {
            "datasets": OmegaConf.to_container(
                multi_dataset_config.get("datasets", []), resolve=True
            ),
            "num_frames": multi_dataset_config.get("num_frames", 24),
            "resolution": multi_dataset_config.get("resolution", [256, 256]),
            "num_queries": config.training.get("num_queries_per_step", 128),
            "batch_size": config.training.get("batch_size", 1),
        }
        train_loader, val_loader = create_multi_dataloaders(
            config=md_config,
            num_workers=config.training.get("num_workers", 0),
        )
    else:
        # Single dataset (Kubric)
        dataset_config = {
            "data_dir": config.dataset.get("data_dir", "data/kubric"),
            "num_frames": config.dataset.get("num_frames", 48),
            "resolution": config.dataset.get("resolution", [256, 256]),
            "num_queries": config.training.get("num_queries_per_step", 2048),
            "batch_size": config.training.get("batch_size", 4),
            "augmentation": config.dataset.get("augmentation", {}),
        }

        train_loader, val_loader = create_kubric_dataloaders(
            config=dataset_config,
            num_workers=config.training.get("num_workers", 4),
        )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Setup logger
    logger = setup_logger(config, local_rank)

    # Build trainer
    print("\nInitializing trainer...")

    # Merge training config with optimizer and scheduler configs
    trainer_config = OmegaConf.to_container(config.training, resolve=True)
    trainer_config["optimizer"] = OmegaConf.to_container(config.get("optimizer", {}), resolve=True)
    trainer_config["scheduler"] = OmegaConf.to_container(config.get("scheduler", {}), resolve=True)

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
        # Check if resuming from checkpoint
        resume_from = config.training.get("resume_from", None)
        trainer.train(resume_from=resume_from)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        if logger is not None and hasattr(logger, "finish"):
            logger.finish()

        if distributed:
            cleanup_distributed()

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
