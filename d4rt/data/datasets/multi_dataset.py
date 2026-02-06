"""Multi-dataset loader for D4RT training.

Combines multiple datasets (Kubric, PointOdyssey, etc.) with weighted sampling.
"""

from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .kubric import KubricDataset
from .pointodyssey import PointOdysseyDataset


class MultiDataset(Dataset):
    """Combined dataset that samples from multiple sources."""

    def __init__(
        self,
        datasets: List[Dataset],
        weights: List[float] = None,
    ):
        """Initialize multi-dataset.

        Args:
            datasets: List of dataset instances
            weights: Sampling weights for each dataset (normalized automatically)
        """
        self.datasets = datasets
        self.dataset_sizes = [len(d) for d in datasets]
        self.total_size = sum(self.dataset_sizes)

        # Compute cumulative indices for mapping
        self.cumulative_sizes = []
        cumsum = 0
        for size in self.dataset_sizes:
            self.cumulative_sizes.append(cumsum)
            cumsum += size

        # Weights for sampling
        if weights is None:
            weights = [1.0] * len(datasets)
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]

        # Create sample weights for WeightedRandomSampler
        self.sample_weights = []
        for i, (dataset, weight) in enumerate(zip(datasets, self.weights)):
            # Weight per sample = dataset_weight / dataset_size
            sample_weight = weight / len(dataset) if len(dataset) > 0 else 0
            self.sample_weights.extend([sample_weight] * len(dataset))

        print(f"MultiDataset: {len(datasets)} datasets, {self.total_size} total samples")
        for i, (d, w) in enumerate(zip(datasets, self.weights)):
            print(f"  [{i}] {d.__class__.__name__}: {len(d)} samples, weight={w:.2f}")

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which dataset this index belongs to
        for i, (cumsum, size) in enumerate(zip(self.cumulative_sizes, self.dataset_sizes)):
            if idx < cumsum + size:
                local_idx = idx - cumsum
                return self.datasets[i][local_idx]

        raise IndexError(f"Index {idx} out of range for MultiDataset of size {self.total_size}")

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Get a weighted random sampler for this dataset."""
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=self.total_size,
            replacement=True,
        )


def create_multi_dataloaders(
    config: Dict,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create multi-dataset train and val dataloaders.

    Args:
        config: Configuration with dataset list and weights
        num_workers: Number of dataloader workers

    Returns:
        train_loader, val_loader
    """
    dataset_configs = config.get("datasets", [])
    num_frames = config.get("num_frames", 24)
    resolution = tuple(config.get("resolution", [256, 256]))
    num_queries = config.get("num_queries", 64)
    batch_size = config.get("batch_size", 1)

    train_datasets = []
    val_datasets = []
    train_weights = []
    val_weights = []

    for ds_config in dataset_configs:
        ds_name = ds_config.get("name")
        ds_weight = ds_config.get("weight", 1.0)
        ds_dir = ds_config.get("data_dir")

        if ds_name == "kubric":
            train_ds = KubricDataset(
                data_dir=ds_dir,
                split="train",
                num_frames=num_frames,
                resolution=resolution,
                num_queries=num_queries,
            )
            val_ds = KubricDataset(
                data_dir=ds_dir,
                split="val",
                num_frames=num_frames,
                resolution=resolution,
                num_queries=num_queries,
            )
        elif ds_name == "pointodyssey":
            # Check for sample split
            split = ds_config.get("train_split", "train")
            val_split = ds_config.get("val_split", "val")

            train_ds = PointOdysseyDataset(
                data_dir=ds_dir,
                split=split,
                num_frames=num_frames,
                resolution=resolution,
                num_queries=num_queries,
            )
            val_ds = PointOdysseyDataset(
                data_dir=ds_dir,
                split=val_split,
                num_frames=num_frames,
                resolution=resolution,
                num_queries=num_queries,
            )
        else:
            raise ValueError(f"Unknown dataset: {ds_name}")

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        train_weights.append(ds_weight)
        val_weights.append(ds_weight)

    # Create multi-datasets
    train_multi = MultiDataset(train_datasets, train_weights)
    val_multi = MultiDataset(val_datasets, val_weights)

    # Create dataloaders with weighted sampling for training
    train_loader = DataLoader(
        train_multi,
        batch_size=batch_size,
        sampler=train_multi.get_weighted_sampler(),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_multi,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
