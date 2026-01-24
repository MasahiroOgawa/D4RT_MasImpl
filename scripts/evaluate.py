"""Comprehensive evaluation script for D4RT models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from omegaconf import OmegaConf

from d4rt.evaluation import evaluate_checkpoint, print_results_table
from d4rt.data.datasets.kubric import create_kubric_dataloaders


def main():
    parser = argparse.ArgumentParser(description='Evaluate D4RT model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='kubric', help='Dataset name (kubric, etc.)')
    parser.add_argument('--data-dir', type=str, default='data/kubric', help='Data directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Dataset split')
    parser.add_argument('--tasks', type=str, nargs='+', default=['tracking', 'depth', 'pose'],
                       choices=['tracking', 'depth', 'pose'],
                       help='Tasks to evaluate')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum samples per task')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--num-frames', type=int, default=48, help='Number of frames')
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256],
                       help='Resolution (H W)')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("D4RT MODEL EVALUATION")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset} ({args.split} split)")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Device: {device}")

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")

    if args.dataset == 'kubric':
        # Create dataloader
        dataset_config = {
            'data_dir': args.data_dir,
            'num_frames': args.num_frames,
            'resolution': args.resolution,
            'num_queries': 128,  # For evaluation
            'batch_size': args.batch_size,
            'augmentation': {},  # No augmentation for evaluation
        }

        if args.split == 'val':
            _, val_loader = create_kubric_dataloaders(
                config=dataset_config,
                num_workers=args.num_workers,
            )
            dataloader = val_loader
        else:
            train_loader, _ = create_kubric_dataloaders(
                config=dataset_config,
                num_workers=args.num_workers,
            )
            dataloader = train_loader

        print(f"✓ Loaded {len(dataloader)} batches")

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Run evaluation
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataloader=dataloader,
        tasks=args.tasks,
        max_samples=args.max_samples,
        device=device,
    )

    # Print results
    print_results_table(results)

    # Save results
    from d4rt.evaluation import D4RTEvaluator
    evaluator = D4RTEvaluator(None, device=device)  # Dummy for save method
    evaluator.save_results(
        {
            'checkpoint': args.checkpoint,
            'dataset': args.dataset,
            'split': args.split,
            'tasks': args.tasks,
            'results': results,
        },
        save_path=args.output,
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
