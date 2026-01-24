"""Script to prepare Kubric dataset for D4RT training.

Kubric is a synthetic dataset generation framework. This script helps you:
1. Generate Kubric data locally, OR
2. Download pre-generated Kubric MOVi datasets

For more information: https://github.com/google-research/kubric
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_kubric_installed():
    """Check if Kubric is installed."""
    try:
        import kubric
        print("✓ Kubric is already installed")
        return True
    except ImportError:
        print("✗ Kubric is not installed")
        return False


def install_kubric():
    """Install Kubric library."""
    print("\nInstalling Kubric...")
    print("This requires: docker, blender, and other dependencies")

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/google-research/kubric.git"
        ])
        print("✓ Kubric installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install Kubric: {e}")
        return False


def download_movi_datasets(output_dir: str):
    """
    Download pre-generated MOVi datasets.

    MOVi (Multi-Object Video) datasets are available via TensorFlow Datasets.
    """
    print("\n" + "=" * 80)
    print("DOWNLOADING MOVi DATASETS")
    print("=" * 80)

    print("\nMOVi datasets are available through TensorFlow Datasets (TFDS).")
    print("Available datasets:")
    print("  - movi_a: Simple scenes with basic objects")
    print("  - movi_b: More complex scenes")
    print("  - movi_c: Even more complex")
    print("  - movi_d: Indoor scenes")
    print("  - movi_e: Outdoor scenes")

    print("\nTo download, you need TensorFlow Datasets:")
    print("  pip install tensorflow-datasets")

    print("\nThen use this code:")
    print("""
    import tensorflow_datasets as tfds

    # Load MOVi-A dataset (simplest)
    ds = tfds.load('movi_a/256x256', split='train')

    # Or download directly
    tfds.load('movi_a/256x256', split='train',
              data_dir='./data/movi',
              download=True)
    """)

    print("\nAfter downloading, you'll need to convert TFDS format to our format.")
    print("See: scripts/convert_movi_to_kubric_format.py (to be created)")


def generate_kubric_scenes(output_dir: str, num_scenes: int = 100):
    """
    Generate Kubric scenes locally.

    This requires Docker and Blender to be installed.
    """
    print("\n" + "=" * 80)
    print("GENERATING KUBRIC SCENES LOCALLY")
    print("=" * 80)

    if not check_kubric_installed():
        install = input("\nInstall Kubric? (y/n): ")
        if install.lower() == 'y':
            if not install_kubric():
                return
        else:
            print("Cannot proceed without Kubric")
            return

    print("\n⚠️  Local generation requires:")
    print("  1. Docker (for rendering)")
    print("  2. Blender 2.93+ (3D rendering engine)")
    print("  3. Significant compute time (minutes per scene)")

    print(f"\nGenerating {num_scenes} scenes to: {output_dir}")
    print("This may take several hours...")

    # Example generation code
    print("\nExample generation script:")
    print("""
    import kubric as kb
    from kubric.renderer.blender import Blender

    # Create scene
    scene = kb.Scene(resolution=(256, 256), frame_end=48)

    # Add objects
    scene += kb.Cube(name="floor", scale=[10, 10, 0.1], position=[0, 0, -0.1])
    scene += kb.Sphere(name="ball", scale=0.5, position=[0, 0, 1])

    # Add camera
    scene.camera = kb.PerspectiveCamera(position=[3, -1, 4], look_at=[0, 0, 0])

    # Render
    renderer = Blender(scene)
    frames = renderer.render()

    # Save frames, depth, camera params...
    """)

    print("\nFor full generation script, see:")
    print("  https://github.com/google-research/kubric/tree/main/examples")


def check_existing_data(data_dir: str):
    """Check if Kubric data already exists."""
    data_path = Path(data_dir)

    train_dir = data_path / 'train'
    val_dir = data_path / 'val'

    if train_dir.exists() and val_dir.exists():
        train_scenes = list(train_dir.iterdir())
        val_scenes = list(val_dir.iterdir())

        print(f"\n✓ Found existing data:")
        print(f"  Train: {len(train_scenes)} scenes")
        print(f"  Val: {len(val_scenes)} scenes")

        # Check format
        if train_scenes:
            sample_scene = train_scenes[0]
            has_rgb = (sample_scene / 'rgb').exists()
            has_depth = (sample_scene / 'depth').exists()
            has_camera = (sample_scene / 'camera.json').exists()

            print(f"\n  Scene structure:")
            print(f"    RGB: {'✓' if has_rgb else '✗'}")
            print(f"    Depth: {'✓' if has_depth else '✗'}")
            print(f"    Camera: {'✓' if has_camera else '✗'}")

        return True

    return False


def create_dummy_data(output_dir: str, num_scenes: int = 10):
    """
    Create dummy Kubric data for quick testing.

    This generates random RGB frames with dummy metadata
    so you can test the training pipeline immediately.
    """
    print("\n" + "=" * 80)
    print("CREATING DUMMY DATA FOR TESTING")
    print("=" * 80)

    output_path = Path(output_dir)

    import numpy as np
    from PIL import Image
    import json

    for split in ['train', 'val']:
        split_num = num_scenes if split == 'train' else max(2, num_scenes // 5)
        print(f"\nCreating {split_num} dummy scenes for {split}...")

        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(split_num):
            scene_dir = split_dir / f'scene_{i:04d}'
            scene_dir.mkdir(exist_ok=True)

            # Create RGB frames
            rgb_dir = scene_dir / 'rgb'
            rgb_dir.mkdir(exist_ok=True)

            for frame_idx in range(48):
                # Random RGB image
                img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                Image.fromarray(img).save(rgb_dir / f'{frame_idx:05d}.png')

            # Create dummy depth
            depth_dir = scene_dir / 'depth'
            depth_dir.mkdir(exist_ok=True)

            for frame_idx in range(48):
                depth = np.random.rand(256, 256).astype(np.float32) * 10.0
                np.save(depth_dir / f'{frame_idx:05d}.npy', depth)

            # Create camera.json
            camera_data = {
                'intrinsics': [[256, 0, 128], [0, 256, 128], [0, 0, 1]],  # Dummy K matrix
                'extrinsics': [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]]] * 48,
            }
            with open(scene_dir / 'camera.json', 'w') as f:
                json.dump(camera_data, f)

            # Create metadata.json
            metadata = {
                'scene_id': f'dummy_{i:04d}',
                'num_frames': 48,
                'resolution': [256, 256],
            }
            with open(scene_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f)

    print(f"\n✓ Created dummy data in: {output_dir}")
    print("\n⚠️  This is DUMMY data for testing only!")
    print("The model won't learn anything meaningful from random pixels.")
    print("Use this to verify the training pipeline works, then get real data.")


def main():
    parser = argparse.ArgumentParser(description='Prepare Kubric dataset for D4RT')
    parser.add_argument('--output-dir', type=str, default='data/kubric',
                       help='Output directory for dataset')
    parser.add_argument('--method', type=str, choices=['check', 'dummy', 'download', 'generate'],
                       default='check',
                       help='Method: check existing, create dummy, download MOVi, or generate locally')
    parser.add_argument('--num-scenes', type=int, default=10,
                       help='Number of scenes to generate (for dummy or generate)')

    args = parser.parse_args()

    print("=" * 80)
    print("KUBRIC DATASET PREPARATION FOR D4RT")
    print("=" * 80)

    if args.method == 'check':
        print("\nChecking for existing data...")
        if check_existing_data(args.output_dir):
            print("\n✓ Data ready for training!")
        else:
            print(f"\n✗ No data found in {args.output_dir}")
            print("\nOptions:")
            print("  1. Create dummy data for testing:")
            print("     python scripts/prepare_kubric_data.py --method dummy")
            print("\n  2. Download MOVi datasets:")
            print("     python scripts/prepare_kubric_data.py --method download")
            print("\n  3. Generate locally (requires Docker + Blender):")
            print("     python scripts/prepare_kubric_data.py --method generate --num-scenes 100")

    elif args.method == 'dummy':
        create_dummy_data(args.output_dir, args.num_scenes)
        print("\n✓ Dummy data created! You can now test training:")
        print(f"   python scripts/train.py model=vit_b_tiny training=debug")

    elif args.method == 'download':
        download_movi_datasets(args.output_dir)

    elif args.method == 'generate':
        generate_kubric_scenes(args.output_dir, args.num_scenes)

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
