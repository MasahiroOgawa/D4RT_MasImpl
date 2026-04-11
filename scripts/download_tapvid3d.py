#!/usr/bin/env python3
"""Download TAP-Vid-3D dataset (DriveTrack minival + PStudio full).

Usage:
    python scripts/download_tapvid3d.py
    python scripts/download_tapvid3d.py --subset drivetrack  # DriveTrack minival only
    python scripts/download_tapvid3d.py --subset pstudio     # PStudio full only
"""

import argparse
import io
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

GCS_BASE = "https://storage.googleapis.com/dm-tapnet/tapvid3d/release_files/v1.0"
PSTUDIO_DATA_URL = "https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip"


def get_file_lists():
    """Get file lists from the official tapvid3d splits."""
    splits_url = "https://raw.githubusercontent.com/google-deepmind/tapnet/main/tapnet/tapvid3d/splits/tapvid3d_splits.py"
    print("Fetching file lists from official tapnet repo...")
    resp = requests.get(splits_url)
    resp.raise_for_status()

    namespace = {}
    exec(resp.text, namespace)

    drivetrack_minival = namespace["MINIVAL_FILES"]["drivetrack"]
    pstudio_all = sorted(
        set(namespace["MINIVAL_FILES"]["pstudio"] + namespace["FULL_EVAL_FILES"]["pstudio"])
    )
    return drivetrack_minival, pstudio_all


def download_file(url, dest, desc=None):
    """Download a single file with progress."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(dest, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name, leave=False) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_file_simple(url, dest):
    """Download without progress bar (for parallel downloads)."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


def download_drivetrack_minival(output_dir):
    """Download DriveTrack minival (50 self-contained .npz files, ~1.5GB)."""
    drivetrack_files, _ = get_file_lists()
    dest_dir = output_dir / "drivetrack"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Skip already downloaded files
    to_download = []
    for fname in drivetrack_files:
        dest = dest_dir / fname
        if dest.exists():
            continue
        to_download.append(fname)

    if not to_download:
        print(f"DriveTrack minival: all {len(drivetrack_files)} files already present.")
        return

    print(f"DriveTrack minival: downloading {len(to_download)}/{len(drivetrack_files)} files...")

    with tqdm(total=len(to_download), desc="DriveTrack minival") as pbar:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    download_file_simple,
                    f"{GCS_BASE}/drivetrack/{fname}",
                    dest_dir / fname,
                ): fname
                for fname in to_download
            }
            for future in as_completed(futures):
                fname = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"\nFailed to download {fname}: {e}")
                pbar.update(1)

    print(f"DriveTrack minival: {len(drivetrack_files)} files in {dest_dir}")


def download_pstudio_full(output_dir):
    """Download PStudio full (156 files, ~1.5GB total).

    PStudio requires merging GCS annotations with video frames from data.zip.
    """
    _, pstudio_files = get_file_lists()
    dest_dir = output_dir / "pstudio"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = [f for f in pstudio_files if (dest_dir / f).exists()]
    if len(existing) == len(pstudio_files):
        # Verify they have images (not just annotations)
        sample = np.load(dest_dir / pstudio_files[0], allow_pickle=True)
        if "images_jpeg_bytes" in sample:
            print(f"PStudio full: all {len(pstudio_files)} files already present.")
            return

    tmp_dir = output_dir / "_pstudio_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download annotation .npz files from GCS
    print(f"PStudio: downloading {len(pstudio_files)} annotation files from GCS...")
    annotations_dir = tmp_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    to_download = [f for f in pstudio_files if not (annotations_dir / f).exists()]
    if to_download:
        with tqdm(total=len(to_download), desc="PStudio annotations") as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(
                        download_file_simple,
                        f"{GCS_BASE}/pstudio/{fname}",
                        annotations_dir / fname,
                    ): fname
                    for fname in to_download
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"\nFailed: {futures[future]}: {e}")
                    pbar.update(1)

    # Step 2: Download and extract data.zip (Dynamic3DGaussians video frames)
    data_dir = tmp_dir / "data"
    data_zip = tmp_dir / "data.zip"

    if not data_dir.exists():
        if not data_zip.exists():
            print("PStudio: downloading Dynamic3DGaussians data.zip (~562MB)...")
            download_file(PSTUDIO_DATA_URL, data_zip, desc="data.zip")

        print("PStudio: extracting data.zip...")
        with zipfile.ZipFile(data_zip, "r") as zf:
            zf.extractall(tmp_dir)
        print(f"Extracted to {data_dir}")
        # Remove zip to save space
        data_zip.unlink()
    else:
        print("PStudio: data directory already extracted.")

    # Step 3: Merge annotations with video frames
    print("PStudio: merging annotations with video frames...")
    for fname in tqdm(pstudio_files, desc="Merging"):
        output_path = dest_dir / fname
        if output_path.exists():
            sample = np.load(output_path, allow_pickle=True)
            if "images_jpeg_bytes" in sample:
                continue

        # Parse filename: {sequence}_{camera_id}.npz
        stem = Path(fname).stem  # e.g. "basketball_5"
        parts = stem.rsplit("_", 1)
        sequence = parts[0]  # e.g. "basketball"
        camera_id = parts[1]  # e.g. "5"

        # Load annotation
        ann_path = annotations_dir / fname
        ann_data = dict(np.load(ann_path, allow_pickle=True))

        # Load video frames
        frames_dir = data_dir / sequence / "ims" / camera_id
        if not frames_dir.exists():
            print(f"\nWarning: frames not found at {frames_dir}, skipping {fname}")
            continue

        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            frame_paths = sorted(frames_dir.glob("*.png"))

        jpeg_bytes_list = []
        for frame_path in frame_paths:
            if frame_path.suffix == ".jpg":
                # Already JPEG - just read raw bytes
                jpeg_bytes_list.append(frame_path.read_bytes())
            else:
                # Convert to JPEG
                from PIL import Image

                img = Image.open(frame_path)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                jpeg_bytes_list.append(buf.getvalue())

        ann_data["images_jpeg_bytes"] = np.array(jpeg_bytes_list, dtype=object)

        # Save merged npz
        np.savez(output_path, **ann_data)

    print("PStudio: cleaning up temporary files...")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"PStudio full: {len(pstudio_files)} files in {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download TAP-Vid-3D dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tapvid3d",
        help="Output directory (default: data/tapvid3d)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["drivetrack", "pstudio", "all"],
        default="all",
        help="Which subset to download (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.subset in ("drivetrack", "all"):
        download_drivetrack_minival(output_dir)

    if args.subset in ("pstudio", "all"):
        download_pstudio_full(output_dir)

    # Summary
    print("\n" + "=" * 50)
    print("Download complete!")
    print(f"Dataset location: {output_dir}")
    for subset in ("drivetrack", "pstudio"):
        subset_dir = output_dir / subset
        if subset_dir.exists():
            count = len(list(subset_dir.glob("*.npz")))
            print(f"  {subset}: {count} files")
    print("=" * 50)


if __name__ == "__main__":
    main()
