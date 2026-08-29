#!/usr/bin/env python3
"""
Download and prepare training/test data for AIGI detection.

This script provides instructions for obtaining public datasets.
Automatic download is limited; most require manual setup.

Recommended datasets:
  - CIFAKE: 10K balanced real/fake (GAN-based)
  - GenImage: 100K (diverse modern generators)
  - OpenFake: 4M (real social media reosts)
"""

import os
import json
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory_structure(root_dir="data"):
    """Create expected directory structure."""
    root = Path(root_dir)
    
    for split in ["train", "test"]:
        for cls in ["real", "fake"]:
            (root / split / cls).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure in {root_dir}/")


def prepare_cifake(dataset_path="~/Downloads/CIFAKE", data_root="data"):
    """
    Prepare CIFAKE dataset.
    
    Download from: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
    Extract to ~/Downloads/CIFAKE/
    """
    dataset_path = Path(dataset_path).expanduser()
    data_root = Path(data_root)
    
    if not dataset_path.exists():
        logger.warning(f"CIFAKE not found at {dataset_path}")
        logger.info("Download from: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        return False
    
    logger.info("Processing CIFAKE...")
    
    # Typical structure: CIFAKE/real/ and CIFAKE/synthetic/
    for split_name, split_dir in [("train", "train"), ("test", "test")]:
        # Copy real
        src_real = dataset_path / f"{split_name}" / "real"
        dst_real = data_root / split_name / "real"
        if src_real.exists():
            copy_images(src_real, dst_real)
        
        # Copy fake
        src_fake = dataset_path / f"{split_name}" / "synthetic"
        dst_fake = data_root / split_name / "fake"
        if src_fake.exists():
            copy_images(src_fake, dst_fake)
    
    logger.info("CIFAKE processed")
    return True


def prepare_genimage(dataset_path="~/Downloads/GenImage", data_root="data"):
    """
    Prepare GenImage dataset.
    
    Download from: https://github.com/donghao51/GenImage
    """
    dataset_path = Path(dataset_path).expanduser()
    data_root = Path(data_root)
    
    if not dataset_path.exists():
        logger.warning(f"GenImage not found at {dataset_path}")
        logger.info("Download from: https://github.com/donghao51/GenImage")
        return False
    
    logger.info("Processing GenImage...")
    
    # GenImage structure: GenImage/images/{real,fake}/
    src_real = dataset_path / "images" / "real"
    src_fake = dataset_path / "images" / "fake"
    
    if src_real.exists():
        dst_real = data_root / "train" / "real"
        copy_images(src_real, dst_real)
    
    if src_fake.exists():
        dst_fake = data_root / "train" / "fake"
        copy_images(src_fake, dst_fake)
    
    logger.info("GenImage processed")
    return True


def copy_images(src_dir, dst_dir, max_images=None):
    """Copy images from src to dst."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in src_dir.rglob("*") if p.suffix.lower() in image_exts]
    
    if max_images:
        images = images[:max_images]
    
    for i, src_file in enumerate(images):
        try:
            dst_file = dst_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
        except Exception as e:
            logger.warning(f"Failed to copy {src_file}: {e}")
    
    logger.info(f"Copied {len(images)} images from {src_dir} to {dst_dir}")


def create_manifest(data_root="data"):
    """Create a manifest of available data."""
    data_root = Path(data_root)
    
    manifest = {"datasets": {}}
    
    for split in ["train", "test"]:
        split_dir = data_root / split
        real_count = len(list((split_dir / "real").rglob("*.*")))
        fake_count = len(list((split_dir / "fake").rglob("*.*")))
        
        manifest["datasets"][split] = {
            "real": real_count,
            "fake": fake_count,
            "total": real_count + fake_count,
        }
    
    manifest_path = data_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest saved to {manifest_path}")
    print("\nData Summary:")
    print(json.dumps(manifest, indent=2))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare datasets for AIGI detection")
    parser.add_argument("--cifake-path", default="~/Downloads/CIFAKE")
    parser.add_argument("--genimage-path", default="~/Downloads/GenImage")
    parser.add_argument("--data-root", default="data")
    
    args = parser.parse_args()
    
    # Create structure
    create_directory_structure(args.data_root)
    
    # Try to process datasets
    logger.info("\nAttempting to process datasets...\n")
    
    cifake_ok = prepare_cifake(args.cifake_path, args.data_root)
    genimage_ok = prepare_genimage(args.genimage_path, args.data_root)
    
    if not (cifake_ok or genimage_ok):
        logger.error("No datasets found!")
        print("\nTo set up data:")
        print("1. Download CIFAKE from Kaggle:")
        print("   https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        print("   Extract to ~/Downloads/CIFAKE/")
        print("\n2. Or download GenImage:")
        print("   git clone https://github.com/donghao51/GenImage")
        print("   Extract to ~/Downloads/GenImage/")
        print("\n3. Then run:")
        print("   python scripts/download_data.py")
        return
    
    # Create manifest
    create_manifest(args.data_root)
    
    logger.info("\nDone! You can now train with:")
    logger.info(f"  python src/train.py --data-dir {args.data_root}")


if __name__ == "__main__":
    main()
