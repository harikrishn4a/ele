"""Data loading and augmentation pipeline for AIGI detection."""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, List
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import cv2


class AIGIDataset(Dataset):
    """
    Load real and AI-generated images from directory structure:
    data/
      ├── train/
      │   ├── real/         (real images)
      │   └── fake/         (AI-generated images)
      └── test/
          ├── real/
          └── fake/
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        augmentation: str = "none",
        image_size: int = 384,
        return_path: bool = False,
    ):
        """
        Args:
            root_dir: path to data/ folder
            split: "train" or "test"
            augmentation: "none", "simple", "advanced", or "pairwise"
            image_size: resize to this size
            return_path: return image path along with (image, label)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.augmentation = augmentation
        self.image_size = image_size
        self.return_path = return_path
        
        self.real_paths = []
        self.fake_paths = []
        self._load_paths()
        
    def _load_paths(self):
        """Load image paths from directory structure."""
        split_dir = self.root_dir / self.split
        
        # Load real images
        real_dir = split_dir / "real"
        if real_dir.exists():
            self.real_paths = sorted([
                p for p in real_dir.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ])
        
        # Load fake images
        fake_dir = split_dir / "fake"
        if fake_dir.exists():
            self.fake_paths = sorted([
                p for p in fake_dir.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ])
        
        print(f"[{self.split}] Found {len(self.real_paths)} real, {len(self.fake_paths)} fake images")
    
    def __len__(self):
        return len(self.real_paths) + len(self.fake_paths)
    
    def __getitem__(self, idx):
        # Alternate between real (label=1) and fake (label=0) or access by index
        if idx < len(self.real_paths):
            image_path = self.real_paths[idx]
            label = 1  # Real
        else:
            image_path = self.fake_paths[idx - len(self.real_paths)]
            label = 0  # Fake
        
        image = self._load_image(image_path)
        image = self._preprocess(image)
        
        if self.return_path:
            return image, label, str(image_path)
        return image, label
    
    def _load_image(self, path):
        """Load image as RGB."""
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a black image as fallback
            return Image.new("RGB", (self.image_size, self.image_size))
    
    def _preprocess(self, image):
        """Apply preprocessing and augmentation."""
        # Resize using crop (not squish) if augmentation is on
        if self.augmentation != "none":
            image = self._crop_resize(image, self.image_size)
        else:
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Apply augmentation based on mode
        if self.augmentation == "simple":
            image = self._augment_simple(image)
        elif self.augmentation == "advanced":
            image = self._augment_advanced(image)
        elif self.augmentation == "pairwise":
            # Return both clean and augmented (handled by caller)
            pass
        
        # Convert to tensor and normalize
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)
        
        return image
    
    def _crop_resize(self, image, size):
        """Crop (don't squish) then resize."""
        w, h = image.size
        min_side = min(w, h)
        
        # Center crop to square
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        image = image.crop((left, top, left + min_side, top + min_side))
        
        # Resize
        image = image.resize((size, size), Image.BILINEAR)
        return image
    
    def _augment_simple(self, image):
        """Simple augmentation: ColorJitter + RandomRotation."""
        # ColorJitter: brightness, contrast, saturation
        enhancer_b = ImageEnhance.Brightness(image)
        image = enhancer_b.enhance(random.uniform(0.8, 1.2))
        
        enhancer_c = ImageEnhance.Contrast(image)
        image = enhancer_c.enhance(random.uniform(0.8, 1.2))
        
        enhancer_s = ImageEnhance.Color(image)
        image = enhancer_s.enhance(random.uniform(0.8, 1.2))
        
        # Random rotation (±15 degrees)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, expand=False)
        
        return image
    
    def _augment_advanced(self, image):
        """Advanced augmentation: degradation pipeline (JPEG, blur, resize, noise, crop, jitter)."""
        # Apply with high probability
        if random.random() > 0.95:  # 95% of the time, apply degradations
            return image
        
        image_np = np.array(image)
        
        # Randomly select 1-3 transformations
        num_transforms = random.randint(1, 3)
        transforms_to_apply = random.sample(
            ["jpeg", "blur", "resize", "noise", "crop", "jitter"],
            min(num_transforms, 6)
        )
        
        for transform_name in transforms_to_apply:
            image_np = self._apply_degradation(image_np, transform_name)
        
        image = Image.fromarray(image_np)
        return image
    
    def _apply_degradation(self, image_np, transform_name):
        """Apply a single degradation transform."""
        h, w = image_np.shape[:2]
        
        if transform_name == "jpeg":
            # JPEG compression at random quality
            quality = random.choice([30, 50, 70, 90])
            image_pil = Image.fromarray(image_np)
            import io
            buf = io.BytesIO()
            image_pil.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            image_pil = Image.open(buf)
            image_np = np.array(image_pil)
        
        elif transform_name == "blur":
            # Gaussian blur
            sigma = random.choice([0.5, 1.0, 2.0])
            image_pil = Image.fromarray(image_np)
            radius = int(sigma * 2)
            image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=radius))
            image_np = np.array(image_pil)
        
        elif transform_name == "resize":
            # Downscale then upscale
            scale = random.choice([0.5, 0.25])
            new_h, new_w = int(h * scale), int(w * scale)
            image_pil = Image.fromarray(image_np)
            image_pil = image_pil.resize((new_w, new_h), Image.BILINEAR)
            image_pil = image_pil.resize((w, h), Image.BILINEAR)
            image_np = np.array(image_pil)
        
        elif transform_name == "noise":
            # Gaussian noise
            sigma = random.choice([0.02, 0.05, 0.10])
            noise = np.random.normal(0, sigma * 255, image_np.shape)
            image_np = np.clip(image_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        elif transform_name == "crop":
            # Center crop (80% of image)
            crop_size = 0.8
            crop_h, crop_w = int(h * crop_size), int(w * crop_size)
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
            image_np = image_np[top:top+crop_h, left:left+crop_w]
            # Resize back to original
            image_pil = Image.fromarray(image_np)
            image_pil = image_pil.resize((w, h), Image.BILINEAR)
            image_np = np.array(image_pil)
        
        elif transform_name == "jitter":
            # Color jitter
            image_pil = Image.fromarray(image_np)
            image_np = self._color_jitter(image_np, brightness=0.2, contrast=0.2, saturation=0.2)
        
        return image_np
    
    def _color_jitter(self, image_np, brightness=0.2, contrast=0.2, saturation=0.2):
        """Apply color jitter (brightness, contrast, saturation)."""
        image_pil = Image.fromarray(image_np)
        
        # Brightness
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image_pil)
            image_pil = enhancer.enhance(random.uniform(1 - brightness, 1 + brightness))
        
        # Contrast
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(image_pil)
            image_pil = enhancer.enhance(random.uniform(1 - contrast, 1 + contrast))
        
        # Saturation
        if random.random() < 0.5:
            enhancer = ImageEnhance.Color(image_pil)
            image_pil = enhancer.enhance(random.uniform(1 - saturation, 1 + saturation))
        
        return np.array(image_pil)


def create_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    augmentation: str = "none",
    shuffle: bool = True,
    return_paths: bool = False,
) -> DataLoader:
    """Create a DataLoader for AIGI detection."""
    dataset = AIGIDataset(
        root_dir=root_dir,
        split=split,
        augmentation=augmentation,
        return_path=return_paths,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )
