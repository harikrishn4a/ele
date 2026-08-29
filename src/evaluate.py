"""Evaluation on transform × severity grid."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score

from src.dataset import AIGIDataset
from src.model import create_model, resolve_device
import torchvision.transforms.functional as TF


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformEvaluator:
    """Evaluate detector robustness across transforms and severity levels."""
    
    def __init__(
        self,
        model_path: str,
        backbone: str = "sigclip",
        device: str = "cuda",
    ):
        self.device = device
        self.backbone = backbone
        
        # Load model
        self.model = create_model(backbone=backbone, device=device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        logger.info(f"Loaded model from {model_path}")
    
    def evaluate_dataset(
        self,
        dataset,
        transform_fn=None,
        transform_name="clean",
        severity=None,
    ) -> Dict:
        """Evaluate on dataset with optional transform."""
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc=f"{transform_name} (sev={severity})"):
                image, label = dataset[idx]
                
                # Apply transform if provided
                if transform_fn is not None:
                    image_pil = TF.to_pil_image(image)
                    image = transform_fn(image_pil)
                    image = TF.to_tensor(image)
                
                # Normalize
                image = TF.normalize(
                    image,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                
                # Forward
                image = image.unsqueeze(0).to(self.device)
                logits = self.model(image)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(fake)
                
                all_probs.append(probs[0])
                all_labels.append(label)
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, all_probs > 0.5)
        
        return {
            "transform": transform_name,
            "severity": severity,
            "auc": auc,
            "accuracy": acc,
            "probs": all_probs,
            "labels": all_labels,
        }
    
    def evaluate_grid(
        self,
        dataset_path: str,
        output_csv: str = "results/eval_grid.csv",
    ):
        """Evaluate on full transform × severity grid."""
        
        dataset = AIGIDataset(
            root_dir=dataset_path,
            split="test",
            augmentation="none",
        )
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        results = []
        
        # Clean baseline
        logger.info("Evaluating clean images...")
        result = self.evaluate_dataset(dataset, transform_fn=None, transform_name="clean")
        results.append(result)
        logger.info(f"  Clean AUC: {result['auc']:.4f}")
        
        # JPEG compression
        logger.info("Evaluating JPEG compression...")
        for quality in [90, 70, 50, 30]:
            transform_fn = lambda img: self._apply_jpeg(img, quality)
            result = self.evaluate_dataset(
                dataset,
                transform_fn=transform_fn,
                transform_name="jpeg",
                severity=quality,
            )
            results.append(result)
            logger.info(f"  JPEG q={quality} AUC: {result['auc']:.4f}")
        
        # Gaussian blur
        logger.info("Evaluating Gaussian blur...")
        for sigma in [0.5, 1.0, 2.0]:
            transform_fn = lambda img, s=sigma: self._apply_blur(img, s)
            result = self.evaluate_dataset(
                dataset,
                transform_fn=transform_fn,
                transform_name="blur",
                severity=sigma,
            )
            results.append(result)
            logger.info(f"  Blur σ={sigma} AUC: {result['auc']:.4f}")
        
        # Resize + upscale
        logger.info("Evaluating resize...")
        for scale in [0.5, 0.25]:
            transform_fn = lambda img, sc=scale: self._apply_resize(img, sc)
            result = self.evaluate_dataset(
                dataset,
                transform_fn=transform_fn,
                transform_name="resize",
                severity=scale,
            )
            results.append(result)
            logger.info(f"  Resize {scale}× AUC: {result['auc']:.4f}")
        
        # Gaussian noise
        logger.info("Evaluating Gaussian noise...")
        for sigma in [0.02, 0.05, 0.10]:
            transform_fn = lambda img, s=sigma: self._apply_noise(img, s)
            result = self.evaluate_dataset(
                dataset,
                transform_fn=transform_fn,
                transform_name="noise",
                severity=sigma,
            )
            results.append(result)
            logger.info(f"  Noise σ={sigma} AUC: {result['auc']:.4f}")
        
        # Center crop
        logger.info("Evaluating crop...")
        transform_fn = lambda img: self._apply_crop(img, 0.8)
        result = self.evaluate_dataset(
            dataset,
            transform_fn=transform_fn,
            transform_name="crop",
            severity="80%",
        )
        results.append(result)
        logger.info(f"  Crop 80% AUC: {result['auc']:.4f}")
        
        # Color jitter
        logger.info("Evaluating color jitter...")
        transform_fn = lambda img: self._apply_jitter(img)
        result = self.evaluate_dataset(
            dataset,
            transform_fn=transform_fn,
            transform_name="jitter",
            severity="±20%",
        )
        results.append(result)
        logger.info(f"  Jitter ±20% AUC: {result['auc']:.4f}")
        
        # Save to CSV
        df = pd.DataFrame([
            {
                "transform": r["transform"],
                "severity": r["severity"],
                "auc": r["auc"],
                "accuracy": r["accuracy"],
            }
            for r in results
        ])
        
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved results to {output_csv}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        clean_auc = df[df["transform"] == "clean"]["auc"].values[0]
        degraded_auc = df[df["transform"] != "clean"]["auc"].mean()
        gap = clean_auc - degraded_auc
        
        print(f"Clean AUC:       {clean_auc:.4f}")
        print(f"Degraded Avg AUC: {degraded_auc:.4f}")
        print(f"Gap:              {gap:.4f}")
        print("\n" + df.to_string(index=False))
        
        return df
    
    # Transform functions
    def _apply_jpeg(self, image, quality):
        import io
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        from PIL import Image
        return Image.open(buf)
    
    def _apply_blur(self, image, sigma):
        from PIL import ImageFilter
        radius = int(sigma * 2)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _apply_resize(self, image, scale):
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h))
        image = image.resize((w, h))
        return image
    
    def _apply_noise(self, image, sigma):
        import numpy as np
        image_np = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, sigma * 255, image_np.shape)
        image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)
        from PIL import Image
        return Image.fromarray(image_np)
    
    def _apply_crop(self, image, crop_ratio):
        w, h = image.size
        crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        image = image.crop((left, top, left + crop_w, top + crop_h))
        image = image.resize((w, h))
        return image
    
    def _apply_jitter(self, image):
        from PIL import ImageEnhance
        import random
        
        # Brightness
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Contrast
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Saturation
        if random.random() < 0.5:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        return image


def main():
    parser = argparse.ArgumentParser(description="Evaluate AIGI detector")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--backbone", default="sigclip")
    parser.add_argument("--dataset", default="data")
    parser.add_argument("--output", default="results/eval.csv")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="auto picks cuda, then Apple MPS, then CPU",
    )
    args = parser.parse_args()
    
    evaluator = TransformEvaluator(
        model_path=args.model,
        backbone=args.backbone,
        device=str(resolve_device(args.device)),
    )
    
    evaluator.evaluate_grid(
        dataset_path=args.dataset,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
