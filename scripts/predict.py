#!/usr/bin/env python3
"""
Inference script: predict on a directory of images.

Usage:
    python scripts/predict.py \
      --image-dir ~/my_images \
      --model models/baseline.pt \
      --backbone sigclip \
      --output predictions.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model, resolve_device


def predict_directory(
    image_dir: str,
    model_path: str,
    backbone: str = "sigclip",
    output_json: str = "predictions.json",
    confidence_threshold: float = 0.5,
    device: str = "cuda",
):
    """
    Predict on all images in a directory.
    
    Args:
        image_dir: directory containing images
        model_path: path to trained model
        backbone: backbone name
        output_json: path to save predictions
        confidence_threshold: threshold for binary decision
        device: auto, cuda, mps, or cpu
    
    Returns:
        dict: predictions
    """
    # Setup
    device = resolve_device(device)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = create_model(backbone=backbone, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Find images
    image_dir = Path(image_dir)
    image_files = sorted([
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ])
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return {"predictions": []}
    
    print(f"Found {len(image_files)} images")
    
    # Predict
    predictions = []
    
    with torch.no_grad():
        for image_path in tqdm(image_files, desc="Predicting"):
            try:
                # Load image
                image = Image.open(image_path).convert("RGB")
                
                # Preprocess
                image = image.resize((384, 384), Image.BILINEAR)
                image_tensor = TF.to_tensor(image)
                image_tensor = TF.normalize(
                    image_tensor,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                image_tensor = image_tensor.unsqueeze(0).to(device)
                
                # Forward
                logits = model(image_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                
                prob_fake = probs[0].item()  # P(fake)
                prob_real = probs[1].item()  # P(real)
                prediction = "real" if prob_real > confidence_threshold else "fake"
                
                predictions.append({
                    "image_path": str(image_path),
                    "prediction": prediction,
                    "confidence": max(prob_real, prob_fake),
                    "prob_real": prob_real,
                    "prob_fake": prob_fake,
                })
            
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                predictions.append({
                    "image_path": str(image_path),
                    "prediction": "error",
                    "confidence": 0.0,
                    "prob_real": 0.0,
                    "prob_fake": 0.0,
                })
    
    # Save
    output = {
        "model": model_path,
        "backbone": backbone,
        "threshold": confidence_threshold,
        "predictions": predictions,
        "summary": {
            "total": len(predictions),
            "real": sum(1 for p in predictions if p["prediction"] == "real"),
            "fake": sum(1 for p in predictions if p["prediction"] == "fake"),
            "error": sum(1 for p in predictions if p["prediction"] == "error"),
        }
    }
    
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nPredictions saved to {output_json}")
    print(f"Summary: {output['summary']}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Predict on image directory")
    parser.add_argument("--image-dir", required=True, help="Directory containing images")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--backbone", default="sigclip")
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="auto picks cuda, then Apple MPS, then CPU",
    )
    
    args = parser.parse_args()
    
    predict_directory(
        image_dir=args.image_dir,
        model_path=args.model,
        backbone=args.backbone,
        output_json=args.output,
        confidence_threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
