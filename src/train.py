"""Training loop for AIGI detector."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from src.dataset import create_dataloader
from src.model import create_model, PairwiseTrainingWrapper
from src.evaluate import evaluate_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device="cuda",
        checkpoint_dir="models",
        pairwise=False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.pairwise = pairwise
        
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / "logs"))
        self.global_step = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]")
        
        for batch_idx, batch in enumerate(pbar):
            if self.pairwise:
                # Pairwise training: each batch has (image, label)
                # We apply augmentation twice and create pairs
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Create augmented copies (this is simplified; ideally done in dataset)
                # For now, just use the standard forward pass
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            else:
                # Standard training
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            
            pbar.set_postfix({
                "loss": loss.item(),
                "acc": correct / labels.size(0)
            })
            
            # Log to tensorboard
            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples
        
        logger.info(f"Epoch {epoch} [train] Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        return avg_loss, avg_acc
    
    def val_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc=f"Epoch {epoch} [val]"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_samples
        
        logger.info(f"Epoch {epoch} [val] Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        self.writer.add_scalar("val/loss", avg_loss, epoch)
        self.writer.add_scalar("val/acc", avg_acc, epoch)
        
        return avg_loss, avg_acc
    
    def train(self, num_epochs, save_every=1, early_stopping_patience=5):
        """Train for num_epochs."""
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.val_epoch(epoch)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(self.model.state_dict(), best_path)
                logger.info(f"New best model: {best_path}, Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train AIGI detector")
    
    # Model
    parser.add_argument("--backbone", default="sigclip", choices=["sigclip", "dinov3-l", "clip-vit"])
    parser.add_argument("--head", default="linear", choices=["linear", "attention", "multi_layer"])
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    
    # Data
    parser.add_argument("--data-dir", default="data", help="Path to data/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Augmentation
    parser.add_argument("--augmentation", default="none", choices=["none", "simple", "advanced"])
    parser.add_argument("--pairwise", action="store_true", help="Use pairwise clean/distorted training")
    
    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stopping", type=int, default=5)
    
    # Output
    parser.add_argument("--output", default="models/baseline.pt")
    parser.add_argument("--checkpoint-dir", default="models")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = create_dataloader(
        root_dir=args.data_dir,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation=args.augmentation,
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        root_dir=args.data_dir,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation="none",
        shuffle=False,
    )
    
    # Create model
    logger.info(f"Creating model: {args.backbone} + {args.head}")
    model = create_model(
        backbone=args.backbone,
        head_type=args.head,
        freeze_backbone=args.freeze_backbone,
        device=device,
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    
    # Only train the head (backbone is frozen)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        pairwise=args.pairwise,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(
        num_epochs=args.epochs,
        save_every=1,
        early_stopping_patience=args.early_stopping,
    )
    
    # Save final model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    logger.info(f"Saved final model to {args.output}")
    
    # Save config
    config = {
        "backbone": args.backbone,
        "head": args.head,
        "augmentation": args.augmentation,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
    }
    config_path = args.output.replace(".pt", "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()
