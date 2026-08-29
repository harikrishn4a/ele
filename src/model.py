"""Model definitions: backbones and classification heads."""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import timm


class AIGIDetector(nn.Module):
    """
    AI-Generated Image Detector: frozen backbone + learnable head.
    
    Architecture:
        Frozen VFM (CLIP/DINOv3/SigLIP) 
        → Extract patch tokens from final layer
        → Pooling (mean / attention / multi-layer)
        → Linear classification head
    """
    
    def __init__(
        self,
        backbone_name: str = "sigclip",
        head_type: str = "linear",
        num_classes: int = 2,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ):
        """
        Args:
            backbone_name: "sigclip", "dinov3-l", "clip-vit"
            head_type: "linear", "attention", "multi_layer"
            num_classes: 2 (real/fake)
            freeze_backbone: freeze backbone weights during training
            pretrained: load pretrained weights
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.head_type = head_type
        self.num_classes = num_classes
        
        # Load backbone
        self.backbone, self.feature_dim, self.patch_dim = self._load_backbone(
            backbone_name, pretrained
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Create head based on type
        if head_type == "linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif head_type == "attention":
            self.pooling = TunableAttentionPooling(self.feature_dim)
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif head_type == "multi_layer":
            # Concatenate features from multiple layers
            self.head = nn.Linear(self.feature_dim * 2, num_classes)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
    
    def _load_backbone(self, backbone_name, pretrained):
        """Load a pretrained vision foundation model."""
        if backbone_name == "sigclip":
            # SigLIP (OpenCLIP)
            try:
                import open_clip
                model, _, transform = open_clip.create_model_and_transforms(
                    "ViT-SO400M-14-SigLIP-384",
                    pretrained="webli"
                )
            except ImportError:
                raise ImportError("Install open_clip: pip install open-clip-torch")
            
            # Extract vision encoder
            backbone = model.visual
            feature_dim = 1152  # ViT-SO400M embedding dim
            patch_dim = 384
        
        elif backbone_name == "dinov3-l":
            # DINOv3-Large
            backbone = timm.create_model("vit_large_patch14_dinov3", pretrained=pretrained)
            feature_dim = 1024
            patch_dim = 14  # 14x14 patches in ViT-L
        
        elif backbone_name == "clip-vit":
            # OpenAI CLIP ViT-L/14
            try:
                import clip
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, _ = clip.load("ViT-L/14", device=device)
            except ImportError:
                raise ImportError("Install clip: pip install openai-clip")
            
            backbone = model.visual
            feature_dim = 768
            patch_dim = 14
        
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        return backbone, feature_dim, patch_dim
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, 3, H, W) image tensor
        
        Returns:
            logits: (B, num_classes)
        """
        # Extract patch tokens
        with torch.no_grad():
            # Get intermediate representations
            if self.backbone_name == "sigclip":
                x = self.backbone(x)
                # x is already pooled; we need patch tokens
                # For SigLIP, we extract from intermediate layer
                features = x  # (B, feature_dim)
            
            elif self.backbone_name == "dinov3-l":
                # DINOv3 has cls token + patch tokens
                x = self.backbone(x)  # (B, 257, 1024) for ViT-L
                features = x.mean(dim=1)  # Global average pool: (B, 1024)
            
            elif self.backbone_name == "clip-vit":
                x = self.backbone(x)  # (B, 768)
                features = x  # Already pooled
        
        # Apply head
        if self.head_type == "linear":
            logits = self.head(features)
        elif self.head_type == "attention":
            # Requires patch tokens; implement if needed
            logits = self.head(features)
        else:
            logits = self.head(features)
        
        return logits
    
    def extract_features(self, x):
        """Extract features (for caching/visualization)."""
        with torch.no_grad():
            if self.backbone_name == "sigclip":
                features = self.backbone(x)
            elif self.backbone_name == "dinov3-l":
                x = self.backbone(x)
                features = x.mean(dim=1)
            elif self.backbone_name == "clip-vit":
                features = self.backbone(x)
            else:
                features = self.backbone(x)
        
        return features


class TunableAttentionPooling(nn.Module):
    """
    Tunable Attention Pooling: learn which patches are important.
    
    Instead of averaging all patches equally, learn attention weights.
    """
    
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: (B, N, D) patch token features
        
        Returns:
            pooled: (B, D) weighted average
        """
        # Compute attention weights
        weights = self.attention(patch_tokens)  # (B, N, 1)
        
        # Weighted average
        pooled = (patch_tokens * weights).sum(dim=1)  # (B, D)
        
        return pooled


class PairwiseTrainingWrapper(nn.Module):
    """
    Wrapper for pairwise clean/distorted training.
    
    Given (clean, augmented) pairs, trains:
        L = CE(pred_clean, label) + α·KL(pred_clean, pred_aug) + β·MSE(feat_clean, feat_aug)
    """
    
    def __init__(self, model: AIGIDetector, alpha=0.5, beta=0.25):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
        self.mse_loss = nn.MSELoss()
    
    def forward(self, x_clean, x_aug, y):
        """
        Args:
            x_clean: (B, 3, H, W) clean images
            x_aug: (B, 3, H, W) augmented images
            y: (B,) labels (0 or 1)
        
        Returns:
            loss: scalar
        """
        # Forward through model
        logits_clean = self.model(x_clean)
        logits_aug = self.model(x_aug)
        
        # Extract features for MSE loss
        feat_clean = self.model.extract_features(x_clean)
        feat_aug = self.model.extract_features(x_aug)
        
        # Compute losses
        ce = self.ce_loss(logits_clean, y)
        
        # KL divergence: predictions should be similar
        probs_clean = torch.softmax(logits_clean, dim=1)
        probs_aug = torch.softmax(logits_aug, dim=1)
        kl = self.kl_loss(probs_aug.log(), probs_clean)
        
        # MSE: features should be similar
        mse = self.mse_loss(feat_clean, feat_aug)
        
        # Combined loss
        loss = ce + self.alpha * kl + self.beta * mse
        
        return loss


def create_model(
    backbone: str = "sigclip",
    head_type: str = "linear",
    freeze_backbone: bool = True,
    device: str = "cuda",
) -> AIGIDetector:
    """Factory function to create a detector model."""
    model = AIGIDetector(
        backbone_name=backbone,
        head_type=head_type,
        freeze_backbone=freeze_backbone,
        pretrained=True,
    )
    model = model.to(device)
    return model
