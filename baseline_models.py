"""
baseline_models.py — Baseline Models for Fair Comparison
=========================================================

Provides a combined baseline method for fair comparison with CT-MUSIQ.
The baseline follows the challenge idea of combining Swin and ResNet.

Available Baseline:
    - 'agaldran_combo': Swin-T + ResNet50 score-level ensemble

Baseline behavior:
    1. Accepts full CT images (not patches/coordinates like CT-MUSIQ)
    2. Uses ImageNet pretrained weights for both backbones
    3. Averages the two predicted scores into a single final score
    4. Returns output dict compatible with CT-MUSIQ loss/training code

Author: Model Comparison Baseline
Project: CT-MUSIQ — Undergraduate Thesis
"""

import torch
import torch.nn as nn
import warnings
from typing import Dict, Tuple

import config

# Try to import timm for Swin transformer
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    warnings.warn("timm not installed. Swin model will not be available.")

# Try to import torchvision for ResNet50
try:
    from torchvision.models import resnet50, ResNet50_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    warnings.warn("torchvision not installed. ResNet50 model will not be available.")


class SwinTransformerQA(nn.Module):
    """
    Swin Transformer-Tiny for CT image quality assessment.
    
    Uses pretrained Swin-T from timm, removes classification head,
    and adds a quality prediction head.
    
    Input: (batch, 3, H, W) normalized CT images (H, W ∈ {224, 384})
    Output: (batch, 1) quality scores in range [0, 4]
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize Swin Transformer model.
        
        Args:
            pretrained: Whether to use ImageNet-1K pretrained weights
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for Swin model. Install: pip install timm")
        
        # Load pretrained Swin Transformer
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0  # Remove classification head, return features
        )
        
        # Get feature dimension from backbone
        # Swin-T outputs 768-dim features
        feature_dim = self.backbone.num_features
        
        # Add quality prediction head
        # Simple head: feature → hidden → quality score
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(256, 1)
        )
        
        self.model_name = 'swin_t'
    
    def forward(
        self,
        images: torch.Tensor,
        coords: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: (batch, 3, H, W) normalized CT images
            coords: (batch, num_patches, 2) patch coordinates (ignored for Swin)
        
        Returns:
            (batch, 1) predicted quality scores
        """
        # Extract features from backbone
        features = self.backbone(images)  # (batch, feature_dim)
        
        # Predict quality score
        quality = self.quality_head(features)  # (batch, 1)
        return quality


class ResNet50QA(nn.Module):
    """
    ResNet50 for CT image quality assessment.
    
    Uses pretrained ResNet50 from torchvision, removes classification head,
    and adds a quality prediction head.
    
    Input: (batch, 3, H, W) normalized CT images (H, W ∈ {224, 384})
    Output: (batch, 1) quality scores in range [0, 4]
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize ResNet50 model.
        
        Args:
            pretrained: Whether to use ImageNet-1K pretrained weights
        """
        super().__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for ResNet50. Install: pip install torchvision")
        
        # Load pretrained ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # Remove classification layer, keep everything up to avgpool
        # This gives us 2048-dim feature vectors
        self.backbone.fc = nn.Identity()
        
        feature_dim = 2048
        
        # Add quality prediction head
        # Simple head: feature → hidden → quality score
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(256, 1)
        )
        
        self.model_name = 'resnet50'
    
    def forward(
        self,
        images: torch.Tensor,
        coords: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: (batch, 3, H, W) normalized CT images
            coords: (batch, num_patches, 2) patch coordinates (ignored for ResNet)
        
        Returns:
            (batch, 1) predicted quality scores
        """
        # Extract features from backbone (up to avgpool)
        features = self.backbone(images)  # (batch, 2048)
        
        # Predict quality score
        quality = self.quality_head(features)  # (batch, 1)
        return quality


class AgaldranComboQA(nn.Module):
    """
    Combined baseline that ensembles Swin-T and ResNet50.

    The final score is the mean of the two model scores.
    Output format mirrors CT-MUSIQ output dictionary so it can be trained
    with the same criterion and loop.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.swin = SwinTransformerQA(pretrained=pretrained)
        self.resnet = ResNet50QA(pretrained=pretrained)
        self.model_name = 'agaldran_combo'

    def forward(self, images: torch.Tensor, coords: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        swin_score = self.swin(images, coords)
        resnet_score = self.resnet(images, coords)
        score = 0.5 * (swin_score + resnet_score)
        return {
            'score': score,
            'scale_scores': []
        }


def create_baseline_model(
    model_name: str = 'agaldran_combo',
    pretrained: bool = True
) -> nn.Module:
    """
    Factory function to create baseline models.
    
    Args:
        model_name: 'agaldran_combo'
        pretrained: Whether to use ImageNet-1K pretrained weights
    
    Returns:
        Model instance
    
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == 'agaldran_combo':
        model = AgaldranComboQA(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'agaldran_combo'")
    
    return model
