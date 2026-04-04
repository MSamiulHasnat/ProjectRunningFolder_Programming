"""
get_model.py — Model Factory Function
======================================

Provides a unified interface to get models (CT-MUSIQ or baseline) for training.

Usage:
    model = get_model('ct_musiq')        # Get CT-MUSIQ model
    model = get_model('agaldran_combo')  # Get combined baseline (Swin + ResNet)

Author: Model Factory
Project: CT-MUSIQ — Undergraduate Thesis
"""

import torch.nn as nn
from typing import Literal


def get_model(
    model_name: Literal['ct_musiq', 'agaldran_combo'] = 'ct_musiq',
    pretrained: bool = True
) -> nn.Module:
    """
    Get a model for training.
    
    All models have a compatible interface:
      - forward(images, coords) → quality_scores
      - Can be trained with the same loss function and optimizer
      - Output shape: (batch_size, 1)
    
    Args:
        model_name: Name of model to load
            - 'ct_musiq': Multi-scale transformer adaptation (your thesis model)
            - 'agaldran_combo': Combined Swin-T + ResNet50 baseline
        pretrained: Whether to use pretrained weights (only affects baselines)
    
    Returns:
        Model instance
    
    Raises:
        ValueError: If model_name is not recognized
    
    Examples:
        >>> model = get_model('ct_musiq')
        >>> model = get_model('agaldran_combo', pretrained=True)
    """
    if model_name == 'ct_musiq':
        # Import here to avoid circular dependency
        from model import create_model
        model = create_model()
        model.model_name = 'ct_musiq'
        
    elif model_name == 'agaldran_combo':
        # Import baseline models
        from baseline_models import create_baseline_model
        model = create_baseline_model(model_name, pretrained=pretrained)
        
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: 'ct_musiq', 'agaldran_combo'"
        )
    
    return model


def get_available_models() -> list:
    """
    Get list of available model names.
    
    Returns:
        List of model names
    """
    return ['ct_musiq', 'agaldran_combo']
