"""
train.py — CT-MUSIQ Training Loop
===================================

Full training pipeline with:
  1. Two-stage training (frozen encoder → full fine-tuning)
  2. Mixed precision (fp16) for 6GB VRAM efficiency
  3. Gradient clipping to prevent exploding gradients
  4. Cosine annealing learning rate scheduler
  5. Early stopping based on validation aggregate score
  6. Comprehensive logging to CSV and console

Two-Stage Training Schedule:
  Stage 1 (Epochs 1-5): Freeze transformer encoder
    - Train only: patch_embed, pos_encoding, prediction_heads
    - LR = 1e-3 (higher for faster convergence of new layers)
    - Purpose: Let CT-specific components initialize before touching
      pretrained weights
    
  Stage 2 (Epochs 6-50): Unfreeze all weights
    - Train entire model end-to-end
    - LR = 1e-4 with cosine annealing to 1e-6
    - Purpose: Fine-tune pretrained features for CT domain

Mixed Precision (torch.cuda.amp):
  - Forward pass in fp16 for speed and memory savings
  - Loss scaling to prevent underflow in fp16 gradients
  - ~40% VRAM reduction, ~30% speedup on RTX 3060

VRAM Management:
  - BATCH_SIZE = 4 (safe for 6GB with AMP)
  - If OOM: reduce to 2, add gradient_accumulation_steps=2
  - Gradient clipping at max_norm=1.0

Usage:
  python train.py                    # Train with default config
  python train.py --lambda_kl 0.05   # Override KL weight
  python train.py --epochs 30        # Override epochs

Author: M Samiul Hasnat, Sichuan University
Project: CT-MUSIQ — Undergraduate Thesis
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import project modules
import config
from dataset import create_dataloaders
from model import create_model, CTMUSIQ
from loss import create_criterion, CTMUSIQLoss

# Import evaluation metrics (will be created in evaluate.py)
# For now, we'll define basic metric computation here
from scipy.stats import pearsonr, spearmanr, kendalltau


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute PLCC, SROCC, KROCC, and aggregate score.
    
    These metrics match the LDCTIQAC 2023 leaderboard format.
    
    Args:
        predictions: Array of predicted quality scores
        targets: Array of ground truth quality scores
        
    Returns:
        Dictionary with PLCC, SROCC, KROCC, and Aggregate scores
    """
    plcc, _ = pearsonr(predictions, targets)
    srocc, _ = spearmanr(predictions, targets)
    krocc, _ = kendalltau(predictions, targets)
    
    return {
        'PLCC': round(plcc, 4),
        'SROCC': round(srocc, 4),
        'KROCC': round(krocc, 4),
        'Aggregate': round(abs(plcc) + abs(srocc) + abs(krocc), 4)
    }


def set_seed(seed: int = config.SEED) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_encoder(model: CTMUSIQ) -> None:
    """
    Freeze transformer encoder weights for Stage 1 training.
    
    Only patch_embed, pos_encoding, and prediction heads remain trainable.
    
    Args:
        model: CT-MUSIQ model
    """
    # Freeze transformer encoder
    for param in model.transformer.parameters():
        param.requires_grad = False
    
    # Freeze [CLS] token
    model.cls_token.requires_grad = False
    
    print("  Frozen: transformer encoder + [CLS] token")
    print("  Trainable: patch_embed, pos_encoding, prediction heads")


def unfreeze_encoder(model: CTMUSIQ) -> None:
    """
    Unfreeze all weights for Stage 2 training.
    
    Args:
        model: CT-MUSIQ model
    """
    for param in model.parameters():
        param.requires_grad = True
    
    print("  Unfrozen: all parameters")


def get_trainable_params(model: CTMUSIQ) -> list:
    """
    Get list of trainable parameters.
    
    Args:
        model: CT-MUSIQ model
        
    Returns:
        List of trainable parameters
    """
    return [p for p in model.parameters() if p.requires_grad]


def train_one_epoch(
    model: CTMUSIQ,
    train_loader,
    criterion: CTMUSIQLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: CT-MUSIQ model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision (ignored if use_amp=False)
        device: Device to train on
        epoch: Current epoch number
        gradient_accumulation_steps: Number of steps to accumulate gradients
        use_amp: Whether to use automatic mixed precision (requires CUDA)
        
    Returns:
        Dictionary with average loss components
    """
    model.train()
    
    total_loss = 0.0
    total_mse = 0.0
    total_kl = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        patches = batch['patches'].to(device)
        coords = batch['coords'].to(device)
        scores = batch['score'].to(device)
        
        # Forward pass (with or without mixed precision)
        if use_amp:
            with autocast('cuda'):
                output = model(patches, coords)
                losses = criterion(output, scores)
                loss = losses['total'] / gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    get_trainable_params(model),
                    max_norm=config.MAX_GRAD_NORM
                )
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Standard training (no mixed precision)
            output = model(patches, coords)
            losses = criterion(output, scores)
            loss = losses['total'] / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    get_trainable_params(model),
                    max_norm=config.MAX_GRAD_NORM
                )
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
        
        # Accumulate losses
        total_loss += losses['total'].item()
        total_mse += losses['mse'].item()
        total_kl += losses['kl'].item()
        num_batches += 1
    
    # Average losses
    avg_losses = {
        'total': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'kl': total_kl / num_batches
    }
    
    return avg_losses


@torch.no_grad()
def validate(
    model: CTMUSIQ,
    val_loader,
    criterion: CTMUSIQLoss,
    device: torch.device
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Validate model on validation set.
    
    Args:
        model: CT-MUSIQ model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (loss_dict, metrics_dict)
    """
    model.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_kl = 0.0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    for batch in val_loader:
        # Move data to device
        patches = batch['patches'].to(device)
        coords = batch['coords'].to(device)
        scores = batch['score'].to(device)
        
        # Forward pass (no mixed precision needed for validation)
        output = model(patches, coords)
        losses = criterion(output, scores)
        
        # Accumulate losses
        total_loss += losses['total'].item()
        total_mse += losses['mse'].item()
        total_kl += losses['kl'].item()
        num_batches += 1
        
        # Collect predictions and targets for metrics
        predictions = output['score'].squeeze(-1).cpu().numpy()
        targets = scores.cpu().numpy()
        
        all_predictions.extend(predictions)
        all_targets.extend(targets)
    
    # Average losses
    avg_losses = {
        'total': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'kl': total_kl / num_batches
    }
    
    # Compute metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    metrics = compute_metrics(all_predictions, all_targets)
    
    return avg_losses, metrics


def save_checkpoint(
    model: CTMUSIQ,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_aggregate: float,
    save_path: str
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: CT-MUSIQ model
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch
        best_aggregate: Best validation aggregate score
        save_path: Path to save checkpoint
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_aggregate': best_aggregate,
        'config': {
            'scales': config.SCALES,
            'lambda_kl': config.LAMBDA_KL,
            'batch_size': config.BATCH_SIZE,
            'seed': config.SEED
        }
    }
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: CTMUSIQ,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    load_path: str,
    device: torch.device
) -> Tuple[int, float]:
    """
    Load model checkpoint.
    
    Args:
        model: CT-MUSIQ model
        optimizer: Optimizer
        scheduler: LR scheduler
        load_path: Path to checkpoint
        device: Device to load to
        
    Returns:
        Tuple of (epoch, best_aggregate)
    """
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_aggregate']


def train(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lambda_kl: float = config.LAMBDA_KL,
    resume_from: Optional[str] = None
) -> None:
    """
    Main training function.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        lambda_kl: KL loss weight
        resume_from: Path to checkpoint to resume from (optional)
    """
    print("="*60)
    print("CT-MUSIQ Training")
    print("="*60)
    
    # Set random seed
    set_seed()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = getattr(torch.cuda.get_device_properties(0), 'total_memory', 
                       getattr(torch.cuda.get_device_properties(0), 'total_mem', 0))
        print(f"VRAM: {vram / 1024**3:.1f} GB")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    
    print(f"  Train: {len(train_loader.dataset)} images")
    print(f"  Val:   {len(val_loader.dataset)} images")
    print(f"  Test:  {len(test_loader.dataset)} images")
    
    # Create model
    print("\nCreating CT-MUSIQ model...")
    model = create_model(
        num_scales=len(config.SCALES),
        pretrained=True,
        device=device
    )
    
    # Create loss criterion
    print(f"\nCreating loss criterion (lambda_kl={lambda_kl})...")
    criterion = create_criterion(lambda_kl=lambda_kl, device=device)
    
    # Create optimizer
    print("\nCreating optimizer...")
    optimizer = optim.AdamW(
        get_trainable_params(model),
        lr=config.LR_STAGE1,  # Start with Stage 1 LR
        weight_decay=0.01
    )
    
    # Create LR scheduler (will be used in Stage 2)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - config.STAGE1_EPOCHS,
        eta_min=config.LR_MIN
    )
    
    # Create GradScaler for mixed precision (only if CUDA available)
    use_amp = device.type == 'cuda'
    if use_amp:
        scaler = GradScaler('cuda')
        print("\n✓ Using mixed precision (fp16) for faster training")
    else:
        scaler = None
        print("\n⚠ CUDA not available — training on CPU (will be slower)")
        print("  To enable GPU training, install CUDA-enabled PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_aggregate = 0.0
    
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        start_epoch, best_aggregate = load_checkpoint(
            model, optimizer, scheduler, resume_from, device
        )
        print(f"  Resumed at epoch {start_epoch + 1}, best aggregate: {best_aggregate:.4f}")
    
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Initialize training log
    log_path = config.TRAINING_LOG_CSV
    if not os.path.exists(log_path):
        log_df = pd.DataFrame(columns=[
            'epoch', 'stage', 'lr',
            'train_loss', 'train_mse', 'train_kl',
            'val_loss', 'val_mse', 'val_kl',
            'PLCC', 'SROCC', 'KROCC', 'Aggregate',
            'time_seconds'
        ])
        log_df.to_csv(log_path, index=False)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # Determine training stage
        if epoch < config.STAGE1_EPOCHS:
            stage = 1
            if epoch == 0:
                print(f"\n--- Stage 1: Frozen Encoder (Epochs 1-{config.STAGE1_EPOCHS}) ---")
                freeze_encoder(model)
                # Set higher LR for Stage 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.LR_STAGE1
        else:
            stage = 2
            if epoch == config.STAGE1_EPOCHS:
                print(f"\n--- Stage 2: Full Fine-tuning (Epochs {config.STAGE1_EPOCHS+1}-{epochs}) ---")
                unfreeze_encoder(model)
                # Set lower LR for Stage 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.LR
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train one epoch
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            use_amp=use_amp
        )
        
        # Validate
        val_losses, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update LR scheduler (only in Stage 2)
        if stage == 2:
            scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} | Stage {stage} | LR: {current_lr:.2e}")
        print(f"  Train — Loss: {train_losses['total']:.4f} (MSE: {train_losses['mse']:.4f}, KL: {train_losses['kl']:.4f})")
        print(f"  Val   — Loss: {val_losses['total']:.4f} (MSE: {val_losses['mse']:.4f}, KL: {val_losses['kl']:.4f})")
        print(f"  Val   — PLCC: {val_metrics['PLCC']:.4f} | SROCC: {val_metrics['SROCC']:.4f} | KROCC: {val_metrics['KROCC']:.4f} | Agg: {val_metrics['Aggregate']:.4f}")
        
        # Check for improvement
        if val_metrics['Aggregate'] > best_aggregate:
            best_aggregate = val_metrics['Aggregate']
            patience_counter = 0
            
            # Save best checkpoint
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_aggregate,
                config.BEST_MODEL_PATH
            )
            print(f"  → New best! Saving checkpoint. (Aggregate: {best_aggregate:.4f})")
        else:
            patience_counter += 1
            print(f"  → No improvement. Patience: {patience_counter}/{config.PATIENCE}")
        
        # Log to CSV
        log_entry = pd.DataFrame([{
            'epoch': epoch + 1,
            'stage': stage,
            'lr': current_lr,
            'train_loss': train_losses['total'],
            'train_mse': train_losses['mse'],
            'train_kl': train_losses['kl'],
            'val_loss': val_losses['total'],
            'val_mse': val_losses['mse'],
            'val_kl': val_losses['kl'],
            'PLCC': val_metrics['PLCC'],
            'SROCC': val_metrics['SROCC'],
            'KROCC': val_metrics['KROCC'],
            'Aggregate': val_metrics['Aggregate'],
            'time_seconds': epoch_time
        }])
        log_entry.to_csv(log_path, mode='a', header=False, index=False)
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered after {config.PATIENCE} epochs without improvement")
            print(f"Best validation aggregate: {best_aggregate:.4f}")
            print(f"{'='*60}")
            break
    
    # Training complete
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best validation aggregate: {best_aggregate:.4f}")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")
    print(f"Training log saved to: {log_path}")
    
    # Return test loader for evaluation
    return model, test_loader, device


def main():
    """
    Main entry point with argument parsing.
    """
    parser = argparse.ArgumentParser(description='Train CT-MUSIQ model')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lambda_kl', type=float, default=config.LAMBDA_KL,
                        help='KL loss weight')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Run training
    model, test_loader, device = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_kl=args.lambda_kl,
        resume_from=args.resume
    )
    
    print("\nTo evaluate on test set, run: python evaluate.py")


if __name__ == "__main__":
    main()
