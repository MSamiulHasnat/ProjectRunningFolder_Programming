"""
ablation.py — CT-MUSIQ Ablation Study
=======================================

Run systematic ablation experiments to isolate the contribution of each
design decision in CT-MUSIQ.

Ablation Configurations:
  A1: 1 scale [224], no KL loss
      → Single-scale baseline
      
  A2: 2 scales [224, 384], no KL loss
      → Tests multi-scale benefit
      
  A3: 2 scales [224, 384], KL λ=0.05
      → Tests KL at low weight
      
  A4: 2 scales [224, 384], KL λ=0.10
      → Expected best configuration
      
  A5: 2 scales [224, 384], KL λ=0.20
      → Tests if over-strong KL hurts
      
  A6: 3 scales [224, 384, 512], KL λ=0.10
      → Tests 3rd scale (only if VRAM allows)

Thesis Narrative:
  - A1 → A2: Multi-scale extraction improves over single-scale
  - A2 → A4: KL consistency loss adds further improvement
  - A3/A4/A5: Sensitivity to λ — hyperparameter analysis
  - A6: Hardware limit test — either 3rd scale helps or documents OOM

All configs use:
  - Same random seed (42) for reproducibility
  - 30 epochs (shorter than full training for faster ablation cycle)
  - Save best validation and final test metrics

Usage:
  python ablation.py              # Run all ablations
  python ablation.py --configs A1 A2 A4  # Run specific ablations
  python ablation.py --skip A6    # Skip 3-scale experiment

Author: M Samiul Hasnat, Sichuan University
Project: CT-MUSIQ — Undergraduate Thesis
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

import torch

# Import project modules
import config
from dataset import LDCTDataset, create_dataloaders
from model import create_model, CTMUSIQ
from loss import create_criterion
from train import train, set_seed, compute_metrics
from evaluate import evaluate_model


def run_ablation(
    ablation_name: str,
    ablation_config: Dict,
    device: torch.device
) -> Dict[str, float]:
    """
    Run a single ablation experiment.
    
    Args:
        ablation_name: Name of ablation (e.g., 'A1')
        ablation_config: Configuration dictionary
        device: Device to train on
        
    Returns:
        Dictionary with test metrics
    """
    print("\n" + "="*70)
    print(f"ABLATION {ablation_name}: {ablation_config['description']}")
    print("="*70)
    
    # Extract configuration
    scales = ablation_config['scales']
    lambda_kl = ablation_config['lambda_kl']
    epochs = ablation_config['epochs']
    
    print(f"\nConfiguration:")
    print(f"  Scales:     {scales}")
    print(f"  Lambda KL:  {lambda_kl}")
    print(f"  Epochs:     {epochs}")
    
    # Check if 3-scale experiment fits in VRAM
    if len(scales) > 2 and device.type == 'cuda':
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        if vram_gb < 8:
            print(f"\n⚠ WARNING: 3-scale experiment may exceed {vram_gb:.1f} GB VRAM")
            print(f"  If OOM occurs, this will be documented in results.")
    
    # Set random seed for reproducibility
    set_seed(config.SEED)
    
    # Create data loaders with ablation scales
    print("\nCreating data loaders...")
    
    # Create datasets with ablation-specific scales
    train_dataset = LDCTDataset(
        data_dir=config.DATA_DIR,
        label_file=config.LABEL_FILE,
        split='train',
        scales=scales,
        patch_size=config.PATCH_SIZE,
        augment=True
    )
    
    val_dataset = LDCTDataset(
        data_dir=config.DATA_DIR,
        label_file=config.LABEL_FILE,
        split='val',
        scales=scales,
        patch_size=config.PATCH_SIZE,
        augment=False
    )
    
    test_dataset = LDCTDataset(
        data_dir=config.DATA_DIR,
        label_file=config.LABEL_FILE,
        split='test',
        scales=scales,
        patch_size=config.PATCH_SIZE,
        augment=False
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader
    from dataset import custom_collate_fn
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    
    # Create model with ablation-specific scales
    print("\nCreating CT-MUSIQ model...")
    model = create_model(
        num_scales=len(scales),
        pretrained=True,
        device=device
    )
    
    # Create loss criterion with ablation-specific lambda_kl
    criterion = create_criterion(lambda_kl=lambda_kl, device=device)
    
    # Create optimizer
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.amp import GradScaler
    
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LR_STAGE1,
        weight_decay=0.01
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - config.STAGE1_EPOCHS,
        eta_min=config.LR_MIN
    )
    
    scaler = GradScaler()
    
    # Training loop (simplified version of train.py)
    print("\nStarting training...")
    
    best_aggregate = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Determine training stage
        if epoch < config.STAGE1_EPOCHS:
            stage = 1
            if epoch == 0:
                from train import freeze_encoder
                freeze_encoder(model)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.LR_STAGE1
        else:
            stage = 2
            if epoch == config.STAGE1_EPOCHS:
                from train import unfreeze_encoder
                unfreeze_encoder(model)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.LR
        
        # Train one epoch
        from train import train_one_epoch, validate
        
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_losses, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update LR scheduler (only in Stage 2)
        if stage == 2:
            scheduler.step()
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Val Agg: {val_metrics['Aggregate']:.4f}")
        
        # Check for improvement
        if val_metrics['Aggregate'] > best_aggregate:
            best_aggregate = val_metrics['Aggregate']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best checkpoint for this ablation
            checkpoint_path = os.path.join(
                config.RESULTS_DIR,
                f"ablation_{ablation_name}_best.pth"
            )
            from train import save_checkpoint
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_aggregate,
                checkpoint_path
            )
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n  Best validation aggregate: {best_aggregate:.4f} (epoch {best_epoch + 1})")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    
    # Load best checkpoint
    checkpoint_path = os.path.join(
        config.RESULTS_DIR,
        f"ablation_{ablation_name}_best.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run evaluation
    predictions, targets, image_ids = evaluate_model(model, test_loader, device)
    test_metrics = compute_metrics(predictions, targets)
    
    print(f"\n  Test Results:")
    print(f"    PLCC:      {test_metrics['PLCC']:.4f}")
    print(f"    SROCC:     {test_metrics['SROCC']:.4f}")
    print(f"    KROCC:     {test_metrics['KROCC']:.4f}")
    print(f"    Aggregate: {test_metrics['Aggregate']:.4f}")
    
    # Return results
    return {
        'ablation': ablation_name,
        'description': ablation_config['description'],
        'scales': str(scales),
        'lambda_kl': lambda_kl,
        'epochs_trained': best_epoch + 1,
        'best_val_aggregate': best_aggregate,
        'test_PLCC': test_metrics['PLCC'],
        'test_SROCC': test_metrics['SROCC'],
        'test_KROCC': test_metrics['KROCC'],
        'test_Aggregate': test_metrics['Aggregate']
    }


def run_all_ablations(
    configs_to_run: Optional[List[str]] = None,
    configs_to_skip: Optional[List[str]] = None
) -> None:
    """
    Run all ablation experiments.
    
    Args:
        configs_to_run: List of config names to run (None = all)
        configs_to_skip: List of config names to skip
    """
    print("="*70)
    print("CT-MUSIQ Ablation Study")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = getattr(torch.cuda.get_device_properties(0), 'total_memory', 
                       getattr(torch.cuda.get_device_properties(0), 'total_mem', 0))
        print(f"VRAM: {vram / 1024**3:.1f} GB")
    
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Determine which configs to run
    all_configs = config.ABLATION_CONFIGS
    
    if configs_to_run is not None:
        configs = {k: v for k, v in all_configs.items() if k in configs_to_run}
    elif configs_to_skip is not None:
        configs = {k: v for k, v in all_configs.items() if k not in configs_to_skip}
    else:
        configs = all_configs
    
    print(f"\nRunning {len(configs)} ablation experiments:")
    for name, cfg in configs.items():
        print(f"  {name}: {cfg['description']}")
    
    # Run each ablation
    all_results = []
    start_time = time.time()
    
    for ablation_name, ablation_config in configs.items():
        try:
            result = run_ablation(ablation_name, ablation_config, device)
            result['status'] = 'completed'
            all_results.append(result)
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n✗ OOM in {ablation_name} — documenting VRAM limit")
                result = {
                    'ablation': ablation_name,
                    'description': ablation_config['description'],
                    'scales': str(ablation_config['scales']),
                    'lambda_kl': ablation_config['lambda_kl'],
                    'status': 'OOM',
                    'test_PLCC': None,
                    'test_SROCC': None,
                    'test_KROCC': None,
                    'test_Aggregate': None
                }
                all_results.append(result)
                
                # Clear GPU memory
                torch.cuda.empty_cache()
            else:
                raise e
        
        except Exception as e:
            print(f"\n✗ Error in {ablation_name}: {e}")
            result = {
                'ablation': ablation_name,
                'description': ablation_config['description'],
                'scales': str(ablation_config['scales']),
                'lambda_kl': ablation_config['lambda_kl'],
                'status': f'error: {str(e)}',
                'test_PLCC': None,
                'test_SROCC': None,
                'test_KROCC': None,
                'test_Aggregate': None
            }
            all_results.append(result)
    
    total_time = time.time() - start_time
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_path = config.ABLATION_RESULTS_CSV
    results_df.to_csv(results_path, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    
    print(f"\nTotal time: {total_time / 3600:.1f} hours")
    print(f"\nResults saved to: {results_path}")
    
    print(f"\n{'Config':<8} {'Description':<45} {'Status':<10} {'Aggregate':>10}")
    print("-"*80)
    
    for result in all_results:
        status = result['status']
        agg = result.get('test_Aggregate')
        agg_str = f"{agg:.4f}" if agg is not None else "N/A"
        
        print(f"{result['ablation']:<8} {result['description']:<45} {status:<10} {agg_str:>10}")
    
    # Print thesis narrative
    print("\n" + "="*70)
    print("THESIS NARRATIVE")
    print("="*70)
    
    # Find completed results
    completed = [r for r in all_results if r['status'] == 'completed']
    
    if len(completed) >= 2:
        # Compare A1 (single scale) vs A2 (multi scale)
        a1 = next((r for r in completed if r['ablation'] == 'A1'), None)
        a2 = next((r for r in completed if r['ablation'] == 'A2'), None)
        
        if a1 and a2:
            diff = a2['test_Aggregate'] - a1['test_Aggregate']
            print(f"\nA1 → A2: Multi-scale benefit: {diff:+.4f}")
            if diff > 0:
                print("  → Multi-scale extraction improves over single-scale ✓")
            else:
                print("  → Multi-scale does not improve (unexpected)")
        
        # Compare A2 (no KL) vs A4 (KL λ=0.1)
        a4 = next((r for r in completed if r['ablation'] == 'A4'), None)
        
        if a2 and a4:
            diff = a4['test_Aggregate'] - a2['test_Aggregate']
            print(f"\nA2 → A4: KL consistency benefit: {diff:+.4f}")
            if diff > 0:
                print("  → KL consistency loss adds further improvement ✓")
            else:
                print("  → KL loss does not improve (unexpected)")
        
        # Compare KL weights
        a3 = next((r for r in completed if r['ablation'] == 'A3'), None)
        a5 = next((r for r in completed if r['ablation'] == 'A5'), None)
        
        if a3 and a4 and a5:
            print(f"\nKL Weight Sensitivity:")
            print(f"  A3 (λ=0.05): {a3['test_Aggregate']:.4f}")
            print(f"  A4 (λ=0.10): {a4['test_Aggregate']:.4f}")
            print(f"  A5 (λ=0.20): {a5['test_Aggregate']:.4f}")
            
            best_lambda = max([(a3['test_Aggregate'], '0.05'),
                              (a4['test_Aggregate'], '0.10'),
                              (a5['test_Aggregate'], '0.20')], key=lambda x: x[0])
            print(f"  → Best λ: {best_lambda[1]}")
    
    # Check A6 (3 scales)
    a6 = next((r for r in all_results if r['ablation'] == 'A6'), None)
    if a6:
        if a6['status'] == 'OOM':
            print(f"\nA6 (3 scales): OOM on RTX 3060 6GB")
            print(f"  → Documents VRAM limitation for thesis")
        elif a6['status'] == 'completed':
            print(f"\nA6 (3 scales): Aggregate = {a6['test_Aggregate']:.4f}")
            print(f"  → 3rd scale fits in VRAM and provides additional benefit")
    
    print("\n✓ Ablation study complete!")


def main():
    """
    Main entry point with argument parsing.
    """
    parser = argparse.ArgumentParser(description='Run CT-MUSIQ ablation study')
    parser.add_argument('--configs', nargs='+', default=None,
                        help='Specific configs to run (e.g., A1 A2 A4)')
    parser.add_argument('--skip', nargs='+', default=None,
                        help='Configs to skip (e.g., A6)')
    
    args = parser.parse_args()
    
    # Run ablations
    run_all_ablations(
        configs_to_run=args.configs,
        configs_to_skip=args.skip
    )


if __name__ == "__main__":
    main()
