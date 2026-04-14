"""
train_surgical_ensemble.py — Master-Assistant Ensemble with SAM
==============================================================

Combines surgical initialization, ensemble heads, and SAM optimizer
for the ultimate push to Rank #1.

Strategy:
  1. Clone global_head into head_1 (Master) and head_2/3 (Assistants).
  2. Use SAM optimizer for robust generalization.
  3. Fine-tune for 30 epochs.
  4. Use Weighted Ensemble Loss.

Author: M Samiul Hasnat & Gemini CLI
Project: CT-MUSIQ — Undergraduate Thesis
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import config
from dataset import LDCTDataset, custom_collate_fn
from model import create_model
from train import validate
from loss import CTMUSIQLoss
from train_hard_mining import identify_hard_samples
from train_sam import SAM

def train_surgical_sam(checkpoint_path, epochs=30, lr=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Master-Assistant SAM training on {device}...")
    
    # 1. Create model
    model = create_model(pretrained=False).to(device)
    
    # 2. Surgical Weight Cloning
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    old_sd = checkpoint['model_state_dict']
    new_sd = model.state_dict()
    
    # Map shared base weights
    shared_keys = [k for k in old_sd if k in new_sd and old_sd[k].shape == new_sd[k].shape]
    for k in shared_keys:
        new_sd[k] = old_sd[k]
        
    # CLONE global_head into head_1, 2, 3
    # Use weights from the 100-epoch hard-mined best
    for i in [1, 2, 3]:
        for layer in [0, 3]: 
            # Note: Hard-mined checkpoint has 'global_head.0.weight' etc.
            new_sd[f'head_{i}.{layer}.weight'] = old_sd[f'global_head.{layer}.weight'].clone()
            new_sd[f'head_{i}.{layer}.bias'] = old_sd[f'global_head.{layer}.bias'].clone()
            
    model.load_state_dict(new_sd)
    print("  ✓ Surgical initialization complete")
    
    # 3. Setup datasets
    train_dataset = LDCTDataset(config.DATA_DIR, config.LABEL_FILE, 'train', config.SCALES, augment=True)
    val_dataset = LDCTDataset(config.DATA_DIR, config.LABEL_FILE, 'val', config.SCALES, augment=False)
    
    # Refresh hard samples for this specific head configuration
    train_loader_inf = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    hard_train_ids = identify_hard_samples(model, train_loader_inf, device)
    for img_id in hard_train_ids:
        train_dataset.sample_weights[img_id] = 3.0
        
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    # 4. SAM Optimizer
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=lr, weight_decay=0.01)
    
    criterion = CTMUSIQLoss(lambda_kl=config.LAMBDA_KL).to(device)
    best_aggregate = 2.7341 # Current leaderboard candidate peak
    
    print(f"Starting Ultimate Fine-tuning. Target: > {best_aggregate:.4f}")
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            patches = batch['patches'].to(device)
            coords = batch['coords'].to(device)
            targets = batch['score'].to(device)
            weights = batch['sample_weight'].to(device)
            
            def get_loss():
                output = model(patches, coords)
                preds = output['score'].squeeze(-1)
                ensemble = output['ensemble_scores']
                
                # Weighted Ensemble Loss
                # Master (head_1) + Assistants (head_2, head_3)
                loss_global = (weights * (preds - targets)**2).mean()
                loss_heads = 0.0
                for h_score in ensemble:
                    loss_heads += (weights * (h_score.squeeze(-1) - targets)**2).mean()
                
                loss_kl = criterion.compute_kl_loss(output['scale_scores'], output['score'])
                return loss_global + 0.2 * (loss_heads / 3.0) + config.LAMBDA_KL * loss_kl

            # SAM First Pass
            loss = get_loss()
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # SAM Second Pass
            loss = get_loss()
            loss.backward()
            optimizer.second_step(zero_grad=True)
            
            pbar.set_postfix({'loss': loss.item()})
            
        # Validate
        _, val_metrics = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} Val Aggregate: {val_metrics['Aggregate']:.4f} (Best: {best_aggregate:.4f})")
        
        if val_metrics['Aggregate'] > best_aggregate:
            best_aggregate = val_metrics['Aggregate']
            print("  ★ NEW CHAMPION MODEL DETECTED ★")
            torch.save({
                'epoch': 200 + epoch,
                'model_state_dict': model.state_dict(),
                'best_aggregate': best_aggregate,
                'metrics': val_metrics,
                'ultimate_ensemble': True,
                'config': checkpoint.get('config', {})
            }, os.path.join(config.RESULTS_DIR, "ct_musiq_ultimate_best.pth"))

if __name__ == "__main__":
    train_surgical_sam("results/ct_musiq/ct_musiq_best.pth")
