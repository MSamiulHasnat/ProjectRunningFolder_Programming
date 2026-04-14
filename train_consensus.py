"""
train_consensus.py — Expert Consensus Fine-tuning
=================================================

Trains the new Multi-Head Consensus architecture using knowledge
from the best hard-mining model.

Strategy:
  1. Load the best hard-mining model (120 ep).
  2. Map shared weights (transformer, patch_embed, scale_heads).
  3. Initialize new Multi-Head experts.
  4. Train with Consensus Loss:
     L = L_mse(global) + 0.5 * sum(L_mse(expert_i)) + lambda * L_kl
  5. Use Hard Sample Mining weights for 30 epochs.

Author: M Samiul Hasnat & Gemini CLI
Project: CT-MUSIQ — Undergraduate Thesis
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import config
from dataset import LDCTDataset, custom_collate_fn
from model import create_model
from train import validate, compute_metrics
from loss import CTMUSIQLoss
from train_hard_mining import identify_hard_samples

def train_consensus(checkpoint_path, epochs=30, lr=5e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Consensus training on {device}...")
    
    # 1. Load model with NEW architecture
    # Note: create_model will now create the Multi-Head version
    model = create_model(pretrained=False).to(device)
    
    # 2. Load weights from hard-mining model (partially)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Filter state dict to load only compatible layers
    curr_model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                       if k in curr_model_dict and v.shape == curr_model_dict[k].shape}
    
    curr_model_dict.update(pretrained_dict)
    model.load_state_dict(curr_model_dict)
    print(f"  ✓ Loaded {len(pretrained_dict)} layers from {checkpoint_path}")
    print(f"  ⚠ Initialized Multi-Head experts randomly")
    
    # 3. Setup datasets with Hard Sample weights
    train_dataset = LDCTDataset(config.DATA_DIR, config.LABEL_FILE, 'train', config.SCALES, augment=True)
    val_dataset = LDCTDataset(config.DATA_DIR, config.LABEL_FILE, 'val', config.SCALES, augment=False)
    
    # Identify hard samples using the current model state
    train_loader_inf = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    hard_train_ids = identify_hard_samples(model, train_loader_inf, device)
    
    for img_id in hard_train_ids:
        train_dataset.sample_weights[img_id] = 3.0
        
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    # 4. Optimizer and Criterion
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion_base = CTMUSIQLoss(lambda_kl=config.LAMBDA_KL).to(device)
    
    best_aggregate = checkpoint['best_aggregate']
    print(f"Starting Consensus Fine-tuning. Baseline: {best_aggregate:.4f}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            patches = batch['patches'].to(device)
            coords = batch['coords'].to(device)
            targets = batch['score'].to(device)
            weights = batch['sample_weight'].to(device)
            
            optimizer.zero_grad()
            
            output = model(patches, coords)
            preds = output['score'].squeeze(-1)
            experts = output['expert_scores'] # List of [B, 1]
            
            # Global Loss (Weighted MSE)
            loss_global = (weights * (preds - targets)**2).mean()
            
            # Expert Losses (Weighted MSE)
            loss_experts = 0.0
            for exp_score in experts:
                loss_experts += (weights * (exp_score.squeeze(-1) - targets)**2).mean()
            
            # KL Loss
            loss_kl = criterion_base.compute_kl_loss(output['scale_scores'], output['score'])
            
            # Combined Loss
            # Expert feedback helps the CLS token learn better multi-faceted features
            loss = loss_global + 0.3 * (loss_experts / 3.0) + config.LAMBDA_KL * loss_kl
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validate
        val_loss_dict, val_metrics = validate(model, val_loader, criterion_base, device)
        print(f"Epoch {epoch+1} Val Aggregate: {val_metrics['Aggregate']:.4f} (Best: {best_aggregate:.4f})")
        
        if val_metrics['Aggregate'] > best_aggregate:
            best_aggregate = val_metrics['Aggregate']
            print(f"  New Rank #1 Candidate! Saving...")
            torch.save({
                'epoch': 120 + epoch,
                'model_state_dict': model.state_dict(),
                'best_aggregate': best_aggregate,
                'metrics': val_metrics,
                'consensus': True,
                'config': checkpoint.get('config', {})
            }, os.path.join(config.RESULTS_DIR, "ct_musiq_consensus_best.pth"))

if __name__ == "__main__":
    cp = "results/ct_musiq_hard_mining_best.pth"
    if not os.path.exists(cp):
        print(f"Checkpoint not found: {cp}")
        sys.exit(1)
    train_consensus(cp)
