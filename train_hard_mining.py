"""
train_hard_mining.py — Fine-tuning with Hard Sample Mining
==========================================================

Identifies images where the model currently has high error and
increases their loss weight during a final fine-tuning phase.

Strategy:
  1. Load the best checkpoint (100 ep).
  2. Run inference on train/val sets to find "hard" samples.
  3. Increase loss weight for top 10% hardest samples.
  4. Fine-tune for 15-20 epochs with a low learning rate.

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
import pandas as pd
from tqdm import tqdm

import config
from dataset import LDCTDataset, custom_collate_fn
from model import create_model
from train import train_one_epoch, validate, compute_metrics
from loss import CTMUSIQLoss

def identify_hard_samples(model, loader, device, threshold_percentile=90):
    """Run inference to identify samples with highest absolute error."""
    model.eval()
    errors = []
    image_ids = []
    
    print(f"Identifying hard samples in {len(loader.dataset)} images...")
    with torch.no_grad():
        for batch in tqdm(loader):
            patches = batch['patches'].to(device)
            coords = batch['coords'].to(device)
            targets = batch['score'].to(device)
            ids = batch['image_id']
            
            output = model(patches, coords)
            preds = output['score'].squeeze(-1)
            
            abs_error = torch.abs(preds - targets).cpu().numpy()
            errors.extend(abs_error)
            image_ids.extend(ids)
            
    # Find threshold for top X%
    threshold = np.percentile(errors, threshold_percentile)
    hard_ids = [img_id for img_id, err in zip(image_ids, errors) if err >= threshold]
    
    print(f"  Threshold (P{threshold_percentile}): {threshold:.4f}")
    print(f"  Found {len(hard_ids)} hard samples.")
    return hard_ids

def train_hard_mining(checkpoint_path, epochs=20, lr=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fine-tuning on {device}...")
    
    # 1. Load model and checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Setup datasets
    train_dataset = LDCTDataset(config.DATA_DIR, config.LABEL_FILE, 'train', config.SCALES, augment=True)
    val_dataset = LDCTDataset(config.DATA_DIR, config.LABEL_FILE, 'val', config.SCALES, augment=False)
    
    # Inference loaders (no shuffle)
    train_loader_inf = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    val_loader_inf = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    # 3. Identify hard samples
    hard_train_ids = identify_hard_samples(model, train_loader_inf, device)
    hard_val_ids = identify_hard_samples(model, val_loader_inf, device)
    
    # 4. Update weights in training dataset (Triple weight for hard samples)
    for img_id in hard_train_ids:
        train_dataset.sample_weights[img_id] = 3.0
        
    # 5. Setup for fine-tuning
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    criterion = CTMUSIQLoss(lambda_kl=config.LAMBDA_KL).to(device)
    
    best_aggregate = checkpoint['best_aggregate']
    print(f"Starting fine-tuning. Baseline Aggregate: {best_aggregate:.4f}")
    
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
            
            # Weighted MSE manually
            loss_mse = (weights * (preds - targets)**2).mean()
            
            # KL Loss using criterion helper
            loss_kl = criterion.compute_kl_loss(output['scale_scores'], output['score'])
            
            loss = loss_mse + config.LAMBDA_KL * loss_kl
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validate using existing validate function
        val_loss_dict, val_metrics = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} Val Aggregate: {val_metrics['Aggregate']:.4f} (Best: {best_aggregate:.4f})")
        
        if val_metrics['Aggregate'] > best_aggregate:
            best_aggregate = val_metrics['Aggregate']
            print(f"  New Best! Saving...")
            torch.save({
                'epoch': 100 + epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_aggregate': best_aggregate,
                'metrics': val_metrics,
                'hard_mining': True,
                'config': checkpoint.get('config', {})
            }, os.path.join(config.RESULTS_DIR, "ct_musiq_hard_mining_best.pth"))

if __name__ == "__main__":
    cp = "results/ct_musiq/ct_musiq_best.pth"
    if not os.path.exists(cp):
        print(f"Checkpoint not found at {cp}")
        sys.exit(1)
    train_hard_mining(cp)
