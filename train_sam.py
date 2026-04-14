"""
train_sam.py — Fine-tuning with Sharpness-Aware Minimization (SAM)
==============================================================

Uses the SAM optimizer to find flatter minima, which typically improves 
generalization performance on small datasets like this one.

Strategy:
  1. Load the best hard-mining model.
  2. Implement SAM optimizer (two-pass gradient).
  3. Fine-tune for 15 epochs with low LR (1e-5).
  4. Use Hard Sample Mining weights for final refinement.

Author: M Samiul Hasnat & Gemini CLI
Project: CT-MUSIQ — Undergraduate Thesis
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import config
from dataset import LDCTDataset, custom_collate_fn
from model import create_model
from train import validate
from loss import CTMUSIQLoss
from train_hard_mining import identify_hard_samples

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "sharpness"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual step
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

def train_sam(checkpoint_path, epochs=15, lr=1e-5, rho=0.05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"SAM fine-tuning on {device} (rho={rho})...")
    
    # 1. Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Setup datasets
    train_dataset = LDCTDataset(config.DATA_DIR, config.LABEL_FILE, 'train', config.SCALES, augment=True)
    val_dataset = LDCTDataset(config.DATA_DIR, config.LABEL_FILE, 'val', config.SCALES, augment=False)
    
    # Identify hard samples
    train_loader_inf = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    hard_train_ids = identify_hard_samples(model, train_loader_inf, device)
    for img_id in hard_train_ids:
        train_dataset.sample_weights[img_id] = 3.0
        
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    # 3. SAM Optimizer
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=lr, weight_decay=0.01)
    
    criterion = CTMUSIQLoss(lambda_kl=config.LAMBDA_KL).to(device)
    best_aggregate = checkpoint['best_aggregate']
    
    print(f"Starting SAM training. Baseline: {best_aggregate:.4f}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            patches = batch['patches'].to(device)
            coords = batch['coords'].to(device)
            targets = batch['score'].to(device)
            weights = batch['sample_weight'].to(device)
            
            # FIRST PASS
            output = model(patches, coords)
            loss = (weights * (output['score'].squeeze(-1) - targets)**2).mean() + \
                   config.LAMBDA_KL * criterion.compute_kl_loss(output['scale_scores'], output['score'])
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # SECOND PASS
            output = model(patches, coords)
            loss = (weights * (output['score'].squeeze(-1) - targets)**2).mean() + \
                   config.LAMBDA_KL * criterion.compute_kl_loss(output['scale_scores'], output['score'])
            loss.backward()
            optimizer.second_step(zero_grad=True)
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validate
        _, val_metrics = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} Val Aggregate: {val_metrics['Aggregate']:.4f} (Best: {best_aggregate:.4f})")
        
        if val_metrics['Aggregate'] > best_aggregate:
            best_aggregate = val_metrics['Aggregate']
            print(f"  → NEW SAM BEST! Saving...")
            torch.save({
                'epoch': 160 + epoch,
                'model_state_dict': model.state_dict(),
                'best_aggregate': best_aggregate,
                'metrics': val_metrics,
                'sam_optimized': True,
                'config': checkpoint.get('config', {})
            }, os.path.join(config.RESULTS_DIR, "ct_musiq_sam_best.pth"))

if __name__ == "__main__":
    cp = "results/ct_musiq_hard_mining_best.pth"
    train_sam(cp)
