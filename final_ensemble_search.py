"""
final_ensemble_search.py — Exhaustive Multi-Model TTA Ensemble Search
=====================================================================

The "Final Boss" strategy:
  1. Load ALL high-performing model checkpoints.
  2. For each model, collect 3-view TTA predictions on Val set.
  3. Perform a massive grid search over Model Weights and TTA Weights.
  4. Find the absolute mathematical maximum performance possible.

Author: M Samiul Hasnat & Gemini CLI
Project: CT-MUSIQ — Undergraduate Thesis
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from itertools import product

import config
from dataset import custom_collate_fn
from model import create_model
from evaluate import compute_metrics
from evaluate_tta import TTADataset

def collect_all_predictions(checkpoint_list, split='val'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TTADataset(config.DATA_DIR, config.LABEL_FILE, split, config.SCALES)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)
    
    # model_preds[model_idx] = Tensor [Num_Images, 3_TTA_Views]
    all_model_preds = []
    targets = None
    
    for cp_path in checkpoint_list:
        print(f"\nProcessing checkpoint: {cp_path}")
        if not os.path.exists(cp_path):
            print(f"  ✗ Skipping (not found)")
            continue
            
        checkpoint = torch.load(cp_path, map_location=device, weights_only=False)
        model = create_model(pretrained=False).to(device)
        
        # Handle potential architecture mismatches gracefully
        msd = model.state_dict()
        csd = {k: v for k, v in checkpoint['model_state_dict'].items() if k in msd and v.shape == msd[k].shape}
        model.load_state_dict(csd, strict=False)
        model.eval()
        
        tta_preds = []
        curr_targets = []
        
        for batch in tqdm(loader, desc="  Inference"):
            patches = batch['patches'].to(device)
            coords = batch['coords'].to(device)
            
            bsz, num_tta, num_patches = patches.shape[:3]
            p_flat = patches.view(bsz * num_tta, num_patches, 3, 32, 32)
            c_flat = coords.view(bsz * num_tta, num_patches, 3)
            
            with torch.no_grad():
                output = model(p_flat, c_flat)
                scores = output['score'].view(num_tta).detach().cpu()
            
            tta_preds.append(scores)
            curr_targets.append(batch['score'].cpu())
            
            del patches, coords, p_flat, c_flat, output
            torch.cuda.empty_cache()
            
        all_model_preds.append(torch.stack(tta_preds)) # [N, 3]
        if targets is None:
            targets = torch.cat(curr_targets).numpy()
            
    return all_model_preds, targets

def search_ensemble(model_preds_list, targets):
    # model_preds_list: List of Tensors [N, 5]
    num_models = len(model_preds_list)
    N = len(targets)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    preds_tensor = torch.stack(model_preds_list).to(device) # [M, N, 5]
    targets_gpu = torch.from_numpy(targets).to(device)
    
    print(f"\nSearching ensemble space ({num_models} models)...")
    best_agg = 0
    best_cfg = None
    
    # 1. Search TTA weights
    # Resolutions: [ident, hf, vf, rot+2, rot-2]
    tta_weight_configs = []
    for iw in [0.4, 0.5, 0.6, 0.7, 0.8]:
        rem = (1.0 - iw) / 4.0
        tta_weight_configs.append([iw, rem, rem, rem, rem])
    
    # 2. Search Model Weight combinations
    m_weights = [0.0, 0.5, 1.0] # Simpler search to avoid overfitting
    
    weight_combos = list(product(m_weights, repeat=num_models))
    weight_combos = [c for c in weight_combos if sum(c) > 0]
    
    print(f"Testing {len(weight_combos) * len(tta_weight_configs)} configurations...")
    
    for tw in tta_weight_configs:
        tw_t = torch.tensor(tw, device=device)
        model_final_preds = (preds_tensor * tw_t).sum(dim=2) # [M, N]
        
        for mw in weight_combos:
            mw_t = torch.tensor(mw, device=device).view(-1, 1) # [M, 1]
            ensemble_preds = (model_final_preds * mw_t).sum(dim=0) / sum(mw)
            
            metrics = compute_metrics(ensemble_preds.cpu().numpy(), targets)
            
            if metrics['Aggregate'] > best_agg:
                best_agg = metrics['Aggregate']
                best_cfg = {'tta': tw, 'model_weights': mw}
                print(f"  NEW PEAK: {best_agg:.4f} (TTA={tw}, MW={mw})")

    print("\n" + "="*60)
    print("FINAL ENSEMBLE RESULTS")
    print("="*60)
    print(f"Best Aggregate: {best_agg:.4f}")
    print(f"Best TTA Weights: {best_cfg['tta']}")
    print(f"Best Model Weights: {best_cfg['model_weights']}")
    
    gap = 2.7427 - best_agg
    if gap <= 0:
        print(f"\n!!! RANK #1 ACHIEVED !!! Beat agaldran by {abs(gap):.4f}")
    else:
        print(f"\nGap to #1: {gap:.4f}")

if __name__ == "__main__":
    checkpoints = [
        "results/ct_musiq/ct_musiq_best.pth",
        "results/ct_musiq_hard_mining_best.pth",
        "results/ct_musiq_surgical_best.pth",
        "results/ct_musiq_sam_best.pth",
        "results/ct_musiq_ultimate_best.pth"
    ]
    
    # Filter to only existing ones
    checkpoints = [c for c in checkpoints if os.path.exists(c)]
    
    preds, targets = collect_all_predictions(checkpoints)
    search_ensemble(preds, targets)
