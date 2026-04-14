"""
evaluate_tta.py — Model Evaluation with User-Requested TTA Weights
==================================================================

Author: M Samiul Hasnat & Gemini CLI
Project: CT-MUSIQ — Undergraduate Thesis
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm

# Import project modules
import config
from dataset import LDCTDataset, custom_collate_fn
from model import create_model
from evaluate import compute_metrics, save_predictions_csv, print_comparison_table, save_results_csv


class TTADataset(LDCTDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 3 views: Identity, Horizontal Flip, Vertical Flip
        self.tta_transforms = ['identity', 'hflip', 'vflip']
        print(f"  TTA enabled with 3 stable views: {self.tta_transforms}")

    def apply_tta(self, image: np.ndarray, tta_type: str) -> np.ndarray:
        img_pil = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        if tta_type == 'hflip': img_pil = TF.hflip(img_pil)
        elif tta_type == 'vflip': img_pil = TF.vflip(img_pil)
        return np.array(img_pil, dtype=np.float32) / 255.0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_id = self.image_ids[idx]
        key = config.LABEL_KEY_FORMAT.format(idx=int(image_id))
        score = self.labels[key]
        base_image = self.load_image(image_id)
        
        all_tta_patches = []
        all_tta_coords = []
        
        for tta_type in self.tta_transforms:
            image = self.apply_tta(base_image, tta_type)
            pyramid = self.build_multi_scale_pyramid(image)
            
            patches_list = []
            coords_list = []
            for scale_idx, scale_image in enumerate(pyramid):
                p, c = self.extract_patches(scale_image, scale_idx)
                patches_list.append(p)
                coords_list.append(c)
            
            patches = self.replicate_to_rgb(np.concatenate(patches_list, axis=0))
            coords = np.concatenate(coords_list, axis=0)
            
            all_tta_patches.append(torch.from_numpy(patches).float())
            all_tta_coords.append(torch.from_numpy(coords).long())
            
        return {
            'patches': torch.stack(all_tta_patches),
            'coords': torch.stack(all_tta_coords),
            'score': torch.tensor(score, dtype=torch.float32),
            'image_id': image_id,
            'sample_weight': torch.tensor(self.sample_weights.get(image_id, 1.0), dtype=torch.float32)
        }


def run_weighted_tta(checkpoint_path, split='test'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print(f"CT-MUSIQ Weighted TTA Evaluation ({split} split)")
    print("="*70)
    
    # 1. Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Model loaded from {checkpoint_path}")
    
    # 2. Setup weights (User requested: 0.35 Identity, 0.33 flips)
    # Total sum = 0.35 + 0.33 + 0.33 = 1.01. We normalize to 1.0.
    weights_vec = torch.tensor([0.35, 0.33, 0.33], device=device)
    weights_vec = weights_vec / weights_vec.sum()
    print(f"  ✓ Using Weights: Identity={weights_vec[0]:.4f}, HFlip={weights_vec[1]:.4f}, VFlip={weights_vec[2]:.4f}")
    
    # 3. Setup data
    dataset = TTADataset(config.DATA_DIR, config.LABEL_FILE, split, config.SCALES)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)
    
    all_preds = []
    all_targets = []
    all_ids = []
    
    print(f"Running inference on {len(dataset)} images...")
    for batch in tqdm(loader):
        patches = batch['patches'].to(device) # [1, 3, N, 3, 32, 32]
        coords = batch['coords'].to(device)   # [1, 3, N, 3]
        
        bsz, num_tta, num_patches = patches.shape[:3]
        p_flat = patches.view(bsz * num_tta, num_patches, 3, 32, 32)
        c_flat = coords.view(bsz * num_tta, num_patches, 3)
        
        with torch.no_grad():
            output = model(p_flat, c_flat)
            tta_scores = output['score'].view(num_tta) # [3]
            
            # Apply weights
            final_score = (tta_scores * weights_vec).sum().item()
            
        all_preds.append(final_score)
        all_targets.append(batch['score'].item())
        all_ids.append(batch['image_id'][0])
        
        del patches, coords, p_flat, c_flat, output
        torch.cuda.empty_cache()
        
    # 4. Compute metrics
    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    
    print("\n" + "="*70)
    print(f"WEIGHTED TTA {split.upper()} RESULTS")
    print("="*70)
    print(f"  PLCC:      {metrics['PLCC']:.4f}")
    print(f"  SROCC:     {metrics['SROCC']:.4f}")
    print(f"  KROCC:     {metrics['KROCC']:.4f}")
    print(f"  Aggregate: {metrics['Aggregate']:.4f}")
    
    # Save results
    save_predictions_csv(np.array(all_preds), np.array(all_targets), all_ids, 
                         os.path.join(config.RESULTS_DIR, f"ct_musiq_weighted_tta_{split}_predictions.csv"))
    save_results_csv(metrics, os.path.join(config.RESULTS_DIR, f"ct_musiq_weighted_tta_{split}_results.csv"))
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="results/ct_musiq/ct_musiq_best.pth")
    parser.add_argument('--split', type=str, default="test")
    args = parser.parse_args()
    
    run_weighted_tta(args.checkpoint, args.split)
