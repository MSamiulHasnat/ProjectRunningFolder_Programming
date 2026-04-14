"""
evaluate.py -- Model Evaluation Script
======================================

Evaluate trained models on the test set.
Computes metrics matching the LDCTIQAC 2023 leaderboard format:
  - PLCC: Pearson Linear Correlation Coefficient
  - SROCC: Spearman Rank-Order Correlation Coefficient
  - KROCC: Kendall Rank-Order Correlation Coefficient
  - Aggregate: |PLCC| + |SROCC| + |KROCC|

Works with CT-MUSIQ and one combined baseline method.

Usage:
  python evaluate.py                                    # Evaluate CT-MUSIQ
  python evaluate.py --model agaldran_combo             # Evaluate combined baseline
  python evaluate.py --model ct_musiq --checkpoint path.pth
  python evaluate.py --tta                              # TTA with horizontal flip
  python evaluate.py --tta --calibrate                  # TTA + val-set calibration

Author: M Samiul Hasnat, Sichuan University
Project: CT-MUSIQ -- Undergraduate Thesis
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import torch
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.isotonic import IsotonicRegression

# Import project modules
import config
from dataset import create_dataloaders
from model import create_model, CTMUSIQ
from get_model import get_model


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
    plcc, plcc_p = pearsonr(predictions, targets)
    srocc, srocc_p = spearmanr(predictions, targets)
    krocc, krocc_p = kendalltau(predictions, targets)
    
    return {
        'PLCC': round(plcc, 4),
        'PLCC_p': plcc_p,
        'SROCC': round(srocc, 4),
        'SROCC_p': srocc_p,
        'KROCC': round(krocc, 4),
        'KROCC_p': krocc_p,
        'Aggregate': round(abs(plcc) + abs(srocc) + abs(krocc), 4)
    }


def build_baseline_images_from_patches(
    patches: torch.Tensor,
    coords: torch.Tensor,
    target_scale_idx: int = 0
) -> torch.Tensor:
    """
    Reconstruct one full image per sample from patch tokens of a given scale.

    Baseline models need 4D images [B, 3, H, W], while this project dataloader
    provides patch tokens [B, N, 3, P, P] plus coordinates [B, N, 3].
    We reconstruct using scale 0 patches (default 224x224 path).
    """
    device = patches.device
    bsz, _, channels, patch_h, patch_w = patches.shape
    images = []

    for b in range(bsz):
        mask = coords[b, :, 0] == target_scale_idx
        p = patches[b][mask]
        c = coords[b][mask]

        if p.numel() == 0:
            raise ValueError("No patches found for baseline reconstruction at selected scale")

        max_row = int(c[:, 1].max().item()) + 1
        max_col = int(c[:, 2].max().item()) + 1
        img = torch.zeros((channels, max_row * patch_h, max_col * patch_w), device=device, dtype=patches.dtype)

        for i in range(p.shape[0]):
            row = int(c[i, 1].item())
            col = int(c[i, 2].item())
            r0 = row * patch_h
            c0 = col * patch_w
            img[:, r0:r0 + patch_h, c0:c0 + patch_w] = p[i]

        images.append(img)

    return torch.stack(images, dim=0)


def flip_patches_horizontal(
    patches: torch.Tensor,
    coords: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Horizontally flip all patches and mirror their column coordinates.

    For each scale with grid_size G, the column index is remapped:
        col_new = (G - 1) - col_old

    Args:
        patches: [B, N, 3, P, P]
        coords:  [B, N, 3]  -- (scale_idx, row_idx, col_idx)

    Returns:
        (patches_flipped, coords_flipped) with same shapes
    """
    patches_flipped = torch.flip(patches, dims=[-1])  # flip width dimension
    coords_flipped = coords.clone()

    col_vals = coords_flipped[:, :, 2].clone()  # [B, N] — col indices

    for scale_i, scale in enumerate(config.SCALES):
        grid_size = scale // config.PATCH_SIZE
        mask = coords[:, :, 0] == scale_i  # [B, N] bool mask
        col_vals[mask] = (grid_size - 1) - col_vals[mask]

    coords_flipped[:, :, 2] = col_vals
    return patches_flipped, coords_flipped


def calibrate_predictions(
    train_preds: np.ndarray,
    train_targets: np.ndarray,
    test_preds: np.ndarray
) -> np.ndarray:
    """
    Apply isotonic regression calibration fitted on val-set predictions.

    Isotonic regression learns a monotone mapping from raw predictions to
    calibrated scores, preserving rank order while correcting scale/bias.
    This cannot harm SROCC/KROCC and typically improves PLCC.

    Args:
        train_preds:  Val-set raw predictions
        train_targets: Val-set ground-truth scores
        test_preds:   Test-set raw predictions to calibrate

    Returns:
        Calibrated test predictions
    """
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(train_preds, train_targets)
    return ir.predict(test_preds)


@torch.no_grad()
def evaluate_model(
    model,
    test_loader,
    device: torch.device,
    model_type: str = 'ct_musiq',
    use_tta: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Run model on test/val set and collect predictions.

    Args:
        model: Trained model
        test_loader: Data loader (test or val)
        device: Device to run on
        model_type: Type of model ('ct_musiq', 'agaldran_combo')
        use_tta: If True, average predictions from original + horizontal flip

    Returns:
        Tuple of (predictions, targets, image_ids)
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_image_ids = []

    for batch in test_loader:
        if model_type == 'ct_musiq':
            patches = batch['patches'].to(device)
            coords = batch['coords'].to(device)
            scores = batch['score'].to(device)
            image_ids = batch['image_id']

            output = model(patches, coords)
            pred = output['score'].squeeze(-1)

            if use_tta:
                patches_f, coords_f = flip_patches_horizontal(patches, coords)
                output_f = model(patches_f, coords_f)
                pred = (pred + output_f['score'].squeeze(-1)) / 2.0

        else:
            patches = batch['patches'].to(device)
            coords = batch['coords'].to(device)
            images = build_baseline_images_from_patches(patches, coords, target_scale_idx=0)
            scores = batch['score'].to(device)
            image_ids = batch['image_id']

            output = model(images, None)
            pred = output['score'].squeeze(-1)

        all_predictions.extend(pred.cpu().numpy())
        all_targets.extend(scores.cpu().numpy())
        all_image_ids.extend(image_ids)

    return np.array(all_predictions), np.array(all_targets), all_image_ids


def save_predictions_csv(
    predictions: np.ndarray,
    targets: np.ndarray,
    image_ids: List[str],
    save_path: str
) -> None:
    """
    Save per-image predictions to CSV.
    
    Args:
        predictions: Predicted quality scores
        targets: Ground truth quality scores
        image_ids: Image identifiers
        save_path: Path to save CSV
    """
    df = pd.DataFrame({
        'image_id': image_ids,
        'predicted': predictions,
        'target': targets,
        'error': predictions - targets,
        'abs_error': np.abs(predictions - targets)
    })
    
    df.to_csv(save_path, index=False)
    print(f"  Predictions saved to: {save_path}")


def print_comparison_table(metrics: Dict[str, float]) -> None:
    """
    Print comparison table with Lee et al. 2025 published results.
    
    Args:
        metrics: Dictionary with computed metrics
    """
    print("\n" + "="*80)
    print("COMPARISON WITH PUBLISHED RESULTS (Lee et al. 2025, Medical Image Analysis)")
    print("="*80)
    
    # Published results from Lee et al. 2025 Table 3
    published_results = [
        ("agaldran", 2.7427, 0.9491, 0.9495, 0.8440),
        ("RPI_AXIS", 2.6843, 0.9434, 0.9414, 0.7995),
        ("CHILL@UK", 2.6719, 0.9402, 0.9387, 0.7930),
        ("FeatureNet", 2.6550, 0.9362, 0.9338, 0.7851),
        ("Team Epoch", 2.6202, 0.9278, 0.9232, 0.7691),
        ("gabybaldeon", 2.5671, 0.9143, 0.9096, 0.7432),
        ("SNR baseline", 2.4026, 0.8226, 0.8748, 0.7052),
        ("BRISQUE", 2.1219, 0.7500, 0.7863, 0.5856),
    ]
    
    # Print header
    print(f"\n{'Model':<20} {'Aggregate':>10} {'PLCC':>8} {'SROCC':>8} {'KROCC':>8} {'Source':<15}")
    print("-"*80)
    
    # Print published results
    for model_name, agg, plcc, srocc, krocc in published_results:
        print(f"{model_name:<20} {agg:>10.4f} {plcc:>8.4f} {srocc:>8.4f} {krocc:>8.4f} {'Lee et al. 2025':<15}")
    
    # Print our results
    print("-"*80)
    print(f"{'CT-MUSIQ (ours)':<20} {metrics['Aggregate']:>10.4f} {metrics['PLCC']:>8.4f} {metrics['SROCC']:>8.4f} {metrics['KROCC']:>8.4f} {'This work':<15}")
    
    # Find ranking
    all_aggregates = [r[1] for r in published_results] + [metrics['Aggregate']]
    all_aggregates_sorted = sorted(all_aggregates, reverse=True)
    rank = all_aggregates_sorted.index(metrics['Aggregate']) + 1
    
    print("\n" + "-"*80)
    print(f"CT-MUSIQ Rank: #{rank} out of {len(published_results) + 1} methods")
    print("-"*80)


def save_results_csv(metrics: Dict[str, float], save_path: str) -> None:
    """
    Save results table to CSV for thesis inclusion.
    
    Args:
        metrics: Dictionary with computed metrics
        save_path: Path to save CSV
    """
    # Published results from Lee et al. 2025
    results = [
        {"Model": "agaldran", "Aggregate": 2.7427, "PLCC": 0.9491, "SROCC": 0.9495, "KROCC": 0.8440, "Source": "Lee et al. 2025"},
        {"Model": "RPI_AXIS", "Aggregate": 2.6843, "PLCC": 0.9434, "SROCC": 0.9414, "KROCC": 0.7995, "Source": "Lee et al. 2025"},
        {"Model": "CHILL@UK", "Aggregate": 2.6719, "PLCC": 0.9402, "SROCC": 0.9387, "KROCC": 0.7930, "Source": "Lee et al. 2025"},
        {"Model": "FeatureNet", "Aggregate": 2.6550, "PLCC": 0.9362, "SROCC": 0.9338, "KROCC": 0.7851, "Source": "Lee et al. 2025"},
        {"Model": "Team Epoch", "Aggregate": 2.6202, "PLCC": 0.9278, "SROCC": 0.9232, "KROCC": 0.7691, "Source": "Lee et al. 2025"},
        {"Model": "gabybaldeon", "Aggregate": 2.5671, "PLCC": 0.9143, "SROCC": 0.9096, "KROCC": 0.7432, "Source": "Lee et al. 2025"},
        {"Model": "SNR baseline", "Aggregate": 2.4026, "PLCC": 0.8226, "SROCC": 0.8748, "KROCC": 0.7052, "Source": "Lee et al. 2025"},
        {"Model": "BRISQUE", "Aggregate": 2.1219, "PLCC": 0.7500, "SROCC": 0.7863, "KROCC": 0.5856, "Source": "Lee et al. 2025"},
        {"Model": "CT-MUSIQ (ours)", "Aggregate": metrics['Aggregate'], "PLCC": metrics['PLCC'], "SROCC": metrics['SROCC'], "KROCC": metrics['KROCC'], "Source": "This work"},
    ]
    
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"  Results table saved to: {save_path}")


def evaluate(
    model_type: str = 'ct_musiq',
    checkpoint_path: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE,
    use_tta: bool = False,
    use_calibration: bool = False
) -> Dict[str, float]:
    """
    Main evaluation function.

    Args:
        model_type: Type of model ('ct_musiq', 'agaldran_combo')
        checkpoint_path: Path to model checkpoint (default: auto-detected)
        batch_size: Batch size for evaluation
        use_tta: If True, average predictions with horizontal flip TTA
        use_calibration: If True, apply isotonic regression calibration using val set

    Returns:
        Dictionary with computed metrics
    """
    print("="*70)
    print(f"{model_type.upper()} Evaluation")
    if use_tta:
        print("  TTA: enabled (horizontal flip)")
    if use_calibration:
        print("  Calibration: enabled (isotonic regression on val set)")
    print("="*70)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Determine checkpoint path -- prefer the per-model subdirectory
    if checkpoint_path is None:
        if model_type == 'ct_musiq':
            per_model_path = os.path.join(config.RESULTS_DIR, 'ct_musiq', 'ct_musiq_best.pth')
            if os.path.exists(per_model_path):
                checkpoint_path = per_model_path
            else:
                checkpoint_path = config.BEST_MODEL_PATH  # legacy fallback
        else:
            model_results_dir = os.path.join(config.RESULTS_DIR, model_type)
            checkpoint_path = os.path.join(model_results_dir, f'{model_type}_best.pth')

    if not os.path.exists(checkpoint_path):
        print(f"\nFAIL Checkpoint not found: {checkpoint_path}")
        print(f"  Please train {model_type} first: python train.py --model {model_type}")
        sys.exit(1)

    print(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best aggregate: {checkpoint['best_aggregate']:.4f}")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True
    )
    print(f"  Val images:  {len(val_loader.dataset)}")
    print(f"  Test images: {len(test_loader.dataset)}")

    # Create model
    print(f"\nCreating {model_type.upper()} model...")
    if model_type == 'ct_musiq':
        checkpoint_config = checkpoint.get('config', {})
        scales = checkpoint_config.get('scales', config.SCALES)
        num_scales = len(scales)
        model = create_model(
            num_scales=num_scales,
            pretrained=False,
            device=device
        )
    else:
        model = get_model(model_type, pretrained=False)
        model = model.to(device)

    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing:
        print(f"  WARN Missing keys (random init, not critical): {missing}")
    if unexpected:
        print(f"  WARN Unexpected keys (ignored): {unexpected}")
    print("  OK Model weights loaded")

    # Optional calibration: fit isotonic regression on val set
    calibrator = None
    if use_calibration:
        print("\nFitting calibration on val set...")
        val_preds, val_targets, _ = evaluate_model(
            model, val_loader, device, model_type=model_type, use_tta=use_tta
        )
        val_metrics_raw = compute_metrics(val_preds, val_targets)
        print(f"  Val raw  -- PLCC: {val_metrics_raw['PLCC']:.4f}  SROCC: {val_metrics_raw['SROCC']:.4f}  "
              f"KROCC: {val_metrics_raw['KROCC']:.4f}  Agg: {val_metrics_raw['Aggregate']:.4f}")

        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(val_preds, val_targets)
        calibrator = ir
        val_preds_cal = ir.predict(val_preds)
        val_metrics_cal = compute_metrics(val_preds_cal, val_targets)
        print(f"  Val calib -- PLCC: {val_metrics_cal['PLCC']:.4f}  SROCC: {val_metrics_cal['SROCC']:.4f}  "
              f"KROCC: {val_metrics_cal['KROCC']:.4f}  Agg: {val_metrics_cal['Aggregate']:.4f}")

    # Evaluate on test set
    print("\nRunning evaluation on test set...")
    predictions, targets, image_ids = evaluate_model(
        model, test_loader, device, model_type=model_type, use_tta=use_tta
    )

    if calibrator is not None:
        predictions = calibrator.predict(predictions)

    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    
    # Print results
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(f"\n  PLCC:      {metrics['PLCC']:.4f}  (p={metrics['PLCC_p']:.2e})")
    print(f"  SROCC:     {metrics['SROCC']:.4f}  (p={metrics['SROCC_p']:.2e})")
    print(f"  KROCC:     {metrics['KROCC']:.4f}  (p={metrics['KROCC_p']:.2e})")
    print(f"  Aggregate: {metrics['Aggregate']:.4f}")
    
    # Print prediction statistics
    print(f"\n  Prediction statistics:")
    print(f"    Mean:   {predictions.mean():.4f}")
    print(f"    Std:    {predictions.std():.4f}")
    print(f"    Min:    {predictions.min():.4f}")
    print(f"    Max:    {predictions.max():.4f}")
    
    print(f"\n  Target statistics:")
    print(f"    Mean:   {targets.mean():.4f}")
    print(f"    Std:    {targets.std():.4f}")
    print(f"    Min:    {targets.min():.4f}")
    print(f"    Max:    {targets.max():.4f}")
    
    # Print comparison table
    print_comparison_table(metrics)
    
    # Save results
    print("\nSaving results...")
    
    # Create results directory and model-specific subdirectory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    if model_type != 'ct_musiq':
        model_results_dir = os.path.join(config.RESULTS_DIR, model_type)
        os.makedirs(model_results_dir, exist_ok=True)
    
    # Save per-image predictions
    predictions_path = os.path.join(
        config.RESULTS_DIR if model_type == 'ct_musiq' else os.path.join(config.RESULTS_DIR, model_type),
        f"{model_type}_predictions.csv"
    )
    save_predictions_csv(predictions, targets, image_ids, predictions_path)
    
    # Save results table
    results_path = os.path.join(
        config.RESULTS_DIR if model_type == 'ct_musiq' else os.path.join(config.RESULTS_DIR, model_type),
        f"{model_type}_results.csv"
    )
    save_results_csv(metrics, results_path)
    
    print("\nOK Evaluation complete!")
    
    return metrics


def main():
    """
    Main entry point with argument parsing.
    """
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument(
        '--model',
        type=str,
        choices=['ct_musiq', 'agaldran_combo'],
        default='ct_musiq',
        help='Model to evaluate (default: ct_musiq)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint (default: auto-detected from results/ct_musiq/)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config.BATCH_SIZE,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--tta',
        action='store_true',
        default=False,
        help='Enable test-time augmentation (horizontal flip averaging)'
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        default=False,
        help='Apply isotonic regression calibration fitted on the val set'
    )

    args = parser.parse_args()

    # Run evaluation
    metrics = evaluate(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        use_tta=args.tta,
        use_calibration=args.calibrate
    )


if __name__ == "__main__":
    main()
