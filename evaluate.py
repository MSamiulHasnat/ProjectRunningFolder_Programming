"""
evaluate.py — CT-MUSIQ Evaluation Script
==========================================

Evaluate trained CT-MUSIQ model on the test set.
Computes metrics matching the LDCTIQAC 2023 leaderboard format:
  - PLCC: Pearson Linear Correlation Coefficient
  - SROCC: Spearman Rank-Order Correlation Coefficient
  - KROCC: Kendall Rank-Order Correlation Coefficient
  - Aggregate: |PLCC| + |SROCC| + |KROCC|

Outputs:
  1. Console summary with metrics
  2. CSV file with per-image predictions
  3. Comparison table with Lee et al. 2025 published results

Usage:
  python evaluate.py                        # Evaluate best checkpoint
  python evaluate.py --checkpoint path.pth  # Evaluate specific checkpoint

Author: M Samiul Hasnat, Sichuan University
Project: CT-MUSIQ — Undergraduate Thesis
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import torch
from scipy.stats import pearsonr, spearmanr, kendalltau

# Import project modules
import config
from dataset import create_dataloaders
from model import create_model, CTMUSIQ


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


@torch.no_grad()
def evaluate_model(
    model: CTMUSIQ,
    test_loader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Run model on test set and collect predictions.
    
    Args:
        model: Trained CT-MUSIQ model
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        Tuple of (predictions, targets, image_ids)
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_image_ids = []
    
    for batch in test_loader:
        # Move data to device
        patches = batch['patches'].to(device)
        coords = batch['coords'].to(device)
        scores = batch['score'].to(device)
        image_ids = batch['image_id']
        
        # Forward pass
        output = model(patches, coords)
        
        # Collect predictions and targets
        predictions = output['score'].squeeze(-1).cpu().numpy()
        targets = scores.cpu().numpy()
        
        all_predictions.extend(predictions)
        all_targets.extend(targets)
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
    checkpoint_path: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE
) -> Dict[str, float]:
    """
    Main evaluation function.
    
    Args:
        checkpoint_path: Path to model checkpoint (default: best_model.pth)
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with computed metrics
    """
    print("="*60)
    print("CT-MUSIQ Evaluation")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = config.BEST_MODEL_PATH
    
    if not os.path.exists(checkpoint_path):
        print(f"\n✗ Checkpoint not found: {checkpoint_path}")
        print("  Please train the model first: python train.py")
        sys.exit(1)
    
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    checkpoint_config = checkpoint.get('config', {})
    scales = checkpoint_config.get('scales', config.SCALES)
    num_scales = len(scales)
    
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best aggregate: {checkpoint['best_aggregate']:.4f}")
    print(f"  Scales: {scales}")
    
    # Create test data loader
    print("\nCreating test data loader...")
    _, _, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True
    )
    print(f"  Test images: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating CT-MUSIQ model...")
    model = create_model(
        num_scales=num_scales,
        pretrained=False,  # We'll load from checkpoint
        device=device
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("  ✓ Model weights loaded")
    
    # Evaluate
    print("\nRunning evaluation on test set...")
    predictions, targets, image_ids = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
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
    
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Save per-image predictions
    predictions_path = os.path.join(config.RESULTS_DIR, "test_predictions.csv")
    save_predictions_csv(predictions, targets, image_ids, predictions_path)
    
    # Save results table
    results_path = config.TEST_RESULTS_CSV
    save_results_csv(metrics, results_path)
    
    print("\n✓ Evaluation complete!")
    
    return metrics


def main():
    """
    Main entry point with argument parsing.
    """
    parser = argparse.ArgumentParser(description='Evaluate CT-MUSIQ model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: best_model.pth)')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
