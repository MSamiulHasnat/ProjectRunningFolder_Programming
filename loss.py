"""
loss.py — CT-MUSIQ Loss Functions
==================================

Combined loss for training CT-MUSIQ:
  1. Primary Loss: MSE between predicted and ground truth quality scores
  2. Scale-Consistency Loss: KL divergence between per-scale predictions
     and the global prediction

The KL loss encourages agreement across scales, preventing the model from
learning scale-specific shortcuts. This is a key innovation over vanilla MUSIQ.

Loss Formula:
  L_total = L_mse + λ * L_kl
  
Where:
  - L_mse = MSE(global_score, target_score)
  - L_kl = mean(KL(scale_i || global) for each scale i)
  - λ = LAMBDA_KL (default: 0.1)

Score-to-Distribution Conversion:
  Scalar scores are converted to soft probability distributions using a
  Gaussian kernel centered at the score value. This enables KL divergence
  computation between predictions.

Author: M Samiul Hasnat, Sichuan University
Project: CT-MUSIQ — Undergraduate Thesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

# Import project configuration
import config


class ScoreToDistribution(nn.Module):
    """
    Convert scalar quality scores to soft probability distributions.
    
    Uses a Gaussian kernel centered at the score value to create a smooth
    distribution over discrete bins. This enables KL divergence computation
    between different score predictions.
    
    Example:
        score = 2.5, bins = [0, 0.2, 0.4, ..., 4.0]
        → distribution peaks around bin 12-13 (corresponding to 2.4-2.6)
    
    Args:
        num_bins: Number of discrete bins (default: 20)
        score_min: Minimum score value (default: 0.0)
        score_max: Maximum score value (default: 4.0)
        sigma: Gaussian kernel width (default: 0.5)
    """
    
    def __init__(
        self,
        num_bins: int = config.NUM_BINS,
        score_min: float = config.SCORE_MIN,
        score_max: float = config.SCORE_MAX,
        sigma: float = config.SIGMA
    ):
        super().__init__()
        
        self.num_bins = num_bins
        self.score_min = score_min
        self.score_max = score_max
        self.sigma = sigma
        
        # Create bin centers (fixed, not learnable)
        # Bins are evenly spaced from score_min to score_max
        bin_centers = torch.linspace(score_min, score_max, num_bins)
        self.register_buffer('bin_centers', bin_centers)
    
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Convert scores to probability distributions.
        
        Args:
            scores: Tensor of shape [B] or [B, 1] — quality scores
            
        Returns:
            Tensor of shape [B, num_bins] — probability distributions
        """
        # Ensure scores are 1D: [B]
        if scores.dim() > 1:
            scores = scores.squeeze(-1)
        
        # Expand scores for broadcasting: [B, 1]
        scores = scores.unsqueeze(-1)
        
        # Compute Gaussian kernel: [B, num_bins]
        # Each bin gets a weight based on distance from the score
        # weight = exp(-(score - bin_center)^2 / (2 * sigma^2))
        diff = scores - self.bin_centers.unsqueeze(0)  # [B, num_bins]
        weights = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        
        # Normalize to probability distribution: sum to 1
        distributions = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return distributions


class CTMUSIQLoss(nn.Module):
    """
    Combined loss for CT-MUSIQ training.
    
    Components:
      1. MSE Loss: Primary regression loss for quality score prediction
      2. KL Divergence Loss: Scale-consistency regularization
      
    The KL loss converts scores to soft distributions and measures
    disagreement between per-scale predictions and the global prediction.
    
    Args:
        lambda_kl: Weight for KL loss (default: 0.1)
        num_bins: Number of bins for score distribution (default: 20)
        score_min: Minimum score value (default: 0.0)
        score_max: Maximum score value (default: 4.0)
        sigma: Gaussian kernel width (default: 0.5)
    """
    
    def __init__(
        self,
        lambda_kl: float = config.LAMBDA_KL,
        num_bins: int = config.NUM_BINS,
        score_min: float = config.SCORE_MIN,
        score_max: float = config.SCORE_MAX,
        sigma: float = config.SIGMA
    ):
        super().__init__()
        
        self.lambda_kl = lambda_kl
        
        # Score to distribution converter
        self.score_to_dist = ScoreToDistribution(
            num_bins=num_bins,
            score_min=score_min,
            score_max=score_max,
            sigma=sigma
        )
        
        # KL divergence loss
        # Note: KLDivLoss expects log-probabilities as input
        # reduction='batchmean' averages over batch dimension
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # MSE loss for primary regression
        self.mse_loss = nn.MSELoss()
    
    def compute_mse_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target scores.
        
        Args:
            predicted: Tensor [B, 1] — model predictions
            target: Tensor [B] — ground truth scores
            
        Returns:
            Scalar MSE loss
        """
        # Squeeze predicted to match target shape
        predicted = predicted.squeeze(-1)
        
        return self.mse_loss(predicted, target)
    
    def compute_kl_loss(
        self,
        scale_scores: List[torch.Tensor],
        global_score: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence loss for scale consistency.
        
        Converts each scale's prediction and the global prediction to
        probability distributions, then measures KL divergence between
        each scale distribution and the global distribution.
        
        Args:
            scale_scores: List of [B, 1] tensors — per-scale predictions
            global_score: Tensor [B, 1] — global prediction
            
        Returns:
            Scalar KL loss (averaged across scales)
        """
        # Convert global score to distribution (target for KL)
        global_dist = self.score_to_dist(global_score.squeeze(-1))
        
        # KL divergence is asymmetric: KL(P || Q) where P is target
        # We want scale distributions to match global, so:
        # KL(global || scale_i) — global is target (P), scale is input (Q)
        # KLDivLoss expects: input = log(Q), target = P
        
        # Take log of global distribution (target)
        # Add small epsilon to avoid log(0)
        global_log_dist = torch.log(global_dist + 1e-8)
        
        # Compute KL for each scale
        kl_losses = []
        for scale_score in scale_scores:
            # Convert scale score to distribution
            scale_dist = self.score_to_dist(scale_score.squeeze(-1))
            
            # KL(global || scale) = sum(P * log(P/Q))
            # KLDivLoss(input, target) computes: target * (log(target) - input)
            # So we pass: input = log(scale_dist), target = global_dist
            scale_log_dist = torch.log(scale_dist + 1e-8)
            
            # Compute KL divergence
            kl = self.kl_loss(scale_log_dist, global_dist)
            kl_losses.append(kl)
        
        # Average across scales
        mean_kl = torch.stack(kl_losses).mean()
        
        return mean_kl
    
    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            output: Dictionary from model with:
                'score': Tensor [B, 1] — global prediction
                'scale_scores': List of Tensor [B, 1] — per-scale predictions
            target: Tensor [B] — ground truth quality scores
            
        Returns:
            Dictionary with:
                'total': Total loss (MSE + λ * KL)
                'mse': MSE loss component
                'kl': KL loss component
        """
        global_score = output['score']
        scale_scores = output['scale_scores']
        
        # Compute MSE loss
        mse_loss = self.compute_mse_loss(global_score, target)
        
        # Compute KL loss (only if lambda > 0 and we have scale scores)
        if self.lambda_kl > 0 and len(scale_scores) > 0:
            kl_loss = self.compute_kl_loss(scale_scores, global_score)
        else:
            kl_loss = torch.tensor(0.0, device=target.device)
        
        # Combined loss
        total_loss = mse_loss + self.lambda_kl * kl_loss
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'kl': kl_loss
        }


def create_criterion(
    lambda_kl: float = config.LAMBDA_KL,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> CTMUSIQLoss:
    """
    Factory function to create loss criterion.
    
    Args:
        lambda_kl: Weight for KL loss
        device: Device to place tensors on
        
    Returns:
        CT-MUSIQ loss criterion
    """
    criterion = CTMUSIQLoss(lambda_kl=lambda_kl)
    criterion = criterion.to(device)
    
    return criterion


# =============================================================================
# TESTING / VERIFICATION
# =============================================================================

if __name__ == "__main__":
    """
    Quick test to verify the loss functions.
    Run: python loss.py
    """
    print("="*60)
    print("CT-MUSIQ Loss Functions Test")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Test ScoreToDistribution
    print("\n" + "-"*40)
    print("Testing ScoreToDistribution...")
    print("-"*40)
    
    score_converter = ScoreToDistribution(num_bins=20, sigma=0.5)
    
    # Test with sample scores
    test_scores = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    distributions = score_converter(test_scores)
    
    print(f"\nInput scores: {test_scores.tolist()}")
    print(f"Distribution shape: {distributions.shape}")
    print(f"Distribution sums (should be ~1.0): {distributions.sum(dim=-1).tolist()}")
    
    # Show which bins have highest probability for each score
    bin_centers = score_converter.bin_centers
    for i, score in enumerate(test_scores):
        max_bin = distributions[i].argmax()
        print(f"  Score {score:.1f} → peak at bin {max_bin} (center={bin_centers[max_bin]:.2f})")
    
    # Test CTMUSIQLoss
    print("\n" + "-"*40)
    print("Testing CTMUSIQLoss...")
    print("-"*40)
    
    criterion = CTMUSIQLoss(lambda_kl=0.1)
    criterion = criterion.to(device)
    
    # Create dummy model output
    batch_size = 4
    dummy_output = {
        'score': torch.randn(batch_size, 1).to(device),
        'scale_scores': [
            torch.randn(batch_size, 1).to(device),
            torch.randn(batch_size, 1).to(device)
        ]
    }
    
    # Create dummy targets
    dummy_targets = torch.tensor([2.5, 1.8, 3.2, 0.5]).to(device)
    
    # Compute loss
    losses = criterion(dummy_output, dummy_targets)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Lambda KL: {config.LAMBDA_KL}")
    print(f"\nLoss components:")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  MSE loss:   {losses['mse'].item():.4f}")
    print(f"  KL loss:    {losses['kl'].item():.4f}")
    print(f"  KL weight:  {config.LAMBDA_KL * losses['kl'].item():.4f}")
    
    # Verify loss is differentiable
    print("\nTesting gradient flow...")
    dummy_output['score'].requires_grad_(True)
    losses = criterion(dummy_output, dummy_targets)
    losses['total'].backward()
    
    if dummy_output['score'].grad is not None:
        print(f"  ✓ Gradients computed successfully")
        print(f"  Gradient norm: {dummy_output['score'].grad.norm().item():.4f}")
    else:
        print(f"  ✗ No gradients computed!")
    
    # Test with single scale (no KL loss)
    print("\n" + "-"*40)
    print("Testing with single scale (KL disabled)...")
    print("-"*40)
    
    criterion_no_kl = CTMUSIQLoss(lambda_kl=0.0)
    dummy_output_single = {
        'score': torch.randn(batch_size, 1).to(device),
        'scale_scores': []  # No scale scores
    }
    
    losses_no_kl = criterion_no_kl(dummy_output_single, dummy_targets)
    print(f"\nLoss components (no KL):")
    print(f"  Total loss: {losses_no_kl['total'].item():.4f}")
    print(f"  MSE loss:   {losses_no_kl['mse'].item():.4f}")
    print(f"  KL loss:    {losses_no_kl['kl'].item():.4f}")
    
    print("\n✓ Loss functions test complete!")
