"""
model.py — CT-MUSIQ Model Architecture
=======================================

Architectural adaptation of MUSIQ (Ke et al., ICCV 2021) for no-reference
perceptual image quality assessment of Low-Dose CT images.

Architecture Overview:
  1. Patch Embedding: Conv2d(3, 768, 32, 32) — loads pretrained ViT-B/32 weights
  2. Hash-based Positional Encoding: Sum of 3 nn.Embedding lookups
     (scale_idx, row_idx, col_idx) — handles variable multi-scale sequences
  3. Transformer Encoder: 12 layers, 8 heads, d_model=768, FFN=3072
  4. Global [CLS] Head: Linear(768, 1) → final quality score
  5. Per-scale Heads: Average pool per scale → Linear(768, 1) → scale scores

Key Innovations over vanilla MUSIQ:
  - Hash-based positional encoding for multi-scale patch sequences
  - Per-scale prediction heads for scale-consistency KL loss
  - Grayscale-to-RGB replication for pretrained weight compatibility

VRAM Budget (RTX 3060 6GB):
  - Model weights: ~1.5 GB
  - Activations (batch=4): ~1.5-2 GB
  - Optimizer states: ~1.5 GB
  - Mixed precision (fp16) saves ~40%

Author: M Samiul Hasnat, Sichuan University
Project: CT-MUSIQ — Undergraduate Thesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import warnings

# Import timm for pretrained ViT weights
try:
    import timm
except ImportError:
    warnings.warn("timm not installed. Pretrained weights will not be loaded.")
    timm = None

# Import project configuration
import config


class HashPositionalEncoding(nn.Module):
    """
    Hash-based spatial positional encoding for multi-scale patch sequences.
    
    Standard sinusoidal encodings assume a fixed sequence length, but our
    multi-scale pyramid produces variable-length sequences (49 + 144 = 193
    patches). Hash encoding handles this by decomposing position into three
    independent components:
    
      pos_enc = scale_embed(scale_idx) + row_embed(row_idx) + col_embed(col_idx)
    
    This allows the model to learn:
      - Scale-specific features (224 vs 384 resolution)
      - Spatial position within each scale's grid
      - Compositional relationships across scales
    
    Args:
        num_scales: Number of resolution scales (default: 2)
        max_grid_size: Maximum grid dimension (default: 13 for 384/32=12)
        d_model: Embedding dimension (default: 768)
    """
    
    def __init__(
        self,
        num_scales: int = 2,
        max_grid_size: int = 13,
        d_model: int = config.D_MODEL
    ):
        super().__init__()
        
        # Three separate embedding tables
        # scale_embed: learns scale-specific features (224 vs 384)
        self.scale_embed = nn.Embedding(num_scales, d_model)
        
        # row_embed: learns vertical position features
        self.row_embed = nn.Embedding(max_grid_size, d_model)
        
        # col_embed: learns horizontal position features
        self.col_embed = nn.Embedding(max_grid_size, d_model)
        
        # Initialize with small random values
        nn.init.normal_(self.scale_embed.weight, std=0.02)
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute positional encoding for a batch of patch coordinates.
        
        Args:
            coords: Tensor of shape [B, N, 3] where each row is
                    (scale_idx, row_idx, col_idx)
                    
        Returns:
            Positional encodings of shape [B, N, d_model]
        """
        # Extract coordinate components
        scale_idx = coords[:, :, 0]  # [B, N]
        row_idx = coords[:, :, 1]    # [B, N]
        col_idx = coords[:, :, 2]    # [B, N]
        
        # Look up embeddings and sum
        # Each embedding returns [B, N, d_model]
        pos_enc = (
            self.scale_embed(scale_idx) +
            self.row_embed(row_idx) +
            self.col_embed(col_idx)
        )
        
        return pos_enc


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer using Conv2d.
    
    Converts image patches into token embeddings. Uses a Conv2d with
    kernel_size=stride=patch_size to efficiently extract non-overlapping
    patches and project them to d_model dimensions.
    
    This layer is compatible with pretrained ViT-B/32 weights from timm.
    
    Args:
        patch_size: Side length of square patches (default: 32)
        in_channels: Number of input channels (default: 3 for RGB)
        d_model: Embedding dimension (default: 768)
    """
    
    def __init__(
        self,
        patch_size: int = config.PATCH_SIZE,
        in_channels: int = 3,
        d_model: int = config.D_MODEL
    ):
        super().__init__()
        
        # Conv2d acts as patch extractor + linear projection
        # Input: [B*N, 3, 32, 32] → Output: [B*N, 768, 1, 1]
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.patch_size = patch_size
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project patches to embeddings.
        
        Args:
            x: Tensor of shape [B*N, 3, 32, 32] (flattened batch of patches)
            
        Returns:
            Tensor of shape [B*N, d_model] (patch embeddings)
        """
        # Apply convolution: [B*N, 3, 32, 32] → [B*N, 768, 1, 1]
        x = self.proj(x)
        
        # Flatten spatial dimensions: [B*N, 768, 1, 1] → [B*N, 768]
        x = x.flatten(1)
        
        return x


class CTMUSIQ(nn.Module):
    """
    CT-MUSIQ: CT-adapted MUSIQ Transformer for brain CT quality assessment.
    
    Full model architecture:
      1. Patch Embedding: Convert patches to token embeddings
      2. Hash Positional Encoding: Add spatial + scale information
      3. Prepend [CLS] token for global representation
      4. Transformer Encoder: 6 layers of self-attention
      5. Global Head: [CLS] → quality score
      6. Per-scale Heads: Average pool per scale → scale scores
    
    Args:
        num_scales: Number of resolution scales (default: 2)
        d_model: Embedding dimension (default: 768)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 6)
        patch_size: Patch size for embedding (default: 32)
        max_grid_size: Max grid dimension for positional encoding (default: 13)
        dropout: Dropout rate (default: 0.1)
        pretrained: Whether to load pretrained ViT weights (default: True)
    """
    
    def __init__(
        self,
        num_scales: int = len(config.SCALES),
        d_model: int = config.D_MODEL,
        num_heads: int = config.NUM_HEADS,
        num_layers: int = config.NUM_LAYERS,
        patch_size: int = config.PATCH_SIZE,
        max_grid_size: int = config.MAX_GRID_SIZE,
        dropout: float = config.DROPOUT,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Store configuration
        self.num_scales = num_scales
        self.d_model = d_model
        self.num_layers = num_layers
        
        # =========================================================================
        # 1. PATCH EMBEDDING
        # =========================================================================
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=3,  # RGB (grayscale replicated)
            d_model=d_model
        )
        
        # =========================================================================
        # 2. HASH-BASED POSITIONAL ENCODING
        # =========================================================================
        # Core innovation: handles variable-length multi-scale sequences
        self.pos_encoding = HashPositionalEncoding(
            num_scales=num_scales,
            max_grid_size=max_grid_size,
            d_model=d_model
        )
        
        # =========================================================================
        # 3. LEARNABLE [CLS] TOKEN
        # =========================================================================
        # [CLS] token aggregates global information across all patches
        # Shape: [1, 1, d_model] — will be expanded to batch size
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # =========================================================================
        # 4. TRANSFORMER ENCODER
        # =========================================================================
        # Standard ViT encoder: multi-head self-attention + FFN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=config.FFN_DIM,  # 3072 = 4 * 768
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # Input shape: [B, N, D]
            norm_first=True    # Pre-norm for better training stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer norm after transformer
        self.norm = nn.LayerNorm(d_model)
        
        # =========================================================================
        # 5. PREDICTION HEADS
        # =========================================================================
        # Global head: [CLS] token -> final quality score
        self.global_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Per-scale heads: for KL consistency loss during training
        # Each scale gets its own prediction head
        self.scale_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
            for _ in range(num_scales)
        ])
        
        # =========================================================================
        # 6. LOAD PRETRAINED WEIGHTS
        # =========================================================================
        if pretrained and timm is not None:
            self._load_pretrained_weights()
        elif pretrained and timm is None:
            warnings.warn("timm not available. Using random initialization.")
        
        # Print model statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"CT-MUSIQ Model Initialized:")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size:           {total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    
    def _load_pretrained_weights(self) -> None:
        """
        Load pretrained weights from ViT-B/32.
        
        Attempts to load patch embedding and transformer encoder weights
        from timm's pretrained ViT-B/32. Only loads weights that match
        in shape — incompatible layers are kept with random initialization.
        """
        try:
            print("\nLoading pretrained ViT-B/32 weights...")
            
            # Load pretrained ViT-B/32 from timm
            pretrained_vit = timm.create_model(
                'vit_base_patch32_224',
                pretrained=True
            )
            
            # Get state dicts
            pretrained_dict = pretrained_vit.state_dict()
            model_dict = self.state_dict()
            
            # Track what gets loaded
            loaded_keys = []
            skipped_keys = []
            
            # Try to load patch embedding weights
            # Pretrained: patch_embed.proj.weight shape is [768, 3, 32, 32]
            # Our model: same shape — should match!
            if 'patch_embed.proj.weight' in pretrained_dict:
                pretrained_weight = pretrained_dict['patch_embed.proj.weight']
                if pretrained_weight.shape == self.patch_embed.proj.weight.shape:
                    model_dict['patch_embed.proj.weight'] = pretrained_weight
                    loaded_keys.append('patch_embed.proj.weight')
                else:
                    skipped_keys.append('patch_embed.proj.weight (shape mismatch)')
            
            if 'patch_embed.proj.bias' in pretrained_dict:
                pretrained_bias = pretrained_dict['patch_embed.proj.bias']
                if pretrained_bias.shape == self.patch_embed.proj.bias.shape:
                    model_dict['patch_embed.proj.bias'] = pretrained_bias
                    loaded_keys.append('patch_embed.proj.bias')
            
            # Try to load transformer encoder weights
            # Map pretrained encoder layers to our encoder
            for i in range(self.num_layers):
                # Pretrained has 12 layers, we use 6 — load first 6
                if i >= 12:
                    break
                    
                prefix = f'blocks.{i}.'
                our_prefix = f'transformer.layers.{i}.'
                
                # Map attention weights
                attn_mappings = {
                    'attn.qkv.weight': 'self_attn.in_proj_weight',
                    'attn.qkv.bias': 'self_attn.in_proj_bias',
                    'attn.proj.weight': 'self_attn.out_proj.weight',
                    'attn.proj.bias': 'self_attn.out_proj.bias',
                    'norm1.weight': 'norm1.weight',
                    'norm1.bias': 'norm1.bias',
                    'norm2.weight': 'norm2.weight',
                    'norm2.bias': 'norm2.bias',
                    'mlp.fc1.weight': 'linear1.weight',
                    'mlp.fc1.bias': 'linear1.bias',
                    'mlp.fc2.weight': 'linear2.weight',
                    'mlp.fc2.bias': 'linear2.bias',
                }
                
                for pretrained_name, our_name in attn_mappings.items():
                    pretrained_key = prefix + pretrained_name
                    our_key = our_prefix + our_name
                    
                    if pretrained_key in pretrained_dict and our_key in model_dict:
                        if pretrained_dict[pretrained_key].shape == model_dict[our_key].shape:
                            model_dict[our_key] = pretrained_dict[pretrained_key]
                            loaded_keys.append(our_key)
            
            # Load the updated state dict
            self.load_state_dict(model_dict, strict=False)
            
            print(f"  ✓ Loaded {len(loaded_keys)} pretrained layers")
            if skipped_keys:
                print(f"  ⚠ Skipped {len(skipped_keys)} incompatible layers")
            
        except Exception as e:
            warnings.warn(f"Failed to load pretrained weights: {e}")
            print(f"  ✗ Using random initialization")
    
    def forward(
        self,
        patches: torch.Tensor,
        coords: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CT-MUSIQ.
        
        Args:
            patches: Tensor of shape [B, N, 3, 32, 32]
                     B = batch size, N = number of patches (193)
            coords: Tensor of shape [B, N, 3]
                    Each row: (scale_idx, row_idx, col_idx)
                    
        Returns:
            Dictionary with:
                'score': Tensor [B, 1] — final quality prediction
                'scale_scores': List of Tensor [B, 1] — per-scale predictions
        """
        batch_size, num_patches = patches.shape[:2]
        
        # =========================================================================
        # 1. PATCH EMBEDDING
        # =========================================================================
        # Reshape: [B, N, 3, 32, 32] → [B*N, 3, 32, 32]
        patches_flat = patches.view(batch_size * num_patches, *patches.shape[2:])
        
        # Embed patches: [B*N, 3, 32, 32] → [B*N, 768]
        patch_embeddings = self.patch_embed(patches_flat)
        
        # Reshape back: [B*N, 768] → [B, N, 768]
        patch_embeddings = patch_embeddings.view(batch_size, num_patches, -1)
        
        # =========================================================================
        # 2. ADD POSITIONAL ENCODING
        # =========================================================================
        # Hash-based encoding: [B, N, 3] → [B, N, 768]
        pos_enc = self.pos_encoding(coords)
        
        # Add to patch embeddings
        embeddings = patch_embeddings + pos_enc
        
        # =========================================================================
        # 3. PREPEND [CLS] TOKEN
        # =========================================================================
        # Expand [CLS] token to batch size: [1, 1, 768] → [B, 1, 768]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate: [B, 1, 768] + [B, N, 768] → [B, N+1, 768]
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # =========================================================================
        # 4. TRANSFORMER ENCODER
        # =========================================================================
        # Self-attention over all tokens including [CLS]
        # Input/Output: [B, N+1, 768]
        encoded = self.transformer(embeddings)
        encoded = self.norm(encoded)
        
        # =========================================================================
        # 5. EXTRACT [CLS] AND SCALE TOKENS
        # =========================================================================
        # [CLS] token is at position 0
        cls_output = encoded[:, 0, :]  # [B, 768]
        
        # Patch tokens are at positions 1 onwards
        patch_outputs = encoded[:, 1:, :]  # [B, N, 768]
        
        # =========================================================================
        # 6. GLOBAL & SPECIALIZED PREDICTIONS
        # =========================================================================
        # [CLS] -> quality score
        global_score = self.global_head(cls_output)  # [B, 1]
        
        # =========================================================================
        # 7. PER-SCALE PREDICTIONS (for KL loss)
        # =========================================================================
        scale_scores = []
        
        # Track which patches belong to which scale
        patch_idx = 0
        for scale_idx, scale in enumerate(config.SCALES):
            # Calculate number of patches for this scale
            grid_size = scale // config.PATCH_SIZE
            num_scale_patches = grid_size * grid_size
            
            # Extract patches for this scale
            scale_tokens = patch_outputs[:, patch_idx:patch_idx + num_scale_patches, :]
            
            # Average pool over patches in this scale
            scale_pooled = scale_tokens.mean(dim=1)  # [B, 768]
            
            # Predict scale-specific score
            scale_score = self.scale_heads[scale_idx](scale_pooled)  # [B, 1]
            scale_scores.append(scale_score)
            
            patch_idx += num_scale_patches
        
        return {
            'score': global_score,
            'scale_scores': scale_scores
        }


def create_model(
    num_scales: int = len(config.SCALES),
    pretrained: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> CTMUSIQ:
    """
    Factory function to create and initialize CT-MUSIQ model.
    
    Args:
        num_scales: Number of resolution scales
        pretrained: Whether to load pretrained ViT weights
        device: Device to place model on
        
    Returns:
        Initialized CT-MUSIQ model
    """
    model = CTMUSIQ(
        num_scales=num_scales,
        pretrained=pretrained
    )
    
    model = model.to(device)
    
    return model


# =============================================================================
# TESTING / VERIFICATION
# =============================================================================

if __name__ == "__main__":
    """
    Quick test to verify the model architecture.
    Run: python model.py
    """
    print("="*60)
    print("CT-MUSIQ Model Architecture Test")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    print("\nCreating CT-MUSIQ model...")
    model = CTMUSIQ(
        num_scales=len(config.SCALES),
        pretrained=False  # Skip pretrained for quick test
    )
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    num_patches = 193  # 49 + 144
    
    print(f"\nCreating dummy input:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num patches: {num_patches}")
    
    # Random patches: [B, N, 3, 32, 32]
    dummy_patches = torch.randn(batch_size, num_patches, 3, 32, 32).to(device)
    
    # Random coordinates: [B, N, 3]
    # Scale 0: 49 patches (7x7 grid), Scale 1: 144 patches (12x12 grid)
    dummy_coords = torch.zeros(batch_size, num_patches, 3, dtype=torch.long).to(device)
    
    # Fill in coordinates
    patch_idx = 0
    for scale_idx, scale in enumerate(config.SCALES):
        grid_size = scale // config.PATCH_SIZE
        for row in range(grid_size):
            for col in range(grid_size):
                dummy_coords[:, patch_idx, 0] = scale_idx
                dummy_coords[:, patch_idx, 1] = row
                dummy_coords[:, patch_idx, 2] = col
                patch_idx += 1
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(dummy_patches, dummy_coords)
    
    print(f"\nOutput:")
    print(f"  Global score shape: {output['score'].shape}")
    print(f"  Global score values: {output['score'].squeeze()}")
    print(f"  Number of scale scores: {len(output['scale_scores'])}")
    for i, scale_score in enumerate(output['scale_scores']):
        print(f"    Scale {i} score shape: {scale_score.shape}")
        print(f"    Scale {i} score values: {scale_score.squeeze()}")
    
    # Test with mixed precision
    if device.type == 'cuda':
        print("\nTesting with mixed precision (fp16)...")
        with torch.cuda.amp.autocast():
            output_amp = model(dummy_patches, dummy_coords)
        print(f"  ✓ Mixed precision forward pass successful")
        print(f"  Score dtype: {output_amp['score'].dtype}")
    
    print("\n✓ Model architecture test complete!")
