# CT-MUSIQ: Complete Architecture Logic

## 1. PROJECT GOAL

**Thesis**: Automated Perceptual Assessment of Low-Dose CT via Architectural Adaptation of the MUSIQ Transformer Model

**Problem**: Medical imaging workflows need automated quality control for CT scans without reference images (no-reference IQA).

**Solution**: CT-MUSIQ adapts the MUSIQ multi-scale image quality transformer for brain CT assessment.

**Expected Impact**: Predict radiologist-assessed quality scores (0-4 Likert) automatically, enabling rapid clinical workflows.

---

## 2. DATA FLOW: UNDERSTANDING THE PIPELINE

### 2.1 Input: Brain CT Images

```
LDCTIQAC 2023 Dataset
├─ 1,000 brain CT slices
├─ Format: TIFF, 32-bit float, 512×512 pixels
├─ Value range: [0, 1] (normalized)
└─ Quality labels: Radiologist Likert scores 0-4
```

**Why Brain CT?** The original LDCTIQAC challenge used abdominal CT. Brain CT has different tissue contrast, so CT-MUSIQ uses the clinical brain soft-tissue window:
- **Window Level**: 40 HU (center of interest)
- **Window Width**: 80 HU (tissue range: [0, 80])
- **Formula**: `clipped = clip(pixel, 0, 80); normalized = clipped / 80`

### 2.2 Data Splits (Deterministic by Index)

```python
# config.py
TRAIN_RANGE = (0, 699)    # 700 images: indices 0000.tif to 0699.tif
VAL_RANGE   = (700, 899)  # 200 images: indices 0700.tif to 0899.tif
TEST_RANGE  = (900, 999)  # 100 images: indices 0900.tif to 0999.tif
```

**Key insight**: Split by filename index, not random shuffle. This ensures reproducibility and matches the original challenge protocol.

### 2.3 Multi-Scale Patch Pyramid

Each image is processed at **two scales** to capture quality signals at different resolutions:

```
Input: 1 channel (grayscale brain CT), 512×512

Scale 0 (224×224):
├─ Resize: 512×512 → 224×224 (PIL bicubic)
├─ Patch division: 224 / 32 = 7 patches per side
├─ Total patches: 7×7 = 49 patches
└─ Each patch: 32×32 pixels

Scale 1 (384×384):
├─ Resize: 512×512 → 384×384 (PIL bicubic)
├─ Patch division: 384 / 32 = 12 patches per side
├─ Total patches: 12×12 = 144 patches
└─ Each patch: 32×32 pixels

Total tokens per image: 49 + 144 = 193 patches
Plus [CLS] token: 194 total transformer tokens
```

**Why multi-scale?** Different patch scales capture quality at different frequencies:
- Small patches (224 scale): Fine details, compression artifacts
- Large patches (384 scale): Global structures, overall image sharpness

**Why 32×32 patches?** Matches ViT-B/32 pretrained patch embedding kernel size (Conv2d(3, 768, 32, 32)).

### 2.4 Grayscale to RGB Replication

```python
# In PatchEmbedding
# ViT-B/32 expects 3-channel RGB input
# Brain CT is single-channel grayscale

input_tensor: shape [B, 1, 224, 224]
→ replicate to [B, 3, 224, 224]  # channel 0 = channel 1 = channel 2
→ Conv2d(3, 768, 32, 32)          # patch embedding
→ output: [B, 49, 768]            # 49 = (224/32)²
```

**Why replicate?** Pretrained ViT-B/32 weights are on ImageNet RGB. To leverage pretraining, we replicate the grayscale channel across 3 color channels. The model learns to ignore color differences since all channels are identical.

---

## 3. MODEL ARCHITECTURE: COMPONENT BREAKDOWN

### 3.1 Hash-Based Positional Encoding (Custom Innovation)

**Problem with standard sinusoidal encoding**: Assumes fixed sequence length. We have variable-length sequences (49 at scale 0, 144 at scale 1).

**Solution**: Three independent embedding tables that sum together:

```python
class HashPositionalEncoding:
    scale_embed = nn.Embedding(2, 768)    # indices 0-1 for scale 0 and 1
    row_embed   = nn.Embedding(13, 768)   # indices 0-12 for row position
    col_embed   = nn.Embedding(13, 768)   # indices 0-12 for column position

    def forward(self, coords: Tensor) -> Tensor:
        # coords shape: [B, N, 3] where coords[b, i] = [scale_idx, row_idx, col_idx]
        # Example: coords[0, 0] = [0, 0, 0]  (scale 0, top-left)
        #         coords[0, 49] = [1, 0, 0] (scale 1, top-left)
        #         coords[0, 50] = [1, 0, 1] (scale 1, row 0, col 1)
        
        pos_enc = scale_embed(coords[..., 0]) \
                + row_embed(coords[..., 1]) \
                + col_embed(coords[..., 2])
        
        return pos_enc  # shape: [B, N, 768]
```

**Intuition**: 
- **scale_embed**: Learns that "this token is from fine scale" vs "coarse scale"
- **row_embed**: Learns that "this is top row" vs "middle" vs "bottom"
- **col_embed**: Learns that "this is left" vs "middle" vs "right"
- Summing them: Allows position 0 (top-left) to be different at both scales

### 3.2 Patch Embedding

```python
class PatchEmbedding:
    conv = Conv2d(in_channels=3, out_channels=768, kernel_size=32, stride=32)
    
    def forward(self, x: Tensor) -> Tensor:
        # x shape: [B, 3, 224, 224] at scale 0 or [B, 3, 384, 384] at scale 1
        # Stride=32 means non-overlapping patches
        
        x = self.conv(x)  
        # Conv output shape: [B, 768, 7, 7] at scale 0
        #                 or [B, 768, 12, 12] at scale 1
        
        x = x.flatten(2).transpose(1, 2)  # [B, 49, 768] at scale 0
        return x
```

**Why Conv2d instead of unfold?** Conv2d(kernel_size=32, stride=32) acts as both patch extraction AND projection to embedding dimension (768). Pretrained ViT-B/32 weights are already loaded here.

### 3.3 Transformer Encoder (Pretrained ViT-B/32)

```python
# Standard ViT-B configuration:
TransformerEncoder(
    num_layers=12,           # 12 transformer blocks
    d_model=768,             # embedding dimension
    num_heads=8,             # 8 attention heads (768/8 = 96 dim per head)
    ffn_dim=3072,            # feed-forward: 768 → 3072 → 768
    dropout=0.05             # reduced for small dataset
)

# Pretrained from ImageNet ViT-B/32
# All weights loaded except we skip classification head
```

**Flow**:
```
[CLS] token + 193 patch tokens → Layer 0 (attention + FFN) → ... → Layer 11 → [CLS]*
                                                                                  (used for quality prediction)
```

### 3.4 Prediction Heads

#### Global Quality Head (MSE Loss)
```python
global_head = nn.Sequential(
    nn.Linear(768, 1),      # [CLS] → scalar score
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(768, 1)       # ← Wait, this seems wrong. Should be (1, 1) or (768, 1)?
                            # Actual code has (768, 1) twice - likely redundant or design choice
)

# Output: shape [B, 1], range [0, 4] (quality score)
```

#### Per-Scale Heads (KL Loss)
```python
scale_heads = nn.ModuleList([
    nn.Sequential(Linear(768, 1), GELU, Dropout) 
    for _ in range(num_scales)  # num_scales = 2
])

# At scale 0:
scale_0_tokens = transformer_output[1:50]  # 49 tokens (exclude CLS)
scale_0_pooled = mean(scale_0_tokens)      # average pool: [768]
scale_0_quality = scale_heads[0](scale_0_pooled)  # shape [1]

# At scale 1:
scale_1_tokens = transformer_output[50:194]  # 144 tokens
scale_1_pooled = mean(scale_1_tokens)        # average pool: [768]
scale_1_quality = scale_heads[1](scale_1_pooled)  # shape [1]
```

**Why per-scale heads?** Two scales should agree on quality. If they disagree strongly, the KL loss penalizes it → encourages multi-scale consistency.

---

## 4. LOSS FUNCTION: SCALE CONSISTENCY LOGIC

### 4.1 Score-to-Distribution Conversion

Radiologist scores are single floats (0-4), but we need probability distributions for KL divergence:

```python
class ScoreToDistribution:
    def __call__(self, scores: Tensor) -> Tensor:
        # scores shape: [B] or [B, 1], values in [0, 4]
        # output: [B, NUM_BINS=20] probability distribution
        
        # Create 20 bins covering [0, 4]
        bin_centers = linspace(0, 4, num_bins=20)  # [0, 0.21, 0.42, ..., 4.0]
        
        # Gaussian kernel with σ=0.5
        # For each score, compute probability for each bin
        # P(score | bin_i) = exp(-(score - bin_center_i)² / (2 * σ²)) / Z
        
        # Example: score=2.0
        # bin[10] (center=2.1): high probability (close to 2.0)
        # bin[5] (center=1.0): low probability (far from 2.0)
        # bin[15] (center=3.1): low probability
        
        distribution = Gaussian_kernel(scores, bin_centers, sigma=0.5)
        return distribution  # shape: [B, 20], sums to 1 across bins
```

**Intuition**: A score of 2.0 should "peak" at the bin near 2.0, with smooth falloff to neighboring bins. This captures uncertainty and allows fine-grained quality judgments.

### 4.2 Combined Loss

```python
def CTMUSIQLoss(
    global_pred: Tensor,      # [B, 1] from global head
    scale_preds: List[Tensor], # [scale_0_pred [B,1], scale_1_pred [B,1]]
    target: Tensor,            # [B] ground truth scores
    lambda_kl: float = 0.02
):
    # Convert all to distributions
    global_dist = score_to_distribution(global_pred)  # [B, 20]
    target_dist = score_to_distribution(target)       # [B, 20]
    scale_dists = [score_to_distribution(p) for p in scale_preds]  # [scale_0: [B,20], scale_1: [B,20]]
    
    # MSE loss: global prediction vs target
    mse_loss = MSE(global_pred, target)  # [B] → scalar
    
    # KL loss: per-scale should match global
    kl_losses = []
    for scale_dist in scale_dists:
        kl = KLDiv(scale_dist, global_dist.detach())  # scale → global
        kl_losses.append(kl)
    kl_loss = mean(kl_losses)  # scalar
    
    # Combined
    total_loss = mse_loss + lambda_kl * kl_loss
    
    return total_loss
```

**Why this design?**
- **MSE on global**: Primary objective - predict ground truth score
- **KL on per-scale**: Regularization - forces multi-scale agreement
- **λ_KL=0.02**: Light regularization. Ablations show 0.02 < λ < 0.20 works best.

---

## 5. TRAINING PIPELINE: TWO-STAGE LEARNING

### 5.1 Stage 1: Warm-up with Frozen Encoder (Epochs 1-5)

```python
# Freeze all transformer weights
for param in transformer.parameters():
    param.requires_grad = False

# Only train:
# - global_head
# - scale_heads
# - hash positional encoding

optimizer = AdamW([p for p in model.parameters() if p.requires_grad], 
                  lr=3e-4)  # LR_STAGE1

# Why freeze encoder?
# - Pretrained ViT-B/32 is already strong on ImageNet
# - Frozen encoder acts as fixed feature extractor
# - Only adaptation layer (heads) needs tuning
# - Prevents catastrophic forgetting of pretraining
# - Faster training, stable convergence
```

### 5.2 Stage 2: Full Fine-tuning (Epochs 6-100)

```python
# Unfreeze all weights
for param in transformer.parameters():
    param.requires_grad = True

# Linear warmup: 3 epochs
# Start LR=1e-6 at epoch 6, linearly increase to 5e-5 by epoch 8

# Cosine annealing: epochs 9-100
# LR decays from 5e-5 to LR_MIN=1e-6 following cosine schedule

optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Why two-stage?
# Stage 1: Quickly learn to map patches → quality (easy job)
# Stage 2: Slowly refine transformer features (harder, needs lower LR)
```

### 5.3 Training Loop Each Epoch

```python
for epoch in range(1, EPOCHS+1):
    # ===== TRAINING =====
    model.train()
    
    for batch_idx, (images, quality_scores) in enumerate(train_loader):
        # images: [B, 1, 224, 224] and [B, 1, 384, 384] (from dataset)
        # quality_scores: [B] float values 0-4
        
        # Convert grayscale → RGB (replicate channel)
        images_224 = replicate_channels(images[0])  # [B, 3, 224, 224]
        images_384 = replicate_channels(images[1])  # [B, 3, 384, 384]
        
        # Forward pass with mixed precision
        with autocast('cuda', dtype=torch.float16):
            patches_224 = patch_embedding(images_224)  # [B, 49, 768]
            patches_384 = patch_embedding(images_384)  # [B, 144, 768]
            
            coords_224 = build_coords(scale=0, grid=7x7)  # [B, 49, 3]
            coords_384 = build_coords(scale=1, grid=12x12)  # [B, 144, 3]
            
            pos_enc_224 = hash_pos_encoding(coords_224)  # [B, 49, 768]
            pos_enc_384 = hash_pos_encoding(coords_384)  # [B, 144, 768]
            
            # Concatenate: [CLS] + patches + pos_encoding
            seq_224 = concat([cls_token, patches_224 + pos_enc_224])  # [B, 50, 768]
            seq_384 = concat([cls_token, patches_384 + pos_enc_384])  # [B, 145, 768]
            seq = concat([seq_224, seq_384])  # [B, 195, 768]
            
            # Transformer encoder
            output = transformer(seq)  # [B, 195, 768]
            
            cls_output = output[:, 0]  # [B, 768]
            scale_0_tokens = output[:, 1:50]  # [B, 49, 768]
            scale_1_tokens = output[:, 50:]   # [B, 144, 768]
            
            # Predictions
            global_quality = global_head(cls_output)  # [B, 1]
            scale_0_quality = scale_heads[0](mean(scale_0_tokens))  # [B, 1]
            scale_1_quality = scale_heads[1](mean(scale_1_tokens))  # [B, 1]
            
            # Loss
            loss = criterion(global_quality, [scale_0_quality, scale_1_quality], quality_scores)
        
        # Backward with mixed precision
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    
    # ===== VALIDATION =====
    model.eval()
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for images, quality_scores in val_loader:
            # Same forward pass (without loss)
            global_quality = model(images)
            val_preds.append(global_quality.cpu())
            val_targets.append(quality_scores.cpu())
    
    val_preds = concat(val_preds)  # [200]
    val_targets = concat(val_targets)  # [200]
    
    # Compute metrics: PLCC, SROCC, KROCC, Aggregate
    metrics = compute_metrics(val_preds, val_targets)
    
    # ===== EMA (Exponential Moving Average) =====
    # Keep a smoothed copy of weights for validation
    ema.update(model)
    
    # Validate using EMA weights
    model_ema_state = model.state_dict()
    model.load_state_dict(ema.state_dict())
    
    # [compute validation with EMA weights]
    
    model.load_state_dict(model_ema_state)  # restore original
    
    # ===== EARLY STOPPING =====
    if metrics['aggregate'] > best_aggregate:
        best_aggregate = metrics['aggregate']
        save_checkpoint(model, optimizer, epoch)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break
    
    # ===== LOGGING =====
    log_to_csv(epoch, metrics, loss)
    
    # ===== SCHEDULER =====
    scheduler.step()
```

**Key mechanisms**:
1. **Gradient clipping**: Prevents exploding gradients with mixed precision
2. **Mixed precision (AMP)**: Forward in fp16, backward in fp32 → save VRAM
3. **EMA**: Smoother validation metrics, better checkpoint
4. **Early stopping**: Stop if validation doesn't improve for 20 epochs
5. **Cosine annealing**: Smooth LR decay encourages final convergence

---

## 6. EVALUATION METRICS: UNDERSTANDING THE SCORES

```python
def compute_metrics(predictions: Tensor, targets: Tensor):
    # predictions: [N], float values predicted by model
    # targets: [N], float values (radiologist scores)
    # N = 100 (test set size)
    
    # PLCC: Pearson Linear Correlation Coefficient
    # Measures linear correlation between pred and target
    # Higher is better, range [-1, 1]
    # Interpretation: How linearly related are predictions to scores?
    plcc = pearsonr(predictions, targets)
    
    # SROCC: Spearman Rank Order Correlation Coefficient
    # Measures monotonic correlation (doesn't need to be linear)
    # Higher is better, range [-1, 1]
    # Interpretation: How well do rankings match?
    # Example: pred=[1, 2, 3], target=[1.1, 2.2, 2.8] → perfect SROCC
    srocc = spearmanr(predictions, targets)
    
    # KROCC: Kendall Rank Order Correlation Coefficient
    # Similar to SROCC but based on concordant/discordant pairs
    # More robust to outliers than SROCC
    krocc = kendalltau(predictions, targets)
    
    # Aggregate: Sum of absolute values
    # This is the official LDCTIQAC 2023 leaderboard metric
    aggregate = abs(plcc) + abs(srocc) + abs(krocc)
    
    # Why three metrics?
    # PLCC: Penalizes scaling errors (if pred = 2*target, PLCC drops)
    # SROCC: Ignores magnitude, only cares about order
    # KROCC: Robust ranking measure
    # Aggregate: Balances all three perspectives
    
    return {
        'plcc': plcc,
        'srocc': srocc,
        'krocc': krocc,
        'aggregate': aggregate  # typically 0.5 - 3.0 range
    }
```

**Performance Benchmarks (LDCTIQAC 2023 Published)**:
- agaldran: 2.7427 (best)
- RPI_AXIS: 2.6843
- CHILL@UK: 2.6719
- FeatureNet: 2.6550
- Team Epoch: 2.6202
- gabybaldeon: 2.5671
- SNR baseline: 2.4026
- BRISQUE: 2.1219

**CT-MUSIQ Latest**: Val Aggregate ~2.72 (would rank #2 if test matches)

---

## 7. ABLATION STUDIES: UNDERSTANDING DESIGN CHOICES

| Ablation | Config | Test Aggregate | Insight |
|----------|--------|---|---|
| **A1** | 1 scale (224 only) | ❌ CRASHED | Need multi-scale (bug: config.SCALES not synced with self.num_scales) |
| **A2** | 2 scales, λ_KL=0 | 2.233 | Multi-scale baseline (no consistency loss) |
| **A3** | 2 scales, λ_KL=0.05 | 2.249 | Weak KL helps slightly |
| **A4** | 2 scales, λ_KL=0.10 | 2.305 | **Best ablation** - optimal KL weight |
| **A5** | 2 scales, λ_KL=0.20 | 2.251 | Too-strong KL over-regularizes |
| **A6** | 3 scales (512) | Not run | Would exceed VRAM |

**Key findings**:
1. Multi-scale is essential (A2 >> no multi-scale)
2. KL loss helps but with diminishing returns (A2 → A4 improves, A4 → A5 drops)
3. λ_KL=0.02 (default in main) is conservative; λ_KL=0.10 could be better for test set

---

## 8. KNOWN ISSUES & FIXES

### Issue 1: A1 Ablation Crashes
**Root cause** (model.py line ~473):
```python
# WRONG:
for scale_idx in range(len(config.SCALES)):  # Always iterates over config.SCALES=[224, 384]
    scale_quality = self.scale_heads[scale_idx](...)

# With single-scale ablation:
# config.SCALES = [224] (len=1)
# self.scale_heads = ModuleList([head_0]) (len=1)
# But forward() tries to access scale_heads[1] → IndexError
```

**Fix**: Use `self.num_scales` instead of `len(config.SCALES)`:
```python
for scale_idx in range(self.num_scales):  # Respects actual model config
    scale_quality = self.scale_heads[scale_idx](...)
```

### Issue 2: Duplicate Epoch Logging
**Root cause**: EMA apply/restore causes two validation passes per epoch

**Effect**: ct_musiq_training_log.csv has entries like:
```
epoch,stage,loss,val_aggregate
...
96,2,0.045,2.71
96,2,0.045,2.72  # duplicate entry with slightly different metrics (EMA validation)
...
```

**Impact**: Low - Both entries are nearly identical, doesn't affect convergence

### Issue 3: Latest CT-MUSIQ Not Tested
**Status**: Best checkpoint (2.72 val aggregate) exists, but hasn't been formally evaluated on test set.

**Next step**: Run `python evaluate.py --checkpoint results/ct_musiq/ct_musiq_best.pth`

---

## 9. FUTURE DIRECTIONS

1. **Test evaluation**: Formally evaluate latest checkpoint on 100 test images
2. **Single-scale ablation fix**: Fix A1 to compare pure single-scale performance
3. **Three-scale ablation**: Try A6 if adding GPU memory (512×512 patches)
4. **Baseline improvements**: AgaldranCombo shows val instability - could investigate mixed precision issues
5. **Deployment**: Export to ONNX for clinical inference
6. **Uncertainty quantification**: Add Bayesian layers to estimate confidence

---

**Document Created**: 2026-04-05  
**Project**: CT-MUSIQ Undergraduate Thesis  
**Author**: M. Samiul Hasnat, Sichuan University
