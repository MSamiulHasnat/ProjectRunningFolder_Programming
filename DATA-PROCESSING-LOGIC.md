# CT-MUSIQ Data Processing: Complete Logic

## 1. TIFF IMAGE LOADING & PREPROCESSING

### 1.1 TIFF Format Details
```
File: dataset/image/0000.tif to dataset/image/0999.tif
Type: GeoTIFF (geospatial TIFF)
Dtype: float32 (32-bit floating point)
Range: [0, 1] or raw HU values [-1024, 3071]
Shape: 512 × 512 pixels
Channel: Single channel (grayscale)
```

**Important**: Different TIFF files may use different value ranges:
- Some normalize to [0, 1]
- Some store raw HU (Hounsfield Units)
- Some store other scales

### 1.2 Loading & Initial Conversion

```python
from PIL import Image
import numpy as np

def load_tiff_image(filepath: str) -> np.ndarray:
    """Load TIFF image and ensure [0, 1] range"""
    
    img = Image.open(filepath)
    img_array = np.array(img, dtype=np.float32)  # shape: (512, 512)
    
    # Determine if already normalized
    if img_array.max() <= 1.0:
        # Already in [0, 1]
        return img_array
    else:
        # Raw HU or other scale, normalize
        # Assuming HU range [-1024, 3071]
        img_array = np.clip(img_array, -1024, 3071)
        img_array = (img_array + 1024) / 4096
        return img_array
```

---

## 2. CT WINDOWING: BRAIN SOFT-TISSUE WINDOW

### 2.1 What is CT Windowing?

CT machines output raw Hounsfield Units (HU), which represent tissue density:
```
HU = 1000 * (μ_tissue - μ_water) / μ_water

Examples:
Air: -1000 HU
Fat: -100 to -50 HU
Water: 0 HU
Brain gray matter: 30-40 HU
Brain white matter: 20-30 HU
Bone: 300-400 HU
```

**Problem**: Full 4096 HU range is too large. Radiologists use **windows** to focus on specific tissue ranges.

### 2.2 Window Parameters

```python
# Original LDCTIQAC 2023 (abdominal CT)
WINDOW_WIDTH_ORIGINAL = 350  # HU
WINDOW_LEVEL_ORIGINAL = 40   # HU

# CT-MUSIQ (brain CT) — ADAPTED
WINDOW_WIDTH = 80   # HU
WINDOW_LEVEL = 40   # HU
```

**Why different window for brain?**
- Brain tissue contrast is narrow
- Abdominal window (width=350) would wash out brain details
- Clinical brain window (width=80, level=40) is standard for neuroradiology

### 2.3 Windowing Formula

```python
WINDOW_MIN = WINDOW_LEVEL - WINDOW_WIDTH / 2  # 40 - 40 = 0 HU
WINDOW_MAX = WINDOW_LEVEL + WINDOW_WIDTH / 2  # 40 + 40 = 80 HU

def apply_ct_window(img: np.ndarray) -> np.ndarray:
    """
    Apply brain soft-tissue window and normalize to [0, 1]
    
    Args:
        img: numpy array with values in HU
    
    Returns:
        numpy array with values in [0, 1]
    """
    # Step 1: Clip to window range
    # Values below WINDOW_MIN become WINDOW_MIN
    # Values above WINDOW_MAX become WINDOW_MAX
    img_clipped = np.clip(img, WINDOW_MIN, WINDOW_MAX)
    
    # Step 2: Normalize to [0, 1]
    # WINDOW_MIN (0 HU) → 0.0
    # WINDOW_MAX (80 HU) → 1.0
    img_normalized = (img_clipped - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    
    return img_normalized

# Example:
# Input HU values: [-10, 0, 20, 40, 80, 100, 120]
# After clipping: [0, 0, 20, 40, 80, 80, 80]
# After normalize: [0.0, 0.0, 0.25, 0.5, 1.0, 1.0, 1.0]
```

### 2.4 Visual Interpretation

```
Display Scale:

Input HU     | After Windowing | Display Color
─────────────┼─────────────────┼──────────────
< 0 HU       | 0.0 (black)     | Black
0-20 HU      | 0.0-0.25        | Dark gray
20-40 HU     | 0.25-0.5        | Medium gray (brain parenchyma)
40-60 HU     | 0.5-0.75        | Light gray
60-80 HU     | 0.75-1.0        | Light gray
> 80 HU      | 1.0 (white)     | White

In this window:
- Gray matter appears as light gray (35-40 HU)
- White matter appears slightly darker (25-30 HU)
- Blood appears brighter (40-50 HU)
- Bone appears white (100+ HU)
```

**Radiologist interpretation**: Proper windowing makes subtle abnormalities visible.

---

## 3. MULTI-SCALE PYRAMID CONSTRUCTION

### 3.1 Scale Ratios

```python
# Original image: 512 × 512
# Scale 0: 224 × 224 (downsampling ratio: 512/224 ≈ 2.29)
# Scale 1: 384 × 384 (downsampling ratio: 512/384 ≈ 1.33)

SCALES = [224, 384]

# Rationale:
# - 224×224: Standard ViT input size, captures fine details
# - 384×384: Larger patches, captures broader context
# - Multi-scale design similar to FPN (Feature Pyramid Networks) in object detection
```

### 3.2 Resizing Strategy

```python
from PIL import Image

def build_pyramid(img_512: np.ndarray) -> List[np.ndarray]:
    """Build multi-scale pyramid from 512×512 image"""
    
    img_pil = Image.fromarray((img_512 * 255).astype(np.uint8), mode='L')  # Convert to PIL
    
    pyramid = []
    for scale in SCALES:
        # Resize using bicubic interpolation (high quality)
        img_scaled = img_pil.resize((scale, scale), Image.BICUBIC)
        img_scaled_array = np.array(img_scaled, dtype=np.float32) / 255.0
        pyramid.append(img_scaled_array)
    
    return pyramid

# Example dimensions:
# Original: (512, 512)
# Scale 0: (224, 224)
# Scale 1: (384, 384)
```

**Why bicubic interpolation?**
- Smoother than nearest-neighbor or bilinear
- Better for medical imaging (preserves edge sharpness)
- Standard in medical image processing

### 3.3 Patch Extraction

```python
def extract_patches(img_scaled: np.ndarray, scale_idx: int) -> Tuple[Tensor, List[Tuple]]:
    """Extract non-overlapping patches and return with coordinates"""
    
    scale = img_scaled.shape[0]  # 224 or 384
    patch_size = 32
    
    num_patches_per_side = scale // patch_size  # 7 or 12
    
    patches = []
    coordinates = []
    
    for row in range(num_patches_per_side):
        for col in range(num_patches_per_side):
            # Extract patch
            y_start = row * patch_size
            y_end = y_start + patch_size
            x_start = col * patch_size
            x_end = x_start + patch_size
            
            patch = img_scaled[y_start:y_end, x_start:x_end]
            patches.append(patch)
            
            # Store coordinate [scale_idx, row, col]
            coordinates.append([scale_idx, row, col])
    
    return patches, coordinates

# Example for 224×224 scale:
# Grid: 7×7 = 49 patches
# Patch coordinates:
# [0, 0, 0] → top-left patch (pixels 0-32, 0-32)
# [0, 0, 1] → top-left-right patch (pixels 0-32, 32-64)
# [0, 6, 6] → bottom-right patch (pixels 192-224, 192-224)

# Example for 384×384 scale:
# Grid: 12×12 = 144 patches
```

**Key insight**: Non-overlapping patches ensure each pixel is covered exactly once per scale.

---

## 4. CHANNEL REPLICATION (GRAYSCALE → RGB)

### 4.1 ViT-B/32 Expectation

```python
# ViT-B/32 is trained on ImageNet (RGB images)
# Expected input: [B, 3, 224, 224] or [B, 3, 384, 384]
#                  ↑ three color channels

# But our data is grayscale: [B, 1, 224, 224]
```

### 4.2 Replication Strategy

```python
def replicate_grayscale_to_rgb(img_grayscale: Tensor) -> Tensor:
    """Replicate single grayscale channel to 3 RGB channels"""
    
    # Input shape: [B, 1, H, W]
    # Output shape: [B, 3, H, W]
    
    # Method: Simple repetition
    img_rgb = img_grayscale.repeat(1, 3, 1, 1)  # Repeat channel dim
    
    # Or equivalently:
    img_rgb = torch.cat([img_grayscale, img_grayscale, img_grayscale], dim=1)
    
    return img_rgb

# Visual:
# Grayscale: [[0.5, 0.6], [0.7, 0.8]] (shape: 1×2×2)
# RGB: [[[0.5, 0.6], [0.7, 0.8]],    (shape: 3×2×2)
#       [[0.5, 0.6], [0.7, 0.8]],
#       [[0.5, 0.6], [0.7, 0.8]]]
#
# R channel = G channel = B channel = original grayscale
```

### 4.3 Why This Works

```python
# ViT-B/32 patch embedding:
# Conv2d(in_channels=3, out_channels=768, kernel_size=32, stride=32)
#
# Convolution formula:
# output[c_out] = sum over c_in,h,w of (kernel[c_out, c_in, h, w] * input[c_in, h, w])
#
# With replicated channels:
# input = [g, g, g]  (R = G = B = grayscale)
#
# output[c_out] = kernel[c_out, 0] * g + kernel[c_out, 1] * g + kernel[c_out, 2] * g
#               = g * (kernel[c_out, 0] + kernel[c_out, 1] + kernel[c_out, 2])
#
# The kernel learns appropriate weights for each channel.
# With replicated input, it learns to ignore channel differences.
#
# Benefit: Leverages pretrained ImageNet weights without fine-tuning patch embedding.
```

---

## 5. BATCH CONSTRUCTION

### 5.1 Custom Collate Function (Rarely Used)

```python
def custom_collate_fn(batch):
    """Handle variable-length patch sequences (not used in main training)"""
    
    # batch: list of (images, coords, label) tuples
    # Issue: Different images produce different number of patches if not fixed-size
    
    batch_images = []
    batch_coords = []
    batch_labels = []
    max_patches = 0
    
    for images, coords, label in batch:
        batch_images.append(images)
        batch_coords.append(coords)
        batch_labels.append(label)
        max_patches = max(max_patches, coords.shape[0])
    
    # Pad shorter sequences
    padded_coords = []
    for coords in batch_coords:
        if coords.shape[0] < max_patches:
            pad_size = max_patches - coords.shape[0]
            padded = torch.cat([coords, torch.zeros(pad_size, 3)])
            padded_coords.append(padded)
        else:
            padded_coords.append(coords)
    
    return batch_images, torch.stack(padded_coords), torch.stack(batch_labels)
```

### 5.2 Default Collation (Used in Main)

```python
# In create_dataloaders():
# collate_fn=default_collate  # PyTorch's default

# Why default works:
# All samples produce EXACTLY 193 patches:
# - Scale 0: 7×7 = 49
# - Scale 1: 12×12 = 144
# - Total: 193 (fixed!)
#
# Since all samples are same size, default_collate handles it fine.
```

---

## 6. TRAIN/VAL/TEST SPLITS

### 6.1 Split Strategy

```python
TRAIN_RANGE = (0, 699)    # 700 images
VAL_RANGE   = (700, 899)  # 200 images
TEST_RANGE  = (900, 999)  # 100 images
```

**Proportions**:
- Training: 70% (largest, for learning)
- Validation: 20% (for hyperparameter tuning, early stopping)
- Test: 10% (final evaluation, never seen during training)

**Why deterministic (by index, not random)?**
- Original LDCTIQAC 2023 challenge used this split
- Ensures exact reproducibility
- Fair comparison with published baselines

### 6.2 Data Leak Prevention

```python
# CRITICAL: Never apply augmentation to val/test data
if split == 'train':
    augmented_image = apply_augmentation(image)
else:
    augmented_image = image  # No augmentation

# Why?
# - Augmentation artificially creates variations
# - Validation should test on REAL data, not augmented
# - Augmentation on val/test would inflate performance metrics
```

---

## 7. NORMALIZATION CHOICE

### 7.1 CT Windowing as Normalization

```python
# Q: Should we also apply ImageNet mean/std normalization?
# A: No.

# ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# These are for natural RGB images.

# Our data:
# - Single channel (replicated to RGB, but all same value)
# - Medical imaging (very different statistics than ImageNet)
# - Already windowed to [0, 1]

# Decision: Use CT windowing [0, 1] directly
# Pretrained ViT-B/32 will adapt to medical image statistics during fine-tuning
```

---

## 8. AUGMENTATION RATIONALE

### 8.1 Augmentation Details Revisited

```python
# 1. Horizontal flip (50%)
#    - Brain CT can be flipped (brain is ~symmetric)
#    - Doubles effective training set size
#    - Common in medical imaging

# 2. Rotation ±5°
#    - Realistic: Gantry tilt in CT acquisition
#    - Mild rotation (not ±45°) to preserve clinical realism
#    - Prevents model from learning spurious alignments

# 3. Crop 85-100%
#    - Simulates partial field-of-view
#    - Trains model to assess quality even with missing borders
#    - Zoom augmentation without explicit scaling

# 4. Brightness/contrast ±5%
#    - Simulates window level variations
#    - Radiologists use different window levels
#    - Small perturbations (±5%) keep it realistic

# 5. Gaussian noise
#    - Simulates dose variations (lower dose = noisier)
#    - Important for low-dose CT assessment
#    - σ ∈ [0.005, 0.02] is subtle but present
```

---

## 9. QUALITY SCORE DISTRIBUTION

### 9.1 Expected Distribution

```python
# Ground truth: Radiologist Likert scores (0-4)
# Distribution: Likely skewed toward 2-3

# Histogram (typical for medical imaging):
# Score 0: 5% (very poor, rare)
# Score 1: 10% (poor)
# Score 2: 25% (acceptable)
# Score 3: 35% (good, most common)
# Score 4: 25% (excellent)

# Why skewed toward high?
# - Radiologists don't acquire very poor images
# - Poor scans are retaken or excluded
# - Dataset selection bias toward acceptable quality
```

### 9.2 Label Averaging

```python
# train.json format:
# {
#   "0000.tif": 2.75,  # Averaged across 3+ radiologists
#   "0001.tif": 3.25,
#   ...
# }

# Benefits of averaging:
# - Reduces inter-rater variability
# - More stable target for training
# - Single float easier than per-rater labels
```

---

## 10. COMPLETE DATA PIPELINE (SINGLE IMAGE)

```python
# Example: Index 0000 (from training set)

# Step 1: Load TIFF
img_512 = load_tiff_image("dataset/image/0000.tif")  # shape (512, 512), [0, 1]

# Step 2: Apply CT windowing
img_512 = apply_ct_window(img_512)  # shape (512, 512), [0, 1], normalized to brain window

# Step 3: Load label
quality_score = labels["0000.tif"]  # 2.75 (float)

# Step 4: Augmentation (training only)
if split == 'train':
    img_512 = apply_augmentation(img_512)  # flip, rotate, crop, etc.

# Step 5: Build multi-scale pyramid
pyramid = [img_224, img_384]  # Each shape (224, 224) and (384, 384)

# Step 6: Extract patches from each scale
patches_224 = [patch_1, patch_2, ..., patch_49]  # 49 patches, each 32×32
patches_384 = [patch_50, patch_51, ..., patch_193]  # 144 patches, each 32×32

# Step 7: Create coordinates
coords_224 = [[0, 0, 0], [0, 0, 1], ..., [0, 6, 6]]  # 49 coordinates
coords_384 = [[1, 0, 0], [1, 0, 1], ..., [1, 11, 11]]  # 144 coordinates
all_coords = coords_224 + coords_384  # 193 total

# Step 8: Replicate grayscale → RGB
tensor_224 = torch.tensor(img_224)  # [1, 224, 224]
tensor_224_rgb = replicate_grayscale(tensor_224)  # [3, 224, 224]
(similar for 384)

# Step 9: Batch collation
batch = {
    'images': [tensor_224_rgb, tensor_384_rgb],  # List of scaled images
    'coords': torch.stack(all_coords),  # [193, 3]
    'labels': torch.tensor(quality_score)  # scalar
}

# Step 10: Model forward pass
global_pred = model(batch)  # [1] scalar prediction ~2.75
```

---

**Dataset Processing Complete**: 2026-04-05  
**Total Data Points**: 1,000 brain CT images  
**Total Tokens Per Image**: 193 patches + 1 CLS = 194 transformer tokens
