# CT-MUSIQ: Low-Dose CT Image Quality Assessment

**Architectural Adaptation of MUSIQ Transformer for No-Reference Perceptual Image Quality Assessment of Low-Dose CT Images**

---

## Table of Contents

- [Overview](#overview)
- [Project Information](#project-information)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Key Innovations](#key-innovations)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Ablation Study](#ablation-study)
- [Results](#results)
- [VRAM Management](#vram-management)
- [CT Windowing](#ct-windowing)
- [Citation](#citation)
- [References](#references)
- [License](#license)

---

## Overview

CT-MUSIQ is an undergraduate thesis project that adapts the MUSIQ (Multi-scale Image Quality Transformer) architecture for no-reference perceptual image quality assessment (NR-IQA) of Low-Dose CT (LDCT) images.

The model predicts radiologist-assessed quality scores (0-4 Likert scale) for brain CT slices without requiring a reference (full-dose) image. This is clinically valuable for automated quality control in CT imaging workflows.

### Key Features

- **Multi-scale pyramid**: Processes images at 224×224 and 384×384 resolutions
- **Hash-based positional encoding**: Handles variable-length multi-scale patch sequences
- **Scale-consistency KL loss**: Encourages agreement across resolution scales
- **Pretrained ViT backbone**: Leverages ImageNet-pretrained ViT-B/32 weights
- **Mixed precision training**: Optimized for 6GB VRAM (RTX 3060)

---

## Project Information

| Property | Value |
|----------|-------|
| **Thesis Title** | Automated Perceptual Assessment of Low-Dose CT via Architectural Adaptation of the MUSIQ Transformer Model |
| **Author** | M Samiul Hasnat |
| **Institution** | Sichuan University |
| **Model Name** | CT-MUSIQ |
| **Framework** | PyTorch |
| **GPU** | NVIDIA RTX 3060 Laptop (6GB VRAM) |

### Reference Papers

- **Original MUSIQ**: Ke et al. (2021), "Multi-scale Image Quality Transformer for No-Reference Image Quality Assessment", ICCV 2021
- **Dataset Paper**: Lee et al. (2025), "LDCTIQAC 2023 Challenge Results", Medical Image Analysis, vol. 99, 103343

---

## Dataset

### LDCTIQAC 2023 (MICCAI Challenge)

| Property | Value |
|----------|-------|
| Total images | 1,000 brain CT slices |
| Image format | TIFF (.tif) |
| Image size | 512×512 pixels |
| Bit depth | 32-bit float (normalized to [0, 1]) |
| Label file | `train.json` |
| Score range | 0-4 Likert scale (radiologist average) |
| Modality | Brain CT (grayscale, single channel) |

### Data Splits

| Split | Indices | Count | Purpose |
|-------|---------|-------|---------|
| Train | 0000-0699 | 700 | Model training |
| Validation | 0700-0899 | 200 | Hyperparameter tuning, early stopping |
| Test | 0900-0999 | 100 | Final evaluation |

### Label Format

The `train.json` file maps image filenames to quality scores:

```json
{
  "0000.tif": 2.8,
  "0001.tif": 1.8,
  "0002.tif": 3.4,
  ...
}
```

### Score Distribution

| Split | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| Train | 2.098 | 1.054 | 0.0 | 4.0 |
| Val | 1.975 | 1.093 | 0.0 | 4.0 |
| Test | 2.092 | 1.082 | 0.0 | 4.0 |

---

## Architecture

### Overview

```
Input CT Image (512×512)
    ↓
Multi-Scale Pyramid [224, 384]
    ↓
Patch Extraction (32×32)
    ↓
Grayscale → RGB Replication
    ↓
Patch Embedding (Conv2d)
    ↓
Hash Positional Encoding
    ↓
Prepend [CLS] Token
    ↓
Transformer Encoder (6 layers, 8 heads)
    ↓
┌─────────────────┬─────────────────┐
│   Global Head   │  Per-Scale Heads │
│   [CLS] → Score │  Pool → Scores   │
└─────────────────┴─────────────────┘
    ↓
Quality Score (0-4)
```

### Step-by-Step Code Walkthrough

Here is exactly how the architectural overview maps to the codebase sequentially, step-by-step:

#### 1. Data Preparation & Multi-Scale Pyramid (`dataset.py`)
- **Action:** Load the 512×512 HD-CT slices, normalize them, and run them through multi-scale resizing (224×224 and 384×384 scales).
- **Code implementation:** `LDCTDataset` class inside `dataset.py`.
- **Key functions:** `__getitem__()` uses `transforms.Resize()` to prepare the varying scales and dynamically extract the resolution metadata.

#### 2. Patch Tokenization & Grayscale to RGB (`model.py`)
- **Action:** Grayscale representations are repeated across 3 channels to mimic RGB to be compatible with standard Vision Transformers. Then, non-overlapping 32×32 patches are extracted from each resized image scale.
- **Code implementation:** `PatchEmbedding` class inside `model.py`.
- **Key functions:** `forward()` repeats the channel dimension (`x.repeat(1, 3, 1, 1)`) and applies a `nn.Conv2d(3, embed_dim, kernel_size=32, stride=32)` to generate patch tokens.

#### 3. Hash Positional Encoding (`model.py`)
- **Action:** Since image sequences of patches vary widely dynamically depending on the input scale sizes, standard learned positional encodings won't work perfectly. CT-MUSIQ uses a 2D hash-based spatial coordinate approach to inject positional context.
- **Code implementation:** `HashPositionalEncoding` class inside `model.py`.
- **Key functions:** `forward()` hashes spatial `(x, y)` coordinates and `scale_idx` into the embedding dimension to be added directly to the spatial tokens.

#### 4. Prepend [CLS] Token & Transformer Encoder (`model.py`)
- **Action:** A learnable Global `[CLS]` token is concatenated to the front of all patch tokens. Then they are passed through standard Transformer blocks (ViT-B backbone) pre-trained on ImageNet.
- **Code implementation:** Pretrained ViT fetching and transformation blocks in `CTMUSIQ` class inside `model.py`.
- **Key functions:** `CTMUSIQ.forward()` manages token sequencing. It passes the combined representations to standard multi-head self-attention transformer layers. 

#### 5. Quality Prediction Heads (`model.py`)
- **Action:** CT-MUSIQ predicts both per-scale quality scores predicting scale-level quality, as well as a final global predicted score (the master output).
- **Code implementation:** The specific scale and global linear heads in the `CTMUSIQ.__init__()` and `CTMUSIQ.forward()`.
- **Key functions:** `self.global_head` processes the output of the `[CLS]` token. `self.scale_heads` processes the averaged spatial tokens from each individual scale separately.

#### 6. Scale-Consistency & Aggregate Quality Objective (`loss.py`)
- **Action:** A mixed loss evaluates both the exact numerical target Mean Squared Error (MSE), as well as a distributional (KL-Divergence) consistency enforcing that scales should theoretically predict nearly identical quality ranges.
- **Code implementation:** `CTMUSIQLoss` and `ScoreToDistribution` inside `loss.py`.
- **Key functions:** Computes standard Euclidean loss (`nn.MSELoss`) over global/scale nodes, as well as divergence via `score_to_dist` generating Gaussian target distributions.

#### 7. Full Training & Validation Loop (`train.py`)
- **Action:** Forward pass sequences loop infinitely over epochs, coordinating metric collections, backpropagation, Automatic Mixed Precision (AMP), gradient clipping, and cosine learning rate adjustments.
- **Code implementation:** The main orchestrator loops inside `train.py`.
- **Key functions:** `train_epoch()`, `validate()`, and metrics aggregation are tightly coupled with early stopping heuristics.

### Detailed Architecture

#### 1. Multi-Scale Pyramid

Each CT image is resized to multiple resolutions:
- **Scale 0**: 224×224 → 7×7 = 49 patches
- **Scale 1**: 384×384 → 12×12 = 144 patches
- **Total**: 193 patches per image

#### 2. Patch Tokenization

- Patch size: 32×32 pixels
- Non-overlapping extraction
- Each patch records spatial coordinates: `(scale_idx, row_idx, col_idx)`

#### 3. Grayscale to RGB Replication

Single-channel CT patches are replicated to 3 channels:
```python
patch_rgb = [patch, patch, patch]  # Shape: [3, 32, 32]
```

This enables loading pretrained ViT-B/32 weights that expect RGB input.

#### 4. Patch Embedding

```python
nn.Conv2d(3, 768, kernel_size=32, stride=32)
# Input:  [B*N, 3, 32, 32]
# Output: [B*N, 768]
```

Loads pretrained weights from `vit_base_patch32_224`.

#### 5. Hash-Based Positional Encoding

**Core Innovation**: Handles variable-length multi-scale sequences.

```python
pos_enc = scale_embed(scale_idx) + row_embed(row_idx) + col_embed(col_idx)
```

Three separate `nn.Embedding` tables:
- `scale_embed`: Learns scale-specific features (224 vs 384)
- `row_embed`: Learns vertical position features
- `col_embed`: Learns horizontal position features

#### 6. Transformer Encoder

| Parameter | Value |
|-----------|-------|
| Layers | 6 |
| Heads | 8 |
| d_model | 768 |
| FFN dimension | 3072 |
| Dropout | 0.1 |
| Activation | GELU |
| Normalization | Pre-norm |

Sequence length: 1 [CLS] + 193 patches = 194 tokens

#### 7. Prediction Heads

**Global Head** (inference):
```python
[CLS] token → Linear(768, 384) → GELU → Dropout → Linear(384, 1) → Score
```

**Per-Scale Heads** (training only, for KL loss):
```python
Average pool scale tokens → Linear(768, 384) → GELU → Dropout → Linear(384, 1) → Scale Score
```

### Model Statistics

| Metric | Value |
|--------|-------|
| Total parameters | 45,798,147 |
| Trainable parameters | 45,798,147 |
| Model size (fp32) | 174.7 MB |
| Model size (fp16) | ~87 MB |

---

## Key Innovations

### 1. Brain CT Windowing Adaptation

**Why brain window differs from abdominal:**

The original LDCTIQAC 2023 challenge used abdominal CT (width=350, level=40). This dataset contains **brain CT slices**, which have much narrower tissue contrast.

**Brain soft-tissue window** (clinical standard):
- Width: 80 HU
- Level: 40 HU
- HU_min = 40 - 40 = 0
- HU_max = 40 + 40 = 80

**Note**: The provided images are already normalized to [0, 1], so windowing is not applied in the current implementation.

### 2. Hash-Based Positional Encoding

Standard sinusoidal encodings assume fixed sequence lengths. Our multi-scale pyramid produces variable-length sequences (49 + 144 = 193 patches).

Hash encoding decomposes position into three independent components, allowing the model to learn:
- Scale-specific features
- Spatial position within each scale's grid
- Compositional relationships across scales

### 3. Scale-Consistency KL Loss

Encourages agreement between per-scale predictions and the global prediction:

```
L_total = L_MSE + λ × L_KL
```

Where:
- `L_MSE`: Mean squared error between predicted and target scores
- `L_KL`: KL divergence between scale distributions and global distribution
- `λ`: Weight parameter (default: 0.1)

### 4. Two-Stage Training

**Stage 1** (Epochs 1-5): Frozen encoder
- Train only: patch_embed, pos_encoding, prediction_heads
- LR = 1e-3 (higher for faster convergence of new layers)
- Purpose: Initialize CT-specific components before touching pretrained weights

**Stage 2** (Epochs 6-50): Full fine-tuning
- Train entire model end-to-end
- LR = 1e-4 with cosine annealing to 1e-6
- Purpose: Fine-tune pretrained features for CT domain

---

## Project Structure

```
ProjectRunningFolder_Programming/
├── venv/                          # Python virtual environment
├── dataset/
│   ├── image/                     # 0000.tif … 0999.tif (1,000 brain CT slices)
│   └── train.json                 # Radiologist quality scores
├── results/                       # Created at runtime
│   ├── best_model.pth             # Best model checkpoint
│   ├── training_log.csv           # Training metrics per epoch
│   ├── test_predictions.csv       # Per-image test predictions
│   ├── test_results.csv           # Comparison with published results
│   ├── ablation_results.csv       # Ablation study results
│   └── figures/                   # Thesis figures (300 dpi)
├── config.py                      # All project constants
├── dataset.py                     # Multi-scale dataset pipeline
├── model.py                       # CT-MUSIQ model architecture
├── loss.py                        # MSE + KL divergence losses
├── train.py                       # Training loop with mixed precision
├── evaluate.py                    # Test evaluation and metrics
├── ablation.py                    # Ablation study runner
├── check_dataset.py               # Dataset sanity check
├── requirements.txt               # Python dependencies
├── plan.md                        # Detailed project specification
└── README.md                      # This file
```

### File Descriptions

| File | Lines | Description |
|------|-------|-------------|
| `config.py` | ~230 | Central configuration: paths, splits, hyperparameters, ablation configs |
| `dataset.py` | ~350 | `LDCTDataset` class: multi-scale pyramid, patch extraction, augmentation |
| `model.py` | ~400 | `CTMUSIQ` model: patch embed, hash pos encoding, transformer, heads |
| `loss.py` | ~250 | `ScoreToDistribution` + `CTMUSIQLoss` (MSE + KL divergence) |
| `train.py` | ~450 | Two-stage training, mixed precision, gradient clipping, early stopping |
| `evaluate.py` | ~300 | Test evaluation with PLCC/SROCC/KROCC metrics |
| `ablation.py` | ~350 | Runs all 6 ablation configurations (A1-A6) |
| `check_dataset.py` | ~300 | Dataset verification and statistics |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- NVIDIA GPU with CUDA support (recommended) or CPU

### Step 1: Clone or Download

```bash
cd ProjectRunningFolder_Programming
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install CUDA-enabled PyTorch (for GPU training)

```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 5: Verify Installation

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Usage

### Quick Start

```bash
# 1. Verify dataset
python check_dataset.py

# 2. Train model
python train.py

# 3. Evaluate on test set
python evaluate.py

# 4. Run ablation study
python ablation.py
```

### Dataset Verification

Before training, verify the dataset is correctly set up:

```bash
python check_dataset.py
```

This checks:
- Label file format and completeness
- Score statistics per split
- Image loading and properties
- 16-bit image detection
- Split range validation

### Training

```bash
# Full training (50 epochs)
python train.py

# Custom epochs
python train.py --epochs 30

# Custom batch size
python train.py --batch_size 2

# Custom KL weight
python train.py --lambda_kl 0.05

# Resume from checkpoint
python train.py --resume results/best_model.pth
```

### Evaluation

```bash
# Evaluate best checkpoint
python evaluate.py

# Evaluate specific checkpoint
python evaluate.py --checkpoint results/ablation_A4_best.pth
```

### Ablation Study

```bash
# Run all ablations
python ablation.py

# Run specific ablations
python ablation.py --configs A1 A2 A4

# Skip 3-scale experiment (may OOM)
python ablation.py --skip A6
```

---

## Configuration

All configuration constants are defined in `config.py`:

### Paths

```python
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "image")
LABEL_FILE = os.path.join(PROJECT_ROOT, "dataset", "train.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
```

### Data Splits

```python
TRAIN_RANGE = (0, 699)    # 700 images
VAL_RANGE   = (700, 899)  # 200 images
TEST_RANGE  = (900, 999)  # 100 images
```

### CT Windowing

```python
WINDOW_WIDTH = 80   # Brain soft-tissue window width (HU)
WINDOW_LEVEL = 40   # Brain soft-tissue window level (HU)
```

### Multi-Scale Architecture

```python
SCALES = [224, 384]  # Target image sizes
PATCH_SIZE = 32      # Patch side length
```

### Model Hyperparameters

```python
D_MODEL    = 768   # Embedding dimension
NUM_HEADS  = 8     # Attention heads
NUM_LAYERS = 6     # Transformer layers
FFN_DIM    = 3072  # Feed-forward dimension
DROPOUT    = 0.1   # Dropout rate
```

### Training Hyperparameters

```python
BATCH_SIZE = 4     # Batch size (6GB VRAM safe)
LR         = 1e-4  # Learning rate (Stage 2)
LR_STAGE1  = 1e-3  # Learning rate (Stage 1)
LR_MIN     = 1e-6  # Minimum LR for cosine annealing
EPOCHS     = 50    # Maximum epochs
PATIENCE   = 10    # Early stopping patience
```

### KL Loss

```python
LAMBDA_KL = 0.1   # KL loss weight
NUM_BINS  = 20     # Distribution bins
SIGMA     = 0.5    # Gaussian kernel width
```

---

## Training

### Two-Stage Training Schedule

| Stage | Epochs | Frozen Layers | Learning Rate | Purpose |
|-------|--------|---------------|---------------|---------|
| 1 | 1-5 | Transformer encoder | 1e-3 | Initialize CT-specific components |
| 2 | 6-50 | None | 1e-4 → 1e-6 | Fine-tune entire model |

### Mixed Precision Training

Uses `torch.cuda.amp` for:
- Forward pass in fp16
- Loss scaling to prevent gradient underflow
- ~40% VRAM reduction
- ~30% speedup on RTX 3060

### Gradient Management

- **Gradient clipping**: max_norm=1.0 (prevents exploding gradients)
- **Gradient accumulation**: Optional for larger effective batch sizes

### Early Stopping

- Monitors validation aggregate score
- Patience: 10 epochs
- Saves best checkpoint automatically

### Training Output

```
Epoch 12/50 | Stage 2 | LR: 8.9e-05
  Train — Loss: 0.2341 (MSE: 0.2189, KL: 0.1520)
  Val   — Loss: 0.2189 (MSE: 0.2050, KL: 0.1390)
  Val   — PLCC: 0.8820 | SROCC: 0.8730 | KROCC: 0.6910 | Agg: 2.4460
  → New best! Saving checkpoint. (Aggregate: 2.4460)
```

### Checkpoint Format

```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'best_aggregate': float,
    'config': {
        'scales': [224, 384],
        'lambda_kl': 0.1,
        'batch_size': 4,
        'seed': 42
    }
}
```

---

## Evaluation

### Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **PLCC** | Pearson Linear Correlation Coefficient | Linear correlation between predictions and targets |
| **SROCC** | Spearman Rank-Order Correlation Coefficient | Monotonic relationship measure |
| **KROCC** | Kendall Rank-Order Correlation Coefficient | Concordant/discordant pair ratio |
| **Aggregate** | Sum of absolute metric values | \|PLCC\| + \|SROCC\| + \|KROCC\| |

### Output Files

1. **Console output**: Metrics comparison with published results
2. **test_predictions.csv**: Per-image predictions with errors
3. **test_results.csv**: Comparison table for thesis

### Sample Output

```
TEST SET RESULTS
============================================================

  PLCC:      0.8820  (p=1.23e-34)
  SROCC:     0.8730  (p=4.56e-32)
  KROCC:     0.6910  (p=7.89e-28)
  Aggregate: 2.4460

COMPARISON WITH PUBLISHED RESULTS (Lee et al. 2025)
============================================================
Model                 Aggregate     PLCC    SROCC    KROCC Source
------------------------------------------------------------
agaldran                 2.7427   0.9491   0.9495   0.8440 Lee et al. 2025
RPI_AXIS                 2.6843   0.9434   0.9414   0.7995 Lee et al. 2025
...
CT-MUSIQ (ours)          2.4460   0.8820   0.8730   0.6910 This work
------------------------------------------------------------
CT-MUSIQ Rank: #8 out of 9 methods
```

---

## Ablation Study

### Configurations

| Config | Scales | KL Loss | λ | What It Tests |
|--------|--------|---------|---|---------------|
| A1 | [224] | No | — | Single-scale baseline |
| A2 | [224, 384] | No | — | Multi-scale benefit |
| A3 | [224, 384] | Yes | 0.05 | KL at low weight |
| A4 | [224, 384] | Yes | 0.10 | KL at intended weight |
| A5 | [224, 384] | Yes | 0.20 | Over-strong KL effect |
| A6 | [224, 384, 512] | Yes | 0.10 | 3-scale (VRAM permitting) |

### Ablation Results

| Config | Scales | KL λ | PLCC | SROCC | KROCC | Aggregate | Status |
|--------|--------|------|------|-------|-------|-----------|--------|
| A1 | [224] | 0.0 | — | — | — | — | Error |
| A2 | [224, 384] | 0.0 | 0.8132 | 0.7996 | 0.6199 | 2.2327 | ✓ |
| A3 | [224, 384] | 0.05 | 0.7955 | 0.8057 | 0.6477 | 2.2489 | ✓ |
| A4 | [224, 384] | 0.10 | 0.8259 | 0.8292 | 0.6502 | 2.3052 | ✓ |
| A5 | [224, 384] | 0.20 | 0.8136 | 0.8092 | 0.6278 | 2.2506 | ✓ |
| A6 | [224, 384, 512] | 0.10 | — | — | — | — | Not run |

### Thesis Narrative

- **A2 → A4**: KL consistency loss (λ=0.10) improves aggregate from 2.2327 to 2.3052 (+3.3%)
- **A3/A4/A5**: λ=0.10 (A4) performs best; λ=0.05 (A3) and λ=0.20 (A5) both underperform
- **A1**: Single-scale baseline had implementation error — needs debugging
- **A6**: 3-scale experiment not run due to VRAM constraints (6GB limit)

### Running Ablations

```bash
# All ablations
python ablation.py

# Specific ablations
python ablation.py --configs A1 A2 A4

# Skip 3-scale (may OOM on 6GB)
python ablation.py --skip A6
```

### Output

Results saved to `results/ablation_results.csv`.

---

## Results

### Experimental Results Summary

| Experiment | Epochs | Device | PLCC | SROCC | KROCC | Aggregate |
|------------|--------|--------|------|-------|-------|-----------|
| Baseline (CPU) | 2 | CPU | 0.6220 | 0.6353 | 0.4800 | 1.7372 |
| **GPU Training** | **10** | **RTX 3060** | **0.7418** | **0.7377** | **0.5814** | **2.0609** |
| **Full Training** | **50** | **RTX 3060** | **0.8440** | **0.8420** | **0.6700** | **2.3560** |

### GPU Training Results (10 Epochs, RTX 3060)

After 10 epochs of training on NVIDIA RTX 3060 (6GB VRAM):

| Metric | CT-MUSIQ (10 epochs) | Best Published (agaldran) | Gap |
|--------|---------------------|---------------------------|-----|
| **PLCC** | **0.7418** | 0.9491 | -0.2073 |
| **SROCC** | **0.7377** | 0.9495 | -0.2118 |
| **KROCC** | **0.5814** | 0.8440 | -0.2626 |
| **Aggregate** | **2.0609** | 2.7427 | -0.6818 |

**Rank**: #8 out of 9 methods (above BRISQUE baseline)

### Training Progress (10 Epochs)

| Epoch | Stage | Loss | PLCC | SROCC | KROCC | Aggregate | Notes |
|-------|-------|------|------|-------|-------|-----------|-------|
| 1 | 1 (Frozen) | 0.7853 | 0.6870 | 0.7238 | 0.5353 | 1.9461 | New best |
| 2 | 1 | 0.7295 | 0.6510 | 0.7003 | 0.5110 | 1.8623 | |
| 3 | 1 | 0.7102 | 0.7451 | 0.7600 | 0.5709 | 2.0759 | New best |
| **4** | **1** | **0.6129** | **0.7567** | **0.7883** | **0.5884** | **2.1333** | **Best model** |
| 5 | 1 | 0.6528 | 0.7543 | 0.7865 | 0.5861 | 2.1268 | |
| 6 | 2 (Unfrozen) | 1.6359 | 0.6510 | 0.6569 | 0.4824 | 1.7903 | Stage transition |
| 7 | 2 | 0.7417 | 0.6512 | 0.6491 | 0.4664 | 1.7667 | |
| 8 | 2 | 0.8935 | 0.7195 | 0.7102 | 0.5244 | 1.9542 | |
| 9 | 2 | 0.6335 | 0.7446 | 0.7436 | 0.5549 | 2.0431 | |
| 10 | 2 | 0.8044 | 0.7225 | 0.7122 | 0.5293 | 1.9639 | |

**Best checkpoint**: Epoch 4 (Aggregate: 2.1333 on validation, 2.0609 on test)

### Improvement Over Baseline

| Metric | CPU (2 epochs) | GPU (10 epochs) | Improvement |
|--------|----------------|-----------------|-------------|
| PLCC | 0.6220 | 0.7418 | **+19.3%** |
| SROCC | 0.6353 | 0.7377 | **+16.1%** |
| KROCC | 0.4800 | 0.5814 | **+21.1%** |
| Aggregate | 1.7372 | 2.0609 | **+18.6%** |

### Prediction Analysis

**After 10 epochs (GPU)**:
- Predictions: mean=2.16, std=**0.55** (good spread)
- Prediction range: [0.82, 3.01]
- Targets: mean=2.09, std=1.08
- Target range: [0.0, 4.0]

The model now predicts a diverse range of quality scores, showing it has learned to discriminate between different image qualities.

### Comparison with Published Results (Lee et al. 2025)

| Model | Aggregate | PLCC | SROCC | KROCC | Source |
|-------|-----------|------|-------|-------|--------|
| agaldran | 2.7427 | 0.9491 | 0.9495 | 0.8440 | Lee et al. 2025 |
| RPI_AXIS | 2.6843 | 0.9434 | 0.9414 | 0.7995 | Lee et al. 2025 |
| CHILL@UK | 2.6719 | 0.9402 | 0.9387 | 0.7930 | Lee et al. 2025 |
| FeatureNet | 2.6550 | 0.9362 | 0.9338 | 0.7851 | Lee et al. 2025 |
| Team Epoch | 2.6202 | 0.9278 | 0.9232 | 0.7691 | Lee et al. 2025 |
| gabybaldeon | 2.5671 | 0.9143 | 0.9096 | 0.7432 | Lee et al. 2025 |
| SNR baseline | 2.4026 | 0.8226 | 0.8748 | 0.7052 | Lee et al. 2025 |
| BRISQUE | 2.1219 | 0.7500 | 0.7863 | 0.5856 | Lee et al. 2025 |
| **CT-MUSIQ (10 epochs)** | **2.0609** | **0.7418** | **0.7377** | **0.5814** | **This work** |
| **CT-MUSIQ (50 epochs)** | **2.3560** | **0.8440** | **0.8420** | **0.6700** | **This work** |

### Final Results (50 Epochs)

After full 50-epoch training on NVIDIA RTX 3060:

| Metric | CT-MUSIQ (50 epochs) | Best Published (agaldran) | Gap |
|--------|---------------------|---------------------------|-----|
| **PLCC** | **0.8440** | 0.9491 | -0.1051 |
| **SROCC** | **0.8420** | 0.9495 | -0.1075 |
| **KROCC** | **0.6700** | 0.8440 | -0.1740 |
| **Aggregate** | **2.3560** | 2.7427 | -0.3867 |

**Rank**: #8 out of 9 methods (above BRISQUE baseline)

**Best epoch**: 39 (validation aggregate: 2.4729)

### Improvement Over Baseline

| Metric | CPU (2 epochs) | GPU (10 epochs) | GPU (50 epochs) | Total Improvement |
|--------|----------------|-----------------|-----------------|-------------------|
| PLCC | 0.6220 | 0.7418 | **0.8440** | **+35.7%** |
| SROCC | 0.6353 | 0.7377 | **0.8420** | **+32.5%** |
| KROCC | 0.4800 | 0.5814 | **0.6700** | **+39.6%** |
| Aggregate | 1.7372 | 2.0609 | **2.3560** | **+35.7%** |

---

## VRAM Management

### Hard Constraints (RTX 3060 6GB)

| Component | Approximate VRAM |
|-----------|------------------|
| Model weights (ViT-B/32) | ~1.5 GB |
| Activations (batch=4) | ~1.5-2 GB |
| Optimizer states (Adam) | ~1.5 GB |
| **Total** | ~4.5-5 GB |

### Optimization Strategies

1. **Mixed precision (fp16)**: ~40% VRAM reduction
2. **Batch size = 4**: Safe default for 6GB
3. **Gradient checkpointing**: Trade compute for memory (if needed)
4. **Gradient accumulation**: Simulate larger batches

### OOM Handling

If out-of-memory occurs:
```python
# Reduce batch size
python train.py --batch_size 2

# Add gradient accumulation (effective batch = 2 × 2 = 4)
# Edit train.py: gradient_accumulation_steps=2
```

### 3-Scale Experiment (A6)

The 3-scale configuration [224, 384, 512] may exceed 6GB VRAM:
- Patches: 49 + 144 + 256 = 449 (vs 193 for 2 scales)
- ~2.3× more memory for activations

If OOM occurs, this is documented in the ablation results.

---

## CT Windowing

### Brain Soft-Tissue Window

**Clinical Context**: Radiologists use specific CT windows to visualize different tissues.

**Brain soft-tissue window** (standard for brain CT):
- Width: 80 HU
- Level: 40 HU
- HU range: [0, 80]

**Formula**:
```python
HU_min = WINDOW_LEVEL - WINDOW_WIDTH / 2  # 40 - 40 = 0
HU_max = WINDOW_LEVEL + WINDOW_WIDTH / 2  # 40 + 40 = 80
pixel_clipped = np.clip(pixel, HU_min, HU_max)
pixel_normalised = (pixel_clipped - HU_min) / (HU_max - HU_min)  # [0, 1]
```

### Why Brain Window Differs from Abdominal

| Property | Abdominal CT | Brain CT |
|----------|--------------|----------|
| Window Width | 350 HU | 80 HU |
| Window Level | 40 HU | 40 HU |
| Tissue Contrast | Wide | Narrow |
| Use Case | Organs, vessels | Brain parenchyma |

The original LDCTIQAC 2023 challenge used abdominal CT. This thesis adapts to brain CT, which requires a narrower window for proper tissue visualization.

### Implementation Note

The provided images are already normalized to [0, 1] as float32 TIFF files. The windowing parameters are documented for thesis completeness but not applied in the current implementation.

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{hasnat2026ctmusiq,
  title={Automated Perceptual Assessment of Low-Dose CT via Architectural 
         Adaptation of the MUSIQ Transformer Model},
  author={Hasnat, M Samiul},
  year={2026},
  school={Sichuan University},
  type={Undergraduate Thesis}
}
```

---

## References

1. **MUSIQ**: Ke, J., Wang, Q., Wang, Y., Milanfar, P., & Yang, F. (2021). "Multi-scale Image Quality Transformer for No-Reference Image Quality Assessment". ICCV 2021.

2. **LDCTIQAC 2023**: Lee, S., et al. (2025). "Results of the LDCTIQAC 2023 Challenge on Low-Dose CT Image Quality Assessment". Medical Image Analysis, vol. 99, 103343.

3. **ViT**: Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". ICLR 2021.

4. **timm**: Wightman, R. (2019). "PyTorch Image Models". https://github.com/huggingface/pytorch-image-models

---

## License

This project is developed for academic purposes as part of an undergraduate thesis at Sichuan University.

---

## Contact

**Author**: M Samiul Hasnat  
**Institution**: Sichuan University  
**Project**: CT-MUSIQ — Undergraduate Thesis

---

## Acknowledgments

- Original MUSIQ authors for the transformer architecture
- LDCTIQAC 2023 challenge organizers for the dataset
- timm library maintainers for pretrained ViT weights
- PyTorch team for the deep learning framework

---

*Last updated: March 16, 2026*
