"""
config.py — CT-MUSIQ Project Configuration
============================================

Central configuration file for the CT-MUSIQ project.  Every constant used
anywhere in the codebase is defined here so that experiments are fully
reproducible and nothing is hard-coded in training / evaluation scripts.

Author : M Samiul Hasnat, Sichuan University
Project: Undergraduate Thesis — CT-MUSIQ
"""

import os

# =============================================================================
# 1. PATHS
# =============================================================================

# Root directory of the project (resolved relative to this file's location)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory containing the 1,000 brain CT .tif images (0000.tif … 0999.tif)
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "image")

# JSON file mapping image filenames to radiologist quality scores
LABEL_FILE = os.path.join(PROJECT_ROOT, "dataset", "train.json")

# Directory where checkpoints, logs, CSVs, and figures are saved (created at runtime)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# =============================================================================
# 2. DATASET FORMAT
# =============================================================================
# These constants encode the actual naming conventions discovered by inspecting
# train.json and the image directory.  The original plan.md assumed 3-digit
# keys without an extension ("000"), but the real dataset uses 4-digit keys
# with a ".tif" suffix ("0000.tif").

# File extension used by the CT images on disk
IMAGE_EXT = ".tif"

# Number of zero-padded digits in the image filename (e.g. "0000" → 4 digits)
IMAGE_ID_DIGITS = 4

# Python format string to build a train.json key from an integer index:
#   LABEL_KEY_FORMAT.format(idx=42)  →  "0042.tif"
LABEL_KEY_FORMAT = "{idx:04d}.tif"

# Python format string to build an image filename from an integer index:
#   IMAGE_FILENAME_FORMAT.format(idx=42)  →  "0042.tif"
IMAGE_FILENAME_FORMAT = "{idx:04d}.tif"

# =============================================================================
# 3. DATA SPLITS
# =============================================================================
# The LDCTIQAC 2023 dataset is split purely by filename index.
# Ranges are **inclusive** on both ends.

TRAIN_RANGE = (0, 699)    # 700 images for training   (0000.tif – 0699.tif)
VAL_RANGE   = (700, 899)  # 200 images for validation  (0700.tif – 0899.tif)
TEST_RANGE  = (900, 999)  # 100 images for testing      (0900.tif – 0999.tif)

TOTAL_IMAGES = 1000       # Expected total number of images in the dataset

# =============================================================================
# 4. CT WINDOWING (Brain Soft-Tissue)
# =============================================================================
# CT windowing clips raw pixel / HU values to a clinically relevant range and
# then normalises to [0, 1].
#
# ** Why brain window differs from the original challenge: **
# The LDCTIQAC 2023 challenge originally used abdominal CT (width=350,
# level=40).  This dataset contains BRAIN CT slices, which have much narrower
# tissue contrast.  The brain soft-tissue window (width=80, level=40) is the
# clinical standard used by radiologists to evaluate brain parenchyma.
# This is a deliberate, thesis-relevant adaptation — document it in the
# Method section.
#
# Window formula:
#   HU_min = WINDOW_LEVEL - WINDOW_WIDTH / 2  →  40 - 40 = 0
#   HU_max = WINDOW_LEVEL + WINDOW_WIDTH / 2  →  40 + 40 = 80
#   pixel  = clip(pixel, HU_min, HU_max)
#   pixel  = (pixel - HU_min) / (HU_max - HU_min)   →  [0.0, 1.0]

WINDOW_WIDTH = 80   # Width of the brain soft-tissue CT window (in HU)
WINDOW_LEVEL = 40   # Centre of the brain soft-tissue CT window (in HU)

# =============================================================================
# 5. QUALITY SCORE RANGE
# =============================================================================
# Radiologist Likert scores range from 0 (worst) to 4 (best).
# These bounds are used for normalisation and for building soft distributions
# in the KL scale-consistency loss.

SCORE_MIN = 0.0  # Minimum possible quality score
SCORE_MAX = 4.0  # Maximum possible quality score

# =============================================================================
# 6. MULTI-SCALE ARCHITECTURE
# =============================================================================
# CT-MUSIQ resizes each CT image to multiple resolutions and extracts non-
# overlapping patches from each.  Using 2 scales is safe for 6 GB VRAM;
# a 3rd scale (512) is reserved for ablation A6 if memory allows.

SCALES = [224, 384]  # Target image sizes (pixels) for the multi-scale pyramid

PATCH_SIZE = 32      # Side length of each square patch (pixels)
                     # 224 / 32 = 7  → 7×7 = 49 patches at scale 0
                     # 384 / 32 = 12 → 12×12 = 144 patches at scale 1
                     # Total tokens per image: 49 + 144 = 193

# =============================================================================
# 7. TRANSFORMER / MODEL HYPERPARAMETERS
# =============================================================================

D_MODEL    = 768   # Embedding dimension (must match ViT-B/32 pretrained weights)
NUM_HEADS  = 8     # Number of attention heads in the transformer encoder
NUM_LAYERS = 12    # Number of transformer encoder layers (full ViT-B depth)
FFN_DIM    = 3072  # Feed-forward network hidden dimension (4 × d_model)
DROPOUT    = 0.05  # Dropout rate in transformer layers (reduced for small dataset)

# Maximum grid size for hash-based positional encoding.
# At 384 / 32 = 12 patches per side, we need indices 0–11.
# Set to 13 to leave a small margin for a potential 3rd scale (512/32 = 16,
# but that would need MAX_GRID_SIZE = 17 — updated dynamically in ablation A6).
MAX_GRID_SIZE = 13

# =============================================================================
# 8. TRAINING HYPERPARAMETERS
# =============================================================================

BATCH_SIZE = 10     # Default batch size — fits within 6 GB VRAM with AMP
LR         = 5e-5  # Base learning rate for Stage 2 (lower for stable fine-tuning)
LR_STAGE1  = 3e-4  # Stage 1 LR (reduced from 1e-3 for stability)
LR_MIN     = 1e-6  # Minimum LR for cosine annealing scheduler
EPOCHS     = 100   # Maximum number of training epochs (extended for better convergence)
PATIENCE   = 20    # Early stopping patience (increased for 100-epoch extended run)

# Two-stage training schedule:
#   Stage 1 (epochs 1–STAGE1_EPOCHS): freeze transformer encoder, train heads only
#   Stage 2 (epochs STAGE1_EPOCHS+1 – EPOCHS): unfreeze all, cosine LR decay
STAGE1_EPOCHS = 5  # Number of warm-up epochs with frozen encoder

# LR warmup at Stage 2 start (linear warmup over this many epochs)
STAGE2_WARMUP_EPOCHS = 3

# Gradient clipping max norm (prevents exploding gradients with mixed precision)
MAX_GRAD_NORM = 1.0

# =============================================================================
# 9. KL SCALE-CONSISTENCY LOSS
# =============================================================================
# The KL loss penalises disagreement between per-scale quality predictions
# and the global prediction head.  Scores are converted to soft probability
# distributions over NUM_BINS bins using a Gaussian kernel with width SIGMA.

LAMBDA_KL = 0.10  # Weight of KL loss — ablation A4 confirmed 0.10 is optimal (Agg 2.305 vs 2.249 no-KL)
NUM_BINS  = 20     # Number of bins for the soft score distribution
SIGMA     = 0.5    # Gaussian kernel width for score → distribution conversion

# =============================================================================
# 10. REPRODUCIBILITY
# =============================================================================

SEED = 42  # Random seed for torch, numpy, and Python's random module

# =============================================================================
# 11. ABLATION CONFIGURATIONS
# =============================================================================
# Each dict defines one ablation experiment.  Keys match constructor /
# training-loop arguments so they can be unpacked directly.

ABLATION_CONFIGS = {
    "A1": {
        "description": "Single-scale baseline (224 only, no KL loss)",
        "scales": [224],
        "lambda_kl": 0.0,
        "epochs": 30,       # Shorter cycle for ablation runs
    },
    "A2": {
        "description": "Two scales, no KL loss",
        "scales": [224, 384],
        "lambda_kl": 0.0,
        "epochs": 30,
    },
    "A3": {
        "description": "Two scales, KL λ=0.05 (low weight)",
        "scales": [224, 384],
        "lambda_kl": 0.05,
        "epochs": 30,
    },
    "A4": {
        "description": "Two scales, KL λ=0.10 (intended best)",
        "scales": [224, 384],
        "lambda_kl": 0.10,
        "epochs": 30,
    },
    "A5": {
        "description": "Two scales, KL λ=0.20 (over-strong KL?)",
        "scales": [224, 384],
        "lambda_kl": 0.20,
        "epochs": 30,
    },
    "A6": {
        "description": "Three scales, KL λ=0.10 (only if VRAM allows)",
        "scales": [224, 384, 512],
        "lambda_kl": 0.10,
        "epochs": 30,
    },
}

# =============================================================================
# 12. LOGGING & CHECKPOINTING
# =============================================================================

# CSV log file appended to every epoch during training
TRAINING_LOG_CSV = os.path.join(RESULTS_DIR, "training_log.csv")

# Best model checkpoint (saved whenever validation aggregate improves)
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")

# Final test-set results table (includes comparison with Lee et al. 2025)
TEST_RESULTS_CSV = os.path.join(RESULTS_DIR, "test_results.csv")

# Ablation summary table
ABLATION_RESULTS_CSV = os.path.join(RESULTS_DIR, "ablation_results.csv")

# Directory for thesis-quality figures (300 dpi PNGs)
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
