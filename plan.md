# CT-MUSIQ: Architectural Adaptation of MUSIQ for Low-Dose CT Image Quality Assessment
## Agent Plan — Antigravity IDE

---

## Project Identity

**Thesis Title:** Automated Perceptual Assessment of Low-Dose CT via Architectural Adaptation of the MUSIQ Transformer Model  
**Author:** M Samiul Hasnat, Sichuan University  
**Model Name:** CT-MUSIQ  
**Dataset:** LDCTIQAC 2023 (MICCAI) — 1,000 brain CT slices, radiologist Likert scores [0–4]  
**Framework:** PyTorch  
**GPU:** NVIDIA RTX 3060 Laptop, 6GB VRAM  
**Reference Paper:** Lee et al. (2025), *Medical Image Analysis*, vol. 99, 103343  
**Original MUSIQ Paper:** Ke et al. (2021), ICCV  

---

## Project Root & Virtual Environment

All work lives inside `ProjectRunningFolder_Programming/`. Always activate the venv before running anything.

```
ProjectRunningFolder_Programming/
├── venv/                        ← virtual environment (never edit manually)
├── dataset/
│   ├── image/                   ← 000.tiff … 999.tiff (1,000 brain CT slices)
│   └── train.json               ← radiologist quality scores for all images
├── config.py
├── dataset.py
├── model.py
├── loss.py
├── train.py
├── evaluate.py
├── ablation.py
├── plan.md                      ← this file
├── README.md
├── requirements.txt
└── results/                     ← created at runtime, holds CSVs and checkpoints
```

**Creating the venv (run once):**
```bash
cd ProjectRunningFolder_Programming
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## Dataset Details

| Property | Value |
|---|---|
| Total images | 1,000 brain CT slices (.tiff format) |
| Image directory | `./dataset/image/` |
| Label file | `./dataset/train.json` |
| Training split | 000.tiff – 699.tiff (700 images) |
| Validation split | 700.tiff – 899.tiff (200 images) |
| Test split | 900.tiff – 999.tiff (100 images) |
| Score range | 0–4 Likert scale (radiologist average) |
| Modality | Brain CT (grayscale, single channel) |

**Filename convention:** Images are named with zero-padded 3-digit indices
(`000.tiff`, `001.tiff`, …, `999.tiff`). The split is determined entirely by
index — no separate split file needed.

**train.json expected structure (verify on first run):**
```json
{
  "000": 3.2,
  "001": 1.8,
  ...
}
```
Keys are the image index as a string (without `.tiff`). Values are averaged
radiologist scores. Print the first 5 entries on startup to confirm format.

---

## Agent Behaviour Rules

- Never silently skip a step. If blocked, say why and propose an alternative.
- Never hallucinate metric values. Only report numbers from actual training runs.
- Always write modular, commented code. Every function must have a docstring.
- When a design choice exists, implement the simpler option first, then offer
  the advanced variant.
- Prefer explicit over implicit. No magic numbers without a named constant.
- After completing each file, print a short summary of what was built and what
  the next step is.
- Always respect the 6GB VRAM constraint. Flag any operation likely to exceed it.

---

## VRAM Budget (RTX 3060 6GB)

| Component | Approx. VRAM |
|---|---|
| Model weights (ViT-B/32 backbone) | ~1.5 GB |
| Activations per batch | ~1.5–2 GB |
| Optimizer states (Adam) | ~1.5 GB |
| **Safe default batch size** | **4 images** |
| Mixed precision (fp16) saves | ~40% reduction |

**Hard rules for staying within 6GB:**
- Always use `torch.cuda.amp.autocast()` and `GradScaler` (mixed precision).
- Default `BATCH_SIZE = 4`. Increase to 6 only after confirming no OOM.
- Start with 2 scales only. Add a 3rd as a conditional ablation config (A6)
  after confirming memory headroom.
- Set `torch.backends.cudnn.benchmark = True` for faster convolution selection.

---

## Phase 0 — Environment Setup

**Goal:** Reproducible venv, dataset verified, first image displayed.

### Steps

1. Create `requirements.txt`:
   ```
   torch>=2.1.0
   torchvision>=0.16.0
   numpy
   scipy
   scikit-learn
   pillow
   pandas
   matplotlib
   tqdm
   einops
   timm
   ```

2. Create `config.py` with all project constants:
   ```python
   DATA_DIR    = "./dataset/image/"
   LABEL_FILE  = "./dataset/train.json"
   RESULTS_DIR = "./results/"

   TRAIN_RANGE = (0, 699)   # inclusive both ends
   VAL_RANGE   = (700, 899)
   TEST_RANGE  = (900, 999)

   SCALES      = [224, 384]  # two scales — safe for 6GB
   PATCH_SIZE  = 32
   BATCH_SIZE  = 4
   LR          = 1e-4
   LR_MIN      = 1e-6
   EPOCHS      = 50
   LAMBDA_KL   = 0.1
   PATIENCE    = 10          # early stopping patience

   # Brain CT standard soft-tissue window
   WINDOW_WIDTH = 80
   WINDOW_LEVEL = 40

   SCORE_MIN = 0
   SCORE_MAX = 4
   SEED      = 42
   ```

   > **Why brain CT window differs from abdominal:** The original challenge
   > used abdominal CT (width=350, level=40). This dataset uses brain CT, which
   > has much narrower tissue contrast. Brain soft-tissue window (width=80,
   > level=40) is the clinical standard and is a deliberate, thesis-relevant
   > adaptation. Document this choice in the thesis Method section.

3. Write a dataset sanity check (as a script block, not a separate file):
   - Load `train.json`, print first 5 entries to confirm key format
   - Confirm all 1,000 keys are present
   - Print score distribution: mean, std, min, max, histogram per split
   - Load one image from each split, apply CT windowing, display side by side
   - Print: image shape, dtype, raw pixel range, post-window range
   - Confirm train/val/test index ranges are non-overlapping

**Exit criteria:** All 1,000 paths resolve. Label stats print. Windowed image
displays as a recognisable brain CT slice.

---

## Phase 1 — Dataset Pipeline (`dataset.py`)

**Goal:** A `Dataset` returning multi-scale patches, coordinates, and quality score.

### CT Windowing (brain-specific)
```python
HU_min = WINDOW_LEVEL - WINDOW_WIDTH / 2   # = 0
HU_max = WINDOW_LEVEL + WINDOW_WIDTH / 2   # = 80
pixel  = np.clip(pixel, HU_min, HU_max)
pixel  = (pixel - HU_min) / (HU_max - HU_min)  # → [0.0, 1.0]
```

### TIFF Loading
- Use `PIL.Image.open()` for .tiff files.
- Convert immediately to `numpy.float32`.
- Check bit depth — brain CT TIFFs may be 16-bit. Normalise raw pixel values
  to float32 range before applying windowing.

### Multi-scale Pyramid
- Produce `len(SCALES)` resized versions using bicubic interpolation.
- Replicate single channel to 3: `image_tensor.repeat(3, 1, 1)`.
  This allows loading ImageNet-pretrained patch embedding weights.

### Patch Tokenisation
- 224×224 image → 7×7 = 49 patches.
- 384×384 image → 12×12 = 144 patches.
- Total tokens per image: 193 patches.
- For each patch, record integer coords `(scale_idx, row_idx, col_idx)`.
- Return all patches concatenated: `[193, 3, 32, 32]`.
- Return all coords: `[193, 3]`.

### Augmentation (training split only)
- Random horizontal flip (p=0.5)
- Random crop retaining 85–100% of area, resize back to original
- Brightness/contrast jitter ±0.05 only — CT noise is diagnostically meaningful,
  do not apply strong photometric augmentation

### Class interface
```python
class LDCTDataset(Dataset):
    def __init__(self, data_dir, label_file, split,
                 scales, patch_size, augment=False):
        """
        Args:
            data_dir:   path to ./dataset/image/
            label_file: path to ./dataset/train.json
            split:      'train', 'val', or 'test'
            scales:     list of int target sizes, e.g. [224, 384]
            patch_size: int, default 32
            augment:    bool, training split only
        """
    def __len__(self): ...

    def __getitem__(self, idx) -> dict:
        # returns:
        # {
        #   'patches':  Tensor [N, 3, 32, 32],
        #   'coords':   Tensor [N, 3],
        #   'score':    Tensor scalar float32,
        #   'image_id': str  e.g. '042'
        # }
```

**Exit criteria:** DataLoader iterates one training epoch without error.
Patches shape `[4, 193, 3, 32, 32]` for batch size 4. Scores in [0, 4].

---

## Phase 2 — Model Architecture (`model.py`)

**Goal:** Full CT-MUSIQ. Build bottom-up in this order:
patch embed → hash positional encoding → transformer encoder → prediction heads.

### 2.1 Patch Embedding
```python
self.patch_embed = nn.Conv2d(3, 768, kernel_size=32, stride=32)
# Input:  [B*N, 3, 32, 32]
# Output: [B*N, 768, 1, 1] → flatten → [B, N, 768]
```
Attempt to load weights from pretrained ViT-B/32. If shapes match, load them.
If not (e.g. channel mismatch), reinitialise that layer only and log a warning.

### 2.2 Hash-based Spatial Positional Encoding
```python
self.scale_embed = nn.Embedding(num_scales, d_model)
self.row_embed   = nn.Embedding(max_grid_size, d_model)  # max_grid_size = 13
self.col_embed   = nn.Embedding(max_grid_size, d_model)

# For each token: pos_enc = scale_embed(s) + row_embed(r) + col_embed(c)
```
This is the core innovation over standard ViT. Standard sinusoidal encodings
assume a fixed sequence length. Hash encoding handles variable-length sequences
from multiple scales with different spatial grids. Explain this explicitly in
the thesis Architecture section.

### 2.3 Transformer Encoder
- Prepend learnable `[CLS]` token: `nn.Parameter(torch.randn(1, 1, 768))`.
- `nn.TransformerEncoder`: 6 layers, 8 heads, FFN=3072, dropout=0.1.
- Load encoder weights from `timm.create_model('vit_base_patch32_224', pretrained=True)`.
- Sequence length = 1 + 193 = 194. Self-attention cost: O(194²) ≈ safe for 6GB.

### 2.4 Prediction Heads
- **Global head:** `nn.Linear(768, 1)` on [CLS] token output → final quality score.
- **Per-scale heads:** Average-pool tokens belonging to each scale separately
  → `nn.Linear(768, 1)` per scale → scale-specific score for KL loss.
- Inference: return global head only.
- Training: return global score + list of scale scores.

### Class interface
```python
class CTMUSIQ(nn.Module):
    def __init__(self, num_scales=2, d_model=768, num_heads=8,
                 num_layers=6, patch_size=32, pretrained=True):
        """
        CT-adapted MUSIQ transformer for brain CT image quality assessment.
        """

    def forward(self, patches, coords):
        """
        Args:
            patches: Tensor [B, N, 3, 32, 32]
            coords:  Tensor [B, N, 3]
        Returns:
            dict:
                'score':        Tensor [B, 1]
                'scale_scores': list[Tensor [B, 1]], one per scale
        """
```

**Exit criteria:** Forward pass on random batch `[4, 193, 3, 32, 32]` completes.
No OOM. Print total trainable parameter count.

---

## Phase 3 — Loss Function (`loss.py`)

**Goal:** MSE primary loss + KL scale-consistency loss.

### Primary loss
```python
L_primary = F.mse_loss(output['score'].squeeze(), target_score)
```

### Scale-consistency KL loss
```python
def score_to_distribution(score, num_bins=20, score_min=0.0,
                           score_max=4.0, sigma=0.5):
    """
    Convert a scalar quality score to a soft probability distribution
    over num_bins evenly-spaced bins between score_min and score_max.
    Uses a Gaussian kernel centred at the score.
    This enables KL divergence to compare predictions across scales.
    """

def scale_consistency_loss(scale_scores, global_score,
                            num_bins=20, lambda_kl=0.1):
    """
    Penalise disagreement between per-scale scores and the global score.
    KL divergence is computed between each scale's distribution and the
    global distribution, then averaged across scales and weighted by lambda_kl.
    """
```

### Combined loss
```python
L_total = L_primary + lambda_kl * L_kl
```

### Implementation note
`nn.KLDivLoss(reduction='batchmean')` requires **log-probabilities** as input.
Use `torch.log(distribution + 1e-8)` to avoid `log(0)` numerical errors.

**Exit criteria:** Loss decreases to near-zero on a 1-image 50-step overfit.
KL term is non-zero and changing during training.

---

## Phase 4 — Training Loop (`train.py`)

**Goal:** Full training with frozen→unfrozen stages, mixed precision, logging.

### Two-stage training
```
Stage 1 (Epochs 1–5):
  - Freeze transformer encoder weights
  - Train: patch_embed + pos_enc + prediction_heads only
  - LR = 1e-3
  - Allows CT-specific components to initialise before touching pretrained weights

Stage 2 (Epochs 6–50):
  - Unfreeze all weights
  - LR = 1e-4, cosine annealing scheduler, LR_min = 1e-6
```

### Mixed precision (mandatory)
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(patches, coords)
    loss   = criterion(output, target)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### Early stopping
Patience = 10 epochs watching validation aggregate score.
Save best checkpoint whenever aggregate score improves.

### Checkpoint format
```python
{
    'epoch':                int,
    'model_state_dict':     dict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'best_aggregate':       float,
    'config': {
        'scales':     SCALES,
        'lambda_kl':  LAMBDA_KL,
        'batch_size': BATCH_SIZE,
        'seed':       SEED
    }
}
```
Save to `./results/best_model.pth`.

### Console log per epoch
```
Epoch 12/50 | LR: 8.9e-05
  Train — Loss: 0.2341
  Val   — Loss: 0.2189 | PLCC: 0.882 | SROCC: 0.873 | KROCC: 0.691 | Agg: 2.447
  → New best! Saving checkpoint.
```
All epoch rows appended to `./results/training_log.csv`.

**Exit criteria:** 10 epochs complete, no OOM. Val aggregate > 1.5 (above random).

---

## Phase 5 — Evaluation (`evaluate.py`)

**Goal:** Test set metrics matching Lee et al. 2025 Table 3 format.

### Metric computation
```python
from scipy.stats import pearsonr, spearmanr, kendalltau

def compute_metrics(predictions: list, targets: list) -> dict:
    """
    Compute PLCC, SROCC, KROCC, and aggregate score.
    All values should be in [0, 4] — do not normalise before calling this.
    """
    plcc,  _ = pearsonr(predictions, targets)
    srocc, _ = spearmanr(predictions, targets)
    krocc, _ = kendalltau(predictions, targets)
    return {
        'PLCC':      round(plcc,  4),
        'SROCC':     round(srocc, 4),
        'KROCC':     round(krocc, 4),
        'Aggregate': round(abs(plcc) + abs(srocc) + abs(krocc), 4)
    }
```

### Output (save as `./results/test_results.csv`)

| Model | Aggregate | PLCC | SROCC | KROCC | Source |
|---|---|---|---|---|---|
| agaldran | 2.7427 | 0.9491 | 0.9495 | 0.8440 | Lee et al. 2025 |
| RPI_AXIS | 2.6843 | 0.9434 | 0.9414 | 0.7995 | Lee et al. 2025 |
| CHILL@UK | 2.6719 | 0.9402 | 0.9387 | 0.7930 | Lee et al. 2025 |
| FeatureNet | 2.6550 | 0.9362 | 0.9338 | 0.7851 | Lee et al. 2025 |
| Team Epoch | 2.6202 | 0.9278 | 0.9232 | 0.7691 | Lee et al. 2025 |
| gabybaldeon | 2.5671 | 0.9143 | 0.9096 | 0.7432 | Lee et al. 2025 |
| SNR baseline | 2.4026 | 0.8226 | 0.8748 | 0.7052 | Lee et al. 2025 |
| BRISQUE | 2.1219 | 0.7500 | 0.7863 | 0.5856 | Lee et al. 2025 |
| **CT-MUSIQ (ours)** | **TBD** | **TBD** | **TBD** | **TBD** | This work |

**Exit criteria:** Metrics on all 100 test images computed and saved. Numbers real.

---

## Phase 6 — Ablation Study (`ablation.py`)

**Goal:** Isolate the contribution of each design decision.

### Configurations

| Config | Scales | KL Loss | λ | What it tests |
|---|---|---|---|---|
| A1 | 1 × 224 | No | — | Single-scale baseline |
| A2 | 2 × [224, 384] | No | — | Does multi-scale help? |
| A3 | 2 × [224, 384] | Yes | 0.05 | KL at low weight |
| A4 | 2 × [224, 384] | Yes | 0.10 | KL at intended weight |
| A5 | 2 × [224, 384] | Yes | 0.20 | Does over-strong KL hurt? |
| A6 | 3 × [224, 384, 512] | Yes | 0.10 | 3 scales (run only if VRAM allows) |

All configs: same seed (42), same data, 30 epochs each (faster ablation cycle).
Save each config's best validation and final test metrics to `./results/ablation_results.csv`.

### Thesis narrative
- **A1 → A2:** Multi-scale extraction improves over single-scale.
- **A2 → A4:** KL consistency loss adds further improvement.
- **A3/A4/A5:** Sensitivity to λ — this is your hyperparameter analysis.
- **A6:** Hardware limit test — either a 3rd scale helps more, or documents OOM.

---

## Phase 7 — Figures for Thesis

Save all figures to `./results/figures/` at 300dpi.

| Filename | Content |
|---|---|
| `architecture.png` | Input pyramid → patches → hash enc → MSTE → heads |
| `training_curves.png` | Loss + PLCC vs epoch (train & val) for best config |
| `scatter_plot.png` | Predicted vs ground-truth scores on 100 test images |
| `score_distribution.png` | Histogram of predicted vs true scores |
| `ablation_bar.png` | Aggregate score per config A1–A5 |
| `failure_cases.png` | 5 images with highest prediction error |

---

## Thesis Argument

> *CT-MUSIQ adapts the MUSIQ multi-scale Transformer for brain CT image quality
> assessment by applying brain-specific CT windowing, hash-based spatial
> positional encoding across resolution scales, and a KL scale-consistency
> regularisation loss. Our ablation study demonstrates that each component
> contributes measurable improvement, yielding a competitive no-reference IQA
> model that approaches full-reference metric performance without requiring
> clean reference images.*

---

## Known Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| OOM with batch size 4 | Medium | Drop to batch size 2 + gradient accumulation steps=2 |
| OOM with 3 scales (A6) | High | Document as finding, stay at 2 scales |
| 16-bit TIFF loading errors | Medium | Check dtype, normalise raw values before windowing |
| train.json key format differs | Low | Print first 5 entries in sanity check, adjust parsing |
| KL loss causes unstable training | Medium | Warm up with λ=0.0 for 5 epochs, then enable |
| Overfitting on 700 training images | Medium | Augmentation + dropout + early stopping |
| Results below top challenge teams | Expected | Frame as competitive NR model; ablation proves contribution |

---

## Definition of Done

- [ ] All 5+ ablation configs trained and evaluated
- [ ] `./results/test_results.csv` contains real test metrics
- [ ] All 6 thesis figures saved as 300dpi PNGs
- [ ] `model.py` is clean, commented, explainable line by line
- [ ] Best checkpoint saved and reloadable from scratch
- [ ] README explains full reproduction from a clean venv
