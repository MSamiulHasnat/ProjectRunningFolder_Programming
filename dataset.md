# Dataset Pipeline (`dataset.py`) - Research Deep Dive

This document provides an exhaustive, section-by-section breakdown of the `dataset.py` file in the CT-MUSIQ project. It is intended for research personnel to understand the design decisions, mathematical justifications, code logic, alternatives, and limitations of the data pipeline.

---

## 1. Overview and Core Philosophy

The `dataset.py` file is responsible for ingesting raw standard Low-Dose CT slices and preparing them dynamically for standard Vision Transformers (ViTs) that expect multi-scale inputs. 

Usually, Vision Transformers (like ViT-B) expect fixed-resolution RGB inputs (e.g., $224 \times 224 \times 3$). CT images differ fundamentally: they are single-channel (grayscale), have variable underlying features, and clinical artifacts exist at both global (macro anatomial) and structural (micro noise/blur) scales.

The dataset pipeline solves this by dynamically taking a $512 \times 512$ image and:
1. Creating a **pyramid** of different resolutions.
2. Chopping them into **patches**.
3. Replicating the channel to mock **RGB data**.
4. Saving structural coordinates to use in the **Hash Positional Encoding** step later.

---

## 2. Exhaustive Code Breakdown

### A. Imports Block
```python
import os
import json
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import config
```
*   `os` and `json`: Handles system file paths and parsing the ground truth `train.json` dictionary.
*   `numpy` and `PIL.Image`: Used for intermediate matrix manipulations. PIL handles the bicubic resizing elegantly before converting back to numpy matrices.
*   `torch`, `Dataset`, `DataLoader`: Core PyTorch dataset API definitions. By inheriting from `torch.utils.data.Dataset`, we can plug this generic class natively into PyTorch standard training loops.
*   `torchvision.transforms.functional as TF`: We use the functional API over standard Transforms because it provides granular lower-level explicit control over bounding boxes without relying on opaque pipeline sequences.
*   `config`: Centralizes our variables to prevent magic numbers in code.

### B. The `LDCTDataset` Initialization (`__init__`)
```python
def __init__(self, data_dir, label_file, split, scales, patch_size=32, augment=False):
```
**Detailed Logic:**
1.  **Split Management:** The class is instantiated dynamically with a `split` string (`'train'`, `'val'`, `'test'`). It directly routes to `config.TRAIN_RANGE` (e.g., indices `0` to `699`).
2.  **Deterministic Allocation:** Instead of doing a `train_test_split` randomized shuffle, it uses fixed strict index ranges (like `0000-0699` is `train`). *Research Insight:* This guarantees that reproducing results across baselines uses the exact identical statistical distribution limits. 
3.  **JSON Mapping & Error Handling:** It iterates over the assigned index limits (e.g., `range(0, 700)`). It formats the index to string literals (e.g., `idx:04d -> "0042.tif"`). If the formatted key exists in the loaded `train.json`, it saves the score in `self.labels` and assigns the filename string to `self.image_ids` array. Warning prints act as strict fault tolerances.
4.  **Grid Precomputation Math:** To prevent recalculating patch grids every loop, it operates on the `scales` list (default `[224, 384]`). 
    *   $224 / 32 = 7$ grid divisions.
    *   $384 / 32 = 12$ grid divisions.
    *   It dynamically maps `self.num_patches = 7*7 + 12*12 = 193`.
    *   This logic makes adding another scale (like $512 \to 16\times16$) fully automated without writing boilerplate.

### C. Image Loading (`load_image`)
```python
def load_image(self, image_id: str) -> np.ndarray:
```
**Detailed Logic:**
1.  Constructs the path using `config.IMAGE_FILENAME_FORMAT` to find `dataset/image/XXXX.tif`.
2.  Throws a `FileNotFoundError` explicitly if missing, halting bad state execution.
3.  Reads the file using `Image.open`, dumping to a `float32` NumPy array. 
4.  **`np.clip(pixel_array, 0.0, 1.0)`**: A deliberate safety lock. Since standard CT files might float slightly off-bounds from artifacts, clipping forces a strict zero-to-one floor/ceiling.
**Alternative Approaches (The HU Windowing Debate):** Most raw clinical medical datasets are Dicom files (`.dcm`) with huge ranges ($-1000$ to $+3000$). Normally, we apply a "brain window" (center $40$, width $80$). However, since the MICCAI challenge mapped everything to standard $32\text{-bit}$ floats inside `[0,1]`, applying windowing *again* is destructive—it destroys micro-pixel variations that represent exact speckle noises.

### D. Data Augmentation (`apply_augmentation`)
```python
def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
```
**Detailed Logic:**
1.  Temporarily converts the Float32 `[0, 1]` matrix to an integer `uint8` `[0, 255]` PIL image. Torchvision functional API works significantly faster and safer on standard image byte layouts.
2.  **Horizontal Flip:** Applies an exact `50%` randomized binomial flip (`np.random.random() < 0.5`). Brain symmetry permits this horizontally (sagittal context), but *not* vertically.
3.  **Random Crop Geometry:** Computes a `crop_fraction` randomly sampled between `[0.85, 1.0]`. 
    *   It multiplies this percentage against image `Height` and `Width` to get new dimensions.
    *   It randomizes the starting top-left `top` and `left` anchors to ensure it does not crop outside the array limits.
    *   Crops it, then crucially **resizes it back** to the original height and width, keeping dimensionality safe for the ViT.
4.  **Contrast/Brightness Jitter:** Modifies coefficients randomly between offsets of `[0.95, 1.05]`.
**Research Insight:** Why so subtle? In medical IQA, harsh augmentations are fatal. If you heavily blur or jitter a high-quality CT image, its ground truth label contradicts the heavily mutated visual state. An alternative approach would be purely synthetic physics-based poisson noise injection where labels are algorithmically degraded.

### E. Multi-Scale Pyramid Generation (`build_multi_scale_pyramid`)
```python
def build_multi_scale_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
```
**Detailed Logic:**
Iterates dynamically over every size provided in `self.scales`. Using `Image.BICUBIC`, it forces the full resolution input down to the exact multi-scale thresholds.
**Why it's needed:** MUSIQ assumes image quality is perceived locally *and* globally. Supplying heavily compressed variants ($224 \times 224$ scaled) lets patches see whole brain regions. Milder compression ($384 \times 384$) lets patches see higher-rez structural noise. 
**Alternative:** Using `Nearest Exact` neighbor. Bicubic interpolation acts as a low-pass filter (slight blurring). Nearest neighbor preserves stark noise profiles better, but usually has inferior gradients during backprop.

### F. Patch Extraction & Coordinate Masking (`extract_patches`)
```python
def extract_patches(self, image: np.ndarray, scale_idx: int) -> Tuple[np.ndarray, np.ndarray]:
```
**Detailed Logic:**
1.  We define `grid_size = scale_size // self.patch_size` (e.g. $224 / 32 = 7$).
2.  We allocate two blank memory grids:
    *   `patches` = floating zeros bounding `[num_patches, 32, 32]`
    *   `coords` = integer zeros bounding `[num_patches, 3]`
3.  Two standard nested loops (`for row in... for col in...`) act as sliding windows. 
4.  Calculates explicit array split limits: `r_start`, `r_end`, `c_start`, `c_end`. 
5.  Populates the `patch_idx` linearly, copying the visual slice.
6.  **Crucial step:** It populates `coords[patch_idx] = [scale_idx, row, col]`.
**Research Insight:** Standard transformers flatten 1D positional tokens. Because CT-MUSIQ mixes up $224$ multi-patches and $384$ multi-patches into one giant 1D stream block, 1D positional encodings completely fail. By stamping the exact scale level (`scale_idx`) and X/Y position (`row, col`), the model utilizes a spatial 2D-hashed positional encoder downstream. 

### G. Grayscale to RGB Mocking (`replicate_to_rgb`)
```python
def replicate_to_rgb(self, patches: np.ndarray) -> np.ndarray:
```
**Detailed Logic:** Uses `np.stack([patches, patches, patches], axis=1)`. 
Instead of modifying data artificially, we duplicate the exact $1\text{-channel}$ grayscale into $3\text{-channels}$.
**Why?** The upcoming ViT parameter weights are rigorously pre-trained on ImageNet RGB data. Rewriting a vision transformer completely from scratch on 1,000 CT slices overfits aggressively. Tricking the ViT into thinking it is seeing RGB imagery by blasting grey into Red, Green, and Blue channels allows it to flawlessly piggyback on ImageNet's low-level optical edge detection priors. 

### H. The Aggregation Flow (`__getitem__`)
```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
```
This is the master pipeline sequence invoked during iterator execution per step:
1. Looks up the `image_id` and corresponding radiologist score.
2. Calls `load_image`.
3. Calls `apply_augmentation` if explicitly asked. 
4. Calls `build_multi_scale_pyramid`.
5. Loops across the returned pyramid arrays and triggers `extract_patches` iteratively, appending matrix chunks to `all_patches` / `all_coords` arrays. 
6. `np.concatenate(all_patches, axis=0)` flattens all nested hierarchy into one monolithic sequential block of $193$ tokens. 
7. Repositions axes into PyTorch format yielding `[193, 3, 32, 32]`. 
8. Emits a uniform PyTorch dictionary for the `DataLoader`.

---

## 3. Data Loaders & Collation Limits

### `create_dataloaders`
Constructs the actual iterable loops. 
*   **`drop_last=True`:** On the training loader, if a batch is uneven (e.g. $700$ mod BatchSize equals remainder), the remainder generates irregular tensor shapes. Dropping the remainder ensures perfect Batch Normalization limits.
*   **`num_workers=0` vs `num_workers=N`:** On Windows environments natively, PyTorch's multithreading triggers `RuntimeError: Broken pipe` due to Python Pickler limitations when moving classes between child processes. Using `0` keeps the operations on the native main thread safely.
*   **`pin_memory=True`:** Locks the dataset array references directly into Non-Paged system ram slots, accelerating NVIDIA CUDA offloading exponentially.

### `custom_collate_fn`
The DataLoader wraps around `custom_collate_fn`. Usually, PyTorch natively stacks uniform numeric outputs correctly. However, dealing with list comprehensions of dictionaries requires mapping logic. A neat loop comprehension `torch.stack([item['patches'] for item in batch])` explicitly gathers out the stacked shapes `[B, 193, 3, 32, 32]`, maintaining dimension uniformity explicitly.

---

## 4. Hardware Limitations & VRAM Design

**Why fix scales to [224, 384] and batch size to 4?**
Standard CT imagery is 512. Passing a large image via standard patches into a Vision Transformer forces quadratic $O(N^2)$ memory limits. 
In order to survive strictly within the thesis hard limits of an **RTX 3060 6GB VRAM**:
*   A 384 maximum cap limits token length strictly to 193 patches.
*   Using PyTorch Automatic Mixed Precision (AMP) logic downstreams halves precision byte thresholds.
*   If later configurations possess 24GB+ VRAM (like an RTX 3090/4090), expanding the parameters simply requires editing `config.py` metrics `SCALES=[224, 384, 512]`. All code dynamically adapts.