"""
dataset.py — CT-MUSIQ Multi-Scale Dataset Pipeline
===================================================

PyTorch Dataset class for loading LDCTIQAC 2023 brain CT images.
Handles multi-scale pyramid construction, patch extraction, and
hash-based coordinate tracking for positional encoding.

Key Design Decisions:
  1. Images are already normalized to [0, 1] — no windowing needed
  2. Grayscale replicated to 3 channels for pretrained ViT compatibility
  3. Two resolution scales: [224, 384] (safe for 6GB VRAM)
  4. 32x32 patches: 49 (224) + 144 (384) = 193 total tokens per image

Author: M Samiul Hasnat, Sichuan University
Project: CT-MUSIQ — Undergraduate Thesis
"""

import os
import json
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# Import project configuration
import config


class LDCTDataset(Dataset):
    """
    PyTorch Dataset for LDCTIQAC 2023 brain CT images.
    
    Loads CT images, constructs multi-scale pyramid, extracts patches,
    and returns patch tensors with spatial coordinates for hash-based
    positional encoding.
    
    Args:
        data_dir: Path to directory containing .tif images (./dataset/image/)
        label_file: Path to train.json with quality scores
        split: One of 'train', 'val', or 'test'
        scales: List of target image sizes, e.g. [224, 384]
        patch_size: Side length of square patches (default: 32)
        augment: Whether to apply data augmentation (train split only)
        
    Returns (per __getitem__):
        dict with keys:
            'patches':  Tensor [N, 3, 32, 32] — all patches concatenated
            'coords':   Tensor [N, 3] — (scale_idx, row_idx, col_idx) per patch
            'score':    Tensor scalar — radiologist quality score [0, 4]
            'image_id': str — e.g. '0042'
    """
    
    def __init__(
        self,
        data_dir: str,
        label_file: str,
        split: str,
        scales: List[int],
        patch_size: int = 32,
        augment: bool = False
    ):
        super().__init__()
        
        # Store configuration
        self.data_dir = data_dir
        self.scales = scales
        self.patch_size = patch_size
        self.augment = augment and (split == 'train')  # Only augment training data
        
        # Validate split argument
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        self.split = split
        
        # Determine index range for this split
        if split == 'train':
            self.idx_range = config.TRAIN_RANGE
        elif split == 'val':
            self.idx_range = config.VAL_RANGE
        else:  # test
            self.idx_range = config.TEST_RANGE
        
        # Load labels from JSON
        with open(label_file, 'r') as f:
            all_labels = json.load(f)
        
        # Filter labels to only include images in this split
        self.labels = {}
        self.image_ids = []
        
        for idx in range(self.idx_range[0], self.idx_range[1] + 1):
            # Build the key format (e.g., "0042.tif")
            key = config.LABEL_KEY_FORMAT.format(idx=idx)
            
            if key in all_labels:
                self.labels[key] = all_labels[key]
                # Store numeric ID without extension for easy reference
                self.image_ids.append(key.replace('.tif', '').replace('.tiff', ''))
            else:
                print(f"Warning: Missing label for {key}")
        
        # Precompute patch grid dimensions for each scale
        # This is used to generate coordinates during __getitem__
        self.patch_grids = []
        for scale in scales:
            grid_size = scale // patch_size
            self.patch_grids.append(grid_size)
        
        # Total number of patches across all scales
        # 224/32 = 7 -> 7×7 = 49 patches
        # 384/32 = 12 -> 12×12 = 144 patches
        # Total: 49 + 144 = 193 patches
        self.num_patches = sum(g * g for g in self.patch_grids)
        
        # Sample weights for hard mining (image_id -> weight)
        self.sample_weights = {}
        
        print(f"Initialized {split} dataset:")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Scales: {scales} -> {self.num_patches} total patches")
        print(f"  Augmentation: {self.augment}")
    
    def __len__(self) -> int:
        """Return the number of images in this split."""
        return len(self.image_ids)
    
    def load_image(self, image_id: str) -> np.ndarray:
        """
        Load a single CT image from disk.
        
        The LDCTIQAC 2023 images are already normalized to [0, 1] range
        as float32 TIFF files. No CT windowing is needed.
        
        Args:
            image_id: Numeric ID string, e.g. '0042'
            
        Returns:
            numpy array of shape (H, W) with dtype float32, values in [0, 1]
        """
        # Build full file path
        filename = config.IMAGE_FILENAME_FORMAT.format(idx=int(image_id))
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image not found: {filepath}")
        
        # Load with PIL and convert to numpy
        img = Image.open(filepath)
        pixel_array = np.array(img, dtype=np.float32)
        
        # Images are already normalized to [0, 1] — no windowing needed
        # Just ensure values are clipped to valid range
        pixel_array = np.clip(pixel_array, 0.0, 1.0)
        
        return pixel_array
    
    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply training augmentation to a single CT image.

        Augmentations are kept mild because CT noise is diagnostically meaningful.
        Operations are done in float32 to preserve precision (no uint8 quantization).

          - Random horizontal flip (p=0.5)
          - Random vertical flip (p=0.5)
          - Random rotation ±7 degrees
          - Random crop retaining 85-100% of area, resize back
          - Brightness/contrast jitter ±0.05 only
          - Mild additive Gaussian noise

        Args:
            image: numpy array of shape (H, W), values in [0, 1]

        Returns:
            Augmented image with same shape and dtype
        """
        h, w = image.shape

        # Random horizontal flip (p=0.5) — in numpy, preserves float32 precision
        if np.random.random() < 0.5:
            image = image[:, ::-1].copy()

        # Random vertical flip (p=0.5) — brain CT axial slices are
        # roughly symmetric top/bottom in terms of quality
        if np.random.random() < 0.5:
            image = image[::-1, :].copy()

        # Small random rotation (±7°) — use PIL for interpolation quality
        # but keep float32 by converting via uint16 to minimize quantization
        if np.random.random() < 0.5:
            angle = float(np.random.uniform(-7.0, 7.0))
            # Use float-mode PIL (mode='F') to avoid 8-bit quantization
            img_pil = Image.fromarray(image, mode='F')
            img_pil = TF.rotate(
                img_pil,
                angle=angle,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0.0
            )
            image = np.array(img_pil, dtype=np.float32)

        # Random crop and resize back (85-100% of area)
        if np.random.random() < 0.5:
            crop_fraction = np.random.uniform(0.85, 1.0)
            crop_h = int(h * crop_fraction)
            crop_w = int(w * crop_fraction)
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
            image = image[top:top + crop_h, left:left + crop_w]
            img_pil = Image.fromarray(image, mode='F')
            img_pil = img_pil.resize((w, h), Image.BICUBIC)
            image = np.array(img_pil, dtype=np.float32)

        # Mild brightness/contrast jitter (±0.05) — in float32 directly
        brightness_delta = np.random.uniform(-0.05, 0.05)
        contrast_factor = 1.0 + np.random.uniform(-0.05, 0.05)
        mean = image.mean()
        image = (image - mean) * contrast_factor + mean + brightness_delta

        # Mild Gaussian noise to simulate scanner noise variability
        noise_std = float(np.random.uniform(0.005, 0.02))
        image = image + np.random.normal(0.0, noise_std, size=image.shape).astype(np.float32)

        return np.clip(image, 0.0, 1.0)
    
    def build_multi_scale_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Resize the input image to multiple target scales.
        
        Uses bicubic interpolation for smooth resizing. Each scale produces
        a square image that will be divided into non-overlapping patches.
        
        Args:
            image: numpy array of shape (H, W), values in [0, 1]
            
        Returns:
            List of resized images, one per scale
        """
        # Convert to PIL for high-quality resizing
        img_pil = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        
        pyramid = []
        for scale in self.scales:
            # Resize to target scale using bicubic interpolation
            resized = img_pil.resize((scale, scale), Image.BICUBIC)
            # Convert back to numpy float32
            resized_array = np.array(resized, dtype=np.float32) / 255.0
            pyramid.append(resized_array)
        
        return pyramid
    
    def extract_patches(self, image: np.ndarray, scale_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract non-overlapping patches from a single-scale image.
        
        Divides the image into a grid of patch_size × patch_size patches.
        Returns patches and their spatial coordinates for hash-based
        positional encoding.
        
        Args:
            image: numpy array of shape (S, S) where S is the scale size
            scale_idx: Index of this scale in self.scales (for coordinates)
            
        Returns:
            patches: numpy array of shape [N, patch_size, patch_size]
            coords: numpy array of shape [N, 3] with (scale_idx, row_idx, col_idx)
        """
        scale_size = image.shape[0]
        grid_size = scale_size // self.patch_size
        
        # Initialize output arrays
        num_patches = grid_size * grid_size
        patches = np.zeros((num_patches, self.patch_size, self.patch_size), dtype=np.float32)
        coords = np.zeros((num_patches, 3), dtype=np.int64)
        
        patch_idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                # Extract patch boundaries
                r_start = row * self.patch_size
                r_end = r_start + self.patch_size
                c_start = col * self.patch_size
                c_end = c_start + self.patch_size
                
                # Extract patch
                patches[patch_idx] = image[r_start:r_end, c_start:c_end]
                
                # Store coordinates: (scale_index, row_index, col_index)
                coords[patch_idx] = [scale_idx, row, col]
                
                patch_idx += 1
        
        return patches, coords
    
    def replicate_to_rgb(self, patches: np.ndarray) -> np.ndarray:
        """
        Replicate single-channel patches to 3 channels.
        
        This allows using pretrained ViT patch embedding weights that expect
        RGB input. The pretrained weights will see the same information in
        all three channels.
        
        Args:
            patches: numpy array of shape [N, H, W] (grayscale)
            
        Returns:
            numpy array of shape [N, 3, H, W] (RGB replica)
        """
        # Stack the same patch 3 times along a new axis
        patches_rgb = np.stack([patches, patches, patches], axis=1)
        return patches_rgb
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process a single image for the model.
        
        Pipeline:
          1. Load grayscale CT image (already normalized to [0, 1])
          2. Apply augmentation if training
          3. Build multi-scale pyramid
          4. Extract patches from each scale
          5. Replicate grayscale to RGB
          6. Concatenate all patches and coordinates
          7. Return as tensors
        
        Args:
            idx: Index into self.image_ids
            
        Returns:
            Dictionary with patches, coords, score, and image_id
        """
        # Get image ID and label
        image_id = self.image_ids[idx]
        key = config.LABEL_KEY_FORMAT.format(idx=int(image_id))
        score = self.labels[key]
        
        # Load image (already normalized to [0, 1])
        image = self.load_image(image_id)
        
        # Apply augmentation if training
        if self.augment:
            image = self.apply_augmentation(image)
        
        # Build multi-scale pyramid
        pyramid = self.build_multi_scale_pyramid(image)
        
        # Extract patches from each scale
        all_patches = []
        all_coords = []
        
        for scale_idx, scale_image in enumerate(pyramid):
            patches, coords = self.extract_patches(scale_image, scale_idx)
            all_patches.append(patches)
            all_coords.append(coords)
        
        # Concatenate patches from all scales
        # Shape: [total_patches, patch_size, patch_size]
        all_patches = np.concatenate(all_patches, axis=0)
        all_coords = np.concatenate(all_coords, axis=0)
        
        # Replicate grayscale to RGB (3 channels)
        # Shape: [total_patches, 3, patch_size, patch_size]
        all_patches = self.replicate_to_rgb(all_patches)
        
        # Convert to PyTorch tensors
        patches_tensor = torch.from_numpy(all_patches).float()
        coords_tensor = torch.from_numpy(all_coords).long()
        score_tensor = torch.tensor(score, dtype=torch.float32)
        
        # Get sample weight (default to 1.0 if not set)
        weight = self.sample_weights.get(image_id, 1.0)
        weight_tensor = torch.tensor(weight, dtype=torch.float32)
        
        return {
            'patches': patches_tensor,   # [N, 3, 32, 32]
            'coords': coords_tensor,     # [N, 3]
            'score': score_tensor,       # scalar
            'weight': weight_tensor,     # scalar
            'image_id': image_id         # str
        }


def create_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test splits.
    
    Args:
        batch_size: Number of images per batch (default: 4 for 6GB VRAM)
        num_workers: Number of data loading workers (0 for Windows compatibility)
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = LDCTDataset(
        data_dir=config.DATA_DIR,
        label_file=config.LABEL_FILE,
        split='train',
        scales=config.SCALES,
        patch_size=config.PATCH_SIZE,
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = LDCTDataset(
        data_dir=config.DATA_DIR,
        label_file=config.LABEL_FILE,
        split='val',
        scales=config.SCALES,
        patch_size=config.PATCH_SIZE,
        augment=False  # No augmentation for validation
    )
    
    test_dataset = LDCTDataset(
        data_dir=config.DATA_DIR,
        label_file=config.LABEL_FILE,
        split='test',
        scales=config.SCALES,
        patch_size=config.PATCH_SIZE,
        augment=False  # No augmentation for testing
    )
    
    # Create DataLoaders
    # Note: num_workers=0 is recommended for Windows to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for consistent batch norm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for testing
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def custom_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collation function for variable-length patch sequences.
    
    Standard collation assumes all items have the same shape, but our
    patches are already fixed-size per image (193 patches). This function
    stacks them into batched tensors.
    
    Args:
        batch: List of dictionaries from __getitem__
        
    Returns:
        Batched dictionary with stacked tensors
    """
    # Stack all patches: [B, N, 3, 32, 32]
    patches = torch.stack([item['patches'] for item in batch])
    
    # Stack all coords: [B, N, 3]
    coords = torch.stack([item['coords'] for item in batch])
    
    # Stack scores: [B]
    scores = torch.stack([item['score'] for item in batch])
    
    # Stack weights: [B]
    weights = torch.stack([item['weight'] for item in batch])
    
    # Collect image IDs as list
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'patches': patches,
        'coords': coords,
        'score': scores,
        'weight': weights,
        'image_id': image_ids
    }


# =============================================================================
# TESTING / VERIFICATION
# =============================================================================

if __name__ == "__main__":
    """
    Quick test to verify the dataset pipeline works correctly.
    Run: python dataset.py
    """
    print("="*60)
    print("CT-MUSIQ Dataset Pipeline Test")
    print("="*60)
    
    # Create a small dataset for testing
    print("\nCreating train dataset...")
    train_dataset = LDCTDataset(
        data_dir=config.DATA_DIR,
        label_file=config.LABEL_FILE,
        split='train',
        scales=config.SCALES,
        patch_size=config.PATCH_SIZE,
        augment=True
    )
    
    print(f"\nDataset length: {len(train_dataset)}")
    
    # Load one sample
    print("\nLoading sample at index 0...")
    sample = train_dataset[0]
    
    print(f"\nSample contents:")
    print(f"  patches shape:  {sample['patches'].shape}")
    print(f"  coords shape:   {sample['coords'].shape}")
    print(f"  score:          {sample['score']:.2f}")
    print(f"  image_id:       {sample['image_id']}")
    
    # Verify patch dimensions
    expected_patches = sum((s // config.PATCH_SIZE) ** 2 for s in config.SCALES)
    actual_patches = sample['patches'].shape[0]
    
    print(f"\nPatch count verification:")
    print(f"  Expected: {expected_patches} (49 from 224 + 144 from 384)")
    print(f"  Actual:   {actual_patches}")
    
    if actual_patches == expected_patches:
        print(f"  ✓ Patch count matches!")
    else:
        print(f"  ✗ Patch count mismatch!")
    
    # Test DataLoader
    print("\nTesting DataLoader with batch_size=2...")
    loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    batch = next(iter(loader))
    print(f"\nBatch contents:")
    print(f"  patches shape:  {batch['patches'].shape}")  # [2, 193, 3, 32, 32]
    print(f"  coords shape:   {batch['coords'].shape}")   # [2, 193, 3]
    print(f"  scores shape:   {batch['score'].shape}")    # [2]
    print(f"  image_ids:      {batch['image_id']}")
    
    print("\n✓ Dataset pipeline test complete!")
