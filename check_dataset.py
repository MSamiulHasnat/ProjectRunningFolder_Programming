"""
check_dataset.py — CT-MUSIQ Dataset Sanity Check
=================================================

Self-contained script to verify the LDCTIQAC 2023 dataset is correctly set up.
Run this before any training to catch data issues early.

Checks performed:
  1. Load train.json and print first 5 entries (confirm key format)
  2. Confirm all 1,000 keys exist
  3. Print score statistics: mean, std, min, max per split
  4. Load one image from each split, apply CT windowing, print metadata
  5. Flag any TIFF files that are 16-bit (needs special handling)

Usage:
  python check_dataset.py

Author: M Samiul Hasnat, Sichuan University
Project: CT-MUSIQ — Undergraduate Thesis
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from collections import defaultdict

# Import project configuration
# We add the project root to sys.path so config.py can be imported
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    import config
except ImportError:
    print("ERROR: Cannot import config.py. Make sure it exists in the project root.")
    sys.exit(1)


def load_labels(label_file: str) -> dict:
    """
    Load the train.json file containing image quality scores.
    
    Args:
        label_file: Path to the JSON file mapping image IDs to quality scores.
        
    Returns:
        Dictionary mapping image ID strings to float quality scores.
        
    Raises:
        FileNotFoundError: If the label file doesn't exist.
        json.JSONDecodeError: If the file isn't valid JSON.
    """
    print(f"Loading labels from: {label_file}")
    
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
    
    with open(label_file, 'r') as f:
        labels = json.load(f)
    
    print(f"  ✓ Loaded {len(labels)} label entries")
    return labels


def check_key_format(labels: dict, num_to_show: int = 5) -> None:
    """
    Print the first N entries to verify the key format matches expectations.
    
    The plan.md originally assumed 3-digit keys ("000"), but the actual dataset
    may use 4-digit keys with extension ("0000.tif"). This function reveals
    the actual format.
    
    Args:
        labels: Dictionary of image labels.
        num_to_show: Number of entries to display.
    """
    print(f"\n{'='*60}")
    print(f"CHECK 1: Key Format Verification (first {num_to_show} entries)")
    print(f"{'='*60}")
    
    # Sort keys to show consistent ordering
    sorted_keys = sorted(labels.keys())[:num_to_show]
    
    print(f"  Expected format: '{config.LABEL_KEY_FORMAT.format(idx=0)}'")
    print(f"  Actual entries:")
    
    for key in sorted_keys:
        value = labels[key]
        print(f"    '{key}' → {value}")
    
    # Check if keys match expected format
    sample_key = sorted_keys[0] if sorted_keys else ""
    expected_sample = config.LABEL_KEY_FORMAT.format(idx=0)
    
    if sample_key == expected_sample:
        print(f"  ✓ Key format matches config: '{expected_sample}'")
    else:
        print(f"  ⚠ Key format differs from config!")
        print(f"    Config expects: '{expected_sample}'")
        print(f"    Actual format:  '{sample_key}'")
        print(f"    → You may need to update config.py LABEL_KEY_FORMAT")


def check_all_keys_exist(labels: dict) -> None:
    """
    Verify that all 1,000 expected image keys are present in the labels.
    
    Args:
        labels: Dictionary of image labels.
    """
    print(f"\n{'='*60}")
    print(f"CHECK 2: Completeness — All {config.TOTAL_IMAGES} Keys Present")
    print(f"{'='*60}")
    
    # Generate expected keys based on config format
    expected_keys = set()
    for idx in range(config.TOTAL_IMAGES):
        expected_keys.add(config.LABEL_KEY_FORMAT.format(idx=idx))
    
    actual_keys = set(labels.keys())
    
    # Find missing and extra keys
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    
    if not missing and not extra:
        print(f"  ✓ All {config.TOTAL_IMAGES} expected keys are present")
        print(f"  ✓ No unexpected keys found")
    else:
        if missing:
            print(f"  ✗ Missing {len(missing)} keys:")
            # Show first 10 missing keys
            for key in sorted(list(missing))[:10]:
                print(f"      - '{key}'")
            if len(missing) > 10:
                print(f"      ... and {len(missing) - 10} more")
        
        if extra:
            print(f"  ⚠ Found {len(extra)} unexpected keys:")
            for key in sorted(list(extra))[:10]:
                print(f"      + '{key}'")
            if len(extra) > 10:
                print(f"      ... and {len(extra) - 10} more")


def get_split_for_index(idx: int) -> str:
    """
    Determine which data split an image index belongs to.
    
    Args:
        idx: Integer image index (0-999).
        
    Returns:
        String: 'train', 'val', or 'test'.
    """
    if config.TRAIN_RANGE[0] <= idx <= config.TRAIN_RANGE[1]:
        return 'train'
    elif config.VAL_RANGE[0] <= idx <= config.VAL_RANGE[1]:
        return 'val'
    elif config.TEST_RANGE[0] <= idx <= config.TEST_RANGE[1]:
        return 'test'
    else:
        return 'unknown'


def compute_split_statistics(labels: dict) -> None:
    """
    Compute and print score statistics (mean, std, min, max) for each split.
    
    Args:
        labels: Dictionary of image labels.
    """
    print(f"\n{'='*60}")
    print(f"CHECK 3: Score Statistics Per Split")
    print(f"{'='*60}")
    
    # Group scores by split
    split_scores = defaultdict(list)
    
    for key, score in labels.items():
        # Extract numeric index from key
        # Handle both "000" and "0000.tif" formats
        try:
            # Remove file extension if present
            idx_str = key.replace('.tif', '').replace('.tiff', '')
            idx = int(idx_str)
            split = get_split_for_index(idx)
            split_scores[split].append(score)
        except ValueError:
            print(f"  ⚠ Cannot parse index from key: '{key}'")
    
    # Print statistics for each split
    for split_name in ['train', 'val', 'test']:
        scores = split_scores.get(split_name, [])
        
        if not scores:
            print(f"\n  {split_name.upper()} split: No scores found!")
            continue
        
        scores_array = np.array(scores)
        
        print(f"\n  {split_name.upper()} split ({len(scores)} images):")
        print(f"    Mean:  {np.mean(scores_array):.4f}")
        print(f"    Std:   {np.std(scores_array):.4f}")
        print(f"    Min:   {np.min(scores_array):.4f}")
        print(f"    Max:   {np.max(scores_array):.4f}")
        print(f"    Range: [{config.SCORE_MIN}, {config.SCORE_MAX}]")
        
        # Check if scores are within expected bounds
        if np.min(scores_array) < config.SCORE_MIN or np.max(scores_array) > config.SCORE_MAX:
            print(f"    ⚠ Some scores outside expected range!")
    
    # Check for 'unknown' split entries
    if 'unknown' in split_scores:
        print(f"\n  ⚠ WARNING: {len(split_scores['unknown'])} images don't belong to any split!")


def apply_ct_windowing(pixel_array: np.ndarray) -> np.ndarray:
    """
    Apply brain CT soft-tissue windowing to raw pixel values.
    
    The brain soft-tissue window (width=80, level=40) clips HU values to the
    range [0, 80] and normalises to [0, 1]. This is narrower than the
    abdominal window used in the original challenge (width=350, level=40).
    
    Args:
        pixel_array: Raw pixel values as numpy array (any numeric dtype).
        
    Returns:
        Windowed and normalised pixel values as float32 in range [0, 1].
    """
    # Calculate HU bounds from window parameters
    hu_min = config.WINDOW_LEVEL - config.WINDOW_WIDTH / 2  # 40 - 40 = 0
    hu_max = config.WINDOW_LEVEL + config.WINDOW_WIDTH / 2  # 40 + 40 = 80
    
    # Convert to float32 for processing
    pixel_float = pixel_array.astype(np.float32)
    
    # Clip values to window range
    pixel_clipped = np.clip(pixel_float, hu_min, hu_max)
    
    # Normalise to [0, 1]
    pixel_normalised = (pixel_clipped - hu_min) / (hu_max - hu_min)
    
    return pixel_normalised


def load_and_inspect_image(image_path: str, split_name: str) -> None:
    """
    Load a single image, apply CT windowing, and print detailed metadata.
    
    Args:
        image_path: Full path to the TIFF image file.
        split_name: Name of the split ('train', 'val', or 'test') for logging.
    """
    print(f"\n  Loading {split_name} sample: {os.path.basename(image_path)}")
    
    if not os.path.exists(image_path):
        print(f"    ✗ File not found: {image_path}")
        return
    
    try:
        # Open image with PIL
        img = Image.open(image_path)
        
        # Convert to numpy array
        pixel_array = np.array(img)
        
        # Print raw image properties
        print(f"    Shape:      {pixel_array.shape}")
        print(f"    Dtype:      {pixel_array.dtype}")
        print(f"    Raw range:  [{pixel_array.min()}, {pixel_array.max()}]")
        
        # Check if image is 16-bit (needs special handling)
        if pixel_array.dtype == np.uint16:
            print(f"    ⚠ 16-bit image detected — values may need rescaling before windowing")
            print(f"      Max possible value for uint16: 65535")
        elif pixel_array.dtype == np.float32 or pixel_array.dtype == np.float64:
            print(f"    ℹ Float image — assuming values are in HU or pre-normalised")
        
        # Apply CT windowing
        windowed = apply_ct_windowing(pixel_array)
        
        print(f"    Post-window range: [{windowed.min():.4f}, {windowed.max():.4f}]")
        
        # Check if windowing produced expected output range
        if windowed.min() < -0.01 or windowed.max() > 1.01:
            print(f"    ⚠ Windowed values outside [0, 1] — check pixel value interpretation")
        
    except Exception as e:
        print(f"    ✗ Error loading image: {e}")


def inspect_sample_images(labels: dict) -> None:
    """
    Load one sample image from each split and inspect its properties.
    
    Args:
        labels: Dictionary of image labels (used to verify key format).
    """
    print(f"\n{'='*60}")
    print(f"CHECK 4: Sample Image Inspection (one per split)")
    print(f"{'='*60}")
    
    # Select one image from each split
    # Use the first image in each range
    sample_indices = {
        'train': config.TRAIN_RANGE[0],  # e.g., 0
        'val': config.VAL_RANGE[0],      # e.g., 700
        'test': config.TEST_RANGE[0]     # e.g., 900
    }
    
    for split_name, idx in sample_indices.items():
        # Build image filename
        filename = config.IMAGE_FILENAME_FORMAT.format(idx=idx)
        image_path = os.path.join(config.DATA_DIR, filename)
        
        load_and_inspect_image(image_path, split_name)


def check_16bit_images(labels: dict, max_to_check: int = 100) -> None:
    """
    Scan images to identify any 16-bit TIFF files that need special handling.
    
    16-bit CT images have pixel values in range [0, 65535] instead of [0, 255].
    These need to be rescaled before CT windowing can be applied correctly.
    
    Args:
        labels: Dictionary of image labels.
        max_to_check: Maximum number of images to check (for speed).
    """
    print(f"\n{'='*60}")
    print(f"CHECK 5: 16-bit Image Detection (checking first {max_to_check} images)")
    print(f"{'='*60}")
    
    images_16bit = []
    images_checked = 0
    
    # Check a sample of images
    sorted_keys = sorted(labels.keys())[:max_to_check]
    
    for key in sorted_keys:
        # Build image path
        filename = key if key.endswith('.tif') or key.endswith('.tiff') else f"{key}{config.IMAGE_EXT}"
        image_path = os.path.join(config.DATA_DIR, filename)
        
        if not os.path.exists(image_path):
            continue
        
        try:
            img = Image.open(image_path)
            pixel_array = np.array(img)
            
            if pixel_array.dtype == np.uint16:
                images_16bit.append(filename)
            
            images_checked += 1
            
        except Exception:
            continue
    
    print(f"  Checked {images_checked} images")
    
    if images_16bit:
        print(f"  ⚠ Found {len(images_16bit)} 16-bit images:")
        for filename in images_16bit[:10]:
            print(f"      - {filename}")
        if len(images_16bit) > 10:
            print(f"      ... and {len(images_16bit) - 10} more")
        print(f"\n  → These images need uint16 → float32 conversion before windowing")
        print(f"    Recommended: pixel_float = pixel_uint16.astype(np.float32) / 65535.0")
    else:
        print(f"  ✓ No 16-bit images found in the checked sample")


def check_split_ranges() -> None:
    """
    Verify that train/val/test index ranges are non-overlapping.
    """
    print(f"\n{'='*60}")
    print(f"CHECK 6: Split Range Validation")
    print(f"{'='*60}")
    
    train_set = set(range(config.TRAIN_RANGE[0], config.TRAIN_RANGE[1] + 1))
    val_set = set(range(config.VAL_RANGE[0], config.VAL_RANGE[1] + 1))
    test_set = set(range(config.TEST_RANGE[0], config.TEST_RANGE[1] + 1))
    
    # Check for overlaps
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    all_indices = train_set | val_set | test_set
    expected_all = set(range(config.TOTAL_IMAGES))
    
    print(f"  Train: {len(train_set)} images (indices {config.TRAIN_RANGE[0]}-{config.TRAIN_RANGE[1]})")
    print(f"  Val:   {len(val_set)} images (indices {config.VAL_RANGE[0]}-{config.VAL_RANGE[1]})")
    print(f"  Test:  {len(test_set)} images (indices {config.TEST_RANGE[0]}-{config.TEST_RANGE[1]})")
    print(f"  Total: {len(all_indices)} images")
    
    if train_val_overlap:
        print(f"  ✗ Overlap between train and val: {sorted(train_val_overlap)}")
    if train_test_overlap:
        print(f"  ✗ Overlap between train and test: {sorted(train_test_overlap)}")
    if val_test_overlap:
        print(f"  ✗ Overlap between val and test: {sorted(val_test_overlap)}")
    
    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print(f"  ✓ No overlaps between splits")
    
    if all_indices == expected_all:
        print(f"  ✓ All {config.TOTAL_IMAGES} indices (0-{config.TOTAL_IMAGES-1}) are covered")
    else:
        missing = expected_all - all_indices
        extra = all_indices - expected_all
        if missing:
            print(f"  ⚠ Missing indices: {sorted(missing)[:10]}...")
        if extra:
            print(f"  ⚠ Extra indices: {sorted(extra)[:10]}...")


def main():
    """
    Main entry point for the dataset sanity check script.
    
    Runs all checks in sequence and prints a summary.
    """
    print("="*60)
    print("CT-MUSIQ Dataset Sanity Check")
    print("="*60)
    print(f"Project root: {config.PROJECT_ROOT}")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Label file: {config.LABEL_FILE}")
    print()
    
    # Track overall status
    all_checks_passed = True
    
    try:
        # Check 1 & 2: Load labels and verify keys
        labels = load_labels(config.LABEL_FILE)
        check_key_format(labels)
        check_all_keys_exist(labels)
        
        # Check 3: Score statistics
        compute_split_statistics(labels)
        
        # Check 4: Sample image inspection
        inspect_sample_images(labels)
        
        # Check 5: 16-bit detection
        check_16bit_images(labels)
        
        # Check 6: Split range validation
        check_split_ranges()
        
    except FileNotFoundError as e:
        print(f"\n✗ FATAL: {e}")
        all_checks_passed = False
    except json.JSONDecodeError as e:
        print(f"\n✗ FATAL: Invalid JSON in label file: {e}")
        all_checks_passed = False
    except Exception as e:
        print(f"\n✗ FATAL: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        all_checks_passed = False
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if all_checks_passed:
        print("  ✓ All checks completed. Review any warnings above.")
        print("  → If no warnings, dataset is ready for training!")
    else:
        print("  ✗ Some checks failed. Fix errors before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
