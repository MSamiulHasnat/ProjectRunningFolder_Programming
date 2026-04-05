# CT-MUSIQ Training Pipeline: Detailed Logic

## 1. INITIALIZATION

### 1.1 Seed for Reproducibility
```python
# config.py: SEED = 42
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Why important?** Ensures exact reproducibility across runs. Any tiny randomness (e.g., dropout, data augmentation order) could lead to different convergence paths.

### 1.2 Device Setup
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# RTX 3060 Laptop (6GB VRAM)
# Mixed precision (fp16) essential for fitting batch_size=10
```

---

## 2. DATA LOADING

### 2.1 Dataset Class (dataset.py: LDCTDataset)

```python
class LDCTDataset:
    def __init__(self, split='train'):
        self.split = split  # 'train', 'val', 'test'
        
        # Determine index range
        if split == 'train':
            start, end = TRAIN_RANGE  # (0, 699)
        elif split == 'val':
            start, end = VAL_RANGE    # (700, 899)
        else:  # test
            start, end = TEST_RANGE   # (900, 999)
        
        self.indices = list(range(start, end + 1))  # inclusive on both ends
        
        # Load JSON labels
        with open(LABEL_FILE) as f:
            self.labels = json.load(f)  # {"0000.tif": 2.5, "0001.tif": 3.2, ...}
    
    def __getitem__(self, idx):
        img_index = self.indices[idx]
        
        # Build filename (4-digit zero-padded)
        filename = f"{img_index:04d}.tif"  # e.g., "0042.tif"
        img_path = os.path.join(DATA_DIR, filename)
        
        # Load TIFF image
        img = Image.open(img_path)  # shape (512, 512), dtype float32
        img = np.array(img)
        
        # Apply CT windowing (brain soft-tissue)
        # Original pixels are in Hounsfield Units [min, max]
        # Window: [WINDOW_LEVEL - WIDTH/2, WINDOW_LEVEL + WIDTH/2] = [0, 80]
        img = np.clip(img, WINDOW_MIN, WINDOW_MAX)  # [0, 80]
        img = (img - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)  # normalize to [0, 1]
        
        # Get quality score
        label = self.labels[filename]  # float in [0, 4]
        
        # Build multi-scale pyramid
        scales_data = []
        patch_coords = []
        
        for scale_idx, scale in enumerate(SCALES):  # [224, 384]
            # Resize to scale
            img_scaled = Image.fromarray((img * 255).astype(np.uint8))
            img_scaled = img_scaled.resize((scale, scale), Image.BICUBIC)
            img_scaled = np.array(img_scaled) / 255.0  # back to [0, 1]
            
            # Convert to tensor (still grayscale, 1 channel)
            img_tensor = torch.from_numpy(img_scaled).float().unsqueeze(0)  # [1, scale, scale]
            
            # Extract patches and coordinates
            num_patches_per_side = scale // PATCH_SIZE  # 7 or 12
            
            coords = []
            for row in range(num_patches_per_side):
                for col in range(num_patches_per_side):
                    # Coordinate: [scale_idx, row, col]
                    coords.append([scale_idx, row, col])
            
            scales_data.append(img_tensor)
            patch_coords.extend(coords)
        
        # Convert to tensors
        images = scales_data  # [img_224, img_384]
        coords = torch.tensor(patch_coords, dtype=torch.long)  # [193, 3]
        label = torch.tensor(label, dtype=torch.float)  # scalar
        
        return images, coords, label
```

### 2.2 Data Augmentation (Training Only)

```python
def apply_augmentation(image: Tensor) -> Tensor:
    # Applied during training to prevent overfitting
    # NOT applied during validation/test
    
    img_array = (image.numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_array)
    
    # 1. Horizontal flip (50% chance)
    if random.random() < 0.5:
        img_pil = ImageOps.mirror(img_pil)
    
    # 2. Rotation ±5° (random angle)
    angle = random.uniform(-5, 5)
    img_pil = img_pil.rotate(angle, fillcolor=128)
    
    # 3. Random crop 85-100%
    crop_ratio = random.uniform(0.85, 1.0)
    crop_size = int(img_pil.size[0] * crop_ratio)
    left = random.randint(0, img_pil.size[0] - crop_size)
    top = random.randint(0, img_pil.size[1] - crop_size)
    img_pil = img_pil.crop((left, top, left + crop_size, top + crop_size))
    img_pil = img_pil.resize((img_pil.size[0], img_pil.size[1]))  # back to original size
    
    # 4. Brightness/contrast jitter
    brightness_factor = random.uniform(0.95, 1.05)
    contrast_factor = random.uniform(0.95, 1.05)
    img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness_factor)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast_factor)
    
    # 5. Gaussian noise (σ ∈ [0.005, 0.02])
    img_array = np.array(img_pil).astype(np.float32) / 255.0
    noise_std = random.uniform(0.005, 0.02)
    img_array += np.random.normal(0, noise_std, img_array.shape)
    img_array = np.clip(img_array, 0, 1)
    
    return torch.from_numpy(img_array).float().unsqueeze(0)
```

**Augmentation strategy**:
- **Horizontal flip**: Brain CTs can be flipped without changing quality assessment (brain is roughly symmetric)
- **Rotation**: ±5° is realistic (gantry tilt in CT acquisition)
- **Crop**: Simulates partial field-of-view
- **Brightness/contrast**: Simulates window level variations
- **Noise**: Simulates dose variations

**Why needed?** 700 training samples is small. Augmentation prevents overfitting and makes model robust to acquisition variations.

### 2.3 DataLoader Creation

```python
def create_dataloaders():
    train_dataset = LDCTDataset(split='train')  # 700 images
    val_dataset = LDCTDataset(split='val')      # 200 images
    test_dataset = LDCTDataset(split='test')    # 100 images
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,  # 10
        shuffle=True,           # shuffle training data
        num_workers=0,          # Windows compatibility (no multiprocessing)
        collate_fn=default_collate_fn  # standard PyTorch collation
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,          # no shuffle for validation
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader

# Collate function combines samples into batch
def default_collate_fn(batch):
    # batch: list of (images, coords, label) tuples
    
    images_batch = []
    coords_batch = []
    labels_batch = []
    
    for images, coords, label in batch:
        images_batch.append(images)
        coords_batch.append(coords)
        labels_batch.append(label)
    
    # All samples produce same number of patches (193) → no padding needed
    # Can safely concatenate
    images_stack = [torch.stack([img[i] for img in images_batch], dim=0) 
                    for i in range(len(images_batch[0]))]
    coords_stack = torch.cat(coords_batch, dim=0)  # [B*193, 3]
    labels_stack = torch.stack(labels_batch)       # [B]
    
    return images_stack, coords_stack, labels_stack
```

---

## 3. MODEL CREATION

### 3.1 Model Instantiation

```python
# get_model.py
def get_model(model_type='ct_musiq', scales=None, lambda_kl=None):
    if model_type == 'ct_musiq':
        model = CTMusiq(
            scales=scales or config.SCALES,      # [224, 384]
            patch_size=config.PATCH_SIZE,        # 32
            d_model=config.D_MODEL,              # 768
            num_heads=config.NUM_HEADS,          # 8
            num_layers=config.NUM_LAYERS,        # 12
            ffn_dim=config.FFN_DIM,              # 3072
            dropout=config.DROPOUT,              # 0.05
            lambda_kl=lambda_kl or config.LAMBDA_KL  # 0.02
        )
        return model
```

### 3.2 Loading Pretrained Weights

```python
# model.py: _load_pretrained_weights()
def _load_pretrained_weights(model):
    # Load ViT-B/32 from timm (trained on ImageNet)
    timm_model = timm.create_model('vit_b_32_plus_256', pretrained=True)
    
    # Extract weights from timm model
    timm_state = timm_model.state_dict()
    
    # Map timm naming → our model naming
    # timm: blocks.i.attn.qkv
    # ours: transformer.layers.i.self_attn.in_proj_weight
    
    mapping = {
        'blocks': 'transformer.layers',
        'attn.qkv': 'self_attn.in_proj',
        'attn.proj': 'self_attn.out_proj',
        'mlp.fc1': 'mlp.0',  # first linear
        'mlp.fc2': 'mlp.2',  # second linear
    }
    
    converted_state = {}
    for timm_key, value in timm_state.items():
        new_key = timm_key
        for old, new in mapping.items():
            new_key = new_key.replace(old, new)
        converted_state[new_key] = value
    
    # Load converted weights (ignore mismatches for head/encoder)
    model.transformer.load_state_dict(converted_state, strict=False)
```

**Why pretrained weights?**
- ViT-B/32 already knows how to extract image features from ImageNet
- Transfer learning: Initialize from strong feature extractor
- Saves ~50 epochs of random-init training
- Better generalization to small datasets (700 train samples)

---

## 4. LOSS FUNCTION & OPTIMIZER

### 4.1 Criterion

```python
# loss.py: CTMUSIQLoss
criterion = CTMUSIQLoss(
    num_bins=config.NUM_BINS,  # 20
    sigma=config.SIGMA,         # 0.5
    lambda_kl=lambda_kl
)

class CTMUSIQLoss(nn.Module):
    def forward(self, global_pred, scale_preds, target):
        # global_pred: [B, 1] scalar predictions from global head
        # scale_preds: [[B, 1], [B, 1]] per-scale predictions
        # target: [B] ground truth quality scores
        
        # Convert to soft distributions
        target_dist = self.score_to_distribution(target.squeeze())  # [B, 20]
        
        # MSE loss
        mse_loss = F.mse_loss(global_pred.squeeze(), target)
        
        # KL loss
        global_dist = self.score_to_distribution(global_pred.squeeze())  # [B, 20]
        kl_loss = 0.0
        
        for scale_pred in scale_preds:
            scale_dist = self.score_to_distribution(scale_pred.squeeze())
            kl_loss += F.kl_div(scale_dist, global_dist.detach(), reduction='batchmean')
        
        kl_loss /= len(scale_preds)
        
        total_loss = mse_loss + self.lambda_kl * kl_loss
        
        return total_loss
```

### 4.2 Optimizer

```python
# train.py: build optimizer
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR_STAGE1 if epoch <= STAGE1_EPOCHS else LR,  # 3e-4 or 5e-5
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-5
)
```

**Why AdamW?**
- Adaptive momentum (adam) + weight decay (W) separately
- Better generalization than plain Adam
- Standard choice for vision transformers

### 4.3 Learning Rate Scheduler

```python
# Stage 1: Fixed LR = 3e-4 (epochs 1-5)
# Stage 2: Linear warmup 3 epochs + cosine decay
if epoch <= STAGE1_EPOCHS:
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR_STAGE1
elif epoch <= STAGE1_EPOCHS + STAGE2_WARMUP_EPOCHS:
    # Linear warmup: LR_MIN → LR over 3 epochs
    warmup_progress = (epoch - STAGE1_EPOCHS) / STAGE2_WARMUP_EPOCHS
    warmup_lr = LR_MIN + (LR - LR_MIN) * warmup_progress
    for param_group in optimizer.param_groups:
        param_group['lr'] = warmup_lr
else:
    # Cosine annealing: LR → LR_MIN over remaining epochs
    T_max = EPOCHS - (STAGE1_EPOCHS + STAGE2_WARMUP_EPOCHS)
    progress = (epoch - STAGE1_EPOCHS - STAGE2_WARMUP_EPOCHS) / T_max
    cosine_lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + np.cos(np.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cosine_lr
```

---

## 5. MIXED PRECISION TRAINING (AMP)

### 5.1 AutoCast for Forward Pass

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(EPOCHS):
    for images, coords, labels in train_loader:
        # Forward in float16
        with autocast('cuda', dtype=torch.float16):
            outputs = model(images, coords)
            loss = criterion(outputs, labels)
        
        # Backward in float32 (automatic with scaler)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step with scaling
        scaler.step(optimizer)
        scaler.update()
```

**Memory savings**:
- float32: 4 bytes per value
- float16: 2 bytes per value
- **50% memory reduction** on activations
- Batch size 10 achievable on 6GB VRAM with AMP

**Safety**:
- Loss is computed in float16 but scaled up (large values)
- Backward pass uses float32 for numerical stability
- GradScaler automatically unscales before optimizer step

---

## 6. EXPONENTIAL MOVING AVERAGE (EMA)

### 6.1 EMA Pattern

```python
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow_state = copy.deepcopy(model.state_dict())
    
    def update(self, model):
        # shadow = decay * shadow + (1 - decay) * current
        # Exponential moving average of weights
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    shadow_param = self.shadow_state[name]
                    shadow_param.copy_(
                        self.decay * shadow_param + (1 - self.decay) * param.data
                    )
    
    def apply(self, model):
        # Swap model weights with EMA weights
        self.backup_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow_state)
    
    def restore(self, model):
        # Restore original weights
        model.load_state_dict(self.backup_state)
```

### 6.2 EMA Validation

```python
# After each training epoch:
for epoch in range(EPOCHS):
    # ... train_one_epoch() ...
    
    # Update EMA
    ema.update(model)
    
    # Validate with EMA weights
    ema.apply(model)  # swap in EMA weights
    
    with torch.no_grad():
        val_metrics = validate(model, val_loader)
    
    ema.restore(model)  # restore original weights
    
    # Save best EMA checkpoint
    if val_metrics['aggregate'] > best_aggregate:
        save_checkpoint(ema.shadow_state, ...)  # save EMA weights
```

**Why EMA?**
- Smoother validation metrics (reduces noise from batch variation)
- Better generalization (ensemble of recent weights)
- Standard practice in modern vision training

---

## 7. EARLY STOPPING

```python
patience = 0
best_aggregate = 0

for epoch in range(EPOCHS):
    # ... train & validate ...
    
    current_aggregate = val_metrics['aggregate']
    
    if current_aggregate > best_aggregate:
        best_aggregate = current_aggregate
        patience = 0
        save_checkpoint(model)
    else:
        patience += 1
        if patience >= PATIENCE:  # 20
            print(f"Early stop at epoch {epoch}")
            break
```

**Patience=20 means**:
- Must improve every 20 epochs
- At 100 epochs total, tolerance for 20 epochs of no improvement
- Typical for medical imaging (slow convergence)

---

## 8. CHECKPOINTING & LOGGING

### 8.1 Checkpoint Format

```python
def save_checkpoint(model, optimizer, epoch, metrics, path):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'config': config.__dict__,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)
    print(f"Saved: {path} (Aggregate: {metrics['aggregate']:.4f})")
```

### 8.2 CSV Logging

```python
import csv

def log_epoch(epoch, stage, loss, metrics):
    csv_path = TRAINING_LOG_CSV
    
    row = {
        'epoch': epoch,
        'stage': stage,  # 1 or 2
        'loss': loss,
        'plcc': metrics['plcc'],
        'srocc': metrics['srocc'],
        'krocc': metrics['krocc'],
        'aggregate': metrics['aggregate'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if epoch == 1:
            writer.writeheader()
        writer.writerow(row)
```

**Log structure** (ct_musiq_training_log.csv):
```
epoch,stage,loss,plcc,srocc,krocc,aggregate,timestamp
1,1,0.892,0.654,0.623,0.411,1.688,2026-04-04T10:15:23
2,1,0.745,0.712,0.698,0.521,1.931,2026-04-04T10:16:15
...
6,2,0.623,0.801,0.789,0.604,2.194,2026-04-04T10:25:00
...
100,2,0.042,0.951,0.946,0.820,2.717,2026-04-04T11:30:00
```

---

## 9. COMPLETE TRAINING LOOP (Pseudocode)

```python
def train():
    # Setup
    train_loader, val_loader, test_loader = create_dataloaders()
    model = get_model()
    criterion = CTMUSIQLoss(lambda_kl=config.LAMBDA_KL)
    ema = ModelEMA(model)
    scaler = GradScaler()
    
    best_aggregate = 0
    patience_counter = 0
    
    # Stage 1 & 2 combined
    for epoch in range(1, config.EPOCHS + 1):
        # ===== STAGE TRANSITIONS =====
        if epoch == 1:
            # Freeze encoder
            for param in model.transformer.parameters():
                param.requires_grad = False
            optimizer = AdamW(model.parameters(), lr=config.LR_STAGE1)
        elif epoch == config.STAGE1_EPOCHS + 1:
            # Unfreeze encoder, switch optimizer
            for param in model.transformer.parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=config.LR)
        
        # ===== TRAINING =====
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, coords, labels) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            coords = coords.to(device)
            labels = labels.to(device)
            
            with autocast('cuda'):
                outputs = model(images, coords)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        
        # ===== VALIDATION =====
        model.eval()
        ema.update(model)
        
        ema.apply(model)
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, coords, labels in val_loader:
                outputs = model([img.to(device) for img in images], coords.to(device))
                val_preds.append(outputs.cpu())
                val_targets.append(labels.cpu())
        ema.restore(model)
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        metrics = compute_metrics(val_preds, val_targets)
        
        # ===== EARLY STOPPING =====
        if metrics['aggregate'] > best_aggregate:
            best_aggregate = metrics['aggregate']
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, metrics)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break
        
        # ===== LOGGING =====
        log_epoch(epoch, 1 if epoch <= config.STAGE1_EPOCHS else 2, epoch_loss, metrics)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={epoch_loss:.4f}, agg={metrics['aggregate']:.4f}")
    
    # ===== TEST EVALUATION =====
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for images, coords, labels in test_loader:
            outputs = model([img.to(device) for img in images], coords.to(device))
            test_preds.append(outputs.cpu())
            test_targets.append(labels.cpu())
    
    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)
    test_metrics = compute_metrics(test_preds, test_targets)
    
    save_test_results(test_metrics, test_preds, test_targets)
    
    return test_metrics
```

---

**Training Completed**: 2026-04-05  
**Best Results**: Validation Aggregate ~2.72 (PLCC 0.95, SROCC 0.95, KROCC 0.82)
