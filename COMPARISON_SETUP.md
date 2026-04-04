# Model Comparison Setup — Updated

## What Changed

The baseline is now treated as a single method:
- `agaldran_combo` = Swin-T + ResNet50 combined inside one model.
- Final prediction is the average of both backbone scores.

This matches your point: the baseline should count as one model/method, not two separate competitors.

## Files Updated

### `baseline_models.py`
- Kept both backbone implementations (`SwinTransformerQA`, `ResNet50QA`) as internal components.
- Added combined model `AgaldranComboQA`.
- `AgaldranComboQA.forward(...)` returns:
  - `score`: ensembled scalar prediction `[B, 1]`
  - `scale_scores`: empty list
- This output format is compatible with your existing `CTMUSIQLoss` and training pipeline.
- Factory now exposes one baseline option:
  - `create_baseline_model('agaldran_combo')`

### `get_model.py`
- Model registry now supports only:
  - `ct_musiq`
  - `agaldran_combo`
- Updated docs and examples accordingly.

### `train.py`
- CLI options updated to:
  - `--model ct_musiq`
  - `--model agaldran_combo`
- Docstrings updated to remove separate `swin_t` / `resnet50` options.
- Existing training flow remains the same.

### `evaluate.py`
- CLI options updated to:
  - `--model ct_musiq`
  - `--model agaldran_combo`
- Docstrings and usage updated for the single combined baseline.

## Correct Usage

Train:
```bash
python train.py --model ct_musiq --epochs 50
python train.py --model agaldran_combo --epochs 50
```

Evaluate:
```bash
python evaluate.py --model ct_musiq
python evaluate.py --model agaldran_combo
```

## Output Locations

- CT-MUSIQ:
  - `results/ct_musiq/ct_musiq_best.pth`
  - `results/ct_musiq/ct_musiq_training_log.csv`
- Combined baseline:
  - `results/agaldran_combo/agaldran_combo_best.pth`
  - `results/agaldran_combo/agaldran_combo_training_log.csv`

## Fairness Note

The comparison still keeps all training/evaluation settings aligned (same data splits, optimizer family, epoch budget, metrics), while changing only model architecture/method.
