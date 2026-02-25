# Captcha Solver

A lightweight captcha recognition model achieving **98.7% accuracy** on 5-character alphanumeric captchas, trained on just 500 images with a proper train/val/test split.

## Results

| Metric | Value |
|---|---|
| Train accuracy | **99.4%** (350 images) |
| Validation accuracy | **100.0%** (75 images) |
| Test accuracy (held-out) | **98.7%** (75 images) |
| Per-character accuracy (test) | **99.7%** |
| Model size (ONNX) | **5.1 MB** |
| Inference runtime | ONNX Runtime (CPU) |
| Character set | `23456789ABCDEFGHJKLMNPQRSTUVWXYZ` (32 chars) |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
uv sync
```

### Inference

```bash
# Single image
uv run python inference.py captcha.png

# Multiple images
uv run python inference.py img1.png img2.png img3.png

# Entire folder (auto-evaluates accuracy if filenames are labels)
uv run python inference.py ProcessedCaptchas/
```

### Train

```bash
# Step 1: Split dataset into train/val/test folders
uv run python split_dataset.py

# Step 2: Train (uses data/train for training, data/val for model selection)
uv run python train_v2.py
```

Outputs are saved to `output/`:
- `best_model_v2.pth` — PyTorch weights
- `captcha_model_v2.onnx` — ONNX model for deployment
- `training_plot_v2.png` — training curves

## Architecture

**Multi-Head CNN** — 5-block CNN backbone with 5 independent classification heads, one per character position.

```
Input Image (1, 64, 160)
        |
  5x [Conv2d + BN + ReLU + MaxPool]
        |
  Feature Map (256, 2, 5)     <-- 5 spatial columns = 5 character positions
        |
  5x [Linear -> ReLU -> Dropout -> Linear]
        |
  5 predictions (one per character)
```

No LSTM, no CTC — each character position is classified independently, which handles repeated characters (e.g., `DD`, `88`) without issues.

## Dataset

Place labeled captcha images in `ProcessedCaptchas/`. Labels are encoded as filenames:

```
ProcessedCaptchas/
  228WH.png    -> label: 228WH
  2BGBB.png    -> label: 2BGBB
  ...
```

Run `uv run python split_dataset.py` to create the train/val/test split:

```
data/
  train/    # 350 images (70%) — used for gradient updates
  val/      #  75 images (15%) — used for model selection (best checkpoint)
  test/     #  75 images (15%) — held-out, evaluated only at the end
```

Requirements:
- Grayscale or RGB PNG images
- Exactly 5 characters per label
- Characters from: `23456789ABCDEFGHJKLMNPQRSTUVWXYZ`

## Project Structure

```
captcha_finetune/
  ProcessedCaptchas/        # Raw dataset (500 labeled images)
  data/
    train/                  # Training split (350 images)
    val/                    # Validation split (75 images)
    test/                   # Held-out test split (75 images)
  output/
    best_model_v2.pth       # Trained PyTorch weights
    captcha_model_v2.onnx   # ONNX model for deployment
    training_plot_v2.png    # Training curves
  split_dataset.py          # Dataset splitter
  train_v2.py               # Training script (multi-head CNN)
  inference.py              # ONNX inference script
  export_onnx.py            # Standalone ONNX export (v1, legacy)
  GUIDELINE.md              # Detailed step-by-step development guide
  pyproject.toml            # Dependencies
```

## ONNX Integration

The exported ONNX model can be used in any language/platform supported by ONNX Runtime:

**Input:** `image` — `float32[batch, 1, 64, 160]`, normalized to `[-1, 1]`

**Outputs:** `char_0` through `char_4` — `float32[batch, 32]` logits

Preprocessing: grayscale, resize to 160x64, `pixel / 255`, then `(x - 0.5) / 0.5`.

Decoding: `argmax` each output head, map index to character.
