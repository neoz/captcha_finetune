# Captcha Solver

A lightweight captcha recognition model achieving **99% accuracy** on 5-character alphanumeric captchas, trained on just 500 images.

## Results

| Metric | Value |
|---|---|
| Full captcha accuracy (held-out test) | **99.0%** |
| Per-character accuracy | **99.8%** |
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

Requirements:
- Grayscale or RGB PNG images
- Exactly 5 characters per label
- Characters from: `23456789ABCDEFGHJKLMNPQRSTUVWXYZ`

## Project Structure

```
captcha_finetune/
  ProcessedCaptchas/        # Dataset (500 labeled images)
  output/
    best_model_v2.pth       # Trained PyTorch weights
    captcha_model_v2.onnx   # ONNX model for deployment
    training_plot_v2.png    # Training curves
  train_v2.py               # Training script
  inference.py              # ONNX inference script
  export_onnx.py            # Standalone ONNX export
  GUIDELINE.md              # Detailed step-by-step development guide
  pyproject.toml            # Dependencies
```

## ONNX Integration

The exported ONNX model can be used in any language/platform supported by ONNX Runtime:

**Input:** `image` — `float32[batch, 1, 64, 160]`, normalized to `[-1, 1]`

**Outputs:** `char_0` through `char_4` — `float32[batch, 32]` logits

Preprocessing: grayscale, resize to 160x64, `pixel / 255`, then `(x - 0.5) / 0.5`.

Decoding: `argmax` each output head, map index to character.
