# Captcha Solver — Finetune Guideline

## Goal

Train a lightweight model to solve text-based captcha images with >90% accuracy, optimized for edge/on-device deployment.

---

## 1. Dataset Analysis

**Source:** `ProcessedCaptchas/` — 500 labeled PNG captcha images.

Labels are encoded in filenames (e.g., `228WH.png` = `228WH`).

| Property | Value |
|---|---|
| Total images | 500 |
| Characters per captcha | 5 (fixed length) |
| Unique characters | 32 (`23456789ABCDEFGHJKLMNPQRSTUVWXYZ`) |
| Excluded characters | `0, 1, I, O` (avoid visual ambiguity) |
| Avg samples per character | ~78 |
| Image style | Grayscale, rotated text, speckle noise, line artifacts |

**Key observation:** Character distribution is relatively balanced (min 64, max 97 per character), which is favorable for training.

---

## 2. Tech Stack Decision

| Component | Choice | Reason |
|---|---|---|
| Package manager | `uv` | Fast, modern Python package manager |
| Training framework | PyTorch | Most flexible, best ecosystem for experimentation |
| Inference runtime | ONNX Runtime | Cross-platform, lightweight, fast on CPU/edge |
| Deployment format | `.onnx` | Open standard, runs anywhere |

### Project Setup

```bash
uv init --no-readme
uv add torch torchvision onnx onnxruntime onnxscript pillow matplotlib numpy
```

---

## 3. Phase 1 — CRNN + CTC (v1)

**Script:** `train.py`

### Architecture

- **CNN backbone:** 5 convolutional blocks (Conv2d + BatchNorm + ReLU + MaxPool), producing feature maps of shape `(B, 256, 1, 10)` — 10 width columns representing sequential positions.
- **Sequence model:** 2-layer Bidirectional LSTM (256 hidden units) processes the 10-column feature sequence.
- **Decoder:** CTC (Connectionist Temporal Classification) loss with greedy decoding.

### Data Augmentation

Heavy augmentation to compensate for the small dataset:
- Random rotation (-15 to 15 degrees)
- Random affine shift
- Random scale (0.85x to 1.15x)
- Gaussian noise
- Gaussian blur
- Erosion/dilation (MinFilter/MaxFilter)
- Brightness/contrast jitter

### Training Config

- Optimizer: Adam (lr=1e-3)
- Scheduler: Cosine annealing over 150 epochs
- Batch size: 32
- Gradient clipping: max norm 5.0

### Results

| Metric | Value |
|---|---|
| Full captcha accuracy | 95.0% |
| Per-character accuracy | 96.4% |
| Model parameters | 1,450,721 |

### Problem Identified

All 5 test errors shared the same root cause — **CTC collapsed consecutive repeated characters**:

| True | Predicted | Issue |
|---|---|---|
| 4DDYR | 4DYR | DD -> D |
| PNHHS | PNHS | HH -> H |
| KAAJB | KAJB | AA -> A |
| CWW72 | CW72 | WW -> W |
| QS889 | QS89 | 88 -> 8 |

CTC decoding works by collapsing repeated predictions and removing blanks. To output `DD`, the model must predict `D, blank, D` — which is hard to learn with limited data.

---

## 4. Phase 1.1 — Multi-Head Classification (v2)

**Script:** `train_v2.py`

### Key Insight

Since captcha length is **fixed at 5**, CTC's variable-length flexibility is unnecessary. Replacing CTC with 5 independent classification heads eliminates the repeat-collapse problem entirely.

### Architecture Change

- **CNN backbone:** Same 5 convolutional blocks, but the 5th pooling layer now produces `(B, 256, 2, 5)` — exactly 5 width columns, one per character position.
- **Classification heads:** 5 independent `Linear(512, 128) -> ReLU -> Dropout -> Linear(128, 32)` heads. Each reads features from its corresponding spatial column.
- **Loss:** Sum of 5 CrossEntropy losses (one per position).

No LSTM, no CTC — simpler, faster, and more accurate for fixed-length output.

### Dataset Split

The v2 training uses a proper 3-way split to avoid data leakage from model selection:

```bash
uv run python split_dataset.py
```

| Split | Count | Role |
|---|---|---|
| `data/train/` | 350 (70%) | Gradient updates |
| `data/val/` | 75 (15%) | Model selection (best checkpoint) |
| `data/test/` | 75 (15%) | Held-out, evaluated only at the end |

The v1 approach used an 80/20 train/test split where the "test" set was also used for model selection every 5 epochs — effectively making it a validation set with no true held-out evaluation.

### Results

| Metric | v1 (CTC) | v2 (Multi-Head) |
|---|---|---|
| Train accuracy | — | **99.4%** |
| Val accuracy | — | **100.0%** |
| Test accuracy (held-out) | 95.0% | **98.7%** |
| Per-character accuracy (test) | 96.4% | **99.7%** |
| Model parameters | 1,450,721 | **1,328,352** |
| Model size | 5.6 MB | **5.1 MB** |

The single remaining test error: `2W7HT` predicted as `2WTHT` — a genuine visual ambiguity between `7` and `T` under heavy rotation and noise.

All 5 previous CTC repeat errors (`DD`, `HH`, `AA`, `WW`, `88`) are now correctly predicted.

Early stopping (patience=30 epochs) halted training at epoch 145 to avoid overfitting. The small train-test gap (99.4% vs 98.7%) confirms good generalization.

---

## 5. ONNX Export

**Script:** `export_onnx.py` (standalone, legacy) or inline in `train_v2.py`

The trained PyTorch model is exported to ONNX format for edge deployment:
- Input: `image` — shape `(batch, 1, 64, 160)`, float32, normalized to [-1, 1]
- Outputs: `char_0` through `char_4` — each shape `(batch, 32)`, logits for 32 character classes

**Output file:** `output/captcha_model_v2.onnx` (5.1 MB)

---

## 6. Inference

**Script:** `inference.py`

Supports three usage modes:

```bash
# Single image
uv run python inference.py captcha.png

# Folder of images
uv run python inference.py ProcessedCaptchas/

# Multiple images
uv run python inference.py img1.png img2.png img3.png
```

If filenames match the 5-character label format, accuracy is automatically calculated.

### Preprocessing Pipeline

1. Load image as grayscale
2. Resize to 160x64
3. Normalize: pixel / 255.0, then (x - 0.5) / 0.5 to [-1, 1]
4. Shape to `(1, 1, 64, 160)` float32

### Decoding

For each of the 5 output heads, take `argmax` and map index to character using the 32-character alphabet.

---

## 7. Final Validation

Held-out test set inference (75 images never used for training or model selection):

```bash
uv run python inference.py data/test/
# Accuracy: 74/75 (98.7%)
```

| Split | Full Accuracy | Char Accuracy |
|---|---|---|
| Train (350) | 99.4% | 99.9% |
| Val (75) | 100.0% | 100.0% |
| **Test (75)** | **98.7%** | **99.7%** |

---

## Project Structure

```
captcha_finetune/
  ProcessedCaptchas/       # Raw dataset (500 labeled captcha images)
  data/
    train/                 # Training split (350 images)
    val/                   # Validation split (75 images)
    test/                  # Held-out test split (75 images)
  output/
    best_model_v2.pth      # PyTorch weights (5.1 MB)
    captcha_model_v2.onnx  # ONNX model for deployment (5.1 MB)
    training_plot_v2.png   # Loss and accuracy curves
  split_dataset.py         # Splits ProcessedCaptchas/ into data/{train,val,test}/
  train.py                 # v1: CRNN + CTC (95% accuracy, deprecated)
  train_v2.py              # v2: Multi-Head Classification (98.7% test accuracy)
  export_onnx.py           # Standalone ONNX export script (v1, legacy)
  inference.py             # ONNX inference script
  pyproject.toml           # uv project config with dependencies
```

---

## Lessons Learned

1. **Match architecture to problem structure.** Fixed-length output does not need CTC. Multi-head classification is simpler, faster, and more accurate for this case.

2. **Heavy data augmentation is critical with small datasets.** 350 training images with aggressive augmentation achieved 98.7% held-out test accuracy without any synthetic data generation.

3. **Error analysis drives improvement.** Identifying that all v1 errors were CTC repeat-collapses pointed directly to the architectural fix, rather than needing more data or hyperparameter tuning.

4. **Start simple, iterate on evidence.** Phase 1 (CTC) was a reasonable starting point. Its failure mode revealed the optimal solution (multi-head), which would have been harder to justify without the comparison.

5. **Use a proper train/val/test split.** Using the "test" set for model selection inflates reported accuracy. A 3-way split (70/15/15) with the test set evaluated only once gives a trustworthy generalization estimate. Early stopping on the validation set also prevents overfitting.
