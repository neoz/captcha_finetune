# Captcha Finetune

## Commands
- `uv run python split_dataset.py` - Split ProcessedCaptchas/ into data/{train,val,test}/
- `uv run python train_v2.py` - Train multi-head CNN model (primary training script)
- `uv run python inference.py <path>` - Run ONNX inference on image or folder
- `uv run python export_onnx.py` - Export v1 (CTC) model to ONNX (legacy)

## Architecture
- Multi-head CNN: 5-block CNN backbone + 5 independent classification heads (train_v2.py)
- Input: grayscale 160x64, normalized to [-1, 1] via (pixel/255 - 0.5) / 0.5
- Output: 5 chars from 32-char alphabet (23456789ABCDEFGHJKLMNPQRSTUVWXYZ)
- train.py is v1 (CRNN+CTC, deprecated) - train_v2.py is current

## Dataset
- Raw: ProcessedCaptchas/ (500 images, labels = filenames)
- Split: data/train/ (350), data/val/ (75), data/test/ (75) via split_dataset.py
- Val used for model selection, test evaluated only at end
