"""Export trained model to ONNX format."""
import os
from pathlib import Path

import torch

from train import CRNN, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH

OUTPUT_DIR = Path("output")
device = torch.device("cpu")

model = CRNN(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth", weights_only=True, map_location=device))
model.eval()

dummy = torch.randn(1, 1, IMG_HEIGHT, IMG_WIDTH).to(device)
onnx_path = OUTPUT_DIR / "captcha_model.onnx"

torch.onnx.export(
    model, dummy, str(onnx_path),
    input_names=["image"],
    output_names=["output"],
    dynamic_axes={"image": {0: "batch"}, "output": {1: "batch"}},
    opset_version=17,
)

onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
print(f"ONNX model saved: {onnx_path} ({onnx_size:.1f} MB)")
