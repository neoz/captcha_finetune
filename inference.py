"""
Captcha solver inference using ONNX Runtime.
Usage:
    python inference.py <image_path_or_folder>

Examples:
    python inference.py captcha.png
    python inference.py ProcessedCaptchas/
    python inference.py img1.png img2.png img3.png
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

CHARS = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
IMG_HEIGHT = 64
IMG_WIDTH = 160
MODEL_PATH = Path("output/captcha_model_v2.onnx")


def load_model(model_path=MODEL_PATH):
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    return session


def preprocess(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
    return arr


def predict(session, image_path):
    inp = preprocess(image_path)
    outputs = session.run(None, {"image": inp})
    chars = []
    for out in outputs:
        idx = np.argmax(out, axis=1)[0]
        chars.append(IDX_TO_CHAR[idx])
    return "".join(chars)


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    session = load_model()

    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.png")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"Not found: {arg}")

    correct = 0
    total_labeled = 0

    for path in paths:
        result = predict(session, path)
        label = path.stem
        has_label = len(label) == 5 and all(c in CHARS for c in label)

        if has_label:
            total_labeled += 1
            match = result == label
            if match:
                correct += 1
            status = "OK" if match else "WRONG"
            print(f"{path.name}  ->  {result}  (label: {label})  [{status}]")
        else:
            print(f"{path.name}  ->  {result}")

    if total_labeled > 0:
        print(f"\nAccuracy: {correct}/{total_labeled} ({correct/total_labeled:.1%})")


if __name__ == "__main__":
    main()
