"""
Split ProcessedCaptchas into train/val/test folders.

Split ratios: 70% train, 15% validation, 15% test
- Train: used for gradient updates
- Val: used for model selection (best checkpoint) during training
- Test: evaluated ONLY at the end, for true accuracy reporting

Files are copied (not moved) so the original ProcessedCaptchas/ stays intact.
"""

import random
import shutil
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (remainder)

SRC_DIR = Path("ProcessedCaptchas")
OUT_DIR = Path("data")

random.seed(SEED)


def main():
    all_files = sorted(SRC_DIR.glob("*.png"))
    print(f"Total images: {len(all_files)}")

    indices = list(range(len(all_files)))
    random.shuffle(indices)

    n = len(indices)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }

    for split_name, split_indices in splits.items():
        split_dir = OUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Clear existing files
        for f in split_dir.glob("*.png"):
            f.unlink()

        for i in split_indices:
            src = all_files[i]
            dst = split_dir / src.name
            shutil.copy2(src, dst)

        print(f"  {split_name}: {len(split_indices)} images -> {split_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
