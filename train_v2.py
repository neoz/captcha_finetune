"""
Multi-Head Classification Captcha Solver - Phase 1.1
Architecture: CNN + 5 independent classification heads (no CTC)
Fixes: repeated character collapse issue from CTC approach
"""

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

# -- Config -------------------------------------------------------------------

CHARS = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)  # 32 classes, no blank token needed

IMG_HEIGHT = 64
IMG_WIDTH = 160
CAPTCHA_LEN = 5
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
SEED = 42

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
TEST_DIR = Path("data/test")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -- Dataset ------------------------------------------------------------------

class CaptchaDataset(Dataset):
    def __init__(self, image_paths, labels, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        label = self.labels[idx]

        if self.augment:
            img = self._augment(img)

        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.tensor(img).unsqueeze(0)  # (1, H, W)

        target = torch.tensor([CHAR_TO_IDX[c] for c in label], dtype=torch.long)
        return img, target

    def _augment(self, img):
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=255, expand=False)

        if random.random() < 0.5:
            dx = random.randint(-8, 8)
            dy = random.randint(-4, 4)
            img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=255)

        if random.random() < 0.3:
            w, h = img.size
            scale = random.uniform(0.85, 1.15)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            result = Image.new("L", (w, h), 255)
            paste_x = (w - new_w) // 2
            paste_y = (h - new_h) // 2
            result.paste(img, (paste_x, paste_y))
            img = result

        if random.random() < 0.4:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 15, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        if random.random() < 0.3:
            if random.random() < 0.5:
                img = img.filter(ImageFilter.MinFilter(3))
            else:
                img = img.filter(ImageFilter.MaxFilter(3))

        if random.random() < 0.4:
            arr = np.array(img, dtype=np.float32)
            alpha = random.uniform(0.7, 1.3)
            beta = random.uniform(-30, 30)
            arr = np.clip(alpha * arr + beta, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img


# -- Model --------------------------------------------------------------------

class MultiHeadCaptchaNet(nn.Module):
    """
    CNN backbone + 5 independent classification heads.
    Each head classifies one character position independently.
    """

    def __init__(self, num_classes, captcha_len):
        super().__init__()
        self.captcha_len = captcha_len

        self.cnn = nn.Sequential(
            # Block 1: (1, 64, 160) -> (32, 32, 80)
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (32, 32, 80) -> (64, 16, 40)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (64, 16, 40) -> (128, 8, 20)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: (128, 8, 20) -> (256, 4, 10)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5: (256, 4, 10) -> (256, 2, 5)
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Global average pooling per character position:
        # After CNN: (B, 256, 2, 5)
        # We treat the width dimension as character positions
        # Pool over the height: (B, 256, 5) -> each position gets a 256-d vector

        # 5 independent classification heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 * 2, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )
            for _ in range(captcha_len)
        ])

    def forward(self, x):
        # CNN: (B, 1, 64, 160) -> (B, 256, 2, 5)
        features = self.cnn(x)
        b, c, h, w = features.size()

        # Reshape: (B, 256, 2, 5) -> (B, 512, 5) by flattening height into channels
        features = features.reshape(b, c * h, w)  # (B, 512, 5)

        # Each head takes the features at its corresponding width position
        outputs = []
        for i, head in enumerate(self.heads):
            pos_feat = features[:, :, i]  # (B, 512)
            outputs.append(head(pos_feat))  # (B, num_classes)

        return outputs  # list of 5 tensors, each (B, num_classes)


# -- Training -----------------------------------------------------------------

def evaluate(model, dataloader, device):
    model.eval()
    correct_full = 0
    correct_chars = 0
    total_chars = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)

            # Decode predictions
            preds = torch.stack([out.argmax(dim=1) for out in outputs], dim=1)  # (B, 5)

            for pred, target in zip(preds, targets):
                total += 1
                pred_str = "".join(IDX_TO_CHAR[p.item()] for p in pred)
                target_str = "".join(IDX_TO_CHAR[t.item()] for t in target)

                if pred_str == target_str:
                    correct_full += 1
                for p, t in zip(pred, target):
                    total_chars += 1
                    if p == t:
                        correct_chars += 1

    full_acc = correct_full / total if total > 0 else 0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    return full_acc, char_acc


def load_split(split_dir):
    """Load image paths and labels from a split directory."""
    files = sorted(Path(split_dir).glob("*.png"))
    paths = [str(f) for f in files]
    labels = [f.stem for f in files]
    for label in labels:
        assert len(label) == CAPTCHA_LEN, f"Bad label length: {label}"
        for c in label:
            assert c in CHAR_TO_IDX, f"Unknown char '{c}' in label '{label}'"
    return paths, labels


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pre-split datasets
    train_paths, train_labels = load_split(TRAIN_DIR)
    val_paths, val_labels = load_split(VAL_DIR)
    test_paths, test_labels = load_split(TEST_DIR)

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    train_ds = CaptchaDataset(train_paths, train_labels, augment=True)
    val_ds = CaptchaDataset(val_paths, val_labels, augment=False)
    test_ds = CaptchaDataset(test_paths, test_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = MultiHeadCaptchaNet(NUM_CLASSES, CAPTCHA_LEN).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count * 4 / 1024 / 1024:.1f} MB float32)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = {"train_loss": [], "val_full_acc": [], "val_char_acc": []}
    best_acc = 0.0
    patience = 30
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)  # list of 5 x (B, num_classes)

            # Sum cross-entropy loss for each character position
            loss = sum(criterion(outputs[i], targets[:, i]) for i in range(CAPTCHA_LEN))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches

        if epoch % 5 == 0 or epoch == 1:
            # Evaluate on VALIDATION set for model selection
            full_acc, char_acc = evaluate(model, val_loader, device)
            history["train_loss"].append(avg_loss)
            history["val_full_acc"].append(full_acc)
            history["val_char_acc"].append(char_acc)

            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Val Full: {full_acc:.1%} | Val Char: {char_acc:.1%} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

            if full_acc > best_acc:
                best_acc = full_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), OUTPUT_DIR / "best_model_v2.pth")
                print(f"  -> New best model saved! ({best_acc:.1%})")
            else:
                epochs_no_improve += 5

            if epochs_no_improve >= patience:
                print(f"  Early stopping: no improvement for {patience} epochs")
                break

    # Final evaluation with best model on HELD-OUT TEST set
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model_v2.pth", weights_only=True))

    val_full, val_char = evaluate(model, val_loader, device)
    test_full, test_char = evaluate(model, test_loader, device)
    train_full, train_char = evaluate(model, train_loader, device)

    print(f"\n{'='*60}")
    print(f"BEST MODEL RESULTS (v2 - Multi-Head):")
    print(f"  Train accuracy: {train_full:.1%} full | {train_char:.1%} char")
    print(f"  Val accuracy:   {val_full:.1%} full | {val_char:.1%} char")
    print(f"  Test accuracy:  {test_full:.1%} full | {test_char:.1%} char")
    print(f"{'='*60}")
    print(f"\n  ** Test accuracy is the TRUE generalization metric **")
    if train_full - test_full > 0.2:
        print(f"  [!] Large train-test gap ({train_full - test_full:.1%}) suggests overfitting")

    # Save training plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_eval = list(range(1, len(history["train_loss"]) + 1))

    ax1.plot(epochs_eval, history["train_loss"], "b-")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Eval Step")
    ax1.set_ylabel("CE Loss")

    ax2.plot(epochs_eval, history["val_full_acc"], "g-", label="Val Full")
    ax2.plot(epochs_eval, history["val_char_acc"], "r-", label="Val Char")
    ax2.axhline(y=test_full, color="g", linestyle="--", alpha=0.5, label=f"Test Full ({test_full:.0%})")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Eval Step")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_plot_v2.png", dpi=100)
    print(f"Training plot saved to {OUTPUT_DIR / 'training_plot_v2.png'}")

    # Error analysis on held-out test set
    print(f"\nError Analysis (misclassified TEST samples):")
    model.eval()
    errors = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.stack([out.argmax(dim=1) for out in outputs], dim=1)

            for pred, target in zip(preds, targets):
                pred_str = "".join(IDX_TO_CHAR[p.item()] for p in pred)
                target_str = "".join(IDX_TO_CHAR[t.item()] for t in target)
                if pred_str != target_str:
                    diff = "".join("^" if p != t else " " for p, t in zip(pred, target))
                    errors.append((target_str, pred_str, diff))

    if errors:
        print(f"  Total errors: {len(errors)}/{len(test_paths)}")
        for true, pred, diff in errors[:30]:
            print(f"  True: {true} | Pred: {pred} | {diff}")
    else:
        print("  No errors on test set! Perfect accuracy.")

    # Export to ONNX
    print(f"\nExporting to ONNX...")
    model.eval()
    dummy = torch.randn(1, 1, IMG_HEIGHT, IMG_WIDTH).to(device)
    onnx_path = OUTPUT_DIR / "captcha_model_v2.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["image"],
        output_names=[f"char_{i}" for i in range(CAPTCHA_LEN)],
        dynamic_axes={"image": {0: "batch"}},
        opset_version=18,
    )
    onnx_size = os.path.getsize(onnx_path)
    data_file = str(onnx_path) + ".data"
    if os.path.exists(data_file):
        onnx_size += os.path.getsize(data_file)
    print(f"ONNX model saved: {onnx_path} ({onnx_size / 1024 / 1024:.1f} MB)")

    return test_full


if __name__ == "__main__":
    test_acc = train()
    if test_acc < 0.9:
        print(f"\n[!] Test accuracy {test_acc:.1%} < 90% target. Further tuning needed.")
    else:
        print(f"\n[OK] Test accuracy {test_acc:.1%} >= 90% target!")
