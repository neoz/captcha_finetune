"""
CRNN + CTC Captcha Solver - Phase 1 Training
Architecture: CNN (custom lightweight) + BiLSTM + CTC Loss
"""

import os
import random
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

# ── Config ──────────────────────────────────────────────────────────────────

CHARS = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 reserved for CTC blank
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank token

IMG_HEIGHT = 64
IMG_WIDTH = 160
CAPTCHA_LEN = 5
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-3
TRAIN_RATIO = 0.8
SEED = 42

DATA_DIR = Path("ProcessedCaptchas")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Dataset ─────────────────────────────────────────────────────────────────

class CaptchaDataset(Dataset):
    def __init__(self, image_paths, labels, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")  # grayscale
        label = self.labels[idx]

        if self.augment:
            img = self._augment(img)

        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        # Normalize to [-1, 1]
        img = (img - 0.5) / 0.5
        img = torch.tensor(img).unsqueeze(0)  # (1, H, W)

        target = torch.tensor([CHAR_TO_IDX[c] for c in label], dtype=torch.long)
        return img, target

    def _augment(self, img):
        # Random rotation (-15 to 15 degrees)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=255, expand=False)

        # Random affine-like shift
        if random.random() < 0.5:
            dx = random.randint(-8, 8)
            dy = random.randint(-4, 4)
            img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=255)

        # Random scale
        if random.random() < 0.3:
            w, h = img.size
            scale = random.uniform(0.85, 1.15)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            # Center crop/pad back to original size
            result = Image.new("L", (w, h), 255)
            paste_x = (w - new_w) // 2
            paste_y = (h - new_h) // 2
            result.paste(img, (paste_x, paste_y))
            img = result

        # Random noise
        if random.random() < 0.4:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 15, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        # Random blur
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Random erosion/dilation
        if random.random() < 0.3:
            if random.random() < 0.5:
                img = img.filter(ImageFilter.MinFilter(3))  # erosion
            else:
                img = img.filter(ImageFilter.MaxFilter(3))  # dilation

        # Random brightness/contrast
        if random.random() < 0.4:
            arr = np.array(img, dtype=np.float32)
            alpha = random.uniform(0.7, 1.3)  # contrast
            beta = random.uniform(-30, 30)     # brightness
            arr = np.clip(alpha * arr + beta, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img


# ── Model ───────────────────────────────────────────────────────────────────

class CRNN(nn.Module):
    """
    Lightweight CRNN for captcha recognition.
    CNN extracts features -> BiLSTM captures sequence -> Linear projects to classes.
    """

    def __init__(self, num_classes):
        super().__init__()

        # CNN feature extractor - lightweight custom backbone
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

            # Block 5: (256, 4, 10) -> (256, 1, 10) - collapse height
            nn.Conv2d(256, 256, (4, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # BiLSTM sequence model
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)

        # Output projection
        self.fc = nn.Linear(256, num_classes)  # 128*2 for bidirectional

    def forward(self, x):
        # CNN: (B, 1, 64, 160) -> (B, 256, 1, 10)
        conv = self.cnn(x)

        # Reshape for RNN: (B, 256, 1, 10) -> (B, 10, 256)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)        # (B, 256, 10)
        conv = conv.permute(0, 2, 1)  # (B, 10, 256)

        # BiLSTM: (B, 10, 256) -> (B, 10, 256)
        rnn_out, _ = self.rnn(conv)

        # Project to classes: (B, 10, 256) -> (B, 10, num_classes)
        output = self.fc(rnn_out)

        # CTC expects (T, B, C)
        output = output.permute(1, 0, 2)
        return output


# ── Training ────────────────────────────────────────────────────────────────

def decode_predictions(output):
    """Greedy CTC decoding."""
    _, max_indices = torch.max(output, dim=2)  # (T, B)
    max_indices = max_indices.permute(1, 0)     # (B, T)

    decoded = []
    for seq in max_indices:
        chars = []
        prev = -1
        for idx in seq:
            idx = idx.item()
            if idx != 0 and idx != prev:  # not blank and not repeat
                if idx in IDX_TO_CHAR:
                    chars.append(IDX_TO_CHAR[idx])
            prev = idx
        decoded.append("".join(chars))
    return decoded


def evaluate(model, dataloader, device):
    model.eval()
    correct_full = 0
    correct_chars = 0
    total_chars = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            output = model(images)
            preds = decode_predictions(output)

            for pred, target in zip(preds, targets):
                target_str = "".join(IDX_TO_CHAR.get(t.item(), "?") for t in target)
                total += 1
                if pred == target_str:
                    correct_full += 1
                # Per-character accuracy
                for p, t in zip(pred.ljust(CAPTCHA_LEN), target_str):
                    total_chars += 1
                    if p == t:
                        correct_chars += 1
                # Account for length mismatch
                if len(pred) != len(target_str):
                    total_chars += abs(len(pred) - len(target_str))

    full_acc = correct_full / total if total > 0 else 0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    return full_acc, char_acc


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    all_files = sorted(DATA_DIR.glob("*.png"))
    all_paths = [str(f) for f in all_files]
    all_labels = [f.stem for f in all_files]

    # Validate labels
    for label in all_labels:
        assert len(label) == CAPTCHA_LEN, f"Bad label length: {label}"
        for c in label:
            assert c in CHAR_TO_IDX, f"Unknown char '{c}' in label '{label}'"

    # Split
    indices = list(range(len(all_paths)))
    random.shuffle(indices)
    split = int(len(indices) * TRAIN_RATIO)
    train_idx, test_idx = indices[:split], indices[split:]

    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    test_paths = [all_paths[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]

    print(f"Train: {len(train_paths)}, Test: {len(test_paths)}")

    train_ds = CaptchaDataset(train_paths, train_labels, augment=True)
    test_ds = CaptchaDataset(test_paths, test_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = CRNN(NUM_CLASSES).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count * 4 / 1024 / 1024:.1f} MB float32)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    # Training loop
    history = {"train_loss": [], "test_full_acc": [], "test_char_acc": []}
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets_flat = targets.reshape(-1).to(device)

            output = model(images)  # (T, B, C)
            T, B = output.size(0), output.size(1)

            input_lengths = torch.full((B,), T, dtype=torch.long)
            target_lengths = torch.full((B,), CAPTCHA_LEN, dtype=torch.long)

            log_probs = F.log_softmax(output, dim=2)
            loss = ctc_loss(log_probs, targets_flat, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            full_acc, char_acc = evaluate(model, test_loader, device)
            history["train_loss"].append(avg_loss)
            history["test_full_acc"].append(full_acc)
            history["test_char_acc"].append(char_acc)

            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Full Acc: {full_acc:.1%} | Char Acc: {char_acc:.1%} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

            if full_acc > best_acc:
                best_acc = full_acc
                torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
                print(f"  -> New best model saved! ({best_acc:.1%})")

    # Final evaluation
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth", weights_only=True))
    full_acc, char_acc = evaluate(model, test_loader, device)
    print(f"\n{'='*60}")
    print(f"BEST MODEL RESULTS:")
    print(f"  Full captcha accuracy: {full_acc:.1%}")
    print(f"  Per-character accuracy: {char_acc:.1%}")
    print(f"{'='*60}")

    # Save training plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_eval = list(range(1, len(history["train_loss"]) + 1))

    ax1.plot(epochs_eval, history["train_loss"], "b-")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Eval Step")
    ax1.set_ylabel("CTC Loss")

    ax2.plot(epochs_eval, history["test_full_acc"], "g-", label="Full Captcha")
    ax2.plot(epochs_eval, history["test_char_acc"], "r-", label="Per-Char")
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Eval Step")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_plot.png", dpi=100)
    print(f"Training plot saved to {OUTPUT_DIR / 'training_plot.png'}")

    # Error analysis: show misclassified examples
    print(f"\nError Analysis (misclassified test samples):")
    model.eval()
    errors = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            output = model(images)
            preds = decode_predictions(output)
            for pred, target in zip(preds, targets):
                target_str = "".join(IDX_TO_CHAR.get(t.item(), "?") for t in target)
                if pred != target_str:
                    errors.append((target_str, pred))

    if errors:
        print(f"  Total errors: {len(errors)}/{len(test_paths)}")
        for true, pred in errors[:20]:
            diff = "".join("^" if p != t else " " for p, t in zip(pred.ljust(CAPTCHA_LEN), true))
            print(f"  True: {true} | Pred: {pred.ljust(CAPTCHA_LEN)} | {diff}")
    else:
        print("  No errors! Perfect accuracy.")

    # Export to ONNX
    print(f"\nExporting to ONNX...")
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

    return best_acc


if __name__ == "__main__":
    best = train()
    if best < 0.9:
        print(f"\n[!] Accuracy {best:.1%} < 90% target. Phase 2 (synthetic data) recommended.")
    else:
        print(f"\n[OK] Accuracy {best:.1%} >= 90% target. Phase 1 complete!")
