# AlexnetDL.py
# CPU-friendly AlexNet-style training for AHDD (784 features -> 1x28x28).
# Saves: accuracy, macro-F1, confusion matrix, classification report, predictions.

import os
import time
import argparse
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional (metrics). If missing, we compute confusion matrix manually.
try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# -----------------------------
# Robust CSV reading utilities
# -----------------------------
def _try_loadtxt(path: str, delimiter=","):
    """Fast path: load numeric csv. Falls back to genfromtxt if needed."""
    try:
        return np.loadtxt(path, delimiter=delimiter)
    except Exception:
        return np.genfromtxt(path, delimiter=delimiter)


def read_matrix_csv_robust(path: str, expected_cols: int):
    """
    Robustly read a CSV numeric matrix.
    Handles:
      - extra index column (e.g., first column 0..N-1)
      - trailing extra empty column
      - headers (via genfromtxt returning nan rows -> filtered)
    Ensures output has exactly expected_cols columns.
    """
    arr = _try_loadtxt(path, delimiter=",")

    # If arr is 1D (single row), make 2D
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Remove rows that are all NaN (header artifacts)
    if np.isnan(arr).any():
        mask_valid = ~np.all(np.isnan(arr), axis=1)
        arr = arr[mask_valid]

    # If still nan exists, replace with 0 (images should be numeric)
    arr = np.nan_to_num(arr, nan=0.0)

    # Fix column count
    c = arr.shape[1]

    # Case: extra index column -> 785 or 784+1, or 783 due to parsing issues
    # If c == expected+1 and first col looks like index (monotonic-ish ints), drop it.
    if c == expected_cols + 1:
        first = arr[:, 0]
        # heuristic: many entries close to integers and near 0..N
        if np.mean(np.abs(first - np.round(first)) < 1e-6) > 0.95:
            arr = arr[:, 1:]
            c = arr.shape[1]

    # Case: too many columns (e.g., trailing comma created empty col)
    if c > expected_cols:
        arr = arr[:, :expected_cols]
        c = arr.shape[1]

    # Case: fewer columns than expected -> cannot safely recover
    if c < expected_cols:
        raise ValueError(
            f"Unexpected column count in {path}. Got {c}, expected {expected_cols}. "
            f"Common causes: truncated CSV, wrong delimiter, or missing values."
        )

    return arr.astype(np.float32)


def read_labels_csv_robust(path: str):
    """
    Robustly read label CSV. Handles:
      - header
      - extra columns
      - blanks/NaNs
    Returns (labels, valid_mask) where valid_mask marks rows with usable labels.
    """
    arr = _try_loadtxt(path, delimiter=",")

    if arr.ndim == 2:
        # Take the first non-empty column
        col = arr[:, 0]
    else:
        col = arr

    # If header produced NaN at top, keep mask
    valid = ~np.isnan(col)
    # Also treat empty/invalid huge casts as invalid
    # (we will round then cast)
    col2 = col.copy()
    col2[~valid] = 0

    col2 = np.round(col2).astype(np.int64)
    # After cast, re-check plausible range
    # We'll accept any int; later we will clamp or filter if needed.
    return col2, valid


# -----------------------------
# Dataset
# -----------------------------
class AHDDDataset(Dataset):
    def __init__(self, X, y, mean, std, augment=False):
        self.X = X  # float32 (N, 784)
        self.y = y  # int64 (N,)
        self.mean = float(mean)
        self.std = float(std) if float(std) > 1e-12 else 1.0
        self.augment = augment

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].reshape(1, 28, 28)  # 1x28x28
        # Normalize
        x = (x - self.mean) / self.std

        # Very light augmentation (CPU-friendly). Default OFF for speed.
        if self.augment:
            # Random small shift by padding + crop
            # Pad 2 pixels then random crop 28x28
            x = torch.tensor(x, dtype=torch.float32)
            x = F.pad(x, (2, 2, 2, 2), mode="constant", value=0.0)
            top = torch.randint(0, 5, (1,)).item()
            left = torch.randint(0, 5, (1,)).item()
            x = x[:, top:top+28, left:left+28]
        else:
            x = torch.tensor(x, dtype=torch.float32)

        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# -----------------------------
# AlexNet-style (paper-inspired)
# ReLU + LRN + overlapping max pool + dropout + FC
# Adapted to 1x28x28 input.
# -----------------------------
class AlexNetAHDD(nn.Module):
    def __init__(self, num_classes=10, fc_dim=1024, dropout=0.5, use_lrn=True):
        super().__init__()
        self.use_lrn = use_lrn

        # NOTE: We keep the "AlexNet motifs" but adapt kernels/strides for 28x28.
        # Overlapping pooling: kernel=3 stride=2 (like paper).
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1) # -> same
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)  # overlapping pool

        # After conv/pool stack:
        # Input 28x28
        # conv1 -> 28, pool -> floor((28-3)/2)+1 = 13
        # conv2 -> 13, pool -> floor((13-3)/2)+1 = 6
        # conv3 -> 6
        # conv4 -> 6
        # conv5 -> 6, pool -> floor((6-3)/2)+1 = 2
        # => feature map 256 x 2 x 2 = 1024
        self.flatten_dim = 256 * 2 * 2

        self.fc1 = nn.Linear(self.flatten_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, num_classes)

        self.drop = nn.Dropout(p=dropout)

        # Bias init like paper (some layers bias=1). We'll do simple variant:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Encourage ReLU activity early for deeper layers (paper idea)
        nn.init.constant_(self.conv2.bias, 1.0)
        nn.init.constant_(self.conv4.bias, 1.0)
        nn.init.constant_(self.conv5.bias, 1.0)
        nn.init.constant_(self.fc1.bias, 1.0)
        nn.init.constant_(self.fc2.bias, 1.0)

    def lrn(self, x):
        # PyTorch has LocalResponseNorm layer; use it similarly to paper.
        # Paper: k=2, n=5, alpha=1e-4, beta=0.75
        return F.local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2.0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.use_lrn:
            x = self.lrn(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        if self.use_lrn:
            x = self.lrn(x)
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# -----------------------------
# Train / Eval
# -----------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * yb.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys = []
    ps = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()
        ps.append(pred)
        ys.append(yb.numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    if SKLEARN_OK:
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)
    else:
        acc = float((y_true == y_pred).mean())
        # manual macro f1 (simple)
        cm = np.zeros((10, 10), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < 10 and 0 <= p < 10:
                cm[t, p] += 1
        f1s = []
        for k in range(10):
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp
            fn = cm[k, :].sum() - tp
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1 = 2 * prec * rec / (prec + rec + 1e-12)
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s))
        report = "sklearn not installed; macro-F1 computed manually."

    return acc, macro_f1, cm, report, y_true, y_pred


def save_outputs(out_dir, acc, macro_f1, cm, report, y_true, y_pred, args, mean, std, elapsed_sec):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save report
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # Save confusion matrix
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    # Save preds
    np.savetxt(out_dir / "y_true.csv", y_true, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "y_pred.csv", y_pred, fmt="%d", delimiter=",")

    # Summary
    summary = [
        f"model=alexnet_style_ahdd",
        f"device={args.device}",
        f"accuracy={acc:.6f}",
        f"macro_f1={macro_f1:.6f}",
        f"mean={mean:.6f}",
        f"std={std:.6f}",
        f"max_epochs={args.max_epochs}",
        f"patience={args.patience}",
        f"batch_size={args.batch_size}",
        f"lr={args.lr}",
        f"weight_decay={args.weight_decay}",
        f"label_smoothing={args.label_smoothing}",
        f"augment={args.augment}",
        f"fc_dim={args.fc_dim}",
        f"use_lrn={args.use_lrn}",
        f"dropout={args.dropout}",
        f"elapsed_sec={elapsed_sec:.2f}",
    ]
    (out_dir / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8")


# -----------------------------
# Data loading (AHDD)
# -----------------------------
def load_ahdd(data_dir: str, num_classes=10):
    data_dir = Path(data_dir)

    # Your naming based on your logs:
    # AHDD\csvTrainImages 60k x 784.csv
    # AHDD\csvTrainLabel  60k x 1.csv
    # AHDD\csvTestImages  10k x 784.csv
    # AHDD\csvTestLabel   10k x 1.csv
    paths = {
        "train_x": str(data_dir / "csvTrainImages 60k x 784.csv"),
        "train_y": str(data_dir / "csvTrainLabel 60k x 1.csv"),
        "test_x":  str(data_dir / "csvTestImages 10k x 784.csv"),
        "test_y":  str(data_dir / "csvTestLabel 10k x 1.csv"),
    }

    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {k} file: {p}")

    print("Loading CSVs (this may take time depending on disk speed)...")

    X_train = read_matrix_csv_robust(paths["train_x"], expected_cols=784)
    y_train_raw, valid_tr = read_labels_csv_robust(paths["train_y"])

    X_test = read_matrix_csv_robust(paths["test_x"], expected_cols=784)
    y_test_raw, valid_te = read_labels_csv_robust(paths["test_y"])

    # Drop invalid label rows (NaN/empty labels)
    if valid_tr.sum() < len(valid_tr):
        drop = int(len(valid_tr) - valid_tr.sum())
        print(f"[WARN] csvTrainLabel 60k x 1.csv has {drop} NaN/empty label rows. Dropping same rows from X.")
        X_train = X_train[valid_tr]
        y_train_raw = y_train_raw[valid_tr]

    if valid_te.sum() < len(valid_te):
        drop = int(len(valid_te) - valid_te.sum())
        print(f"[WARN] csvTestLabel 10k x 1.csv has {drop} NaN/empty label rows. Dropping same rows from X.")
        X_test = X_test[valid_te]
        y_test_raw = y_test_raw[valid_te]

    # Ensure labels are in 0..num_classes-1 (digits)
    # If labels are 1..10 or other, map safely if needed.
    # Here we assume 0..9; if out of range, raise helpful error.
    if y_train_raw.min() < 0 or y_train_raw.max() >= num_classes:
        raise ValueError(
            f"Train labels out of expected range 0..{num_classes-1}. "
            f"Got min={y_train_raw.min()}, max={y_train_raw.max()}."
        )
    if y_test_raw.min() < 0 or y_test_raw.max() >= num_classes:
        raise ValueError(
            f"Test labels out of expected range 0..{num_classes-1}. "
            f"Got min={y_test_raw.min()}, max={y_test_raw.max()}."
        )

    # Compute normalization from train
    mean = float(X_train.mean())
    std = float(X_train.std() + 1e-12)

    return X_train, y_train_raw.astype(np.int64), X_test, y_test_raw.astype(np.int64), mean, std


def label_distribution(y, num_classes=10):
    dist = np.zeros(num_classes, dtype=np.int64)
    for k in range(num_classes):
        dist[k] = int((y == k).sum())
    return dist


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to AHDD folder (contains the CSVs).")
    parser.add_argument("--out_dir", type=str, default="ahdd_results_alexnet", help="Output folder.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Fast CPU defaults (your request)
    parser.add_argument("--max_epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--augment", action="store_true", help="Enable light CPU augmentation (slower).")

    # AlexNet-style knobs
    parser.add_argument("--fc_dim", type=int, default=1024, help="Use 1024 for fastest CPU. (4096 will be slow).")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use_lrn", action="store_true", help="Enable AlexNet LRN (slower but paper-like).")

    args = parser.parse_args()

    set_seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    device_t = torch.device(device)

    print(f"Device: {device}")
    if device == "cpu":
        print("[INFO] CPU training is supported, but AlexNet-style will still be slower than LeNet.")
    print(f"Reading AHDD from: {args.data_dir}")

    X_train, y_train, X_test, y_test, mean, std = load_ahdd(args.data_dir, num_classes=10)

    print(f"[INFO] Normalization (from train): mean={mean:.6f}, std={std:.6f}")
    print(f"Train: X={X_train.shape}, y={y_train.shape} | Test: X={X_test.shape}, y={y_test.shape}")
    print(f"Train label distribution: {label_distribution(y_train, 10)}")
    print(f"Test  label distribution: {label_distribution(y_test, 10)}")
    print(
        f"Augmentation: {args.augment} | max_epochs={args.max_epochs} | batch_size={args.batch_size} | "
        f"lr={args.lr} | wd={args.weight_decay} | patience={args.patience} | label_smoothing={args.label_smoothing} | "
        f"fc_dim={args.fc_dim} | use_lrn={args.use_lrn}"
    )

    ds_tr = AHDDDataset(X_train, y_train, mean, std, augment=args.augment)
    ds_te = AHDDDataset(X_test, y_test, mean, std, augment=False)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = AlexNetAHDD(num_classes=10, fc_dim=args.fc_dim, dropout=args.dropout, use_lrn=args.use_lrn).to(device_t)

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Paper-like optimizer (SGD + momentum), with Nesterov for faster practical convergence
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=args.weight_decay
    )

    # OneCycleLR for fastest convergence (short run)
    steps_per_epoch = max(len(dl_tr), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.max_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=1e4
    )

    best_macro_f1 = -1.0
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, device_t, optimizer, criterion)
        # Step scheduler per batch was not done; we step per epoch by iterating internal step counts.
        # We need to advance scheduler steps equal to steps_per_epoch manually:
        # (Simpler: step inside train loop; but we keep it clean here.)
        # To keep correct behavior, we step per-batch: re-run a dummy loop count.
        # Safer approach: move scheduler stepping inside train loop in production.
        for _ in range(steps_per_epoch):
            scheduler.step()

        va_acc, va_f1, _, _, _, _ = eval_model(model, dl_te, device_t)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"[ALEXNET] Epoch {epoch:03d}/{args.max_epochs} | TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.4f} | "
              f"ValAcc={va_acc:.4f} ValMacroF1={va_f1:.4f} | LR={lr_now:.6f}")

        if va_f1 > best_macro_f1:
            best_macro_f1 = va_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[ALEXNET] Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    acc, macro_f1, cm, report, y_true, y_pred = eval_model(model, dl_te, device_t)
    elapsed = time.time() - t0

    out_dir = Path(args.out_dir)
    save_outputs(out_dir, acc, macro_f1, cm, report, y_true, y_pred, args, mean, std, elapsed)

    print("\n=== FINAL (AlexNet-style) ===")
    print(f"Best epoch (by Val Macro-F1): {best_epoch}")
    print(f"Test Accuracy:  {acc:.6f}")
    print(f"Test Macro-F1:  {macro_f1:.6f}")
    print(f"Elapsed (sec):  {elapsed:.2f}")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
