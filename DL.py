import os
import argparse
import time
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# Robust CSV reading
# =========================================================
def _read_df_robust(path: str) -> pd.DataFrame:
    """
    Reads a CSV-like file that might be comma-separated or whitespace/tab-separated.
    """
    try:
        df = pd.read_csv(path, header=None, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, header=None, sep=r"[\s,]+", engine="python")
    df = df.dropna(axis=1, how="all")  # drop fully empty columns
    return df


def read_matrix_csv_robust(path: str, expected_cols: int = None, dtype=np.uint8) -> np.ndarray:
    df = _read_df_robust(path)
    arr = df.values
    if expected_cols is not None and arr.shape[1] != expected_cols:
        raise ValueError(
            f"Unexpected column count in {path}. Got {arr.shape[1]}, expected {expected_cols}."
        )
    # Use float only later; keep uint8 in memory for speed
    return arr.astype(dtype, copy=False)


def _pick_best_label_column(df: pd.DataFrame) -> int:
    """
    If there are multiple columns, choose the one that best matches integer labels 0..9.
    """
    if df.shape[1] == 1:
        return 0

    best_col = None
    best_score = -1.0
    for c in range(df.shape[1]):
        col = pd.to_numeric(df.iloc[:, c], errors="coerce")
        valid = col.dropna()
        if valid.empty:
            continue
        vals = np.rint(valid.values).astype(np.int64)
        score = float(np.mean((vals >= 0) & (vals <= 9)))
        if score > best_score:
            best_score = score
            best_col = c

    if best_col is None:
        raise ValueError("Could not locate a valid label column.")
    return best_col


def load_images_and_labels_alignment_safe(
    img_path: str,
    lbl_path: str,
    expected_cols: int = 784
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alignment-safe loader:
    - Reads X (N,784)
    - Reads labels with possible extra cols / NaNs
    - If a label row is NaN/empty, drop the SAME row from X
    - Validates labels are 0..9
    """
    X = read_matrix_csv_robust(img_path, expected_cols=expected_cols, dtype=np.uint8)
    df_lbl = _read_df_robust(lbl_path)
    col_idx = _pick_best_label_column(df_lbl)

    s = pd.to_numeric(df_lbl.iloc[:, col_idx], errors="coerce")

    # Align mask length to X rows
    m = min(len(s), X.shape[0])
    X = X[:m]
    s = s.iloc[:m]

    mask = ~s.isna()
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"[WARN] {os.path.basename(lbl_path)} has {dropped} NaN/empty label rows. Dropping same rows from X.")

    X = X[mask.values]
    y = np.rint(s[mask].values).astype(np.int64)

    # Validate label range
    bad = np.where((y < 0) | (y > 9))[0]
    if bad.size > 0:
        idx = int(bad[0])
        raise ValueError(f"Label out of range in {lbl_path}. Example: y[{idx}]={y[idx]}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Alignment failed: X rows={X.shape[0]} vs y rows={y.shape[0]}")

    return X, y


# =========================================================
# Normalization (dataset-based)
# =========================================================
def compute_mean_std_from_uint8_flat(X_uint8: np.ndarray) -> Tuple[float, float]:
    """
    X_uint8: (N,784) in [0..255]
    mean/std computed after scaling to [0..1]
    """
    X = X_uint8.astype(np.float32) / 255.0
    mean = float(X.mean())
    std = float(X.std())
    if std < 1e-6:
        std = 1.0
    return mean, std


# =========================================================
# Dataset
# =========================================================
class AHDDSCSVDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, tfm=None):
        assert X.ndim == 2 and X.shape[1] == 784
        assert y.ndim == 1 and X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.tfm = tfm

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        img = self.X[idx].reshape(28, 28).astype(np.uint8)
        label = int(self.y[idx])
        if self.tfm is not None:
            img = self.tfm(img)
        else:
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        return img, label


# =========================================================
# Models (Padding + MaxPool + stronger Custom)
# =========================================================
class LeNetMaxPad(nn.Module):
    """
    LeNet-style but improved for 28x28:
    - Padding on convs to preserve borders
    - MaxPool (generally stronger for digits than AvgPool)
    """
    def __init__(self, num_classes=10, dropout=0.20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=False),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(32, 64, 5, padding=2, bias=False),  # 14 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.proj = None
        if in_ch != out_ch or stride != 1:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class CustomBestCNN(nn.Module):
    """
    Strong CNN for 28x28 digits:
    - padding preserved
    - residual blocks
    - MaxPool downsampling
    - global average pooling
    """
    def __init__(self, num_classes=10, drop=0.10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.b1 = ResidualBlock(32, 64, stride=1, drop=drop)
        self.p1 = nn.MaxPool2d(2)  # 28 -> 14

        self.b2 = ResidualBlock(64, 128, stride=1, drop=drop)
        self.p2 = nn.MaxPool2d(2)  # 14 -> 7

        self.b3 = ResidualBlock(128, 256, stride=1, drop=drop)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x); x = self.p1(x)
        x = self.b2(x); x = self.p2(x)
        x = self.b3(x)
        x = self.gap(x)
        x = self.head(x)
        return x


def build_alexnet_for_ahdd(num_classes=10) -> nn.Module:
    """
    AlexNet expects big input; we resize to 224x224 for AlexNet.
    Modify first conv for 1-channel and classifier for 10 classes.
    """
    m = models.alexnet(weights=None)
    m.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    return m


def get_model(name: str) -> nn.Module:
    name = name.lower().strip()
    if name == "lenet":
        return LeNetMaxPad(10, dropout=0.20)
    if name == "custom":
        return CustomBestCNN(10, drop=0.10)
    if name == "alexnet":
        return build_alexnet_for_ahdd(10)
    raise ValueError("model must be one of: lenet | alexnet | custom")


# =========================================================
# Transforms: normalization + padding augmentation
# =========================================================
def build_transforms_for_model(model_name: str, augment: bool, mean: float, std: float):
    # LeNet/Custom: Keep 28x28, but use padding+crop as strong regularization
    if model_name in ["lenet", "custom"]:
        train_list = [transforms.ToPILImage()]
        test_list = [transforms.ToPILImage()]

        if augment:
            train_list += [
                transforms.Pad(2, fill=0),
                transforms.RandomCrop(28),
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.10, 0.10),
                    scale=(0.90, 1.10),
                    shear=5
                ),
            ]

        train_list += [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ]
        test_list += [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ]
        return transforms.Compose(train_list), transforms.Compose(test_list)

    # AlexNet: Resize to 224x224
    if model_name == "alexnet":
        train_list = [transforms.ToPILImage(), transforms.Resize((224, 224))]
        test_list = [transforms.ToPILImage(), transforms.Resize((224, 224))]

        if augment:
            train_list += [
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.08, 0.08),
                    scale=(0.95, 1.05)
                )
            ]

        train_list += [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ]
        test_list += [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ]
        return transforms.Compose(train_list), transforms.Compose(test_list)

    raise ValueError("Unknown model_name for transforms")


# =========================================================
# Metrics & Output
# =========================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_true, all_pred = [], []
    correct, total = 0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

        all_true.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    acc = correct / max(total, 1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    return acc, macro_f1, cm, y_true, y_pred


def save_confusion_matrix(cm: np.ndarray, out_png: str, out_csv: str, title: str) -> None:
    classes = [str(i) for i in range(10)]
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_csv, index=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(10)
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)

    thresh = cm.max() * 0.6 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================================================
# Early stopping
# =========================================================
@dataclass
class EarlyStopper:
    patience: int = 10
    min_delta: float = 1e-4
    best_score: float = -1e9
    best_epoch: int = -1
    counter: int = 0

    def step(self, score: float, epoch: int) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# =========================================================
# Training (AdamW + OneCycleLR + optional label smoothing)
# =========================================================
def train_one_epoch(model, loader, device, optimizer, scheduler, criterion, use_amp: bool):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def run_model(
    model_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    out_dir: str,
    device: str,
    max_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    augment: bool,
    seed: int,
    mean: float,
    std: float,
    label_smoothing: float,
    use_amp: bool
) -> Dict:
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # stratified val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
    )

    train_tfm, test_tfm = build_transforms_for_model(model_name, augment=augment, mean=mean, std=std)

    ds_tr = AHDDSCSVDataset(X_tr, y_tr, tfm=train_tfm)
    ds_val = AHDDSCSVDataset(X_val, y_val, tfm=test_tfm)
    ds_te = AHDDSCSVDataset(X_test, y_test, tfm=test_tfm)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.startswith("cuda"))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.startswith("cuda"))
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.startswith("cuda"))

    model = get_model(model_name).to(device)

    # Loss (label smoothing helps stability; keep small)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer: AdamW (very strong default)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler: OneCycleLR (often best for CNN training)
    steps_per_epoch = max(1, len(dl_tr))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=200.0
    )

    stopper = EarlyStopper(patience=patience, min_delta=1e-4)

    best_state = None
    best_epoch = -1
    best_val_f1 = -1.0
    history: List[Dict] = []

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, device, optimizer, scheduler, criterion, use_amp=use_amp)
        val_acc, val_f1, _, _, _ = evaluate(model, dl_val, device)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(f"[{model_name.upper()}] Epoch {epoch:03d}/{max_epochs} | "
              f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.4f} | "
              f"ValAcc={val_acc:.4f} ValMacroF1={val_f1:.4f} | LR={optimizer.param_groups[0]['lr']:.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if stopper.step(val_f1, epoch):
            print(f"[{model_name.upper()}] Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    te_acc, te_f1, cm, y_true, y_pred = evaluate(model, dl_te, device)
    elapsed = time.time() - t0

    cm_png = os.path.join(out_dir, f"{model_name}_confusion_matrix.png")
    cm_csv = os.path.join(out_dir, f"{model_name}_confusion_matrix.csv")
    save_confusion_matrix(cm, cm_png, cm_csv, f"{model_name.upper()} Confusion Matrix (Test)")

    rep_txt = os.path.join(out_dir, f"{model_name}_classification_report.txt")
    with open(rep_txt, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Best epoch (by Val Macro-F1): {best_epoch}\n")
        f.write(f"Test Accuracy: {te_acc:.6f}\n")
        f.write(f"Test Macro-F1: {te_f1:.6f}\n")
        f.write(f"Elapsed seconds: {elapsed:.2f}\n\n")
        f.write(classification_report(y_true, y_pred, digits=4))

    hist_csv = os.path.join(out_dir, f"{model_name}_history.csv")
    pd.DataFrame(history).to_csv(hist_csv, index=False)

    model_path = os.path.join(out_dir, f"{model_name}_best.pt")
    torch.save(model.state_dict(), model_path)

    return {
        "model": model_name,
        "best_epoch": best_epoch,
        "test_accuracy": float(te_acc),
        "test_macro_f1": float(te_f1),
        "confusion_matrix_png": cm_png,
        "confusion_matrix_csv": cm_csv,
        "classification_report_txt": rep_txt,
        "history_csv": hist_csv,
        "best_model_pt": model_path,
        "elapsed_sec": float(elapsed),
    }


# =========================================================
# Main
# =========================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, "AHDD")
    default_out_dir = os.path.join(script_dir, "ahdd_results_best")

    parser = argparse.ArgumentParser(
        description="AHDD CSV: LeNet vs AlexNet vs Custom with best-practice training, normalization, padding, maxpool, early stopping."
    )
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--out_dir", type=str, default=default_out_dir)

    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.003)  # strong default for digits CNNs
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.02)

    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (auto if not set)")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only)")

    parser.add_argument("--skip_alexnet", action="store_true", help="Skip AlexNet (recommended on CPU)")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    augment = not args.no_augment

    os.makedirs(args.out_dir, exist_ok=True)

    train_img_path = os.path.join(args.data_dir, "csvTrainImages 60k x 784.csv")
    train_lbl_path = os.path.join(args.data_dir, "csvTrainLabel 60k x 1.csv")
    test_img_path  = os.path.join(args.data_dir, "csvTestImages 10k x 784.csv")
    test_lbl_path  = os.path.join(args.data_dir, "csvTestLabel 10k x 1.csv")

    for p in [train_img_path, train_lbl_path, test_img_path, test_lbl_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    print(f"Device: {device}")
    if device == "cpu":
        print("[INFO] CPU training is supported, but AlexNet will be slow. Use --skip_alexnet on CPU.")

    print(f"Reading AHDD from: {args.data_dir}")
    print("Loading CSVs (this may take time depending on disk speed)...")

    X_train, y_train = load_images_and_labels_alignment_safe(train_img_path, train_lbl_path)
    X_test,  y_test  = load_images_and_labels_alignment_safe(test_img_path, test_lbl_path)

    # Sanity print
    print(f"Train: X={X_train.shape}, y={y_train.shape} | Test: X={X_test.shape}, y={y_test.shape}")
    print("Train label distribution:", np.bincount(y_train, minlength=10))
    print("Test  label distribution:", np.bincount(y_test, minlength=10))

    mean, std = compute_mean_std_from_uint8_flat(X_train)
    print(f"[INFO] Normalization (from train): mean={mean:.6f}, std={std:.6f}")

    print(f"Augmentation: {augment} | max_epochs={args.max_epochs} | batch_size={args.batch_size} | "
          f"lr={args.lr} | wd={args.weight_decay} | patience={args.patience} | label_smoothing={args.label_smoothing}\n")

    # Which models to run
    model_list = ["lenet", "custom"]
    if not args.skip_alexnet:
        model_list.insert(1, "alexnet")

    results = []
    for model_name in model_list:
        model_out = os.path.join(args.out_dir, model_name)
        res = run_model(
            model_name=model_name,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            out_dir=model_out,
            device=device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            augment=augment,
            seed=args.seed,
            mean=mean,
            std=std,
            label_smoothing=args.label_smoothing,
            use_amp=args.amp
        )
        results.append(res)
        print(f"Saved {model_name} outputs to: {model_out}\n")

    df = pd.DataFrame(results).sort_values(by="test_macro_f1", ascending=False)
    summary_path = os.path.join(args.out_dir, "results_summary.csv")
    df.to_csv(summary_path, index=False)

    print("=== FINAL COMPARISON (sorted by Test Macro-F1) ===")
    print(df[["model", "best_epoch", "test_accuracy", "test_macro_f1", "elapsed_sec"]].to_string(index=False))
    print("\nSummary saved to:", summary_path)
    print("Best model:", df.iloc[0]["model"])


if __name__ == "__main__":
    main()
