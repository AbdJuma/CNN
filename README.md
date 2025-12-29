# Arabic Handwritten Digits Classification (AHDD/MADBase) — CNN Baselines + JumaNet + SVM Feature Pipeline

**Course:** Deep Learning  
**Author:** Abdurrahman Juma  
**Institution:** Birzeit University  

---

## Overview

This repository implements and compares three convolutional neural network (CNN) approaches for Arabic handwritten digit recognition using the **Arabic Handwritten Digits Dataset (AHDD)** (commonly referenced as **MADBase** in the literature). The project includes:

- **LeNet** baseline (digit-scale CNN baseline)  
- **AlexNet** baseline (adapted to grayscale digits)  
- **JumaNet** (custom residual CNN designed for 28×28 digits)  
- **Feature extraction code** (to produce embeddings/features)  
- **SVM kernel source code** (custom kernel + documented modifications) for classification over extracted features  

Evaluation is reported using **Accuracy**, **Macro-F1**, and the **confusion matrix** for the best classifier.

---

## Repository Structure

```
AHDD_Digit_Classification/
├── data/
│   ├── AHDD/                           # Dataset files (CSV images/labels)
│   └── README_DATA.md                  # Notes on dataset placement
├── outputs/
│   ├── lenet/                          # Saved metrics, CM, checkpoints
│   ├── alexnet/
│   ├── jumanet/
│   └── svm/
├── src/
│   ├── train_cnn.py                    # CNN training (LeNet/AlexNet/JumaNet)
│   ├── feature_extract.py              # Feature extraction from trained CNN
│   ├── svm_kernels.py                  # Custom SVM kernels (documented)
│   ├── svm_train.py                    # Train/evaluate SVM on extracted features
│   ├── dataset.py                      # Alignment-safe CSV loading + transforms
│   ├── models.py                       # Architectures (LeNet/AlexNet/JumaNet)
│   └── utils.py                        # Metrics, plotting, reproducibility helpers
├── requirements.txt
└── README.md
```

---

## Installation

### 1) Create environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup (AHDD / MADBase)

Place the dataset CSV files in:

```
data/AHDD/
  csvTrainImages 60k x 784.csv
  csvTrainLabel  60k x 1.csv
  csvTestImages  10k x 784.csv
  csvTestLabel   10k x 1.csv
```

The loader is **alignment-safe**:
- labels are parsed robustly from CSV (handles extra columns and NaNs),
- rows with missing labels are dropped, and the corresponding image rows are dropped as well,
- label range is validated (0–9).

---

## 1) Train CNN Models (LeNet / AlexNet / JumaNet)

### Train JumaNet (recommended)

```bash
python src/train_cnn.py   --model jumanet   --data_dir data/AHDD   --out_dir outputs/jumanet   --max_epochs 90   --batch_size 128   --lr 0.003   --weight_decay 0.0005   --patience 10
```

### Train LeNet

```bash
python src/train_cnn.py   --model lenet   --data_dir data/AHDD   --out_dir outputs/lenet   --max_epochs 90   --batch_size 128   --lr 0.003   --weight_decay 0.0005   --patience 10
```

### Train AlexNet (optional; heavy on CPU)

```bash
python src/train_cnn.py   --model alexnet   --data_dir data/AHDD   --out_dir outputs/alexnet   --max_epochs 20   --batch_size 128   --lr 0.0001   --weight_decay 0.0005   --patience 5
```

Notes:
- **LeNet/JumaNet** operate on **28×28**.
- **AlexNet** is adapted to grayscale and typically uses **224×224 resize** to match original assumptions.
- Training uses **AdamW + OneCycleLR** and early stopping on **Val Macro-F1**.

---

## 2) Feature Extraction (Required)

Some assignments require explicit “feature extraction code”. This repository provides a feature extractor that:

- Loads a trained CNN checkpoint,
- Removes the final classification layer,
- Exports a feature vector for each sample (train/test),
- Saves features as `.npy` (and optional `.csv`) for downstream ML (e.g., SVM).

### Example: extract features from JumaNet

```bash
python src/feature_extract.py   --model jumanet   --checkpoint outputs/jumanet/jumanet_best.pt   --data_dir data/AHDD   --out_dir outputs/features/jumanet   --split test
```

Outputs:
```
outputs/features/jumanet/
  X_test_features.npy
  y_test.npy
  meta.json
```

You can repeat with `--split train` to generate training features.

---

## 3) Train SVM Using a Custom Kernel (Required)

This project includes **custom SVM kernel source code** plus a training script that:

- Loads extracted features (`X_train_features.npy`, `y_train.npy`),
- Applies a kernel function `K(x_i, x_j)` defined in `src/svm_kernels.py`,
- Trains an SVM classifier,
- Evaluates Accuracy / Macro-F1 and exports confusion matrix.

### Train SVM with custom kernel

```bash
python src/svm_train.py   --train_feat outputs/features/jumanet/X_train_features.npy   --train_lbl  outputs/features/jumanet/y_train.npy   --test_feat  outputs/features/jumanet/X_test_features.npy   --test_lbl   outputs/features/jumanet/y_test.npy   --kernel rbf_modified   --C 10.0   --gamma 0.05   --out_dir outputs/svm
```

---

## SVM Kernel Source Code (What is “modified” here?)

The file `src/svm_kernels.py` contains kernels implemented explicitly as functions, for example:

- `linear_kernel(X, Y)`
- `poly_kernel(X, Y, degree, coef0)`
- `rbf_modified_kernel(X, Y, gamma)`

**Documented modifications** typically include:
- enforcing numerical stability (safe epsilon, clipping),
- supporting batch-wise Gram computation (memory-safe for large feature sets),
- adding optional normalization of feature vectors before kernel evaluation,
- exposing hyperparameters cleanly for reproducible runs.

All modifications are described directly in comments inside `svm_kernels.py`, and each kernel is written as a standalone function to satisfy “source code of the SVM kernel” requirements.

---

## Outputs and Reporting

Each run saves:

- `*_history.csv` (epoch-wise metrics + LR)
- `*_classification_report.txt`
- `*_confusion_matrix.csv`
- `*_confusion_matrix.png` (publication-style)
- best model checkpoint `*_best.pt`

For the final report, include:
- Accuracy and Macro-F1 for each model,
- Confusion matrix for the best classifier (typically JumaNet),
- Brief comparison to published baselines (e.g., LeNet/AlexNet results reported in ensemble literature).

---

## Reproducibility

- Fixed seeds for NumPy / PyTorch
- Deterministic CuDNN (when enabled)
- All hyperparameters are captured in logs and exported metadata (`meta.json`)

---

## Key Takeaways

This repository is designed to meet the typical course requirements:
1) implement CNN baselines (LeNet, AlexNet),  
2) design a custom CNN (JumaNet) and document its architecture,  
3) compare models using Accuracy and Macro-F1,  
4) report a confusion matrix for the best classifier,  
5) provide feature extraction code and custom SVM kernel source code.

---

## Citation

If you use this code or results in academic writing, cite the original CNN references:
- LeNet-5: LeCun et al.  
- AlexNet: Krizhevsky et al.  
- Residual learning: He et al.
