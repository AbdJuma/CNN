import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix_jumanet(
    cm: np.ndarray,
    labels=None,
    title="JumaNet Confusion Matrix (Test)",
    cmap="viridis",
    out_path="jumannet_confusion_matrix.png",
    dpi=300,
):
    cm = np.asarray(cm, dtype=np.int64)
    n = cm.shape[0]
    assert cm.shape[0] == cm.shape[1], "Confusion matrix must be square."

    if labels is None:
        labels = [str(i) for i in range(n)]
    else:
        labels = [str(x) for x in labels]

    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    im = ax.imshow(cm, cmap=cmap, interpolation="nearest")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    # Title and axes
    ax.set_title(title, fontsize=22, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted label", fontsize=16, labelpad=10)
    ax.set_ylabel("True label", fontsize=16, labelpad=10)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_yticklabels(labels, fontsize=13)

    # Grid lines (subtle)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.8, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ---- Annotation color logic ----
    # Use a threshold so low-intensity (purple) cells show WHITE text.
    # High-intensity (yellow) cells show BLACK text.
    vmax = cm.max() if cm.size else 1
    threshold = 0.50 * vmax  # tune if you want: 0.45 ~ 0.60

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            # White on dark, black on bright
            txt_color = "white" if val < threshold else "black"
            ax.text(j, i, f"{val}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=txt_color)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    cm = np.array([
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0, 998,   0,   0,   0,   0,   0,   0,   2,   0],
        [0,   1, 996,   1,   1,   0,   1,   0,   0,   0],
        [0,   1,   5, 994,   0,   0,   0,   0,   0,   0],
        [0,   0,   4,   0, 996,   0,   0,   0,   0,   0],
        [0,   0,   3,   0,   0, 996,   0,   1,   0,   0],
        [0,   1,   0,   0,   5,   0, 994,   0,   0,   0],
        [0,   0,   0,   0,   0,   1,   0, 999,   0,   0],
        [0,   0,   2,   0,   0,   1,   0,   0, 996,   1],
        [0,   0,   1,   0,   3,   0,   2,   0,   1, 993],
    ], dtype=np.int64)

    plot_confusion_matrix_jumanet(
        cm,
        labels=list(range(10)),
        title="JumaNet Confusion Matrix (Test)",
        out_path="jumannet_confusion_matrix.png",
        dpi=100,
    )
