import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ==============================
# SETTINGS
# ==============================

BASE_DATA = r"D:\FYP\Emoi-Track\data"
AUTHORS = ["Potter", "Grimms", "HCAndersen"]

TAU_VALUES = np.arange(0.6, 0.99, 0.05)

BEST_T = {
    "Potter": 1.00,
    "Grimms": 1.25,
    "HCAndersen": 2.00
}

BEST_TAU = 0.95  # optimal found earlier

# ==============================
# PLOT STYLE (Conference Ready)
# ==============================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3, 
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

# Colour‑blind safe palette
COLORS = ["#0072B2", "#D55E00", "#009E73"]

plt.figure(figsize=(3.5, 3.0))
plt.grid(True, which="major", axis="both")

# ==============================
# GENERATE CURVES
# ==============================

for idx, author in enumerate(AUTHORS):

    logits = np.load(os.path.join(BASE_DATA, author, "calibration_logits.npy"))
    labels = np.load(os.path.join(BASE_DATA, author, "calibration_labels.npy"))

    T = BEST_T[author]
    scaled_logits = logits / T
    probs = F.softmax(torch.tensor(scaled_logits), dim=1).numpy()

    f1_list = []
    coverage_list = []

    for tau in TAU_VALUES:

        preds = []
        for p in probs:
            if np.max(p) >= tau:
                preds.append(np.argmax(p))
            else:
                preds.append(3)

        preds = np.array(preds)

        macro_f1 = f1_score(labels, preds, average="macro")
        coverage = np.mean(preds != 3)

        f1_list.append(macro_f1)
        coverage_list.append(coverage)

    # Plot curve
    plt.plot(
        coverage_list,
        f1_list,
        marker="o",
        markersize=3,
        linewidth=1.2,
        color=COLORS[idx],
        label=author
    )

    # Mark optimal operating point
    opt_index = np.argmin(np.abs(TAU_VALUES - BEST_TAU))
    plt.scatter(
        coverage_list[opt_index],
        f1_list[opt_index],
        marker="*",
        s=120,
        color=COLORS[idx],
        edgecolors="black",
        zorder=5
    )

# ==============================
# FINAL TOUCHES
# ==============================

plt.xlabel("Coverage")
plt.ylabel("Macro‑F1")
plt.xlim(0.55, 1.0)
plt.ylim(0.45, 0.75)
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig("f1_vs_coverage_conference.png", dpi=600, bbox_inches="tight")
plt.show()
