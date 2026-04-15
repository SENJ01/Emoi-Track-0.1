import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ===============================
# CONFIG
# ===============================

BASE_DIR = "data"
BEST_T = 2.50
TAU_VALUES = np.linspace(0.0, 0.95, 50)
UNKNOWN_CLASS = 3

# Matplotlib styling for conference papers
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.figsize": (6, 5),
})

# ===============================
# LOAD EVALUATION DATA
# ===============================

def load_all_evaluation_logits():
    all_logits = []
    all_labels = []

    for author in os.listdir(BASE_DIR):
        author_path = os.path.join(BASE_DIR, author)

        if not os.path.isdir(author_path):
            continue

        logits_path = os.path.join(author_path, "evaluation_logits_1500.npy")
        labels_path = os.path.join(author_path, "evaluation_labels_1500.npy")

        if os.path.exists(logits_path) and os.path.exists(labels_path):
            print(f"Merging evaluation data from: {author}")

            logits = np.load(logits_path)
            labels = np.load(labels_path)

            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)

    print("Merged Evaluation Shape:", all_logits.shape)

    return all_logits, all_labels


# ===============================
# APPLY TEMPERATURE
# ===============================

def apply_temperature(logits, T):
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    probs = F.softmax(logits_tensor / T, dim=-1)
    return probs.numpy()


# ===============================
# APPLY THRESHOLD
# ===============================

def apply_threshold(probs, tau):
    preds = []
    for p in probs:
        if np.max(p) >= tau:
            preds.append(np.argmax(p))
        else:
            preds.append(UNKNOWN_CLASS)
    return np.array(preds)


# ===============================
# MAIN
# ===============================

def main():

    logits, labels = load_all_evaluation_logits()
    probs = apply_temperature(logits, BEST_T)

    coverages = []
    risks = []

    print("\nComputing risk–coverage curve...\n")

    for tau in TAU_VALUES:

        preds = apply_threshold(probs, tau)
        accept_mask = preds != UNKNOWN_CLASS
        coverage = np.sum(accept_mask) / len(labels)

        if coverage == 0:
            continue

        filtered_preds = preds[accept_mask]
        filtered_labels = labels[accept_mask]

        accuracy = np.mean(filtered_preds == filtered_labels)
        risk = 1.0 - accuracy

        coverages.append(coverage)
        risks.append(risk)

    # Sort by coverage (clean curve)
    coverages = np.array(coverages)
    risks = np.array(risks)

    sorted_idx = np.argsort(coverages)
    coverages = coverages[sorted_idx]
    risks = risks[sorted_idx]

    # ===============================
    # PLOT
    # ===============================

    plt.figure()
    plt.plot(coverages, risks, linewidth=2.5, label="Selective Model")

    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - Accuracy)")
    plt.title("Risk–Coverage Curve (Selective Prediction)")

    plt.xlim(0, 1)
    plt.ylim(0, max(risks) * 1.05)

    # Mark full coverage point
    plt.scatter(coverages[-1], risks[-1], s=60, zorder=5)

    plt.gca().invert_xaxis()

    plt.legend()
    plt.tight_layout()

    plt.savefig("alm_risk_coverage_curve.pdf", dpi=300)
    plt.savefig("alm_risk_coverage_curve.png", dpi=300)


if __name__ == "__main__":
    main()