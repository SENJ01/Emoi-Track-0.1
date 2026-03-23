import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# SETTINGS
BASE_DATA = r"D:\FYP\Emoi-Track\data"
AUTHORS = ["Potter", "Grimms", "HCAndersen"]

T_VALUES = np.arange(0.5, 3.25, 0.25)

# Temporary threshold for Stage 2
TAU = 0.7

# =====================================
# STORAGE
# =====================================

results = {}
best_temperatures = {}
best_scores = {}

# =====================================
# TEMPERATURE SWEEP (WITH REJECTION)
# =====================================

for author in AUTHORS:

    print(f"\nProcessing Author: {author}")

    logits_path = os.path.join(BASE_DATA, author, "calibration_logits.npy")
    labels_path = os.path.join(BASE_DATA, author, "calibration_labels.npy")

    logits = np.load(logits_path)
    labels = np.load(labels_path)

    f1_scores = []

    for T in T_VALUES:

        scaled_logits = logits / T
        probs = F.softmax(torch.tensor(scaled_logits), dim=1).numpy()

        preds = []
        for p in probs:
            max_prob = np.max(p)
            pred_class = np.argmax(p)

            if max_prob >= TAU:
                preds.append(pred_class)
            else:
                preds.append(3)  # UNKNOWN

        preds = np.array(preds)

        macro_f1 = f1_score(labels, preds, average="macro")
        f1_scores.append(macro_f1)

        print(f"T={T:.2f} | Macro-F1={macro_f1:.4f}")

    results[author] = f1_scores

    best_idx = np.argmax(f1_scores)
    best_temperatures[author] = T_VALUES[best_idx]
    best_scores[author] = f1_scores[best_idx]

    print(f"Best T for {author}: {best_temperatures[author]:.2f}")
    print(f"Best Macro-F1: {best_scores[author]:.4f}")

# =====================================
# PLOTTING
# =====================================

plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(10, 7))

for author in AUTHORS:

    plt.plot(
        T_VALUES,
        results[author],
        marker="o",
        linewidth=2,
        label=author
    )

    best_T = best_temperatures[author]
    best_F1 = best_scores[author]

    plt.scatter(best_T, best_F1, s=120, zorder=5)
    plt.annotate(
        f"T={best_T:.2f}\nF1={best_F1:.3f}",
        (best_T, best_F1),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

plt.title("Temperature Scaling with Rejection (Stage 2)")
plt.xlabel("Temperature (T)")
plt.ylabel("Macro-F1 Score")
plt.ylim(0.0, 1.0)
plt.legend()
plt.tight_layout()

output_path = "alm_stage2_temperature_sweep.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("\nSaved figure to:", output_path)
