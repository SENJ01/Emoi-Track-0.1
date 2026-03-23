import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

# =====================================
# SETTINGS
# =====================================

BASE_DATA = r"D:\FYP\Emoi-Track\data"
AUTHORS = ["Potter", "Grimms", "HCAndersen"]
T_VALUES = np.arange(0.5, 3.25, 0.25)

# =====================================
# STYLE (Research Aesthetic)
# =====================================

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12
})

# =====================================
# TEMPERATURE SWEEP (ALL AUTHORS)
# =====================================

results = {}
best_T = {}

for author in AUTHORS:

    logits_path = os.path.join(BASE_DATA, author, "calibration_logits.npy")
    labels_path = os.path.join(BASE_DATA, author, "calibration_labels.npy")

    logits = np.load(logits_path)
    labels = np.load(labels_path)

    f1_scores = []

    for T in T_VALUES:
        scaled_logits = logits / T
        probs = F.softmax(torch.tensor(scaled_logits), dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        macro_f1 = f1_score(labels, preds, average="macro")
        f1_scores.append(macro_f1)

    results[author] = f1_scores

    best_index = np.argmax(f1_scores)
    best_T[author] = T_VALUES[best_index]

    print(f"{author} best T: {best_T[author]:.2f}")

# =====================================
# CONFIDENCE DISTRIBUTION (Example: Potter)
# =====================================

example_author = "Potter"
logits = np.load(os.path.join(BASE_DATA, example_author, "calibration_logits.npy"))

# Before scaling
probs_before = F.softmax(torch.tensor(logits), dim=1).numpy()
max_before = np.max(probs_before, axis=1)

# After scaling using best T
T_opt = best_T[example_author]
scaled_logits = logits / T_opt
probs_after = F.softmax(torch.tensor(scaled_logits), dim=1).numpy()
max_after = np.max(probs_after, axis=1)

# =====================================
# CREATE 2-PANEL FIGURE
# =====================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# -------- LEFT: Temperature Sweep --------
for author in AUTHORS:
    axes[0].plot(
        T_VALUES,
        results[author],
        marker="o",
        linewidth=2,
        label=author
    )

axes[0].set_title("Temperature Scaling Sweep\n(Macro-F1 vs Temperature)")
axes[0].set_xlabel("Temperature (T)")
axes[0].set_ylabel("Macro-F1 Score")
axes[0].legend()
axes[0].set_ylim(0.0, 1.0)

# -------- RIGHT: Confidence Distribution --------
sns.histplot(max_before, bins=20, kde=True, ax=axes[1],
             color="gray", label="Before Scaling", stat="density")

sns.histplot(max_after, bins=20, kde=True, ax=axes[1],
             color="orange", label=f"After Scaling (T={T_opt:.2f})",
             stat="density")

axes[1].set_title(f"Confidence Distribution ({example_author})\nBefore vs After Temperature Scaling")
axes[1].set_xlabel("Max Softmax Probability")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.savefig("temperature_scaling_analysis.png", dpi=300)
plt.show()

print("\nSaved: temperature_scaling_analysis.png")
