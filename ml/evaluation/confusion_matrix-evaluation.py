import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cm = np.array([
    [76, 0, 13, 106],
    [2, 73, 9, 62],
    [1, 2, 171, 52],
    [36, 11, 101, 353]
])

labels = ["Anger", "Fear", "Sadness", "UNKNOWN"]

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar=True,
    linewidths=0.5,
    linecolor="gray"
)

plt.title("Confusion Matrix for Final Classification Evaluation", fontsize=13)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("True Label", fontsize=11)
plt.tight_layout()
plt.savefig("final_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()