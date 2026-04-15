import matplotlib.pyplot as plt

methods = ["Label-Based", "Local Distance", "Trajectory Angle", "NEFI", "Probability-Based"]
scores = [0.52, 0.46, 0.45, 0.45, 0.46]

plt.figure(figsize=(8, 6))
bars = plt.bar(methods, scores, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"])

plt.ylabel("Average F1 Score")
plt.title("Average Shift Detection Performance Across Narratives")
plt.ylim(0, 0.7)
plt.grid(axis="y", alpha=0.3)

for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, score + 0.01, f"{score:.2f}",
             ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("shift_average_bar_chart.png", dpi=300)
plt.show()
