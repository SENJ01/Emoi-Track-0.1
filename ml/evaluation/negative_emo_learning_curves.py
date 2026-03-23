import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# dev folders for each model
models = {
    "BERT": r"D:\Backup\Dataset_emotion\GoEmotions-pytorch\outputs\outputs\bert-base-cased-goemotions-negative\dev",
    "DistilBERT": r"D:\Backup\Dataset_emotion\GoEmotions-pytorch\outputs\distilbert-base-uncased-goemotions-negative\dev",
    "RoBERTa": r"D:\Backup\Dataset_emotion\GoEmotions-pytorch\outputs\roberta-base-goemotions-negative\dev"
}

roberta_folder = models["RoBERTa"]

# Metrics to extract
metrics_to_plot = ["loss", "accuracy", "macro_f1", "micro_f1"]

def extract_metrics(folder):
    """
    Reads all dev txt files in a given folder
    and extracts loss/accuracy/f1 values per checkpoint.
    """
    results = {m: [] for m in metrics_to_plot}
    steps = []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".txt") and file.startswith("dev"):
                path = os.path.join(root, file)
                # Extract checkpoint number from folder name
                step_match = re.search(r"checkpoint-(\d+)", path)
                step = int(step_match.group(1)) if step_match else len(steps) + 1
                steps.append(step)

                with open(path, "r") as f:
                    text = f.read()
                    for metric in metrics_to_plot:
                        match = re.search(rf"{metric}\s*=\s*([0-9.]+)", text)
                        results[metric].append(float(match.group(1)) if match else None)

    # Sort by training step
    combined = list(zip(steps, *[results[m] for m in metrics_to_plot]))
    combined.sort(key=lambda x: x[0])
    sorted_results = {"step": [x[0] for x in combined]}
    for i, metric in enumerate(metrics_to_plot):
        sorted_results[metric] = [x[i+1] for x in combined]

    return sorted_results

# Plot comparison curves
plt.figure(figsize=(12, 8))

for metric in metrics_to_plot:
    plt.clf()
    for model_name, folder in models.items():
        if os.path.exists(folder):
            data = extract_metrics(folder)
            if data["step"]:
                plt.plot(data["step"], data[metric], marker='o', label=model_name)
    plt.title(f"{metric.upper()} Comparison Across Models")
    plt.xlabel("Training Step / Checkpoint")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_path = f"comparison_{metric}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

print("\nAll comparison graphs saved successfully!")

# === PER-CLASS F1 BAR CHART (RoBERTa TEST SET, CHECKPOINT 3000) ===
class_names = ["anger", "fear", "sadness"]
threshold = 0.32  # your RoBERTa threshold

test_probs_path = r"D:\Backup\Dataset_emotion\GoEmotions-pytorch\outputs\roberta-base-goemotions-negative\test\raw_probs.npy"
test_labels_path = r"D:\Backup\Dataset_emotion\GoEmotions-pytorch\outputs\roberta-base-goemotions-negative\test\labels.npy"

probs = np.load(test_probs_path)
labels = np.load(test_labels_path)

preds = (probs >= threshold).astype(int)

per_class_f1 = []
for i, cls in enumerate(class_names):
    f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
    per_class_f1.append(f1)

# Plot bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(class_names, per_class_f1, color=["#ff9999", "#66b3ff", "#99ff99"])
plt.ylim(0, 1)
plt.title("Per-Class Test F1 (RoBERTa, Checkpoint 3000)")
plt.xlabel("Emotion Class")
plt.ylabel("F1 Score")
plt.grid(axis='y', linestyle='--', alpha=0.6)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f"{height:.3f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("roberta_test_per_class_f1_bar_chart.png", dpi=300)
print("Saved: roberta_test_per_class_f1_bar_chart.png")
