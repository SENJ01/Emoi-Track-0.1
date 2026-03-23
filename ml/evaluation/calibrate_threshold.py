import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# 1. LOAD DATA
# Ensure these paths point to GoEmotions test output
probs_path = "outputs/roberta-base-goemotions-negative/test/raw_probs.npy"
labels_path = "outputs/roberta-base-goemotions-negative/test/labels.npy"

probs = np.load(probs_path)
labels = np.load(labels_path)

# Negative Filter labels
label_names = ["anger", "fear", "sadness"] 

# 2. PREPARE GROUND TRUTH
# If labels.npy is one-hot [N, 3], we find the index of the true emotion.
# If a row is all zeros (shouldn't happen with your filter, but good for safety), 
# it defaults to index 3 (Unknown).
y_true = np.argmax(labels, axis=1)

# 3. GLOBAL THRESHOLD SWEEP
def run_global_sweep(probs, y_true):
    # Range of values to experiment
    thresholds = np.arange(0.05, 0.95, 0.05)
    
    sweep_results = []

    print(f"{'Threshold':<10} | {'Macro F1':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 50)

    for t in thresholds:
        preds = []
        for row in probs:
            # If the strongest emotion doesn't beat the threshold, it's UNKNOWN (3)
            if np.max(row) >= t:
                preds.append(np.argmax(row))
            else:
                preds.append(3) # Index for Unknown

        # Calculate metrics (Macro average treats all emotions equally)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        prec = precision_score(y_true, preds, average='macro', zero_division=0)
        rec = recall_score(y_true, preds, average='macro', zero_division=0)

        sweep_results.append({
            "threshold": round(t, 2),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4)
        })

        print(f"{t:<10.2f} | {f1:<10.4f} | {prec:<10.4f} | {rec:<10.4f}")

    return pd.DataFrame(sweep_results)

# Execute Sweep
df_results = run_global_sweep(probs, y_true)

# -----------------------
# 4. FIND OPTIMAL VALUE
# -----------------------
best_row = df_results.loc[df_results['f1'].idxmax()]
best_tau = best_row['threshold']

print("-" * 50)
print(f"EXPERIMENT COMPLETE")
print(f"Optimal Global Threshold (Max F1): {best_tau}")
print(f"Best Macro F1 achieved: {best_row['f1']}")

# 5. VISUALIZE
plt.figure(figsize=(10, 6))
plt.plot(df_results['threshold'], df_results['f1'], label='Macro F1', marker='o', color='blue')
plt.plot(df_results['threshold'], df_results['recall'], label='Macro Recall', linestyle='--', color='green')
plt.axvline(x=best_tau, color='red', linestyle=':', label=f'Optimal Tau ({best_tau})')

plt.title('Threshold Sensitivity Analysis (GoEmotions Calibration)')
plt.xlabel('Global Threshold (τ)')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig("threshold_sweep_analysis.png")
print("Graph saved as 'threshold_sweep_analysis.png'")

# Save results for dissertation table
df_results.to_csv("threshold_experiment_results.csv", index=False)