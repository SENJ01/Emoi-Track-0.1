import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt

# --- CONFIG ---
BASE_PRED_DIR = r"D:\Backup\Dataset_emotion\GoEmotions-pytorch\outputs\predictions"
EMMOOD_DIR = r"D:\Backup\Dataset_emotion\GoEmotions-pytorch\data\Potter\emmood"
CLASSIFICATION_THRESHOLD = 0.5
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6]

mapping = {'A':'anger','D':'anger','F':'fear','Sa':'sadness',
           'N':'unknown','H':'unknown','Su+':'unknown','Su-':'unknown'}

def compute_prob_change(prob_array):
    prob_change = [0.0]
    for i in range(1, len(prob_array)):
        diff = np.linalg.norm(prob_array[i] - prob_array[i-1])
        prob_change.append(diff)
    return prob_change

# --- FIND ALL STORY CSVs ---
all_csvs = [f for f in os.listdir(BASE_PRED_DIR)
            if f.startswith(f"EmoiTrack_T{CLASSIFICATION_THRESHOLD}_") and f.endswith("_ORIGINAL.csv")]

print(f" Found {len(all_csvs)} stories for calibration.")

all_results = []   # per-story threshold results
global_scores = {pt: [] for pt in THRESHOLDS}  # store F1 across stories

for csv_file in all_csvs:
    story_name = csv_file.replace(f"EmoiTrack_T{CLASSIFICATION_THRESHOLD}_", "").replace("_ORIGINAL.csv", "")
    csv_path = os.path.join(BASE_PRED_DIR, csv_file)
    emmood_path = os.path.join(EMMOOD_DIR, f"{story_name}.emmood")

    if not os.path.exists(emmood_path):
        print(f" Skipping {story_name}: missing ground truth.")
        continue

    df = pd.read_csv(csv_path)

    # --- LOAD GROUND TRUTH SHIFTS ---
    gt_pairs = []
    with open(emmood_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            lbls = parts[1].split(':')
            gt_pairs.append([mapping.get(lbls[0], 'unknown'),
                             mapping.get(lbls[1], 'unknown')])

    min_len = min(len(df), len(gt_pairs))

    gt_shifts = [0]
    for i in range(1, min_len):
        change = (gt_pairs[i][0] != gt_pairs[i-1][0]) or (gt_pairs[i][1] != gt_pairs[i-1][1])
        gt_shifts.append(1 if change else 0)

    # --- EXTRACT PROBABILITIES ---
    score_cols = [c for c in df.columns if c.startswith("Score_")]
    probs = df[score_cols].values[:min_len]
    prob_change = compute_prob_change(probs)

    # --- SWEEP THRESHOLDS ---
    story_results = []
    for pt in THRESHOLDS:
        prob_shift = [1 if pc > pt else 0 for pc in prob_change]

        f1 = f1_score(gt_shifts, prob_shift, zero_division=0)
        precision = precision_score(gt_shifts, prob_shift, zero_division=0)
        recall = recall_score(gt_shifts, prob_shift, zero_division=0)
        accuracy = accuracy_score(gt_shifts, prob_shift)

        story_results.append({
            "Story": story_name,
            "Prob_Threshold": pt,
            "Shift_F1": f1,
            "Shift_Precision": precision,
            "Shift_Recall": recall,
            "Shift_Accuracy": accuracy
        })

        global_scores[pt].append(f1)

    story_df = pd.DataFrame(story_results)
    out_csv = os.path.join(BASE_PRED_DIR, f"prob_threshold_sweep_{story_name}.csv")
    story_df.to_csv(out_csv, index=False)
    print(f"📄 Saved: {out_csv}")

    all_results.extend(story_results)

# --- GLOBAL THRESHOLD ANALYSIS ---
global_rows = []
for pt, f1s in global_scores.items():
    if len(f1s) == 0:
        continue
    global_rows.append({
        "Prob_Threshold": pt,
        "Mean_Shift_F1": np.mean(f1s),
        "Std_Shift_F1": np.std(f1s),
        "Num_Stories": len(f1s)
    })

global_df = pd.DataFrame(global_rows)
global_csv = os.path.join(BASE_PRED_DIR, "prob_threshold_sweep_GLOBAL.csv")
global_df.to_csv(global_csv, index=False)

best_row = global_df.sort_values("Mean_Shift_F1", ascending=False).iloc[0]
best_threshold = best_row["Prob_Threshold"]

print("\n GLOBAL THRESHOLD RESULTS")
print(global_df)
print(f"\n Best Global Threshold = {best_threshold:.2f} "
      f"(Mean F1 = {best_row['Mean_Shift_F1']:.3f} over {int(best_row['Num_Stories'])} stories)")

# --- PLOT GLOBAL METRICS ---
plt.figure(figsize=(9, 6))
plt.plot(global_df["Prob_Threshold"], global_df["Mean_Shift_F1"],
         marker='o', color='blue', linewidth=2, label="Mean F1")

plt.fill_between(global_df["Prob_Threshold"],
                 global_df["Mean_Shift_F1"] - global_df["Std_Shift_F1"],
                 global_df["Mean_Shift_F1"] + global_df["Std_Shift_F1"],
                 color='blue', alpha=0.2, label="±1 Std Dev")

plt.axvline(best_threshold, color='red', linestyle='--', label=f"Best = {best_threshold:.2f}")

plt.title("Global Probability Threshold Calibration (Across Stories)")
plt.xlabel("Probability Change Threshold")
plt.ylabel("Mean Shift F1")
plt.legend()
plt.grid(True)
plt.tight_layout()

global_png = os.path.join(BASE_PRED_DIR, "prob_threshold_GLOBAL_curve.png")
plt.savefig(global_png, dpi=300)
plt.show()
print(f"\n📈 Global calibration plot saved to: {global_png}")
