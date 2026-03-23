import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

MODEL_NAME = "roberta-base-emopillars-negative"

def compute_prob_change(prob_array):
    prob_change = [0.0]
    for i in range(1, len(prob_array)):
        diff = np.linalg.norm(prob_array[i] - prob_array[i-1])
        prob_change.append(diff)
    return prob_change

def evaluate_narrative(input_file_path, emmood_dir, threshold):
    threshold_str = str(float(threshold))
    input_path = Path(input_file_path)
    story_name = input_path.stem.replace(".sent", "").replace(".okpuncs", "")

    BASE_DIR = Path(__file__).resolve().parents[2]
    base_pred_dir = BASE_DIR / "outputs" / "predictions"

    before_csv = base_pred_dir / f"EmoiTrack_{MODEL_NAME}_T{threshold_str}_{story_name}_ORIGINAL.csv"
    after_csv  = base_pred_dir / f"EmoiTrack_{MODEL_NAME}_T{threshold_str}_{story_name}_SLIDING_MAXPOOL.csv"

    emmood_path = Path(emmood_dir) / f"{story_name}.emmood"

    #FILE CHECKING
    if not os.path.exists(before_csv):
        print(f"❌ Missing BEFORE file: {before_csv}")
        return
    if not os.path.exists(after_csv):
        print(f"❌ Missing AFTER file: {after_csv}")
        return
    if not os.path.exists(emmood_path):
        print(f"❌ Missing ground truth: {emmood_path}")
        return

    before_df = pd.read_csv(before_csv)
    after_df = pd.read_csv(after_csv)

    #LOAD PROBABILITIES
    score_cols = [col for col in before_df.columns if col.startswith("Score_")]
    probs_before = before_df[score_cols].values
    prob_changes = compute_prob_change(probs_before)

    PROB_SHIFT_THRESHOLD = 0.05
    prob_shift = [1 if pc > PROB_SHIFT_THRESHOLD else 0 for pc in prob_changes]

    #LOAD GROUND TRUTH
    mapping = {'A':'anger','D':'anger','F':'fear','Sa':'sadness',
               'N':'unknown','H':'unknown','Su+':'unknown','Su-':'unknown'}

    gt_labels_all = []
    with open(emmood_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line: continue
            parts = line.split()
            if len(parts) < 2: continue
            lbls = parts[1].split(':')
            gt_labels_all.append([mapping.get(lbls[0], 'unknown'),
                                  mapping.get(lbls[1], 'unknown')])

    min_len = min(len(before_df), len(after_df), len(gt_labels_all))

    preds_before = before_df['Predicted_Emotion'].str.lower().values[:min_len]
    preds_after  = after_df['Predicted_Emotion'].str.lower().values[:min_len]
    gt_pairs = gt_labels_all[:min_len]

    #SHIFT CALCULATION
    gt_shifts = [0]
    for i in range(1, min_len):
        change = (gt_pairs[i][0] != gt_pairs[i-1][0]) or (gt_pairs[i][1] != gt_pairs[i-1][1])
        gt_shifts.append(1 if change else 0)

    shift_label = before_df['Shift_Label'].values[:min_len]
    shift_local = before_df['Shift_Local_Dist'].values[:min_len]
    shift_anchor = before_df['Shift_Anchor_Dist'].values[:min_len]

    #ANGLE-BASED SHIFT ---
    if "Trajectory_Angle" in before_df.columns:
        angle_vals = before_df["Trajectory_Angle"].values[:min_len]
        shift_angle = []
        for a in angle_vals:
            try:
                shift_angle.append(1 if float(a) >= 90 else 0)
            except:
                shift_angle.append(0)  # N/A or NaN
    else:
        shift_angle = [0] * min_len

    print("\n" + "="*80)
    print(f"📊 FULL RESEARCH REPORT — {story_name.upper()}")
    print("="*80)

    print("\n[PHASE 1: EMOTION CLASSIFICATION")
    gt_flat = []
    preds_flat_before = []
    preds_flat_after = []

    for i in range(min_len):
        for human_label in gt_pairs[i]:
            gt_flat.append(human_label)
            preds_flat_before.append(preds_before[i])
            preds_flat_after.append(preds_after[i])

    print("\n[PHASE 1: BEFORE SLIDING (vs BOTH Annotators)]")
    report_before = classification_report(
        gt_flat,
        preds_flat_before,
        zero_division=0,
        output_dict=True
    )

    print(classification_report(gt_flat, preds_flat_before, zero_division=0))

    macro_f1_before = f1_score(gt_flat, preds_flat_before, average="macro", zero_division=0)
    micro_f1_before = f1_score(gt_flat, preds_flat_before, average="micro", zero_division=0)
    accuracy_before = accuracy_score(gt_flat, preds_flat_before)

    print("\n[PHASE 1: AFTER SLIDING]")
    report_after = classification_report(
        gt_flat,
        preds_flat_before,
        zero_division=0,
        output_dict=True
    )

    print(classification_report(gt_flat, preds_flat_before, zero_division=0))

    macro_f1_after = f1_score(gt_flat, preds_flat_after, average="macro", zero_division=0)
    micro_f1_after = f1_score(gt_flat, preds_flat_after, average="micro", zero_division=0)
    accuracy_after = accuracy_score(gt_flat, preds_flat_after)

    # --- DIRECT MATCH RATE ---
    direct_before = (sum(1 for i in range(min_len) if preds_before[i] in gt_pairs[i]) / min_len) * 100
    direct_after  = (sum(1 for i in range(min_len) if preds_after[i]  in gt_pairs[i]) / min_len) * 100

    # --- SHIFT Detection ---
    print("\n[PHASE 2: SHIFT DETECTION ]")
    def evaluate_shift(name, y_true, y_pred):
        print(f"\n[SHIFT — {name}]")
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        print(classification_report(y_true, y_pred, zero_division=0))
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)
        return report, f1, cm

    label_report, f1_label, cm_label = evaluate_shift("LABEL", gt_shifts, shift_label)
    local_report, f1_local, cm_local = evaluate_shift("LOCAL", gt_shifts, shift_local)
    anchor_report, f1_anchor, cm_anchor = evaluate_shift("ANCHOR", gt_shifts, shift_anchor)
    angle_report, f1_angle, cm_angle = evaluate_shift("ANGLE", gt_shifts, shift_angle)
    prob_report, f1_prob, _ = evaluate_shift("PROBABILITY", gt_shifts, prob_shift)


    # SAVE FULL RESEARCH JSON

    full_report = {
        "phase1": {
            "before_sliding": {
                "report": report_before,
                "macro_f1": macro_f1_before,
                "micro_f1": micro_f1_before,
                "accuracy": accuracy_before
            },
            "after_sliding": {
                "report": report_after,
                "macro_f1": macro_f1_after,
                "micro_f1": micro_f1_after,
                "accuracy": accuracy_after
            },
            "direct_match_rate": {
                "before": direct_before,
                "after": direct_after
            }
        },
        "phase2": {
            "label_shift": {
                "report": label_report,
                "f1": f1_label,
                "confusion_matrix": cm_label.tolist()
            },
            "local_distance_shift": {
                "report": local_report,
                "f1": f1_local,
                "confusion_matrix": cm_local.tolist()
            },
            "anchor_distance_shift": {
                "report": anchor_report,
                "f1": f1_anchor,
                "confusion_matrix": cm_anchor.tolist()
            },
            "trajectory_angle_shift": {
                "report": angle_report,
                "f1": f1_angle,
                "confusion_matrix": cm_angle.tolist()
            }
        },
        "phase3": {
            "probability_shift": {
                "report": prob_report,
                "f1": f1_prob
            }
        }
    }

    report_path = base_pred_dir / f"full_research_report_{story_name}.json"

    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=4)

    print(f"\n📊 Full research report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--emmood_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    evaluate_narrative(args.input_file, args.emmood_dir, args.threshold)