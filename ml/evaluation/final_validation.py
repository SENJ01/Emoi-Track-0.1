import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import json
from pathlib import Path
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


def compute_prob_change(prob_array):
    prob_change = [0.0]
    for i in range(1, len(prob_array)):
        diff = np.linalg.norm(prob_array[i] - prob_array[i - 1])
        prob_change.append(diff)
    return prob_change


def evaluate_narrative(input_file_path, emmood_dir, threshold):
    threshold_str = str(float(threshold))
    input_path = Path(input_file_path)
    story_name = input_path.stem.replace(".sent", "").replace(".okpuncs", "")

    BASE_DIR = Path(__file__).resolve().parents[2]
    base_pred_dir = BASE_DIR / "outputs" / "predictions"

    before_csv = base_pred_dir / f"EmoiTrack_T{threshold_str}_{story_name}_ORIGINAL.csv"
    after_csv = (
        base_pred_dir / f"EmoiTrack_T{threshold_str}_{story_name}_SLIDING_MAXPOOL.csv"
    )
    emmood_path = Path(emmood_dir) / f"{story_name}.emmood"

    print(f"\n--- DEBUG START ---")
    print(f"Target Threshold Variable: {threshold}")
    print(f"Target Story Name: {story_name}")
    print(f"Searching for BEFORE: {before_csv}")
    print(f"Searching for AFTER: {after_csv}")
    print(f"--- DEBUG END ---\n")

    # FILE CHECKING
    if not os.path.exists(before_csv):
        print(f"❌ Missing BEFORE file: {before_csv}")
        # DEBUG POINT 2: List actual folder contents to see the mismatch
        print(f"\nDEBUG: Investigating folder: {base_pred_dir}")
        if os.path.exists(base_pred_dir):
            files_in_dir = os.listdir(base_pred_dir)
            print(f"Found {len(files_in_dir)} files in directory.")
            print("First 10 files in folder:")
            for f in files_in_dir[:10]:
                print(f"  - {f}")

            # Specifically look for files containing the story name
            matches = [f for f in files_in_dir if story_name in f]
            print(f"\nFiles matching '{story_name}': {matches}")
        else:
            print(f"🚨 ERROR: The directory {base_pred_dir} does not even exist!")

        print(f"--- DEBUG END ---\n")
        return
        return
    if not os.path.exists(after_csv):
        print(f"❌ Missing AFTER file: {after_csv}")
        return
    if not os.path.exists(emmood_path):
        print(f"❌ Missing ground truth: {emmood_path}")
        return

    before_df = pd.read_csv(before_csv)
    after_df = pd.read_csv(after_csv)

    # LOAD PROBABILITIES
    score_cols = [col for col in before_df.columns if col.startswith("Score_")]
    probs_before = before_df[score_cols].values
    prob_changes = compute_prob_change(probs_before)

    PROB_SHIFT_THRESHOLD = 0.05
    prob_shift = [1 if pc > PROB_SHIFT_THRESHOLD else 0 for pc in prob_changes]

    # LOAD GROUND TRUTH
    mapping = {
        "A": "anger",
        "D": "anger",
        "F": "fear",
        "Sa": "sadness",
        "N": "unknown",
        "H": "unknown",
        "Su+": "unknown",
        "Su-": "unknown",
    }

    gt_labels_all = []
    with open(emmood_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            lbls = parts[1].split(":")
            gt_labels_all.append(
                [mapping.get(lbls[0], "unknown"), mapping.get(lbls[1], "unknown")]
            )

    min_len = min(len(before_df), len(after_df), len(gt_labels_all))

    preds_before = before_df["Predicted_Emotion"].str.lower().values[:min_len]
    preds_after = after_df["Predicted_Emotion"].str.lower().values[:min_len]
    gt_pairs = gt_labels_all[:min_len]

    # SHIFT CALCULATION
    gt_shifts = [0]
    for i in range(1, min_len):
        prev_set = set(gt_pairs[i - 1])
        curr_set = set(gt_pairs[i])
        change = prev_set != curr_set
        gt_shifts.append(1 if change else 0)

    shift_label = before_df["Shift_Label"].values[:min_len]
    shift_local = before_df["Shift_Local_Dist"].values[:min_len]

    # --- NEFI RUPTURE SHIFT ---
    if "Rupture_Flag" in before_df.columns:
        shift_nefi = before_df["Rupture_Flag"].values[:min_len]
    else:
        shift_nefi = [0] * min_len

    # ANGLE-BASED SHIFT ---
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

    print("\n" + "=" * 80)
    print(f"📊 FULL RESEARCH REPORT — {story_name.upper()}")
    print("=" * 80)

    print("\n[PHASE 1: EMOTION CLASSIFICATION]")

    # ---- Convert to single evaluation target using OR logic ----
    correct_before = []
    correct_after = []
    gold_single = []
    pred_single_before = []
    pred_single_after = []

    for i in range(min_len):

        annotator_A, annotator_B = gt_pairs[i]
        pred_b = preds_before[i]
        pred_a = preds_after[i]

        # OR logic: prediction is correct if matches either annotator
        is_correct_before = int(pred_b == annotator_A or pred_b == annotator_B)
        is_correct_after = int(pred_a == annotator_A or pred_a == annotator_B)

        correct_before.append(is_correct_before)
        correct_after.append(is_correct_after)

        # For Macro-F1 over emotion labels:
        # Choose gold label if prediction matches one annotator,
        # otherwise choose first annotator as default reference.
        if pred_b == annotator_A:
            gold_single.append(annotator_A)
        elif pred_b == annotator_B:
            gold_single.append(annotator_B)
        else:
            gold_single.append(annotator_A)

        pred_single_before.append(pred_b)
        pred_single_after.append(pred_a)

    # ---- BEFORE SLIDING ----
    print("\n[BEFORE SLIDING]")
    report_before = classification_report(
        gold_single, pred_single_before, zero_division=0, output_dict=True
    )
    print(classification_report(gold_single, pred_single_before, zero_division=0))

    macro_f1_before = f1_score(
        gold_single, pred_single_before, average="macro", zero_division=0
    )
    micro_f1_before = f1_score(
        gold_single, pred_single_before, average="micro", zero_division=0
    )
    accuracy_before = np.mean(correct_before)

    # ------------------------------
    # CONFUSION MATRIX — BEFORE SLIDING
    # ------------------------------
    cm_before = confusion_matrix(gold_single, pred_single_before)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_before,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["anger", "fear", "sadness", "unknown"],
        yticklabels=["anger", "fear", "sadness", "unknown"],
    )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"{story_name} — Confusion matrix")
    plt.tight_layout()

    cm_path = base_pred_dir / f"confusion_matrix_{story_name}.pdf"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # ---- AFTER SLIDING ----
    print("\n[AFTER SLIDING]")
    report_after = classification_report(
        gold_single, pred_single_after, zero_division=0, output_dict=True
    )
    print(classification_report(gold_single, pred_single_after, zero_division=0))

    macro_f1_after = f1_score(
        gold_single, pred_single_after, average="macro", zero_division=0
    )
    micro_f1_after = f1_score(
        gold_single, pred_single_after, average="micro", zero_division=0
    )
    accuracy_after = np.mean(correct_after)

    # --- SHIFT Detection ---
    print("\n[PHASE 2: SHIFT DETECTION ]")

    def evaluate_shift(name, y_true, y_pred):
        print(f"\n[SHIFT — {name}]")
        report = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )
        print(classification_report(y_true, y_pred, zero_division=0))
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)
        return report, f1, cm

    label_report, f1_label, cm_label = evaluate_shift("LABEL", gt_shifts, shift_label)
    local_report, f1_local, cm_local = evaluate_shift("LOCAL", gt_shifts, shift_local)
    angle_report, f1_angle, cm_angle = evaluate_shift("ANGLE", gt_shifts, shift_angle)
    nefi_report, f1_nefi, cm_nefi = evaluate_shift(
        "NEFI_RUPTURE", gt_shifts, shift_nefi
    )
    prob_report, f1_prob, _ = evaluate_shift("PROBABILITY", gt_shifts, prob_shift)

    # SAVE FULL RESEARCH JSON

    full_report = {
        "phase1": {
            "before_sliding": {
                "report": report_before,
                "macro_f1": macro_f1_before,
                "micro_f1": micro_f1_before,
                "accuracy": accuracy_before,
            },
            "after_sliding": {
                "report": report_after,
                "macro_f1": macro_f1_after,
                "micro_f1": micro_f1_after,
                "accuracy": accuracy_after,
            },
        },
        "phase2": {
            "label_shift": {
                "report": label_report,
                "f1": f1_label,
                "confusion_matrix": cm_label.tolist(),
            },
            "local_distance_shift": {
                "report": local_report,
                "f1": f1_local,
                "confusion_matrix": cm_local.tolist(),
            },
            "trajectory_angle_shift": {
                "report": angle_report,
                "f1": f1_angle,
                "confusion_matrix": cm_angle.tolist(),
            },
            "nefi_rupture_shift": {
                "report": nefi_report,
                "f1": f1_nefi,
                "confusion_matrix": cm_nefi.tolist(),
            },
        },
        "phase3": {"probability_shift": {"report": prob_report, "f1": f1_prob}},
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

    # DEBUG POINT 3: Check argparse results
    print(f"DEBUG: Script started with threshold: {args.threshold}")

    evaluate_narrative(args.input_file, args.emmood_dir, args.threshold)
