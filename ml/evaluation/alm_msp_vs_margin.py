import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

# ===============================
# CONFIG
# ===============================

BASE_DIR = "data"          # or full path if needed
T = 2.5                    # fixed global temperature
UNKNOWN_CLASS = 3          # make sure this matches label encoding

# ===============================
# LOAD + MERGE ALL AUTHORS
# ===============================

def load_all_calibration_logits():
    all_logits = []
    all_labels = []

    for author in os.listdir(BASE_DIR):
        author_path = os.path.join(BASE_DIR, author)

        if not os.path.isdir(author_path):
            continue

        logits_path = os.path.join(author_path, "calibration_logits.npy")
        labels_path = os.path.join(author_path, "calibration_labels.npy")

        if os.path.exists(logits_path) and os.path.exists(labels_path):
            print(f"Merging calibration data from: {author}")

            logits = np.load(logits_path)
            labels = np.load(labels_path)

            all_logits.append(logits)
            all_labels.append(labels)

    if len(all_logits) == 0:
        raise ValueError("No calibration files found.")

    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)

    print("\nMerged Shape:")
    print("Logits:", all_logits.shape)
    print("Labels:", all_labels.shape)

    return all_logits, all_labels

# ===============================
# TEMPERATURE SCALING
# ===============================

def apply_temperature(logits, T):
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    probs = F.softmax(logits_tensor / T, dim=-1)
    return probs.numpy()

# ===============================
# MAIN
# ===============================

def main():

    logits, labels = load_all_calibration_logits()
    probs = apply_temperature(logits, T)

    print("\nTemperature fixed at T =", T)
    print("=====================================")

    # =====================================================
    # METHOD 1 — MSP
    # =====================================================

    best_f1_msp = 0
    best_tau = None

    print("\n--- MSP Sweep (Full System) ---")

    for tau in np.arange(0.60, 0.96, 0.05):

        max_probs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)

        final_preds = preds.copy()
        final_preds[max_probs < tau] = UNKNOWN_CLASS

        macro_f1 = f1_score(labels, final_preds, average="macro")
        coverage = np.mean(max_probs >= tau)

        print(f"τ={tau:.2f} | Macro-F1={macro_f1:.4f} | Coverage={coverage:.2%}")

        if macro_f1 > best_f1_msp:
            best_f1_msp = macro_f1
            best_tau = tau

    # =====================================================
    # METHOD 2 — Margin (Full-System Evaluation)
    # =====================================================

    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    p1 = sorted_probs[:, 0]
    p2 = sorted_probs[:, 1]
    margins = p1 - p2

    best_f1_margin = 0
    best_delta = None

    print("\n--- Margin Sweep (Full System) ---")

    for delta in np.arange(0.05, 0.51, 0.05):

        preds = np.argmax(probs, axis=1)
        final_preds = preds.copy()

        accept_mask = margins >= delta
        final_preds[~accept_mask] = UNKNOWN_CLASS

        macro_f1 = f1_score(labels, final_preds, average="macro")
        coverage = np.mean(accept_mask)

        print(f"δ={delta:.2f} | Macro-F1={macro_f1:.4f} | Coverage={coverage:.2%}")

        if macro_f1 > best_f1_margin:
            best_f1_margin = macro_f1
            best_delta = delta

    # =====================================================
    # FINAL RESULT
    # =====================================================

    print("\n=====================================")
    print("FINAL COMPARISON (Full-System Evaluation)")
    print(f"MSP     → Macro-F1={best_f1_msp:.4f} | τ={best_tau}")
    print(f"Margin  → Macro-F1={best_f1_margin:.4f} | δ={best_delta}")
    print("=====================================")


if __name__ == "__main__":
    main()