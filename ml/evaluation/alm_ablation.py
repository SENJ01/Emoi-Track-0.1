import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

# ===============================
# CONFIG
# ===============================

BASE_DIR = "data"
BEST_T = 2.50
BEST_TAU = 0.75
UNKNOWN_CLASS = 3

# ===============================
# LOAD + MERGE
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
            logits = np.load(logits_path)
            labels = np.load(labels_path)

            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)

    return all_logits, all_labels

# ===============================
# HELPERS
# ===============================

def apply_temperature(logits, T):
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    probs = F.softmax(logits_tensor / T, dim=-1)
    return probs.numpy()

def apply_threshold(probs, tau):
    preds = []
    for p in probs:
        if np.max(p) >= tau:
            preds.append(np.argmax(p))
        else:
            preds.append(UNKNOWN_CLASS)
    return np.array(preds)

# ===============================
# MAIN
# ===============================

def main():

    logits, labels = load_all_calibration_logits()

    # 1️⃣ Baseline (T=1, no threshold)
    probs_base = apply_temperature(logits, 1.0)
    preds_base = np.argmax(probs_base, axis=1)
    f1_base = f1_score(labels, preds_base, average="macro")

    # 2️⃣ Temperature Only
    probs_temp = apply_temperature(logits, BEST_T)
    preds_temp = np.argmax(probs_temp, axis=1)
    f1_temp = f1_score(labels, preds_temp, average="macro")

    # 3️⃣ Temperature + Threshold
    preds_full = apply_threshold(probs_temp, BEST_TAU)
    f1_full = f1_score(labels, preds_full, average="macro")

    print("\n====== ABLATION RESULTS ======")
    print(f"Baseline (T=1, no τ):       {f1_base:.4f}")
    print(f"Temperature Only (T=2.5):   {f1_temp:.4f}")
    print(f"T + τ (T=2.5, τ=0.75):      {f1_full:.4f}")
    print("================================")

if __name__ == "__main__":
    main()
