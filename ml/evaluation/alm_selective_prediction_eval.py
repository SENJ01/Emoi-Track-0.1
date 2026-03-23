import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

# ===============================
# CONFIG
# ===============================

BASE_DIR = "data"                # data/<Author>/
BEST_T = 2.50                    
TAU_VALUES = [0.0, 0.88, 0.90]   # Report these 3
UNKNOWN_CLASS = 3

# ===============================
# LOAD + MERGE ALL EVALUATION DATA
# ===============================

def load_all_evaluation_logits():
    all_logits = []
    all_labels = []

    for author in os.listdir(BASE_DIR):
        author_path = os.path.join(BASE_DIR, author)

        if not os.path.isdir(author_path):
            continue

        logits_path = os.path.join(author_path, "evaluation_logits.npy")
        labels_path = os.path.join(author_path, "evaluation_labels.npy")

        if os.path.exists(logits_path) and os.path.exists(labels_path):
            print(f"Merging evaluation data from: {author}")

            logits = np.load(logits_path)
            labels = np.load(labels_path)

            all_logits.append(logits)
            all_labels.append(labels)

    if len(all_logits) == 0:
        raise ValueError("No evaluation files found.")

    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)

    print("\nMerged Evaluation Shape:")
    print("Logits:", all_logits.shape)
    print("Labels:", all_labels.shape)

    return all_logits, all_labels

# ===============================
# APPLY TEMPERATURE
# ===============================

def apply_temperature(logits, T):
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    probs = F.softmax(logits_tensor / T, dim=-1)
    return probs.numpy()

# ===============================
# APPLY THRESHOLD
# ===============================

def apply_threshold(probs, tau):
    preds = []
    for p in probs:
        max_prob = np.max(p)
        if max_prob >= tau:
            preds.append(np.argmax(p))
        else:
            preds.append(UNKNOWN_CLASS)
    return np.array(preds)

# ===============================
# MAIN
# ===============================

def main():

    logits, labels = load_all_evaluation_logits()
    probs = apply_temperature(logits, BEST_T)

    print(f"\nSelective Prediction Evaluation (T={BEST_T})\n")

    print("-------------------------------------------------------------")
    print("Tau\tCoverage\tMacro-F1\tMicro-F1")
    print("-------------------------------------------------------------")

    for tau in TAU_VALUES:

        preds = apply_threshold(probs, tau)

        accept_mask = preds != UNKNOWN_CLASS
        coverage = np.sum(accept_mask) / len(labels)

        if tau == 0.0:
            # No rejection baseline
            macro_f1 = f1_score(labels, preds, average="macro")
            micro_f1 = f1_score(labels, preds, average="micro")
        else:
            # Remove rejected samples before computing F1
            filtered_preds = preds[accept_mask]
            filtered_labels = labels[accept_mask]

            macro_f1 = f1_score(filtered_labels, filtered_preds, average="macro")
            micro_f1 = f1_score(filtered_labels, filtered_preds, average="micro")

        print(f"{tau:.2f}\t{coverage:.3f}\t\t{macro_f1:.4f}\t\t{micro_f1:.4f}")

    print("-------------------------------------------------------------")


if __name__ == "__main__":
    main()