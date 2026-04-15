import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ===============================
# CONFIG
# ===============================

BASE_DIR = "data"
BEST_T = 2.50              # <-- use best global T
TAU_VALUES = np.arange(0.40, 0.95, 0.02)
UNKNOWN_CLASS = 3

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

        logits_path = os.path.join(author_path, "calibration_logits_1500.npy")
        labels_path = os.path.join(author_path, "calibration_labels_1500.npy")

        if os.path.exists(logits_path) and os.path.exists(labels_path):
            print(f"Merging calibration data from: {author}")

            logits = np.load(logits_path)
            labels = np.load(labels_path)

            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)

    print("\nMerged Shape:")
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

    logits, labels = load_all_calibration_logits()

    probs = apply_temperature(logits, BEST_T)

    best_tau = None
    best_f1 = 0
    f1_scores = []

    print(f"\nSweeping Threshold (τ) with T={BEST_T}\n")

    for tau in TAU_VALUES:
        preds = apply_threshold(probs, tau)
        macro_f1 = f1_score(labels, preds, average="macro")
        f1_scores.append(macro_f1)

        print(f"τ={tau:.2f} | Macro-F1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_tau = tau

    print("\n==============================")
    print(f"Best τ: {best_tau:.2f}")
    print(f"Best Macro-F1: {best_f1:.4f}")
    print("==============================")

        # =====================================
    # COVERAGE ANALYSIS AT BEST TAU
    # =====================================

    print("\nCoverage Analysis at Best τ")

    final_preds = apply_threshold(probs, best_tau)

    accept_mask = final_preds != UNKNOWN_CLASS

    total_samples = len(labels)
    accepted = np.sum(accept_mask)
    rejected = total_samples - accepted
    coverage = accepted / total_samples

    print(f"Total samples: {total_samples}")
    print(f"Accepted (anger/fear/sadness): {accepted}")
    print(f"Rejected (UNKNOWN): {rejected}")
    print(f"Coverage: {coverage:.2%}")

    print("\nPer-Class Coverage:")

    for cls in np.unique(labels):
        class_mask = labels == cls
        class_total = np.sum(class_mask)
        class_accepted = np.sum(accept_mask & class_mask)

        if class_total > 0:
            print(f"Class {cls}: {class_accepted/class_total:.2%}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(TAU_VALUES, f1_scores, marker="o")
    plt.xlabel("Threshold (τ)")
    plt.ylabel("Macro F1")
    plt.title(f"Threshold Sweep (T={BEST_T})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("alm_threshold_sweep.png")

    print("\nSaved figure: alm_threshold_sweep.png")


if __name__ == "__main__":
    main()