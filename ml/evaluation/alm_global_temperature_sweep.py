import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ===============================
# CONFIG
# ===============================

BASE_DIR = "data"        # data/<Author>/calibration_logits.npy
T_VALUES = np.arange(0.5, 3.25, 0.25)
TAU = 0.75               # fixed global threshold
UNKNOWN_CLASS = 3        # index of UNKNOWN

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

    if len(all_logits) == 0:
        raise ValueError("No calibration files found.")

    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)

    print("\nMerged Shape:")
    print("Logits:", all_logits.shape)
    print("Labels:", all_labels.shape)

    return all_logits, all_labels

# ===============================
# TEMPERATURE APPLICATION
# ===============================

def apply_temperature(logits, T):
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    probs = F.softmax(logits_tensor / T, dim=-1)
    return probs.numpy()

# ===============================
# THRESHOLD REJECTION
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
# MAIN GLOBAL SWEEP
# ===============================

def main():

    logits, labels = load_all_calibration_logits()

    best_T = None
    best_f1 = 0
    f1_scores = []

    print("\nSweeping Global Temperature (with threshold)...\n")

    for T in T_VALUES:
        probs = apply_temperature(logits, T)
        preds = apply_threshold(probs, TAU)

        macro_f1 = f1_score(labels, preds, average="macro")
        f1_scores.append(macro_f1)

        print(f"T={T:.2f} | Macro-F1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_T = T

    print("\n==============================")
    print(f"Best Global T: {best_T:.2f}")
    print(f"Best Macro-F1: {best_f1:.4f}")
    print("==============================")

    # ===============================
    # Plot
    # ===============================

    plt.figure(figsize=(8, 5))
    plt.plot(T_VALUES, f1_scores, marker="o")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Macro F1")
    plt.title("Global Temperature Sweep (All Authors + Threshold)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("alm_global_temperature_sweep.png")

    print("\nSaved figure: alm_global_temperature_sweep.png")

if __name__ == "__main__":
    main()
