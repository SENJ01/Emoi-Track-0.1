import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# ===============================
# CONFIG
# ===============================

BASE_DIR = "data"
BEST_T = 2.50
TAU_VALUES = [0.0, 0.76]  # for 1500; change to [0.0, 0.88] for 3000
# TAU_VALUES = [0.0, 0.88]
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

        logits_path = os.path.join(author_path, f"evaluation_logits_1500.npy")
        labels_path = os.path.join(author_path, f"evaluation_labels_1500.npy")

        if os.path.exists(logits_path) and os.path.exists(labels_path):
            print(f"Merging evaluation data from: {author}")

            logits = np.load(logits_path)
            labels = np.load(labels_path)

            all_logits.append(logits)
            all_labels.append(labels)

    if len(all_logits) == 0:
        raise ValueError(f"No evaluation files found for checkpoint.")

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
# REPORT METRICS
# ===============================

def report_metrics(labels, preds, tau):
    accept_mask = preds != UNKNOWN_CLASS
    coverage = np.sum(accept_mask) / len(labels)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")

    print("-------------------------------------------------------------")
    print(f"Checkpoint: 1500")
    print(f"Tau: {tau:.2f}")
    print(f"Coverage: {coverage:.3f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}")

    print("\nPer-Class Coverage:")
    for cls in np.unique(labels):
        class_mask = labels == cls
        class_total = np.sum(class_mask)
        class_accepted = np.sum(accept_mask & class_mask)
        if class_total > 0:
            print(f"Class {cls}: {class_accepted / class_total:.2%}")

    unknown_true = np.sum(labels == UNKNOWN_CLASS)
    unknown_pred = np.sum(preds == UNKNOWN_CLASS)
    unknown_correct = np.sum((labels == UNKNOWN_CLASS) & (preds == UNKNOWN_CLASS))

    print("\nUNKNOWN Behavior:")
    print(f"True UNKNOWN samples: {unknown_true}")
    print(f"Predicted UNKNOWN samples: {unknown_pred}")
    if unknown_true > 0:
        print(f"UNKNOWN Recall / Acceptance: {unknown_correct / unknown_true:.2%}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))

    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4))

# ===============================
# MAIN
# ===============================

def main():
    logits, labels = load_all_evaluation_logits()
    probs = apply_temperature(logits, BEST_T)

    print(f"\nSelective Prediction Evaluation (T={BEST_T})\n")

    for tau in TAU_VALUES:
        preds = apply_threshold(probs, tau)
        report_metrics(labels, preds, tau)

if __name__ == "__main__":
    main()