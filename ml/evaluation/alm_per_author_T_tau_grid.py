import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

# =====================================
# SETTINGS
# =====================================

BASE_DATA = r"D:\FYP\Emoi-Track\data"
AUTHORS = ["Potter", "Grimms", "HCAndersen"]

T_VALUES = np.arange(0.5, 3.25, 0.25)
TAU_VALUES = np.arange(0.6, 0.99, 0.05)

# =====================================
# GRID SEARCH
# =====================================

for author in AUTHORS:

    print(f"\nProcessing Author: {author}")

    logits_path = os.path.join(BASE_DATA, author, "calibration_logits.npy")
    labels_path = os.path.join(BASE_DATA, author, "calibration_labels.npy")

    logits = np.load(logits_path)
    labels = np.load(labels_path)

    best_f1 = 0
    best_T = None
    best_tau = None

    for T in T_VALUES:

        scaled_logits = logits / T
        probs = F.softmax(torch.tensor(scaled_logits), dim=1).numpy()

        for tau in TAU_VALUES:

            preds = []
            for p in probs:
                max_prob = np.max(p)
                pred_class = np.argmax(p)

                if max_prob >= tau:
                    preds.append(pred_class)
                else:
                    preds.append(3)  # UNKNOWN class

            preds = np.array(preds)

            macro_f1 = f1_score(labels, preds, average="macro")

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_T = T
                best_tau = tau

    print(f"Best T for {author}: {best_T:.2f}")
    print(f"Best tau for {author}: {best_tau:.2f}")
    print(f"Best Macro-F1: {best_f1:.4f}")

    # =====================================
    # COMPUTE COVERAGE AT BEST (T, tau)
    # =====================================

    scaled_logits = logits / best_T
    probs = F.softmax(torch.tensor(scaled_logits), dim=1).numpy()

    preds = []
    for p in probs:
        if np.max(p) >= best_tau:
            preds.append(np.argmax(p))
        else:
            preds.append(3)  # UNKNOWN

    preds = np.array(preds)

    total_samples = len(preds)
    non_unknown = np.sum(preds != 3)

    coverage = non_unknown / total_samples

    print(f"Coverage: {coverage*100:.2f}%")
