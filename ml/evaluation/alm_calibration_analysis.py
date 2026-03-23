import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ml.evaluation.alm_calibration_metrics import compute_ece

# =====================================
# SETTINGS
# =====================================

BASE_DATA = r"D:\FYP\Emoi-Track\data"
AUTHORS = ["Potter", "Grimms", "HCAndersen"]

# Use your best T per author from grid search
BEST_T = {
    "Potter": 1.00,
    "Grimms": 1.25,
    "HCAndersen": 2.00
}

# =====================================
# ANALYSIS
# =====================================

for author in AUTHORS:

    print(f"\nProcessing Author: {author}")

    logits_path = os.path.join(BASE_DATA, author, "calibration_logits.npy")
    labels_path = os.path.join(BASE_DATA, author, "calibration_labels.npy")

    logits = np.load(logits_path)
    labels = np.load(labels_path)

    # -----------------------------
    # BEFORE TEMPERATURE SCALING
    # -----------------------------
    probs_base = F.softmax(torch.tensor(logits), dim=1).numpy()
    ece_before = compute_ece(probs_base, labels)

    # -----------------------------
    # AFTER TEMPERATURE SCALING
    # -----------------------------
    T = BEST_T[author]
    scaled_logits = logits / T
    probs_scaled = F.softmax(torch.tensor(scaled_logits), dim=1).numpy()
    ece_after = compute_ece(probs_scaled, labels)

    # -----------------------------
    # AUROC FOR ERROR DETECTION
    # -----------------------------
    confidences = np.max(probs_scaled, axis=1)
    predictions = np.argmax(probs_scaled, axis=1)

    errors = (predictions != labels).astype(int)

    # Lower confidence should indicate error
    auroc = roc_auc_score(errors, -confidences)

    # -----------------------------
    # PRINT RESULTS
    # -----------------------------
    print(f"ECE Before: {ece_before:.4f}")
    print(f"ECE After:  {ece_after:.4f}")
    print(f"AUROC (Error Detection): {auroc:.4f}")
