import os
import argparse
import json
import numpy as np
import torch
import pandas as pd
import time
from pathlib import Path
from transformers import RobertaTokenizer, RobertaConfig
from ml.models.model import RobertaForMultiLabelClassification
from ml.data.data_loader import GoEmotionsProcessor
from scipy.spatial.distance import cosine
from collections import deque

from ml.utils.report_utils import (
    ensure_directory,
    save_results_json,
    generate_emotion_trajectory_plot,
    generate_pdf_report
)

class AttrDict:
    def __init__(self, d):
        self.__dict__.update(d)

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- CONFIGURATION ---
GLOBAL_THRESHOLD = 0.88
WINDOW_SIZE = 3  # k=3 → r=1
HALF_WINDOW = WINDOW_SIZE // 2

BASE_DIR = Path(__file__).resolve().parents[2]
ROBERTA_PATH = BASE_DIR / "outputs" / "roberta-base-goemotions-negative" / "checkpoint-3000"
OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)


def load_model(taxonomy):
    BASE_DIR = Path(__file__).resolve().parents[2]
    OUTPUT_DIR = BASE_DIR / "outputs"

    # Load config
    config_path = BASE_DIR / "ml" / "config" / f"{taxonomy}.json"
    with open(config_path) as f:
        cfg = AttrDict(json.load(f))

    processor = GoEmotionsProcessor(cfg)
    label_list = processor.get_labels()

    # 🔧 IMPORTANT: Use Path object, then convert to string
    ROBERTA_PATH = OUTPUT_DIR / "roberta-base-goemotions-negative" / "checkpoint-3000"

    config = RobertaConfig.from_pretrained(str(ROBERTA_PATH), num_labels=len(label_list))
    tokenizer = RobertaTokenizer.from_pretrained(str(ROBERTA_PATH))
    model = RobertaForMultiLabelClassification.from_pretrained(str(ROBERTA_PATH), config=config)

    return model, tokenizer, label_list

def predict_sentence_with_embedding(model, tokenizer, text):
    encoded = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model.roberta(**encoded)
        last_hidden = outputs.last_hidden_state
        cls_embedding = last_hidden[:, 0, :]
        logits = model.classifier(cls_embedding)
        probs = torch.softmax(logits / 2.5, dim=1).cpu().numpy()[0]
        emb = cls_embedding.cpu().numpy()[0]
    return probs, emb


def get_window_indices(n, idx):
    start = max(0, idx - HALF_WINDOW)
    end = min(n, idx + HALF_WINDOW + 1)
    return list(range(start, end))


def apply_threshold(probs, label_list):
    labels = []
    for p in probs:
        if np.max(p) >= GLOBAL_THRESHOLD:
            labels.append(label_list[np.argmax(p)])
        else:
            labels.append("unknown")
    return labels

#computes trajectory angles
def compute_trajectory_angles(embeddings):
    angles = [0.0, 0.0]
    classes = ["N/A", "N/A"]

    for i in range(2, len(embeddings)):
        v1 = embeddings[i-1] - embeddings[i-2]
        v2 = embeddings[i] - embeddings[i-1]

        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))

        if angle < 30:
            cls = "smooth"
        elif angle < 90:
            cls = "gentle"
        elif angle < 150:
            cls = "strong"
        else:
            cls = "reversal"

        angles.append(angle)
        classes.append(cls)

    return angles, classes

def compute_momentum_deviation(embeddings):
    """
    Predict embedding using simple momentum:
    e_hat_i = e_{i-1} + (e_{i-1} - e_{i-2})
    Deviation = ||e_i - e_hat_i||
    Returns normalized deviation [0,1]
    """
    deviations = [0.0, 0.0]

    raw_dev = []

    for i in range(2, len(embeddings)):
        v_prev = embeddings[i-1] - embeddings[i-2]
        e_hat = embeddings[i-1] + v_prev
        dev = np.linalg.norm(embeddings[i] - e_hat)
        raw_dev.append(dev)
        deviations.append(dev)

    # Normalize to [0,1]
    if len(raw_dev) > 0:
        min_d = min(raw_dev)
        max_d = max(raw_dev)
        range_d = max_d - min_d + 1e-8
        deviations = [
            (d - min_d) / range_d if i >= 2 else 0.0
            for i, d in enumerate(deviations)
        ]

    return deviations

def build_output_rows(segments, probs, labels, local_distances, angles=None, angle_classes=None, deviations=None):
    rows = []
    for i in range(len(segments)):
        label_shift = 1 if (i > 0 and labels[i] != labels[i - 1]) else 0
        local_shift = 1 if local_distances[i] > 0.35 else 0

        # --- NEFI components ---
        C = 0.0
        D = 0.0
        S = label_shift

        if angles is not None and i < len(angles):
            C = angles[i] / 180.0  # normalize angle to [0,1]

        if deviations is not None and i < len(deviations):
            D = deviations[i]

        NEFI = np.sqrt(C**2 + D**2 + S**2)

        
        row = {
            "Sentence_ID": i,
            "Text": segments[i],
            "Predicted_Emotion": labels[i],
            "Shift_Label": label_shift,
            "Shift_Local_Dist": local_shift,
            "Raw_Local": round(local_distances[i], 4),
            "Trajectory_Angle": round(float(angles[i]), 2) if angles else 0.0,
            "Momentum_Deviation": round(float(D), 4),
            "NEFI": round(float(NEFI), 4)
        }

        for j, score in enumerate(probs[i]):
            row[f"Score_{j}"] = round(float(score), 4)

        rows.append(row)
    return rows

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", default="negative_emo_roberta")
    parser.add_argument("--input_file", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found at {input_path}")
        return

    model, tokenizer, label_list = load_model(args.taxonomy)

    # --- STEP: NARRATIVE NORMALIZATION (50–200 Rule) ---
    with open(input_path, encoding="utf8") as f:
        segments = [l.strip() for l in f if l.strip()]

    if len(segments) < 50:
        print(f"Skipping {input_path.name}: Story is too short ({len(segments)} sentences).")
        return

    if len(segments) > 200:
        print(f"Capping {input_path.name} at 200 sentences (Removing resolution noise).")
        segments = segments[:200]

    print(f"Processing {len(segments)} sentences...")
    print(f"Threshold: {GLOBAL_THRESHOLD}")
    print(f"Sliding Window Size: {WINDOW_SIZE}")
    print("Strategy: Max-Pooling over window (classification only)")

    # --- ORIGINAL (USED FOR SHIFT DETECTION) ---
    original_probs = []
    original_embeddings = []

    for sent in segments:
        probs, emb = predict_sentence_with_embedding(model, tokenizer, sent)
        original_probs.append(probs)
        original_embeddings.append(emb)

    original_probs = np.array(original_probs)
    original_embeddings = np.array(original_embeddings)
    original_labels = apply_threshold(original_probs, label_list)

    # --- Local semantic distance ---
    local_distances = [0.0]
    for i in range(1, len(original_embeddings)):
        dist = cosine(original_embeddings[i], original_embeddings[i - 1])
        local_distances.append(dist)

    #compute angles for ORIGINAL ONLY
    angles, angle_classes = compute_trajectory_angles(original_embeddings)

    # compute momentum deviation
    deviations = compute_momentum_deviation(original_embeddings)

    #pass angles ONLY here
    original_rows = build_output_rows(
        segments, original_probs, original_labels,
        local_distances, angles=angles, angle_classes=angle_classes,
        deviations=deviations
    )

    nefi_values = np.array([r["NEFI"] for r in original_rows])
    threshold = nefi_values.mean() + nefi_values.std()

    for r in original_rows:
        r["Rupture_Flag"] = 1 if r["NEFI"] > threshold else 0

    # --- SLIDING WINDOW (classification only, not used for shifts) ---
    sliding_probs = []
    n = len(segments)

    # for i in range(n):
    #     window_indices = get_window_indices(n, i)
    #     window_probs = []

    #     for j in window_indices:
    #         probs, _ = predict_sentence_with_embedding(model, tokenizer, segments[j])
    #         window_probs.append(probs)

    #         if j == i:  # extra weight to center sentence
    #             window_probs.append(probs)

    #     window_probs = np.array(window_probs)
    #     pooled_probs = np.max(window_probs, axis=0)
    #     sliding_probs.append(pooled_probs)

    for i in range(n):
        window_indices = get_window_indices(n, i)
        window_probs = []

        for j in window_indices:
            probs = original_probs[j]   # reuse already computed probs
            window_probs.append(probs)

            if j == i:  # extra weight to center sentence
                window_probs.append(probs)

        window_probs = np.array(window_probs)
        pooled_probs = np.max(window_probs, axis=0)
        sliding_probs.append(pooled_probs)

    sliding_probs = np.array(sliding_probs)
    sliding_labels = apply_threshold(sliding_probs, label_list)

    # Dummy distances (not meaningful here, but needed for CSV format)
    zeros = [0.0] * len(segments)
    sliding_rows = build_output_rows(
        segments, sliding_probs, sliding_labels, zeros, zeros
    )

    # Save outputs
    clean_name = input_path.stem.replace(".sent", "")
    original_csv = OUTPUT_DIR / f"EmoiTrack_T{GLOBAL_THRESHOLD}_{clean_name}_ORIGINAL.csv"
    sliding_csv = OUTPUT_DIR / f"EmoiTrack_T{GLOBAL_THRESHOLD}_{clean_name}_SLIDING_MAXPOOL.csv"

    pd.DataFrame(original_rows).to_csv(original_csv, index=False)
    pd.DataFrame(sliding_rows).to_csv(sliding_csv, index=False)

    print("\nSaved outputs:")
    print(f"  ➤ Original (used for shifts): {original_csv}")
    print(f"  ➤ Sliding window (classification only): {sliding_csv}")

    # Create summary
    summary = {
        "story_name": clean_name,
        "total_sentences": len(original_rows),
        "emotion_counts": dict(
            pd.DataFrame(original_rows)["Predicted_Emotion"].value_counts()
        ),
        "avg_local_shift": float(np.mean([r["Raw_Local"] for r in original_rows])),
        "avg_nefi": float(np.mean([r["NEFI"] for r in original_rows]))
    }

    # Ensure output directory exists
    output_dir = ensure_directory(OUTPUT_DIR)

    # Save JSON
    json_path = output_dir / f"latest_results.json"
    save_results_json(json_path, summary, original_rows)

    # Generate plot
    plot_path = output_dir / f"latest_trajectory.png"
    generate_emotion_trajectory_plot(original_rows, plot_path)

    # Generate PDF
    pdf_path = output_dir / f"latest_report.pdf"
    generate_pdf_report(clean_name, summary, plot_path, pdf_path)

    print("  ➤ JSON Results:", json_path)
    print("  ➤ Trajectory Plot:", plot_path)
    print("  ➤ PDF Report:", pdf_path)

    end_time = time.time()
    total_pipeline_time = round(end_time - start_time, 2)

    # Retrieve the data saved to the JSON file
    if json_path.exists():
        with open(json_path, 'r', encoding="utf8") as f:
            results_data = json.load(f)

        # Add the execution time to the dictionary
        results_data['execution_time'] = total_pipeline_time

        # Save it back to the same file
        with open(json_path, 'w', encoding="utf8") as f:
            json.dump(results_data, f, indent=4)

    print(f"\n⏱️ Total pipeline runtime: {total_pipeline_time} seconds")
    print(f"✅ Metadata injected into {json_path.name}")

if __name__ == "__main__":
    main()
