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
    generate_pdf_report,
)


class AttrDict:
    def __init__(self, d):
        self.__dict__.update(d)


# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- CONFIGURATION ---
GLOBAL_THRESHOLD = 0.76
WINDOW_SIZE = 5  # k=3 → r=1
HALF_WINDOW = WINDOW_SIZE // 2

BASE_DIR = Path(__file__).resolve().parents[2]
# ROBERTA_PATH = (
#     BASE_DIR / "outputs" / "roberta-base-goemotions-negative" / "checkpoint-3000"
# )
ROBERTA_PATH = (
    BASE_DIR
    / "ml"
    / "outputs"
    / "roberta-base-goemotions-negative-final"
    / "checkpoint-1500"
)
# ROBERTA_PATH = BASE_DIR / "outputs" / "roberta-base-emopillars-negative" / "checkpoint-15000"
OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"
# OUTPUT_DIR = BASE_DIR / "outputs" / "predictions_1500"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)


def load_model(taxonomy):
    BASE_DIR = Path(__file__).resolve().parents[2]
    OUTPUT_DIR = BASE_DIR / "outputs"

    # Load config
    # config_path = BASE_DIR / "ml" / "config" / f"{taxonomy}.json"
    # with open(config_path) as f:
    #     cfg = AttrDict(json.load(f))

    # processor = GoEmotionsProcessor(cfg)
    # label_list = processor.get_labels()
    config_path = BASE_DIR / "ml" / "config" / f"{taxonomy}.json"
    with open(config_path) as f:
        cfg = AttrDict(json.load(f))

    cfg.data_dir = str(BASE_DIR / "data" / "negative_emo")

    processor = GoEmotionsProcessor(cfg)
    label_list = processor.get_labels()

    # Use Path object, then convert to string
    # ROBERTA_PATH = OUTPUT_DIR / "roberta-base-goemotions-negative" / "checkpoint-3000"
    ROBERTA_PATH = Path(
        r"D:\FYP\Emoi-Track\ml\outputs\roberta-base-goemotions-negative-final\checkpoint-1500"
    )
    # ROBERTA_PATH = Path(r"D:\FYP\Emoi-Track\outputs\roberta-base-emopillars-negative\checkpoint-15000")
    config = RobertaConfig.from_pretrained(
        str(ROBERTA_PATH), num_labels=len(label_list)
    )
    tokenizer = RobertaTokenizer.from_pretrained(str(ROBERTA_PATH))
    model = RobertaForMultiLabelClassification.from_pretrained(
        str(ROBERTA_PATH), config=config
    )

    return model, tokenizer, label_list


def predict_sentence_with_embedding(model, tokenizer, text):
    encoded = tokenizer(
        text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        # Extract contextual embedding and compute probabilities with temperature scaling
        outputs = model.roberta(**encoded)
        last_hidden = outputs.last_hidden_state
        # [CLS] token representation
        cls_embedding = last_hidden[:, 0, :]
        logits = model.classifier(cls_embedding)
        # Apply temperature scaling (T = 2.5) before softmax
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
        # Apply global confidence threshold to filter uncertain predictions
        if np.max(p) >= GLOBAL_THRESHOLD:
            labels.append(label_list[np.argmax(p)])  # assign highest-probability label
        else:
            labels.append("unknown")  # reject low-confidence predictions
    return labels


# Compute trajectory angles using consecutive embedding movement vectors
def compute_trajectory_angles(embeddings):
    angles = [0.0, 0.0]
    classes = ["N/A", "N/A"]

    for i in range(2, len(embeddings)):
        v1 = embeddings[i - 1] - embeddings[i - 2]
        v2 = embeddings[i] - embeddings[i - 1]

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

    # Estimate expected movement using simple momentum and measure deviation
    deviations = [0.0, 0.0]
    raw_dev = []

    for i in range(2, len(embeddings)):
        v_prev = embeddings[i - 1] - embeddings[i - 2]
        e_hat = embeddings[i - 1] + v_prev
        dev = np.linalg.norm(embeddings[i] - e_hat)
        raw_dev.append(dev)
        deviations.append(dev)

    # Normalize momentum deviation values to the range [0,1]
    if len(raw_dev) > 0:
        min_d = min(raw_dev)
        max_d = max(raw_dev)
        range_d = max_d - min_d + 1e-8
        deviations = [
            (d - min_d) / range_d if i >= 2 else 0.0 for i, d in enumerate(deviations)
        ]

    return deviations


def build_output_rows(
    segments,
    probs,
    labels,
    local_distances,
    angles=None,
    angle_classes=None,
    deviations=None,
):
    rows = []
    for i in range(len(segments)):
        # Derive discrete shift indicators from label change and local distance
        label_shift = 1 if (i > 0 and labels[i] != labels[i - 1]) else 0
        local_shift = 1 if local_distances[i] > 0.35 else 0

        # Initialize NEFI components
        C = 0.0
        D = 0.0
        S = label_shift  # binary label-shift signal

        # Normalize trajectory angle to the range [0,1]
        if angles is not None and i < len(angles):
            C = angles[i] / 180.0

        # Use normalized momentum deviation
        if deviations is not None and i < len(deviations):
            D = deviations[i]

        # Compute composite rupture score
        NEFI = np.sqrt(C**2 + D**2 + S**2)

        # Store segment-level outputs and shift indicators
        row = {
            "Sentence_ID": i,
            "Text": segments[i],
            "Predicted_Emotion": labels[i],
            "Shift_Label": label_shift,
            "Shift_Local_Dist": local_shift,
            "Raw_Local": round(local_distances[i], 4),
            "Trajectory_Angle": round(float(angles[i]), 2) if angles else 0.0,
            "Momentum_Deviation": round(float(D), 4),
            "NEFI": round(float(NEFI), 4),
        }
        # Store probability scores for each target emotion class
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

    # Read input file and construct ordered sentence segments
    with open(input_path, encoding="utf8") as f:
        segments = [l.strip() for l in f if l.strip()]  # remove empty lines

    if len(segments) < 50:
        print(
            f"Skipping {input_path.name}: Story is too short ({len(segments)} sentences)."
        )
        return

    if len(segments) > 200:
        print(
            f"Capping {input_path.name} at 200 sentences (Removing resolution noise)."
        )
        segments = segments[:200]

    print(f"Processing {len(segments)} sentences...")
    print(f"Threshold: {GLOBAL_THRESHOLD}")
    print(f"Sliding Window Size: {WINDOW_SIZE}")
    print("Strategy: Max-Pooling over window (classification only)")

    # ORIGINAL (USED FOR SHIFT DETECTION)
    original_probs = []
    original_embeddings = []

    for sent in segments:
        probs, emb = predict_sentence_with_embedding(model, tokenizer, sent)
        original_probs.append(probs)
        original_embeddings.append(emb)

    original_probs = np.array(original_probs)
    original_embeddings = np.array(original_embeddings)
    original_labels = apply_threshold(original_probs, label_list)

    # Compute local semantic distance between consecutive segment embeddings
    local_distances = [0.0]

    for i in range(1, len(original_embeddings)):
        dist = cosine(original_embeddings[i], original_embeddings[i - 1])
        local_distances.append(dist)

    # compute angles for ORIGINAL ONLY
    angles, angle_classes = compute_trajectory_angles(original_embeddings)

    # compute momentum deviation
    deviations = compute_momentum_deviation(original_embeddings)

    # pass angles ONLY here
    original_rows = build_output_rows(
        segments,
        original_probs,
        original_labels,
        local_distances,
        angles=angles,
        angle_classes=angle_classes,
        deviations=deviations,
    )

    # Compute adaptive rupture threshold based on NEFI distribution
    nefi_values = np.array([r["NEFI"] for r in original_rows])
    threshold = nefi_values.mean() + nefi_values.std()

    # Flag segments exceeding threshold as rupture points
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
            probs = original_probs[j]  # reuse already computed probs
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
    original_csv = (
        OUTPUT_DIR / f"EmoiTrack_T{GLOBAL_THRESHOLD}_{clean_name}_ORIGINAL.csv"
    )
    sliding_csv = (
        OUTPUT_DIR / f"EmoiTrack_T{GLOBAL_THRESHOLD}_{clean_name}_SLIDING_MAXPOOL.csv"
    )

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
        "avg_nefi": float(np.mean([r["NEFI"] for r in original_rows])),
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
        with open(json_path, "r", encoding="utf8") as f:
            results_data = json.load(f)

        # Add the execution time to the dictionary
        results_data["execution_time"] = total_pipeline_time

        # Save it back to the same file
        with open(json_path, "w", encoding="utf8") as f:
            json.dump(results_data, f, indent=4)

    print(f"\n⏱️ Total pipeline runtime: {total_pipeline_time} seconds")
    print(f"✅ Metadata injected into {json_path.name}")


if __name__ == "__main__":
    main()
