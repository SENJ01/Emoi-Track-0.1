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
HF_REVISION = "main"

BASE_DIR = Path(__file__).resolve().parents[2]
ROBERTA_PATH = (
    BASE_DIR
    / "ml"
    / "outputs"
    / "roberta-base-goemotions-negative-final"
    / "checkpoint-1500"
)
OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)


def load_model(taxonomy):
    """Load model, tokenizer, and label list for the selected taxonomy."""
    base_dir = Path(__file__).resolve().parents[2]
    config_path = base_dir / "ml" / "config" / f"{taxonomy}.json"

    with open(config_path, encoding="utf-8") as f:
        cfg = AttrDict(json.load(f))

    cfg.data_dir = str(base_dir / "data" / "negative_emo")

    processor = GoEmotionsProcessor(cfg)
    label_list = processor.get_labels()

    config = RobertaConfig.from_pretrained(
        str(ROBERTA_PATH),
        revision=HF_REVISION,
        num_labels=len(label_list),
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        str(ROBERTA_PATH),
        revision=HF_REVISION,
    )
    model = RobertaForMultiLabelClassification.from_pretrained(
        str(ROBERTA_PATH),
        revision=HF_REVISION,
        config=config,
    )

    return model, tokenizer, label_list


def predict_sentence_with_embedding(model, tokenizer, text):
    """Predict class probabilities and extract the CLS embedding."""
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        # Extract contextual embedding and compute probabilities
        outputs = model.roberta(**encoded)
        last_hidden = outputs.last_hidden_state
        # [CLS] token representation
        cls_embedding = last_hidden[:, 0, :]
        logits = model.classifier(cls_embedding)

        # Apply temperature scaling before softmax
        probs = torch.softmax(logits / 2.5, dim=1).cpu().numpy()[0]
        emb = cls_embedding.cpu().numpy()[0]

    return probs, emb


def get_window_indices(n, idx):
    """Return indices inside the centered sliding window."""
    start = max(0, idx - HALF_WINDOW)
    end = min(n, idx + HALF_WINDOW + 1)
    return list(range(start, end))


def apply_threshold(probs, label_list):
    """Convert probabilities to labels using a global confidence threshold."""
    labels = []

    for p in probs:
        # Apply global confidence threshold to filter uncertain predictions
        if np.max(p) >= GLOBAL_THRESHOLD:
            labels.append(label_list[np.argmax(p)])  # assign highest-probability label
        else:
            labels.append("unknown")  # reject low-confidence predictions
    return labels


def compute_trajectory_angles(embeddings):
    """Compute trajectory angles between consecutive embedding movements."""
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
    """Measure deviation from a simple momentum-based embedding estimate."""
    deviations = [0.0, 0.0]
    raw_dev = []

    for i in range(2, len(embeddings)):
        v_prev = embeddings[i - 1] - embeddings[i - 2]
        e_hat = embeddings[i - 1] + v_prev
        dev = np.linalg.norm(embeddings[i] - e_hat)
        raw_dev.append(dev)
        deviations.append(dev)

    # Normalize values to [0, 1]
    if raw_dev:
        min_d = min(raw_dev)
        max_d = max(raw_dev)
        range_d = max_d - min_d + 1e-8
        deviations = [
            (d - min_d) / range_d if i >= 2 else 0.0
            for i, d in enumerate(deviations)
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
    """Build row-wise output records for CSV export."""
    rows = []

    for i in range(len(segments)):
        # Derive shift indicators
        label_shift = 1 if (i > 0 and labels[i] != labels[i - 1]) else 0
        local_shift = 1 if local_distances[i] > 0.35 else 0

        # Initialize NEFI components
        c_val = 0.0
        d_val = 0.0
        s_val = label_shift

        if angles is not None and i < len(angles):
            c_val = angles[i] / 180.0

        if deviations is not None and i < len(deviations):
            d_val = deviations[i]

        nefi = np.sqrt(c_val**2 + d_val**2 + s_val**2)

        row = {
            "Sentence_ID": i,
            "Text": segments[i],
            "Predicted_Emotion": labels[i],
            "Shift_Label": label_shift,
            "Shift_Local_Dist": local_shift,
            "Raw_Local": round(local_distances[i], 4),
            "Trajectory_Angle": round(float(angles[i]), 2) if angles else 0.0,
            "Momentum_Deviation": round(float(d_val), 4),
            "NEFI": round(float(nefi), 4),
        }

        # Store probability scores for each class
        for j, score in enumerate(probs[i]):
            row[f"Score_{j}"] = round(float(score), 4)

        rows.append(row)

    return rows


def main():
    """Run narrative prediction and save CSV, JSON, plot, and PDF outputs."""
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

    # Read input file and keep non-empty lines
    with open(input_path, encoding="utf8") as f:
        segments = [line.strip() for line in f if line.strip()]

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

    # Original predictions used for shift detection
    original_probs = []
    original_embeddings = []

    for sent in segments:
        probs, emb = predict_sentence_with_embedding(model, tokenizer, sent)
        original_probs.append(probs)
        original_embeddings.append(emb)

    original_probs = np.array(original_probs)
    original_embeddings = np.array(original_embeddings)
    original_labels = apply_threshold(original_probs, label_list)

    # Compute local semantic distance between consecutive embeddings
    local_distances = [0.0]
    for i in range(1, len(original_embeddings)):
        dist = cosine(original_embeddings[i], original_embeddings[i - 1])
        local_distances.append(dist)

    angles, angle_classes = compute_trajectory_angles(original_embeddings)
    deviations = compute_momentum_deviation(original_embeddings)

    original_rows = build_output_rows(
        segments,
        original_probs,
        original_labels,
        local_distances,
        angles=angles,
        angle_classes=angle_classes,
        deviations=deviations,
    )

    # Compute rupture threshold from NEFI distribution
    nefi_values = np.array([r["NEFI"] for r in original_rows])
    threshold = nefi_values.mean() + nefi_values.std()

    for row in original_rows:
        row["Rupture_Flag"] = 1 if row["NEFI"] > threshold else 0

    # Sliding-window classification only
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
        segments,
        sliding_probs,
        sliding_labels,
        zeros,
        zeros,
    )

    # Save CSV outputs
    clean_name = input_path.stem.replace(".sent", "")
    original_csv = OUTPUT_DIR / f"EmoiTrack_T{GLOBAL_THRESHOLD}_{clean_name}_ORIGINAL.csv"
    sliding_csv = (
        OUTPUT_DIR / f"EmoiTrack_T{GLOBAL_THRESHOLD}_{clean_name}_SLIDING_MAXPOOL.csv"
    )

    pd.DataFrame(original_rows).to_csv(original_csv, index=False)
    pd.DataFrame(sliding_rows).to_csv(sliding_csv, index=False)

    print("\nSaved outputs:")
    print(f"  ➤ Original (used for shifts): {original_csv}")
    print(f"  ➤ Sliding window (classification only): {sliding_csv}")

    summary = {
        "story_name": clean_name,
        "total_sentences": len(original_rows),
        "emotion_counts": dict(
            pd.DataFrame(original_rows)["Predicted_Emotion"].value_counts()
        ),
        "avg_local_shift": float(np.mean([r["Raw_Local"] for r in original_rows])),
        "avg_nefi": float(np.mean([r["NEFI"] for r in original_rows])),
    }

    output_dir = ensure_directory(OUTPUT_DIR)

    # Save JSON
    json_path = output_dir / "latest_results.json"
    save_results_json(json_path, summary, original_rows)

    # Generate plot
    plot_path = output_dir / "latest_trajectory.png"
    generate_emotion_trajectory_plot(original_rows, plot_path)

    # Generate PDF
    pdf_path = output_dir / "latest_report.pdf"
    generate_pdf_report(clean_name, summary, plot_path, pdf_path)

    print("  ➤ JSON Results:", json_path)
    print("  ➤ Trajectory Plot:", plot_path)
    print("  ➤ PDF Report:", pdf_path)

    end_time = time.time()
    total_pipeline_time = round(end_time - start_time, 2)

    # Add execution time into the saved JSON
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