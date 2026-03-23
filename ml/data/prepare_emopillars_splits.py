import json
import csv
import os

# Target emotions
TARGET_EMOTIONS = {"anger", "fear", "sadness"}

# Raw Emopillars directory
INPUT_DIR = r"D:\FYP\Emoi-Track\data\emopillars_raw"

# Where processed CSVs will be saved
OUTPUT_DIR = r"D:\FYP\Emoi-Track\data\emopillars_negative"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_split(split_name):
    input_path = os.path.join(INPUT_DIR, f"{split_name}.jsonl")
    output_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")

    kept_samples = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", newline="", encoding="utf-8") as outfile:

        writer = csv.writer(outfile)
        writer.writerow(["text", "label"])

        for line in infile:
            sample = json.loads(line)

            text = sample["utterance"]
            labels = []

            # all_emotions_mapped is a list like:
            # [["anger", 0.4], ["sadness", 0.8], ...]
            for emotion, score in sample["all_emotions_mapped"]:
                if emotion in TARGET_EMOTIONS and score > 0:
                    labels.append(emotion)

            if not labels:
                continue  # skip samples with no target emotions

            writer.writerow([text, ",".join(labels)])
            kept_samples += 1

    print(f"{split_name}.jsonl → kept {kept_samples} samples")


# Process each predefined split
process_split("train")
process_split("dev")
process_split("test")

print("All splits processed successfully.")
