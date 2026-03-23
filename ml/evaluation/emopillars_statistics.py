import pandas as pd
from collections import Counter
import os

# Input data location
DATA_DIR = r"D:\FYP\Emoi-Track\data\emopillars_negative"

# Output location
OUTPUT_DIR = r"D:\FYP\Emoi-Track\outputs\analysis\Emopillars"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = [
    "emopillars_negative_train.csv",
    "emopillars_negative_dev.csv",
    "emopillars_negative_test.csv"
]

for file in FILES:
    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path)

    cleaned_labels = []

    for label_str in df["label"]:
        # Split into individual labels
        labels = label_str.split(",")

        # Remove duplicates using set
        unique_labels = set(labels)

        # Sort to normalize order (anger,fear == fear,anger)
        normalized = ",".join(sorted(unique_labels))

        cleaned_labels.append(normalized)

    combo_counts = Counter(cleaned_labels)

    stats_df = pd.DataFrame(
        combo_counts.items(),
        columns=["Label_Combination", "Count"]
    ).sort_values(by="Count", ascending=False)

    output_path = os.path.join(
        OUTPUT_DIR,
        file.replace(".csv", "_label_stats.csv")
    )

    stats_df.to_csv(output_path, index=False)

    print(f"Saved cleaned stats: {output_path}")
