import pandas as pd
from pathlib import Path

CLASS_ORDER = ["anger", "fear", "sadness"]

# Base paths
base_dir = Path(__file__).resolve().parent.parent  # GoEmotions-pytorch root
data_dir = base_dir / "data/negative_emo"
output_dir = base_dir / "dataset_analysis/analysis_outputs"
output_dir.mkdir(exist_ok=True)

# Load splits
train_df = pd.read_csv(data_dir / "train_negative_emo.csv")
dev_df = pd.read_csv(data_dir / "dev_negative_emo.csv")
test_df = pd.read_csv(data_dir / "test_negative_emo.csv")

all_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

def explode_labels(df):
    df = df.copy()
    df["label"] = df["label"].astype(str).str.split(",")
    return df.explode("label")

def count_by_class(df, class_order):
    counts = explode_labels(df)["label"].value_counts().to_dict()
    ordered_counts = {cls: counts.get(cls, 0) for cls in class_order}
    return pd.DataFrame({
        "Class": class_order,
        "Count": [ordered_counts[c] for c in class_order]
    })

# 1. Per-class counts per split
train_class_counts = count_by_class(train_df, CLASS_ORDER)
train_class_counts.to_csv(output_dir / "train_class_counts.csv", index=False)

dev_class_counts = count_by_class(dev_df, CLASS_ORDER)
dev_class_counts.to_csv(output_dir / "dev_class_counts.csv", index=False)

test_class_counts = count_by_class(test_df, CLASS_ORDER)
test_class_counts.to_csv(output_dir / "test_class_counts.csv", index=False)

# 2. Overall per-class counts
overall_class_counts = count_by_class(all_df, CLASS_ORDER)
overall_class_counts.columns = ["Class", "Total_Count"]
overall_class_counts.to_csv(output_dir / "overall_class_counts.csv", index=False)

# 3. Label combination counts (multi-label patterns)
combo_counts = all_df["label"].value_counts().reset_index()
combo_counts.columns = ["Label_Combination", "Count"]
combo_counts.to_csv(output_dir / "label_combinations_all.csv", index=False)

multi_label_only = combo_counts[combo_counts["Label_Combination"].str.contains(",")]
multi_label_only.to_csv(output_dir / "label_combinations_multi_only.csv", index=False)

# 4. Dataset size summary
dataset_sizes = pd.DataFrame({
    "Split": ["Train", "Dev", "Test", "Total"],
    "Samples": [len(train_df), len(dev_df), len(test_df), len(all_df)]
})
dataset_sizes.to_csv(output_dir / "dataset_sizes.csv", index=False)

# Print summary
print("\n=== Dataset Sizes ===")
print(dataset_sizes)

print("\n=== Overall Class Distribution ===")
print(overall_class_counts)

print("\n=== Multi-label Combinations ===")
print(multi_label_only)

print(f"\nAll statistics saved to: {output_dir.resolve()}")
