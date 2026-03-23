import pandas as pd
import os

# -----------------------
# Config
# -----------------------
ORIGINAL_DATA_DIR = "data/original/"
CSV_FILES = [
    os.path.join(ORIGINAL_DATA_DIR, "goemotions_1.csv"),
    os.path.join(ORIGINAL_DATA_DIR, "goemotions_2.csv"),
    os.path.join(ORIGINAL_DATA_DIR, "goemotions_3.csv")
]
OUTPUT_FILE = os.path.join(ORIGINAL_DATA_DIR, "goemotions_full.csv")

# -----------------------
# Load and concatenate
# -----------------------
dfs = []
for file in CSV_FILES:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"Loaded {file} with {len(df)} rows")
    else:
        print(f"⚠️ File not found: {file}")

full_df = pd.concat(dfs, ignore_index=True)
print(f"Total rows after combining: {len(full_df)}")
print("Columns in the combined dataset:", full_df.columns.tolist())

# Save combined CSV
full_df.to_csv(OUTPUT_FILE, index=False)
print(f"Combined CSV saved as: {OUTPUT_FILE}")
