import pandas as pd
import os

# Directory containing the original GoEmotions CSV files
ORIGINAL_DATA_DIR = "data/original/"

# Individual source files to be merged
CSV_FILES = [
    os.path.join(ORIGINAL_DATA_DIR, "goemotions_1.csv"),
    os.path.join(ORIGINAL_DATA_DIR, "goemotions_2.csv"),
    os.path.join(ORIGINAL_DATA_DIR, "goemotions_3.csv")
]

# Output path for the combined dataset
OUTPUT_FILE = os.path.join(ORIGINAL_DATA_DIR, "goemotions_full.csv")

# Load each CSV file and store it in a list
dfs = []
for file in CSV_FILES:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"Loaded {file} with {len(df)} rows")
    else:
        print(f"File not found: {file}")

# Merge all source files into a single dataset
full_df = pd.concat(dfs, ignore_index=True)
print(f"Total rows after combining: {len(full_df)}")
print("Columns in the combined dataset:", full_df.columns.tolist())

# Save the combined dataset for further preprocessing
full_df.to_csv(OUTPUT_FILE, index=False)
print(f"Combined CSV saved as: {OUTPUT_FILE}")
