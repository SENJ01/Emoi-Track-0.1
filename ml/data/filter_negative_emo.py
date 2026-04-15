import pandas as pd
import os
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize

# -----------------------
# Ensure Punkt tokenizer is available
# -----------------------
PUNKT_DIR = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(PUNKT_DIR):
    os.makedirs(PUNKT_DIR)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=PUNKT_DIR)

nltk.data.path.append(PUNKT_DIR)

# Input file and output directory for the negative-emotion subset
ORIGINAL_DATA_PATH = "data/original/goemotions_full.csv"
NEGATIVE_EMO_DIR = "data/negative_emo/"

# Target negative emotion classes used in EMOi-Track
NEGATIVE_LABELS = {"anger", "fear", "sadness"}

DEV_SIZE = 0.1
TEST_SIZE = 0.1
RANDOM_STATE = 42

os.makedirs(NEGATIVE_EMO_DIR, exist_ok=True)

# Load the combined dataset
df = pd.read_csv(ORIGINAL_DATA_PATH)
print(f"Original dataset rows: {len(df)}")

# Extract the negative labels present in each row
def get_neg_labels(row):
    labels = [emo for emo in NEGATIVE_LABELS if row.get(emo, 0) == 1]
    if labels:
        return ",".join(labels)
    return None

# Create a new column containing only the selected negative labels
df["neg_labels"] = df.apply(get_neg_labels, axis=1)

# Remove rows that do not contain any of the target labels
df = df.dropna(subset=["neg_labels"])
print(f"Rows after negative filter: {len(df)}")

# Convert each record into sentence-level units
rows = []
for _, row in df.iterrows():
    text = str(row["text"])  # ensure it's string
    labels = row["neg_labels"]

    # Apply sentence tokenization
    # use the full text if tokenization fails
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [text]
    
    # Store each sentence as a separate training instance
    for sent in sentences:
        rows.append({"text": sent, "label": labels})

# Create a new dataframe containing the segmented dataset
df_neg = pd.DataFrame(rows)
print(f"Rows after sentence segmentation: {len(df_neg)}")

# Split the processed dataset into training, validation, and test sets
train_df, temp_df = train_test_split(
    df_neg, test_size=DEV_SIZE + TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)
dev_df, test_df = train_test_split(
    temp_df, test_size=TEST_SIZE / (DEV_SIZE + TEST_SIZE), random_state=RANDOM_STATE, shuffle=True
)

# Save the generated dataset partitions
train_df.to_csv(os.path.join(NEGATIVE_EMO_DIR, "train_negative_emo.csv"), index=False)
dev_df.to_csv(os.path.join(NEGATIVE_EMO_DIR, "dev_negative_emo.csv"), index=False)
test_df.to_csv(os.path.join(NEGATIVE_EMO_DIR, "test_negative_emo.csv"), index=False)

print("Negative-only train/dev/test CSVs created successfully!")
print(f"Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")