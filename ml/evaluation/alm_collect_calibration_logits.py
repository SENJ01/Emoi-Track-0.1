import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from ml.models.model import RobertaForMultiLabelClassification

# ==========================
# ARGUMENTS
# ==========================
parser = argparse.ArgumentParser()
parser.add_argument("--author", type=str, required=True)
args = parser.parse_args()

BASE_DATA = r"D:\FYP\Emoi-Track\data"
MODEL_PATH = r"D:\FYP\Emoi-Track\outputs\roberta-base-goemotions-negative\checkpoint-3000"

AUTHOR_DIR = os.path.join(BASE_DATA, args.author)
CALIB_LIST_PATH = os.path.join(AUTHOR_DIR, "calibration_file_list.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# ALM → PILLAR MAPPING
# ==========================
# ALM merged 6-class codes:
# 2 = Angry-Disgusted
# 3 = Fearful
# 4 = Happy
# 6 = Sad
# 7 = Surprised

ALM_TO_PILLAR = {
    2: 0,  # Angry-Disgusted → anger
    3: 1,  # Fearful → fear
    6: 2,  # Sad → sadness
}

UNKNOWN_LABEL = 3  # for Happy (4) and Surprised (7)

# ==========================
# LOAD MODEL
# ==========================
config = AutoConfig.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = RobertaForMultiLabelClassification.from_pretrained(
    MODEL_PATH,
    config=config
)

model.to(device)
model.eval()

all_logits = []
all_labels = []

# ==========================
# LOAD CALIBRATION STORIES
# ==========================
with open(CALIB_LIST_PATH, "r") as f:
    story_files = [line.strip() for line in f.readlines()]

for story_file in tqdm(story_files):

    story_name = story_file.replace(".agree", "")

    sent_path = os.path.join(BASE_DATA, args.author, "sent", story_name + ".sent.okpuncs")
    gold_path = os.path.join(BASE_DATA, args.author, "agree-sent", story_file)

    with open(sent_path, "r", encoding="utf-8") as sf:
        sentences = sf.readlines()

    with open(gold_path, "r", encoding="utf-8") as gf:
        gold_lines = gf.readlines()

    for gold_line in gold_lines:

        parts = gold_line.strip().split("@")
        if len(parts) < 3:
            continue

        sentence_index = int(parts[0])
        alm_label = int(parts[1])

        if sentence_index >= len(sentences):
            continue

        sentence = sentences[sentence_index].strip()

        # Map ALM label → pillar index
        if alm_label in ALM_TO_PILLAR:
            label = ALM_TO_PILLAR[alm_label]
        else:
            label = UNKNOWN_LABEL  # Happy or Surprised

        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs[1].squeeze().cpu().numpy()

        all_logits.append(logits)
        all_labels.append(label)

# ==========================
# SAVE
# ==========================
all_logits = np.array(all_logits)
all_labels = np.array(all_labels)

np.save(os.path.join(AUTHOR_DIR, "calibration_logits.npy"), all_logits)
np.save(os.path.join(AUTHOR_DIR, "calibration_labels.npy"), all_labels)

print("Saved calibration logits.")
print("Logits shape:", all_logits.shape)
print("Label distribution:", np.unique(all_labels, return_counts=True))
