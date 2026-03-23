import os
import random
import argparse

# ARGUMENT PARSER
parser = argparse.ArgumentParser()
parser.add_argument("--author", type=str, required=True,
                    help="Author name (e.g., Potter, Andersen, Grimm)")
parser.add_argument("--target_per_class", type=int, default=5)
args = parser.parse_args()

# CONFIG
BASE_DIR = r"D:\FYP\Emoi-Track\data"
AGREE_DIR = os.path.join(BASE_DIR, args.author, "agree-sent")
OUTPUT_DIR = os.path.join(BASE_DIR, args.author)

TARGET_PER_CLASS = args.target_per_class
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# LABEL MAP
label_map = {
    2: 0,  # anger
    3: 1,  # fear
    6: 2,  # sadness
    4: 3,  # UNKNOWN
    7: 3   # UNKNOWN
}

# GET FILES
agree_files = [f for f in os.listdir(AGREE_DIR) if f.endswith(".agree")]
random.shuffle(agree_files)

class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
calibration_files = set()

# ITERATE OVER STORIES
for filename in agree_files:
    
    if all(count >= TARGET_PER_CLASS for count in class_counts.values()):
        break
    
    filepath = os.path.join(AGREE_DIR, filename)
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    story_used = False
    
    for line in lines:
        try:
            parts = line.strip().split("@")
            label_code = int(parts[1])
        except:
            continue
        
        if label_code not in label_map:
            continue
        
        mapped_label = label_map[label_code]
        
        if class_counts[mapped_label] < TARGET_PER_CLASS:
            class_counts[mapped_label] += 1
            story_used = True
    
    if story_used:
        calibration_files.add(filename)

# EVALUATION FILES
evaluation_files = [
    f for f in agree_files if f not in calibration_files
]

# SAVE OUTPUT
with open(os.path.join(OUTPUT_DIR, "calibration_file_list.txt"), "w") as f:
    for file in sorted(calibration_files):
        f.write(file + "\n")

with open(os.path.join(OUTPUT_DIR, "evaluation_file_list.txt"), "w") as f:
    for file in sorted(evaluation_files):
        f.write(file + "\n")

print("Calibration class counts:", class_counts)
print("Calibration stories:", len(calibration_files))
print("Evaluation stories:", len(evaluation_files))
