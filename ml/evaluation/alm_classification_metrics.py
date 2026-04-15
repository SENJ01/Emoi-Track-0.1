import os
import json
import pandas as pd

JSON_FILES = [
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_a_story.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_the_white_snake.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_96_the_three_little_birds.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_163_the_glass_coffin.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_ginger_and_pickles.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_clever_elsie.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_por_duck.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_cat-skin.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_shirtcol.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_daisy.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_hansel_and_gretel.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_the_twelve_dancing_princesses.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_the_golden_bird.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_114_the_cunning_little_tailor.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_the_tale_of_peter_rabbit.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_91_the_gnome.json",
]

OUTPUT_CSV = r"D:\FYP\Emoi-Track\outputs\predictions\phase1_clean_table.csv"

EMOTIONS = ["anger", "fear", "sadness", "unknown"]


def get_metric(report, emotion, metric):
    if emotion in report and isinstance(report[emotion], dict):
        return report[emotion].get(metric, "-")
    return "-"


def clean_float(val):
    if isinstance(val, (int, float)):
        return round(val, 4)
    return "-"


def clean_support(val):
    if isinstance(val, (int, float)):
        return int(val)
    return "-"


def extract_story_name(path):
    return os.path.basename(path).replace("full_research_report_", "").replace(".json", "")


rows = []

for path in JSON_FILES:
    if not os.path.exists(path):
        print(f"Skipping: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    before = data.get("phase1", {}).get("before_sliding", {})
    report = before.get("report", {})

    row = {
        "story": extract_story_name(path),
        "accuracy": round(before.get("accuracy", 0), 4),
        "macro_f1": round(before.get("macro_f1", 0), 4),
    }

    for emo in EMOTIONS:
        f1_val = get_metric(report, emo, "f1-score")
        support_val = get_metric(report, emo, "support")

        row[f"{emo}_f1"] = clean_float(f1_val)
        row[f"{emo}_support"] = clean_support(support_val)

    rows.append(row)

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_CSV, index=False)

print("CSV saved at:", OUTPUT_CSV)
print(df)