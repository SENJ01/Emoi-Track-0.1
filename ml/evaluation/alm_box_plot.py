import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

label = []
local = []
angle = []
nefi = []
prob = []

for file_path in JSON_FILES:
    path = Path(file_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        label.append(float(data["phase2"]["label_shift"]["report"]["1"]["f1-score"]))
        local.append(float(data["phase2"]["local_distance_shift"]["report"]["1"]["f1-score"]))
        angle.append(float(data["phase2"]["trajectory_angle_shift"]["report"]["1"]["f1-score"]))
        nefi.append(float(data["phase2"]["nefi_rupture_shift"]["report"]["1"]["f1-score"]))
        prob.append(float(data["phase3"]["probability_shift"]["report"]["1"]["f1-score"]))

        print(f"Loaded: {path.name}")

    except Exception as e:
        print(f"Error in {path.name}: {e}")

print("\nCounts:")
print("Label:", len(label))
print("Local:", len(local))
print("Angle:", len(angle))
print("NEFI:", len(nefi))
print("Prob:", len(prob))

print("\n=== DEBUG MEANS ===")
print("Label mean:", np.mean(label))
print("Local mean:", np.mean(local))
print("Angle mean:", np.mean(angle))
print("NEFI mean:", np.mean(nefi))
print("Prob mean:", np.mean(prob))

data = [label, local, angle, nefi, prob]

plt.figure(figsize=(9, 6))
plt.boxplot(
    data,
    tick_labels=[
        "Label-Based",
        "Local Distance",
        "Trajectory Angle",
        "NEFI Rupture",
        "Probability-Based"
    ],
    showmeans=True
)

plt.ylabel("F1 Score (Shift Class)")
plt.title("Boxplot of Shift Detection Performance Across Narratives")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("shift_boxplot.png", dpi=300)
plt.show()