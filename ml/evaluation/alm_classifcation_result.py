import pandas as pd
import matplotlib.pyplot as plt

# CSV path
CSV_PATH = r"D:\FYP\Emoi-Track\outputs\predictions\phase1_table.csv"

# Load data
df = pd.read_csv(CSV_PATH)

# -------------------------------
# CLEAN DATA
# -------------------------------
df.replace("-", pd.NA, inplace=True)

cols = ["anger_f1", "fear_f1", "sadness_f1", "unknown_f1", "macro_f1", "accuracy"]
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# -------------------------------
# AVG F1 PER EMOTION
# -------------------------------
avg_f1 = {
    "Anger": df["anger_f1"].mean(),
    "Fear": df["fear_f1"].mean(),
    "Sadness": df["sadness_f1"].mean(),
    "UNKNOWN": df["unknown_f1"].mean(),
}

labels = list(avg_f1.keys())
values = list(avg_f1.values())

# -------------------------------
# COLORS (custom per emotion)
# -------------------------------
colors = [
    "#e74c3c",  # Anger → red
    "#3498db",  # Fear → blue
    "#9b59b6",  # Sadness → purple
    "#7f8c8d"   # UNKNOWN → gray
]

# -------------------------------
# PLOT
# -------------------------------
plt.figure()

bars = plt.bar(labels, values, color=colors)

plt.xlabel("Emotion Class")
plt.ylabel("Average F1 Score")
plt.title("Average F1 Score per Emotion Class")

# -------------------------------
# ADD VALUES ON TOP OF BARS
# -------------------------------
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha='center',
        va='bottom'
    )

plt.tight_layout()

# Save figure
plt.savefig(r"D:\FYP\Emoi-Track\outputs\predictions\avg_f1_per_emotion.png")

print("✅ Chart saved with labels and colors!")