import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

# =========================================================
# GLOBAL STYLE
# =========================================================
sns.set_theme(style="whitegrid")
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 120

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "ml/outputs/roberta-base-goemotions-negative-final/training_history.csv"
OUT_DIR = BASE_DIR / "thesis_figures"
OUT_DIR.mkdir(exist_ok=True)

BEST_STEP = 1500
SMOOTH_WINDOW = 5  # try 3 or 5

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

required_cols = [
    "step",
    "train_loss",
    "train_accuracy",
    "train_macro_f1",
    "train_micro_f1",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "val_micro_f1",
    "learning_rate",
]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# =========================================================
# SMOOTH SELECTED COLUMNS
# =========================================================
smooth_cols = [
    "train_loss", "val_loss",
    "train_accuracy", "val_accuracy",
    "train_macro_f1", "val_macro_f1",
    "train_micro_f1", "val_micro_f1",
    "learning_rate"
]

for col in smooth_cols:
    df[f"{col}_smooth"] = df[col].rolling(
        window=SMOOTH_WINDOW,
        center=True,
        min_periods=1
    ).mean()

x = df["step"]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def add_best_line(best_x: int):
    plt.axvline(
        best_x,
        linestyle="--",
        linewidth=1.5,
        color="grey",
        alpha=0.9,
        label=f"Selected checkpoint ({best_x})"
    )

def finish_plot(xlabel: str, ylabel: str, title: str, filename: str):

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close()

# =========================================================
# 1. TRAINING VS VALIDATION LOSS
# =========================================================
plt.figure(figsize=(8, 5))
sns.lineplot(x=x, y=df["train_loss_smooth"], linewidth=2.2, color="blue", label="Training Loss")
sns.lineplot(x=x, y=df["val_loss_smooth"], linewidth=2.2, linestyle="--", color="red", label="Validation Loss")
add_best_line(BEST_STEP)
finish_plot(
    xlabel="Training Step",
    ylabel="Loss",
    title="Training vs Validation Loss Across Training Steps",
    filename="training_validation_loss.png"
)

# =========================================================
# 2. TRAINING VS VALIDATION MACRO-F1
# =========================================================
plt.figure(figsize=(8, 5))
sns.lineplot(x=x, y=df["train_macro_f1_smooth"], linewidth=2.2, color="blue", label="Training Macro-F1")
sns.lineplot(x=x, y=df["val_macro_f1_smooth"], linewidth=2.2, linestyle="--", color="red", label="Validation Macro-F1")
add_best_line(BEST_STEP)
finish_plot(
    xlabel="Training Step",
    ylabel="Macro-F1",
    title="Training vs Validation Macro-F1 Across Training Steps",
    filename="training_validation_macro_f1.png"
)

# =========================================================
# 3. TRAINING VS VALIDATION ACCURACY
# =========================================================
plt.figure(figsize=(8, 5))
sns.lineplot(x=x, y=df["train_accuracy_smooth"], linewidth=2.2, color="blue", label="Training Accuracy")
sns.lineplot(x=x, y=df["val_accuracy_smooth"], linewidth=2.2, linestyle="--", color="red", label="Validation Accuracy")
add_best_line(BEST_STEP)
finish_plot(
    xlabel="Training Step",
    ylabel="Accuracy",
    title="Training vs Validation Accuracy Across Training Steps",
    filename="training_validation_accuracy.png"
)

# =========================================================
# 4. VALIDATION METRICS ONLY
# =========================================================
plt.figure(figsize=(8, 5))
sns.lineplot(x=x, y=df["val_accuracy_smooth"], linewidth=2.2, color="blue", label="Validation Accuracy")
sns.lineplot(x=x, y=df["val_macro_f1_smooth"], linewidth=2.2, linestyle="--", color="orange", label="Validation Macro-F1")
sns.lineplot(x=x, y=df["val_micro_f1_smooth"], linewidth=2.2, linestyle="-.", color="purple", label="Validation Micro-F1")
add_best_line(BEST_STEP)
finish_plot(
    xlabel="Training Step",
    ylabel="Score",
    title="Validation Metrics Across Training Steps",
    filename="validation_metrics_across_steps.png"
)

# =========================================================
# 5. LEARNING RATE SCHEDULE
# =========================================================
plt.figure(figsize=(8, 5))
sns.lineplot(x=x, y=df["learning_rate_smooth"], linewidth=2.2, color="black", label="Learning Rate")
add_best_line(BEST_STEP)
finish_plot(
    xlabel="Training Step",
    ylabel="Learning Rate",
    title="Learning Rate Schedule Across Training Steps",
    filename="learning_rate_schedule.png"
)

print(f"Saved all figures to: {OUT_DIR.resolve()}")
print("Generated files:")
print("- training_validation_loss.png")
print("- training_validation_macro_f1.png")
print("- training_validation_accuracy.png")
print("- validation_metrics_across_steps.png")
print("- learning_rate_schedule.png")