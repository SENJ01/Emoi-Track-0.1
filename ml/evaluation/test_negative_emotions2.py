import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, accuracy_score
from tabulate import tabulate   # pretty table

# ---------- Load dataset ----------
csv_path = "eng_dataset.csv"
df = pd.read_csv(csv_path)

# df = df.head(10)

texts = df["content"].tolist()
y_true = df["sentiment"].str.lower().tolist()  # gold labels

# ---------- Load pre-trained GoEmotions Ekman model ----------
model_name = "monologg/bert-base-cased-goemotions-ekman"
tokenizer_name_or_path = "monologg/bert-base-cased-goemotions-ekman"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None  # replaces return_all_scores=True
)

# ---------- Define negative labels ----------
NEGATIVE_LABELS = ["anger", "fear", "sadness"]

# ---------- Run predictions ----------
y_pred = []
results = []

for text in texts:
    pred = classifier(text)
    # Keep only negative emotions
    negative_pred = [x for x in pred[0] if x['label'].lower() in NEGATIVE_LABELS]

    if negative_pred:   # pick the label with max confidence
        best = max(negative_pred, key=lambda x: x['score'])
        y_pred.append(best['label'].lower())
    else:
        y_pred.append("none")

    results.append({
        "text": text,
        "predictions": negative_pred
    })

# ---------- Save predictions ----------
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/negative_emotions_results.csv", index=False)

# ---------- Performance report ----------
report = classification_report(y_true, y_pred, labels=NEGATIVE_LABELS, output_dict=True)
accuracy = accuracy_score(y_true, y_pred)

# Add accuracy into the report
report["accuracy"] = {"precision": None, "recall": None, "f1-score": accuracy, "support": len(y_true)}

# Save performance as CSV
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("outputs/performance_metrics.csv", index=True)

# Pretty print to console
print("\n=== Classification Report ===")
print(tabulate(report_df, headers="keys", tablefmt="pretty", floatfmt=".3f"))
print("\nFiles saved: 'outputs/negative_emotions_results.csv' and 'outputs/performance_metrics.csv'")
