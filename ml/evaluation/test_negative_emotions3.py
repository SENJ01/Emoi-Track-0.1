import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support

# 1. Load your CSV
df = pd.read_csv("goemotions_1.csv")

# Extract text + labels (GoEmotions has multiple annotators -> treat columns as binary indicators)
texts = df["text"].tolist()

# Multi-hot to single-label (take first negative emotion that is 1, or neutral if none)
NEGATIVE_LABELS = ["anger", "fear", "sadness"]

y_true = []
for _, row in df.iterrows():
    found = None
    for label in NEGATIVE_LABELS:
        if row[label] == 1:
            found = label
            break
    if found:
        y_true.append(found)
    else:
        y_true.append("neutral")

# 2. Load pretrained model
model_name = "monologg/bert-base-cased-goemotions-ekman"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True # To get all scores, can simplify
)

# 3. Run predictions
y_pred = []
for text in texts:
    pred = classifier(text)
    negative_pred = [x for x in pred[0] if x['label'].lower() in NEGATIVE_LABELS]
    if negative_pred:
        best = max(negative_pred, key=lambda x: x['score'])
        y_pred.append(best['label'].lower())
    else:
        y_pred.append("neutral")

# 4. Evaluation
print(" Classification Report (per class):")
print(classification_report(y_true, y_pred, labels=NEGATIVE_LABELS))

# Macro-average across negative emotions only
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=NEGATIVE_LABELS, average="macro"
)
print(f"\n Macro-average across anger, fear, sadness:")
print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# 5. Save results
df_results = pd.DataFrame({"text": texts, "true": y_true, "pred": y_pred})
df_results.to_csv("goemotions_negative_eval.csv", index=False)
print("\n Results saved to goemotions_negative_eval.csv")
