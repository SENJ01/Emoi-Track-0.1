import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, accuracy_score
from tabulate import tabulate   # pretty table
import os

# ---------- Load dataset ----------
csv_path = "data/negative_emo/negative_isear.csv"  
df = pd.read_csv(csv_path)

texts = df["content"].tolist()
y_true = df["sentiment"].str.lower().tolist()  # gold labels

# ----------labels ----------
NEGATIVE_LABELS = ["anger", "fear", "sadness"]

# ---------- Models to test ----------
models_to_test = [
    # "monologg/bert-base-cased-goemotions-ekman",  # GoEmotions Ekman
    # "cardiffnlp/twitter-roberta-base-emotion",   # Twitter RoBERTa Emotion
    # "j-hartmann/emotion-english-distilroberta-base", # DistilRoBERTa emotions
    # "joeddav/distilbert-base-uncased-go-emotions-student"
    # "tae898/emoberta-base"
    "SamLowe/roberta-base-go_emotions"
    # "arpanghoshal/EmoRoBERTa",
    # "ayoubkirouane/BERT-Emotions-Classifier"
]

# Create output folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# ---------- Loop through each model ----------
for model_name in models_to_test:
    print(f"\n=== Testing model: {model_name} ===")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None  # replaces return_all_scores=True
    )
    
    # ---------- Run predictions ----------
    y_pred = []
    results = []

    for text in texts:
        pred = classifier(text)
        # Keep only negative emotions
        negative_pred = [x for x in pred[0] if x['label'].lower() in NEGATIVE_LABELS]

        if negative_pred:  # pick the label with max confidence
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
    pred_file = f"outputs/{model_name.split('/')[-1]}_negative_emotions_results.csv"
    results_df.to_csv(pred_file, index=False)
    
    # ---------- Performance report ----------
    report = classification_report(y_true, y_pred, labels=NEGATIVE_LABELS, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)

    # Add accuracy into the report
    report["accuracy"] = {"precision": None, "recall": None, "f1-score": accuracy, "support": len(y_true)}
    
    # Save performance as CSV
    report_df = pd.DataFrame(report).transpose()
    metrics_file = f"outputs/{model_name.split('/')[-1]}_performance_metrics.csv"
    report_df.to_csv(metrics_file, index=True)
    
    # Pretty print to console
    print("\n=== Classification Report ===")
    print(tabulate(report_df, headers="keys", tablefmt="pretty", floatfmt=".3f"))
    print(f"\nFiles saved: '{pred_file}' and '{metrics_file}'\n")
