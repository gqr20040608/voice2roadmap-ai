
"""Compare TextBlob and RoBERTa sentiment predictions."""

from pathlib import Path

import pandas as pd
from transformers import pipeline

from src.pipeline import prepare_reviews


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "english_reviews_sample.csv"
OUTPUT = ROOT / "data" / "sentiment_comparison.csv"

# Load customer reviews
raw = pd.read_csv(SOURCE)

# Existing TextBlob baseline
baseline = prepare_reviews(raw)

print("Loading RoBERTa model...", flush=True)

classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device="cpu"
)

print("Analyzing customer reviews...", flush=True)

texts = baseline["review"].fillna("").astype(str).tolist()

results = classifier(
    texts,
    batch_size=16,
    truncation=True,
    max_length=512
)

# Store predictions
comparison = baseline.copy()

comparison["textblob_sentiment"] = baseline["sentiment"]

comparison["roberta_sentiment"] = [
    item["label"].capitalize()
    for item in results
]

comparison["roberta_score"] = [
    round(float(item["score"]), 3)
    for item in results
]

# Compare model outputs
print("\n=== TextBlob Sentiment ===")
print(comparison["textblob_sentiment"].value_counts())

print("\n=== RoBERTa Sentiment ===")
print(comparison["roberta_sentiment"].value_counts())

print("\n=== Model Agreement ===")

agreement = (
    comparison["textblob_sentiment"]
    == comparison["roberta_sentiment"]
).mean()

print(f"Agreement: {agreement:.1%}")

print("\n=== Rating vs RoBERTa ===")

print(
    pd.crosstab(
        comparison["rating"],
        comparison["roberta_sentiment"]
    )
)

print("\n=== Changed Predictions ===")

changed = comparison[
    comparison["textblob_sentiment"]
    != comparison["roberta_sentiment"]
]

for _, row in changed.head(8).iterrows():
    print("\nReview:", str(row["review"])[:160])
    print("TextBlob:", row["textblob_sentiment"])
    print("RoBERTa:", row["roberta_sentiment"])

comparison.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print("Sentiment comparison completed!")
