
from pathlib import Path

import pandas as pd
from ftfy import fix_text
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "real_reviews_sample.csv"
OUTPUT = ROOT / "data" / "english_reviews_sample.csv"

df = pd.read_csv(SOURCE)

# Repair text encoding issues
original_text = df["review"].fillna("").astype(str)

df["review"] = original_text.apply(fix_text)
df["review"] = (
    df["review"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

fixed_count = (original_text != df["review"]).sum()

# Detect English reviews
def is_english(text):
    try:
        predictions = detect_langs(text)
        return any(
            p.lang == "en" and p.prob >= 0.90
            for p in predictions
        )
    except LangDetectException:
        return False


english_mask = df["review"].apply(is_english)
english = df.loc[english_mask].copy()

english.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)

print("\n=== Language Cleaning Results ===")
print("Original reviews:", len(df))
print("Text records changed:", fixed_count)
print("English reviews:", len(english))
print("Excluded or uncertain:", len(df) - len(english))
print("Saved:", OUTPUT)
print("\nCleaning completed!")
