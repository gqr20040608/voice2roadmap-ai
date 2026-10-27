"""Data validation and explainable baseline text analysis."""
from __future__ import annotations

import re
import pandas as pd
from textblob import TextBlob

THEMES: dict[str, list[str]] = {
    "Login & Access": ["log in", "login", "sign in", "password", "account locked", "authentication", "2fa"],
    "Performance": ["slow", "lag", "load", "loading", "freeze", "crash", "speed", "stuck"],
    "Payments & Pricing": ["price", "pricing", "charge", "charged", "payment", "billing", "refund", "checkout", "subscription"],
    "Notifications": ["notification", "alert", "reminder", "email update", "push message"],
    "Usability & Design": ["design", "navigation", "menu", "layout", "confusing", "button", "interface", "dark mode"],
    "Customer Support": ["support", "help desk", "agent", "customer service", "response time", "ticket"],
    "Accessibility": ["accessibility", "screen reader", "contrast", "font size", "keyboard navigation"],
}

# A small explicit vocabulary supports more intuitive demo behavior when
# off-the-shelf lexicons miss software jargon. This is NOT a trained model.
NEGATIVE_TERMS = ["crash", "broken", "stuck", "fail", "terrible", "awful", "bug", "unusable", "never works", "frustrating"]
POSITIVE_TERMS = ["love", "great", "excellent", "easy", "helpful", "smooth", "perfect", "works well"]


def detect_theme(text: str) -> str:
    text = text.lower()
    matches = []
    for theme, keywords in THEMES.items():
        hits = sum(bool(re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", text)) for kw in keywords)
        if hits:
            matches.append((hits, theme))
    return sorted(matches, key=lambda t: (-t[0], t[1]))[0][1] if matches else "Other / General"


def detect_sentiment(text: str) -> tuple[str, float]:
    polarity = float(TextBlob(text).sentiment.polarity)
    lowered = text.lower()
    neg = sum(t in lowered for t in NEGATIVE_TERMS)
    pos = sum(t in lowered for t in POSITIVE_TERMS)
    adjusted = max(-1.0, min(1.0, polarity + 0.18 * (pos - neg)))
    if adjusted > 0.1:
        label = "Positive"
    elif adjusted < -0.1:
        label = "Negative"
    else:
        label = "Neutral"
    return label, round(adjusted, 3)


def prepare_reviews(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded CSV. Review is required; date/product/rating are optional."""
    aliases = {
        "content": "review", "text": "review", "review_text": "review",
        "score": "rating", "stars": "rating", "timestamp": "date", "at": "date",
        "app": "product", "app_name": "product", "id": "review_id",
    }
    df = raw.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    for origin, target in aliases.items():
        if origin in df.columns and target not in df.columns:
            df = df.rename(columns={origin: target})
    if "review" not in df.columns:
        raise ValueError("CSV needs a review column (also accepted: content, text, review_text).")
    df["review"] = df["review"].fillna("").astype(str).str.strip()
    df = df[df["review"].str.len() > 0].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No non-empty reviews found in the uploaded CSV.")
    if "review_id" not in df.columns:
        df["review_id"] = [f"REV-{i+1:05d}" for i in range(len(df))]
    else:
        df["review_id"] = df["review_id"].fillna("").astype(str)
        df.loc[df["review_id"].eq(""), "review_id"] = [f"REV-{i+1:05d}" for i in df.index[df["review_id"].eq("")]]
    # Duplicate review IDs break traceability, so make repeated IDs unique.
    duplicated = df["review_id"].duplicated(keep=False)
    if duplicated.any():
        df.loc[duplicated, "review_id"] = [f"REV-{i+1:05d}" for i in df.index[duplicated]]
    if "product" not in df.columns:
        df["product"] = "Uploaded Product"
    df["product"] = df["product"].fillna("Unknown").astype(str)
    if "date" not in df.columns:
        df["date"] = pd.NaT
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "rating" not in df.columns:
        df["rating"] = float("nan")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df.loc[~df["rating"].between(1, 5), "rating"] = float("nan")
    sentiment = df["review"].apply(detect_sentiment)
    df["sentiment"] = sentiment.apply(lambda v: v[0])
    df["sentiment_score"] = sentiment.apply(lambda v: v[1])
    df["theme"] = df["review"].apply(detect_theme)
    return df[["review_id", "date", "product", "rating", "review", "sentiment", "sentiment_score", "theme"]]
