"""Evidence retrieval based on word and bigram overlap (TF-IDF)."""
from __future__ import annotations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_reviews(reviews: pd.DataFrame, question: str, limit: int = 5) -> pd.DataFrame:
    if reviews.empty or not question.strip():
        return reviews.head(0).copy()
    corpus = reviews["review"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)
    try:
        x = vectorizer.fit_transform(corpus)
        q = vectorizer.transform([question])
    except ValueError:  # empty vocabulary
        return reviews.head(0).copy()
    scores = cosine_similarity(q, x)[0]
    positions = scores.argsort()[::-1][:max(1, limit)]
    positions = [int(i) for i in positions if scores[i] > 0]
    if not positions:
        return reviews.head(0).copy()
    found = reviews.iloc[positions].copy()
    found["relevance_score"] = [round(float(scores[i]), 3) for i in positions]
    return found
