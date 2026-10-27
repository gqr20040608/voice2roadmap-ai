"""Search actual review evidence using locally computed sentence embeddings."""

from functools import lru_cache

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model():
    # Loaded only once per running Python process.
    return SentenceTransformer(MODEL)


@lru_cache(maxsize=4)
def _corpus_embeddings(texts: tuple[str, ...]) -> np.ndarray:
    # Caches up to four filtered review sets for faster repeat searches.
    return _model().encode(
        list(texts), batch_size=32, show_progress_bar=False,
        normalize_embeddings=True,
    )


def retrieve_reviews_semantic(
    reviews: pd.DataFrame,
    question: str,
    limit: int = 8,
    min_similarity: float = 0.25,
) -> pd.DataFrame:
    """Return original review rows ordered by semantic cosine similarity.

    Relevance scores are NOT confidence probabilities. This model is tuned
    primarily for English text. Never produces fabricated quotations.
    """
    if reviews.empty or not question.strip():
        return reviews.head(0).assign(
            relevance_score=pd.Series(dtype="float64")
        )

    texts = tuple(reviews["review"].fillna("").astype(str).tolist())
    vectors = _corpus_embeddings(texts)
    query_vector = _model().encode(
        [question], normalize_embeddings=True, show_progress_bar=False
    )[0]
    scores = np.asarray(vectors @ query_vector, dtype=float)
    positions = np.argsort(scores)[::-1]
    chosen = [int(i) for i in positions if scores[i] >= min_similarity][
        :max(1, int(limit))
    ]
    if not chosen:
        return reviews.head(0).assign(
            relevance_score=pd.Series(dtype="float64")
        )
    result = reviews.iloc[chosen].copy()
    result["relevance_score"] = np.round(scores[chosen], 3)
    return result
