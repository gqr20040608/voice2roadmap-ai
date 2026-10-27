"""Add optional local embedding search to Voice2Roadmap Evidence Explorer.

Place in voice2roadmap-ai/scripts/ and run:
    python scripts/enable_semantic_evidence_search.py

Safeguards: backs up app.py; validates syntax; leaves Qwen and analytics untouched;
refuses to overwrite files when an unexpected project version is detected.
"""

from datetime import datetime
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
MODULE = ROOT / "src" / "semantic_retrieval.py"
MARKER = '    # Semantic evidence search (local MiniLM).'

MODULE_CODE = '''"""Search actual review evidence using locally computed sentence embeddings."""

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
'''

OLD_DESCRIPTION = (
    '    st.caption("The search engine uses TF-IDF similarity and only returns actual rows '
    'in the uploaded dataset. It does not invent quotes.")\n'
)
NEW_DESCRIPTION = '''    # Semantic evidence search (local MiniLM).
    from src.semantic_retrieval import retrieve_reviews_semantic
    st.caption("Search retrieves only original uploaded review rows. "
               "Embedding relevance scores are cosine similarities, not confidence probabilities.")
    search_method = st.radio(
        "Evidence search method",
        ["TF-IDF baseline", "Embedding semantic search"],
        index=1 if theme_mode == "Embedding AI (experimental)" else 0,
        horizontal=True,
    )
'''
OLD_SEARCH = '        hits = retrieve_reviews(filtered, query, limit=8)\n'
NEW_SEARCH = '''        if search_method == "Embedding semantic search":
            with st.spinner("Searching original reviews using local embeddings..."):
                hits = retrieve_reviews_semantic(filtered, query, limit=8)
        else:
            hits = retrieve_reviews(filtered, query, limit=8)
'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one {label}, found {count}. No files changed. "
            "Share this error to adapt the script to your version."
        )
    return source.replace(old, new, 1)


def main():
    if not APP.is_file() or not (ROOT / "src" / "retrieval.py").is_file():
        raise FileNotFoundError(
            "Put this script in voice2roadmap-ai/scripts/ (with app.py and src/retrieval.py)."
        )
    app = APP.read_text(encoding="utf-8")
    if MARKER in app and MODULE.exists():
        print("Semantic evidence search is already installed. No changes needed.")
        return
    if MARKER in app or MODULE.exists():
        raise RuntimeError("Partial installation detected. No project files changed.")
    if 'theme_mode = st.radio("Theme classification"' not in app:
        raise RuntimeError("Theme classification toggle missing; no files changed.")

    new_app = replace_once(app, OLD_DESCRIPTION, NEW_DESCRIPTION, "TF-IDF caption")
    new_app = replace_once(new_app, OLD_SEARCH, NEW_SEARCH, "TF-IDF retrieval call")
    compile(new_app, str(APP), "exec")
    compile(MODULE_CODE, str(MODULE), "exec")

    backup_dir = ROOT.parent / "voice2roadmap-backups" / (
        "semantic-evidence-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(APP, backup_dir / "app.py")
    MODULE.write_text(MODULE_CODE, encoding="utf-8")
    APP.write_text(new_app, encoding="utf-8")

    print("Backup:", backup_dir)
    print("Semantic evidence search enabled successfully!")
    print("Evidence Explorer now offers TF-IDF and local Embedding search.")
    print("Your Qwen model, requirements and review data were not changed.")


if __name__ == "__main__":
    main()
