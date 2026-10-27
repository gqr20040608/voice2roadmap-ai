import pandas as pd
import pytest
from src.pipeline import prepare_reviews, detect_theme
from src.analytics import build_backlog
from src.retrieval import retrieve_reviews


def test_normalization_and_scores():
    df = prepare_reviews(pd.DataFrame({"content": ["Terrible crash every time I log in", "I love the smooth dashboard"]}))
    assert len(df) == 2
    assert df.iloc[0]["sentiment"] == "Negative"
    assert df.iloc[1]["sentiment"] == "Positive"
    assert set(["review_id", "theme", "date"]).issubset(df.columns)


def test_error_for_missing_review():
    with pytest.raises(ValueError, match="review column"):
        prepare_reviews(pd.DataFrame({"rating": [3, 5]}))


def test_backlog_traceability():
    df = prepare_reviews(pd.DataFrame({"review_id": ["A1", "A2"], "review": ["Terrible crash and freeze", "App crashes, awful loading speed"]}))
    backlog = build_backlog(df)
    assert backlog.iloc[0]["theme"] == "Performance"
    assert "A1" in backlog.iloc[0]["evidence_ids"]
    assert backlog.iloc[0]["status"].startswith("Proposed")


def test_retrieval_returns_real_review_ids():
    df = prepare_reviews(pd.DataFrame({"review_id": ["A1", "A2"], "review": ["Login password reset fails", "Beautiful screen layout"]}))
    found = retrieve_reviews(df, "password login", limit=2)
    assert len(found) >= 1
    assert found.iloc[0]["review_id"] == "A1"


def test_unknown_topic_fallback():
    assert detect_theme("Very nice") == "Other / General"
