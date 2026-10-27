"""Voice2Roadmap AI — Feedback-to-requirements decision-support MVP."""
from __future__ import annotations
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import build_backlog, make_summary
from src.pipeline import prepare_reviews
from src.retrieval import retrieve_reviews
from src.ai import generate_ai_brief

st.set_page_config(page_title="Voice2Roadmap AI", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2rem; max-width: 1390px;}
h1, h2, h3 {letter-spacing: -0.02em;}
div[data-testid="stMetric"] {padding: 16px; background: #f4f7ff; border: 1px solid #e4ebfa; border-radius: 13px;}
div[data-testid="stMetricLabel"] {color: #596881;}
div[data-testid="stMetricValue"] {font-weight: 750;}
.note {color:#52617d;font-size:14px;line-height:1.5;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).resolve().parent / "data" / "demo_reviews.csv")

@st.cache_data(show_spinner=False)
def analyze(raw: pd.DataFrame) -> pd.DataFrame:
    return prepare_reviews(raw)

@st.cache_data(show_spinner=False)
def apply_embedding_themes(reviews: pd.DataFrame) -> pd.DataFrame:
    """Classify themes locally; preserve sentiment and review evidence IDs."""
    from src.semantic_classifier import classify_themes

    result = reviews.copy()
    predictions = classify_themes(result["review"].tolist())
    result["baseline_theme"] = result["theme"]
    result["theme"] = predictions["semantic_theme"].to_numpy()
    result["similarity_score"] = predictions["similarity_score"].to_numpy()
    result["review_needed"] = predictions["review_needed"].to_numpy()
    return result


@st.cache_resource(show_spinner=False)
def load_roberta_model():
    """Load and cache the local sentiment model."""
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device="cpu"
    )


@st.cache_data(show_spinner=False)
def apply_roberta_sentiment(reviews: pd.DataFrame) -> pd.DataFrame:
    """Replace TextBlob predictions with RoBERTa predictions."""

    result = reviews.copy()

    classifier = load_roberta_model()

    texts = (
        result["review"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    predictions = classifier(
        texts,
        batch_size=16,
        truncation=True,
        max_length=512
    )

    # Keep the original TextBlob results for comparison
    result["baseline_sentiment"] = result["sentiment"]

    result["sentiment"] = [
        item["label"].capitalize()
        for item in predictions
    ]

    result["roberta_score"] = [
        round(float(item["score"]), 3)
        for item in predictions
    ]

    return result




with st.sidebar:
    st.title("🧭 Voice2Roadmap")
    st.caption("Customer voice → evidence → product decisions")
    st.divider()
    source = st.radio("Data source", ["Built-in demo", "Upload your CSV"], index=0)
    if source == "Upload your CSV":
        st.caption("Required: `review` (or `content` / `text`). Optional: `review_id`, `date`, `product`, `rating`.")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to get started, or switch to the demo.")
            st.stop()
        try:
            raw = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Couldn't read CSV: {exc}")
            st.stop()
    else:
        raw = load_demo()
        st.caption("Demo dataset is SYNTHETIC. Do not claim these are real customer reviews.")
    try:
        reviews = analyze(raw)
    except (ValueError, TypeError) as exc:
        st.error(f"Data validation failed: {exc}")
        st.stop()

    sentiment_mode = st.radio(
        "Sentiment analysis model",
        [
            "TextBlob baseline",
            "RoBERTa AI (experimental)"
        ],
        index=0,
        help="Compare rule-based sentiment with local RoBERTa AI."
    )

    if sentiment_mode == "RoBERTa AI (experimental)":
        try:
            with st.spinner("Analyzing sentiment with RoBERTa..."):
                reviews = apply_roberta_sentiment(reviews)
        except Exception as exc:
            st.error(f"RoBERTa sentiment analysis failed: {exc}")
            st.stop()


    theme_mode = st.radio("Theme classification", [
        "Keyword baseline", "Embedding AI (experimental)"
    ], index=0, help="Experimental English-language theme predictions. Not verified accuracy.")
    if theme_mode == "Embedding AI (experimental)":
        try:
            with st.spinner("Analyzing themes with the local embedding model..."):
                reviews = apply_embedding_themes(reviews)
        except Exception as exc:
            st.error(f"Embedding classification failed: {exc}")
            st.stop()
    products = ["All products"] + sorted(reviews["product"].unique().tolist())
    chosen_product = st.selectbox("Product", products)
    sentiments = st.multiselect("Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
    st.divider()

    st.caption(
        "Local AI analysis: TextBlob or RoBERTa sentiment, "
        "keyword or embedding-based themes, and optional "
        "Qwen3 decision briefs via Ollama. "
        "No paid API key required."
    )


filtered = reviews.copy()
if chosen_product != "All products":
    filtered = filtered[filtered["product"] == chosen_product]
filtered = filtered[filtered["sentiment"].isin(sentiments)]
backlog = build_backlog(filtered)

st.title("Voice2Roadmap AI")
st.caption("Select the baseline or experimental embedding model in the sidebar. "
           "Embedding similarity is not a calibrated confidence probability.")
st.caption("Product intelligence · NLP analytics · evidence-backed requirements")
if source == "Built-in demo":
    st.info("**Portfolio demo:** the built-in customer reviews are fictional, generated for product testing. Upload a properly licensed real-world dataset for an evidence-based case study.", icon="ℹ️")
st.write(make_summary(filtered, backlog))
st.divider()

if filtered.empty:
    st.warning("No reviews match the selected filters. Change the filters in the sidebar.")
    st.stop()

n = len(filtered)
negative = int((filtered["sentiment"] == "Negative").sum())
positive = int((filtered["sentiment"] == "Positive").sum())
ratings = filtered["rating"].dropna()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Feedback analyzed", f"{n:,}")
k2.metric("Negative feedback", f"{negative/n:.1%}")
k3.metric("Positive feedback", f"{positive/n:.1%}")
k4.metric("Average rating", f"{ratings.mean():.2f} / 5" if len(ratings) else "N/A")

st.write("")
overview, requirements, evidence, ai_brief = st.tabs(["📊 Overview", "📋 Requirements Backlog", "🔎 Evidence Explorer", "✨ AI Decision Brief"])

with overview:
    c1, c2 = st.columns([1.05, 1])
    with c1:
        counts = filtered["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0).reset_index()
        counts.columns = ["Sentiment", "Reviews"]
        pie = px.pie(counts, values="Reviews", names="Sentiment", title="Customer sentiment distribution", hole=0.65,
                     color="Sentiment", color_discrete_map={"Positive":"#2ca68a", "Neutral":"#adb7cb", "Negative":"#ed7385"})
        pie.update_layout(template="plotly_white", margin=dict(t=60,b=20,l=10,r=10), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(pie, use_container_width=True)
    with c2:
        topic_counts = filtered["theme"].value_counts().reset_index()
        topic_counts.columns = ["Theme", "Mentions"]
        fig = px.bar(topic_counts.sort_values("Mentions"), x="Mentions", y="Theme", orientation="h",
                     title="Most discussed themes", color_discrete_sequence=["#5277ef"])
        fig.update_layout(template="plotly_white", margin=dict(t=60,b=20,l=10,r=10), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    if filtered["date"].notna().any():
        temp = filtered.dropna(subset=["date"]).copy()
        temp["Month"] = temp["date"].dt.to_period("M").dt.to_timestamp()
        temp["Is negative"] = temp["sentiment"].eq("Negative").astype(int)
        trend = temp.groupby("Month", as_index=False).agg(negative_share=("Is negative", "mean"), reviews=("review_id", "count"))
        trend["Negative share (%)"] = (trend["negative_share"] * 100).round(1)
        line = px.line(trend, x="Month", y="Negative share (%)", markers=True, title="Negative-feedback trend (within selected reviews)")
        line.update_traces(line_color="#5277ef")
        line.update_layout(template="plotly_white", margin=dict(t=60,b=20,l=20,r=20), yaxis_range=[0,100])
        st.plotly_chart(line, use_container_width=True)
    else:
        st.caption("No dates provided: trend chart is unavailable for this dataset.")
    st.caption("Sentiment is a heuristic NLP classification. Review frequency is not a representative estimate of all users.")

with requirements:
    st.subheader("Proposed product requirements")
    st.caption("Scores combine relative theme frequency and negative share. These are transparent **triage signals**, not validated impact estimates or RICE scores. All requirements need human review.")
    columns = ["requirement_id", "theme", "mentions", "negative_reviews", "negative_rate", "signal_score", "proposed_requirement", "example_acceptance_criterion", "evidence_ids", "status"]
    st.dataframe(backlog[columns], hide_index=True, use_container_width=True, column_config={
        "negative_rate": st.column_config.NumberColumn("Negative %", format="%.1f%%"),
        "signal_score": st.column_config.ProgressColumn("Signal", min_value=0,max_value=100,format="%.1f"),
        "proposed_requirement": "Draft requirement",
        "example_acceptance_criterion": "Example acceptance criterion",
        "evidence_ids": "Review evidence IDs",
    })
    st.download_button("Download requirements backlog (CSV)", data=backlog.to_csv(index=False).encode("utf-8"), file_name="voice2roadmap_backlog.csv", mime="text/csv")
    # Keep unclassified feedback visible, without ranking it as a requirement.
    uncategorized = filtered[filtered["theme"] == "Other / General"]
    if not uncategorized.empty:
        with st.expander(f"Uncategorized review queue ({len(uncategorized):,}) — not ranked"):
            st.caption(
                "These reviews need better classification or human review. "
                "They are excluded from requirement priority scores, not deleted. "
                "Showing the first 50 reviews."
            )
            st.dataframe(
                uncategorized[["review_id", "product", "review", "sentiment"]].head(50),
                hide_index=True,
                use_container_width=True,
            )

    st.markdown("#### Evidence traceability")
    if not backlog.empty:
        selected_req = st.selectbox("Inspect requirement", backlog["requirement_id"].tolist())
        row = backlog.loc[backlog["requirement_id"] == selected_req].iloc[0]
        st.write(f"**{row['proposed_requirement']}**")
        st.write(f"**Example acceptance criterion:** {row['example_acceptance_criterion']}")
        ids = [s.strip() for s in row["evidence_ids"].split(",") if s.strip()]
        st.dataframe(filtered[filtered["review_id"].isin(ids)][["review_id","product","review","sentiment"]], hide_index=True, use_container_width=True)

with evidence:
    st.subheader("Search original customer evidence")
    # Semantic evidence search (local MiniLM).
    from src.semantic_retrieval import retrieve_reviews_semantic
    st.caption("Search retrieves only original uploaded review rows. "
               "Embedding relevance scores are cosine similarities, not confidence probabilities.")
    search_method = st.radio(
        "Evidence search method",
        ["TF-IDF baseline", "Embedding semantic search"],
        index=1 if theme_mode == "Embedding AI (experimental)" else 0,
        horizontal=True,
    )
    query = st.text_input("Question or search terms", placeholder="Example: Why are customers complaining about login crashes?")
    if query:
        if search_method == "Embedding semantic search":
            with st.spinner("Searching original reviews using local embeddings..."):
                hits = retrieve_reviews_semantic(filtered, query, limit=8)
        else:
            hits = retrieve_reviews(filtered, query, limit=8)
        if hits.empty:
            st.info("No close text match found in the selected reviews. Try different terms.")
        else:
            st.dataframe(hits[["review_id","product","theme","sentiment","review","relevance_score"]], hide_index=True, use_container_width=True)
    else:
        st.dataframe(filtered[["review_id","date","product","rating","theme","sentiment","review"]].head(60), hide_index=True, use_container_width=True)
    st.download_button("Download analyzed reviews (CSV)", data=filtered.to_csv(index=False).encode("utf-8"), file_name="voice2roadmap_analyzed_reviews.csv", mime="text/csv")


with ai_brief:
    st.subheader("Local AI Decision Brief")

    st.write(
        "Generate an AI-powered product analysis report "
        "using the local Qwen3 model."
    )

    st.success(
        "Local AI Mode: Qwen3 4B via Ollama — No paid API required."
    )

    if st.button("Generate AI decision brief", type="primary"):

        # Remove the previous report before generating a new one.
        st.session_state.pop("ai_brief", None)

        with st.spinner("Qwen is analyzing customer feedback..."):
            try:
                from src.local_ai import generate_local_brief


                dataset_type = (
                    "Synthetic demonstration reviews generated for portfolio testing"
                    if source == "Built-in demo"
                    else "User-uploaded review dataset. Original provenance is not "
                         "independently verified. Reviews may come from multiple "
                         "unrelated apps and must not be treated as one product."
                )

                report = generate_local_brief(
                    filtered,
                    backlog,
                    dataset_type=dataset_type
                )

                st.session_state["ai_brief"] = report

            except Exception as exc:
                st.error(
                    "Local AI request failed. Make sure Ollama "
                    f"is running. Details: {exc}"
                )

    report = st.session_state.get("ai_brief")

    if report:
        st.warning(
            "AI-generated draft. Verify all statistics, "
            "review IDs and recommendations before use."
        )

        st.markdown(report)

        st.download_button(
            "Download AI Report",
            data=report,
            file_name="voice2roadmap_ai_report.md",
            mime="text/markdown"
        )

st.divider()
st.caption("Voice2Roadmap AI · Demonstration / portfolio project · Human review required before implementation")
