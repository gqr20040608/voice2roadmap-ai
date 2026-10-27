"""Voice2Roadmap AI — Feedback-to-requirements decision-support MVP."""
from __future__ import annotations
import os
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import build_backlog, make_summary
from src.pipeline import prepare_reviews
from src.retrieval import retrieve_reviews
from src.ai import generate_ai_brief

st.set_page_config(
    page_title="Voice2Roadmap AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #FFFFFF;
    color: #172033;
    font-family: -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
}

/* Main content */
.block-container {
    max-width: 1440px;
    padding-top: 2.4rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Typography */
h1, h2, h3 {
    color: #172033;
    letter-spacing: -0.035em;
}

h1 {
    font-size: 2.4rem !important;
    font-weight: 750 !important;
    margin-bottom: 0.7rem !important;
}

h2 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

h3 {
    font-size: 1.15rem !important;
    font-weight: 650 !important;
}

p {
    line-height: 1.65;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F7F9FC;
    border-right: 1px solid #E5EAF1;
}

section[data-testid="stSidebar"] h1 {
    font-size: 1.45rem !important;
    letter-spacing: -0.03em;
}

section[data-testid="stSidebar"] label {
    font-weight: 550;
}

/* KPI cards */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 20px 22px;
    box-shadow: 0 2px 8px rgba(20, 35, 60, 0.025);
}

div[data-testid="stMetricLabel"] {
    color: #64748B;
    font-size: 0.85rem;
    font-weight: 500;
}

div[data-testid="stMetricValue"] {
    color: #172033;
    font-size: 2rem;
    font-weight: 730;
    letter-spacing: -0.04em;
}

/* Navigation tabs */
button[data-baseweb="tab"] {
    font-weight: 550;
    font-size: 0.9rem;
    padding: 12px 18px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #345DE3;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 8px;
}

/* Buttons */
.stButton > button[kind="primary"] {
    background-color: #345DE3;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button[kind="primary"]:hover {
    background-color: #244BC6;
}

/* Tables */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Horizontal dividers */
hr {
    border-color: #E9EDF3;
}

/* General explanatory text */
.note {
    color: #64748B;
    font-size: 14px;
    line-height: 1.6;
}

/* Responsive layout */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1.1rem;
        padding-right: 1.1rem;
        padding-top: 1.5rem;
    }
}


/* Page header */
.page-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #64748B;
    margin-bottom: 12px;
}

/* Dataset provenance notice */
.dataset-notice {
    background: #F8FAFC;
    border-left: 3px solid #5277EF;
    border-radius: 4px;
    padding: 13px 18px;
    margin-top: 12px;
    margin-bottom: 20px;
}

.dataset-notice strong {
    font-size: 0.72rem;
    letter-spacing: 0.07em;
    color: #334155;
}

.dataset-notice p {
    margin: 5px 0 0 0;
    font-size: 0.85rem;
    color: #64748B;
    line-height: 1.5;
}

/* Analysis summary */
.analysis-summary-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-top: 8px;
    margin-bottom: 4px;
}


/* Sidebar branding */
.sidebar-kicker {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #64748B;
    margin-bottom: 8px;
}

section[data-testid="stSidebar"] h1 {
    font-size: 1.35rem !important;
    font-weight: 750 !important;
    margin-bottom: 0.2rem !important;
}

section[data-testid="stSidebar"] hr {
    margin-top: 12px;
    margin-bottom: 12px;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #334155;
}


/* Disabled primary buttons */
.stButton > button[kind="primary"]:disabled {
    background-color: #E9EDF4 !important;
    color: #94A3B8 !important;
    border: 1px solid #DCE3EC !important;
    opacity: 1 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
}

.stButton > button[kind="primary"]:disabled:hover {
    background-color: #E9EDF4 !important;
    color: #94A3B8 !important;
    border-color: #DCE3EC !important;
}


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



@st.cache_data(ttl=20, show_spinner=False)
def is_ollama_available() -> bool:
    """Check whether the configured local Ollama model is available."""
    try:
        with urlopen(
            "http://127.0.0.1:11434/api/tags",
            timeout=1.5
        ) as response:
            if response.status != 200:
                return False

            import json
            data = json.load(response)

            return any(
                model.get("name") == "qwen3:4b-instruct"
                for model in data.get("models", [])
            )
    except Exception:
        return False




with st.sidebar:

    st.markdown(
        '<div class="sidebar-kicker">ANALYTICS WORKSPACE</div>',
        unsafe_allow_html=True
    )
    st.title("Voice2Roadmap AI")
    st.caption("Customer feedback intelligence")
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


st.markdown(
    '<div class="page-eyebrow">'
    'VOICE2ROADMAP AI / PRODUCT INTELLIGENCE'
    '</div>',
    unsafe_allow_html=True
)

st.title("Customer Feedback Overview")

st.caption(
    "Analyze customer sentiment, discover recurring issues, "
    "and prioritize evidence-backed product requirements."
)

if source == "Built-in demo":
    st.markdown(
        """
        <div class="dataset-notice">
            <strong>SYNTHETIC DEMO DATA</strong>
            <p>
                This dataset contains fictional customer reviews
                for demonstration purposes. Findings do not
                represent actual customer behavior.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="dataset-notice">
            <strong>USER-UPLOADED DATA</strong>
            <p>
                Dataset provenance and redistribution rights
                have not been independently verified.
                Validate the source before making product decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="analysis-summary-label">ANALYSIS SUMMARY</div>',
    unsafe_allow_html=True
)

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

overview, requirements, evidence, ai_brief = st.tabs([
    "Overview",
    "Requirements Backlog",
    "Evidence Explorer",
    "AI Decision Brief"
])


with overview:
    c1, c2 = st.columns([1.05, 1])
    with c1:
        counts = filtered["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0).reset_index()
        counts.columns = ["Sentiment", "Reviews"]

        pie = px.pie(
            counts,
            values="Reviews",
            names="Sentiment",
            hole=0.72,
            title="Customer sentiment distribution",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#268A81",
                "Neutral": "#A7B2C4",
                "Negative": "#D16B78"
            },
        )

        pie.update_traces(
            textinfo="percent",
            textposition="outside",
            textfont=dict(size=12, color="#334155"),
            marker=dict(
                line=dict(color="#FFFFFF", width=2)
            ),
            hovertemplate=(
                "%{label}<br>"
                "%{value} reviews<br>"
                "%{percent}<extra></extra>"
            ),
            sort=False,
        )

        pie.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(t=65, b=50, l=35, r=35),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#334155"
            ),
            title=dict(
                font=dict(size=16, color="#172033"),
                x=0,
                xanchor="left",
            ),
            legend=dict(
                orientation="h",
                y=-0.12,
                x=0.5,
                xanchor="center",
                font=dict(size=12)
            ),
            showlegend=True,
        )

        st.plotly_chart(pie, use_container_width=True)

    with c2:
        topic_counts = filtered["theme"].value_counts().reset_index()
        topic_counts.columns = ["Theme", "Mentions"]

        fig = px.bar(
            topic_counts.sort_values("Mentions"),
            x="Mentions",
            y="Theme",
            orientation="h",
            text="Mentions",
            title="Most discussed themes",
            color_discrete_sequence=["#5773CF"],
        )

        fig.update_traces(
            textposition="outside",
            textfont=dict(
                size=12,
                color="#334155"
            ),
            marker=dict(
                line=dict(width=0)
            ),
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>"
                "%{x} reviews"
                "<extra></extra>"
            ),
        )

        fig.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(t=65, b=50, l=10, r=40),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#334155"
            ),
            title=dict(
                font=dict(size=16, color="#172033"),
                x=0,
                xanchor="left",
            ),
            showlegend=False,
            bargap=0.28,
        )

        fig.update_xaxes(
            title_text="Number of reviews",
            showgrid=True,
            gridcolor="#EDF1F6",
            zeroline=False,
            range=[0, topic_counts["Mentions"].max() * 1.18],
        )

        fig.update_yaxes(
            title_text=None,
            showgrid=False,
            categoryorder="total ascending",
            tickfont=dict(size=11, color="#64748B"),
        )

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

    st.markdown(
        '<div class="page-eyebrow">REQUIREMENTS MANAGEMENT</div>',
        unsafe_allow_html=True
    )

    st.subheader("Prioritized Requirements")

    st.caption(
        "Evidence-backed product improvement opportunities, "
        "ranked by feedback frequency and negative sentiment."
    )

    st.write("")

    # Requirements summary
    total_requirements = len(backlog)

    highest_signal = (
        float(backlog["signal_score"].max())
        if not backlog.empty
        else 0.0
    )

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Total requirements",
            f"{total_requirements:,}"
        )

    with r2:
        st.metric(
            "Highest signal",
            f"{highest_signal:.1f} / 100"
            if not backlog.empty
            else "N/A"
        )

    with r3:
        st.metric(
            "Review status",
            "Proposed"
        )

    st.write("")

    st.markdown("#### Requirements Overview")

    st.caption(
        "Signal scores combine relative theme frequency and "
        "negative feedback share. They are heuristic triage "
        "indicators, not validated business impact estimates "
        "or RICE scores. Human review is required."
    )

    columns = [
        "requirement_id",
        "theme",
        "mentions",
        "negative_reviews",
        "negative_rate",
        "signal_score",
        "proposed_requirement",
        "example_acceptance_criterion",
        "evidence_ids",
        "status"
    ]

    st.dataframe(
        backlog[columns],
        hide_index=True,
        use_container_width=True,
        height=390,
        column_config={
            "requirement_id": st.column_config.TextColumn(
                "Requirement ID",
                width="small"
            ),
            "theme": st.column_config.TextColumn(
                "Issue Theme",
                width="medium"
            ),
            "mentions": st.column_config.NumberColumn(
                "Mentions",
                format="%d"
            ),
            "negative_reviews": st.column_config.NumberColumn(
                "Negative Reviews",
                format="%d"
            ),
            "negative_rate": st.column_config.NumberColumn(
                "Negative %",
                format="%.1f%%"
            ),
            "signal_score": st.column_config.ProgressColumn(
                "Priority Signal",
                min_value=0,
                max_value=100,
                format="%.1f"
            ),
            "proposed_requirement": st.column_config.TextColumn(
                "Proposed Requirement",
                width="large"
            ),
            "example_acceptance_criterion": st.column_config.TextColumn(
                "Acceptance Criterion",
                width="large"
            ),
            "evidence_ids": st.column_config.TextColumn(
                "Evidence IDs",
                width="medium"
            ),
            "status": st.column_config.TextColumn(
                "Review Status",
                width="medium"
            )
        }
    )

    st.download_button(
        "Download requirements backlog (CSV)",
        data=backlog.to_csv(index=False).encode("utf-8"),
        file_name="voice2roadmap_backlog.csv",
        mime="text/csv"
    )

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


    st.divider()

    st.markdown(
        '<div class="page-eyebrow">EVIDENCE TRACEABILITY</div>',
        unsafe_allow_html=True
    )

    st.subheader("Requirement Details")

    st.caption(
        "Select a requirement to inspect its proposed scope, "
        "acceptance criterion, and supporting customer reviews."
    )

    if not backlog.empty:

        selected_req = st.selectbox(
            "Select requirement",
            backlog["requirement_id"].tolist()
        )

        row = backlog.loc[
            backlog["requirement_id"] == selected_req
        ].iloc[0]

        with st.container(border=True):

            st.markdown(
                f"### {row['requirement_id']} — {row['theme']}"
            )

            st.caption(
                f"Status: {row['status']}"
            )

            st.divider()

            st.markdown("**PROPOSED REQUIREMENT**")
            st.write(row["proposed_requirement"])

            st.markdown("**ACCEPTANCE CRITERION**")
            st.write(row["example_acceptance_criterion"])

            st.divider()

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Theme mentions",
                    f"{int(row['mentions']):,}"
                )

            with c2:
                st.metric(
                    "Negative feedback",
                    f"{float(row['negative_rate']):.1f}%"
                )

            with c3:
                st.metric(
                    "Priority signal",
                    f"{float(row['signal_score']):.1f} / 100"
                )

        st.markdown("#### Supporting Review Evidence")

        st.caption(
            "These original review records are linked to "
            "the selected requirement through their review IDs."
        )

        ids = [
            s.strip()
            for s in str(row["evidence_ids"]).split(",")
            if s.strip()
        ]

        supporting_reviews = filtered[
            filtered["review_id"].astype(str).isin(ids)
        ]

        if supporting_reviews.empty:
            st.info(
                "No matching review evidence is available "
                "for the selected requirement."
            )
        else:
            st.dataframe(
                supporting_reviews[
                    ["review_id", "product", "sentiment", "review"]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "review_id": "Review ID",
                    "product": "Product",
                    "sentiment": "Sentiment",
                    "review": st.column_config.TextColumn(
                        "Original Customer Review",
                        width="large"
                    )
                }
            )
    else:
        st.info(
            "No ranked requirements are available "
            "for the current filters."
        )



with evidence:

    st.markdown(
        '<div class="page-eyebrow">EVIDENCE RETRIEVAL</div>',
        unsafe_allow_html=True
    )

    st.subheader("Customer Evidence Explorer")

    st.caption(
        "Search original customer feedback, inspect relevant reviews, "
        "and trace findings back to their source records."
    )

    st.write("")

    from src.semantic_retrieval import retrieve_reviews_semantic

    st.markdown("#### Search Configuration")

    search_method = st.radio(
        "Retrieval method",
        [
            "TF-IDF baseline",
            "Embedding semantic search"
        ],
        index=1 if theme_mode == "Embedding AI (experimental)" else 0,
        horizontal=True,
    )

    query = st.text_input(
        "Search customer feedback",
        placeholder="Example: The app crashes when I try to log in",
    )

    st.caption(
        "TF-IDF uses lexical similarity. Embedding search uses "
        "semantic similarity. Relevance scores are ranking signals, "
        "not calibrated confidence probabilities."
    )

    st.divider()

    if query.strip():

        st.markdown("#### Search Results")

        if search_method == "Embedding semantic search":
            with st.spinner("Searching customer reviews..."):
                hits = retrieve_reviews_semantic(
                    filtered,
                    query.strip(),
                    limit=8
                )
        else:
            hits = retrieve_reviews(
                filtered,
                query.strip(),
                limit=8
            )

        if hits.empty:
            st.info(
                "No matching reviews were found. "
                "Try different search terms or another retrieval method."
            )
        else:
            st.caption(
                f"{len(hits)} matching reviews returned using "
                f"{search_method}."
            )

            st.dataframe(
                hits[
                    [
                        "review_id",
                        "product",
                        "theme",
                        "sentiment",
                        "review",
                        "relevance_score"
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "review_id": st.column_config.TextColumn(
                        "Review ID",
                        width="small"
                    ),
                    "product": "Product",
                    "theme": "Issue Theme",
                    "sentiment": "Sentiment",
                    "review": st.column_config.TextColumn(
                        "Original Customer Review",
                        width="large"
                    ),
                    "relevance_score": st.column_config.NumberColumn(
                        "Relevance",
                        format="%.3f"
                    )
                }
            )

    else:

        st.markdown("#### Review Library")

        st.caption(
            "Enter a search query above to find relevant evidence. "
            "The table below previews the first 60 reviews "
            "from the current selection."
        )

        st.dataframe(
            filtered[
                [
                    "review_id",
                    "date",
                    "product",
                    "rating",
                    "theme",
                    "sentiment",
                    "review"
                ]
            ].head(60),
            hide_index=True,
            use_container_width=True,
            column_config={
                "review_id": "Review ID",
                "date": "Date",
                "product": "Product",
                "rating": "Rating",
                "theme": "Issue Theme",
                "sentiment": "Sentiment",
                "review": st.column_config.TextColumn(
                    "Original Customer Review",
                    width="large"
                )
            }
        )

    st.divider()

    st.download_button(
        "Download analyzed reviews (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="voice2roadmap_analyzed_reviews.csv",
        mime="text/csv"
    )



with ai_brief:

    st.markdown(
        '<div class="page-eyebrow">AI REPORT GENERATION</div>',
        unsafe_allow_html=True
    )

    st.subheader("AI Decision Brief")

    st.caption(
        "Generate evidence-grounded product recommendations "
        "using a locally hosted language model."
    )

    st.write("")

    # Check local Ollama availability
    ollama_ready = is_ollama_available()

    with st.container(border=True):

        st.markdown("#### Model Configuration")

        model_col, status_col = st.columns(2)

        with model_col:
            st.metric(
                "Language model",
                "Qwen3 4B"
            )

        with status_col:
            st.metric(
                "Model status",
                "Available" if ollama_ready else "Unavailable"
            )

        st.divider()

        if ollama_ready:

            st.markdown("**Local inference available**")

            st.caption(
                "Qwen3 is accessible through Ollama. "
                "Reports combine deterministic analytics "
                "with AI-generated qualitative recommendations. "
                "No paid API key is required."
            )

        else:

            st.markdown("**Local execution required**")

            st.caption(
                "AI report generation requires Ollama and the "
                "qwen3:4b-instruct model running on the same "
                "machine as the application. "
                "The public cloud deployment does not include "
                "this local inference service."
            )

            st.caption(
                "Dashboard analytics, sentiment classification, "
                "theme analysis, requirements management, and "
                "evidence retrieval remain available."
            )

        st.write("")

        if st.button(
            "Generate Decision Brief",
            type="primary",
            disabled=not ollama_ready,
            use_container_width=True
        ):

            # Clear previous report before generating a new one
            st.session_state.pop("ai_brief", None)

            with st.spinner("Generating product decision brief..."):

                try:
                    from src.local_ai import generate_local_brief

                    dataset_type = (
                        "Synthetic demonstration reviews "
                        "generated for portfolio testing"
                        if source == "Built-in demo"
                        else
                        "User-uploaded review dataset. "
                        "Original provenance is not independently "
                        "verified. Reviews may come from multiple "
                        "unrelated apps and must not be treated "
                        "as one product."
                    )

                    report = generate_local_brief(
                        filtered,
                        backlog,
                        dataset_type=dataset_type
                    )

                    st.session_state["ai_brief"] = report

                except Exception as exc:

                    st.error(
                        "AI report generation failed. "
                        f"Details: {exc}"
                    )

    # Show reports only when local Ollama is available
    report = (
        st.session_state.get("ai_brief")
        if ollama_ready
        else None
    )

    if report:

        st.divider()

        st.markdown(
            '<div class="page-eyebrow">GENERATED REPORT</div>',
            unsafe_allow_html=True
        )

        st.subheader("Product Decision Report")

        st.caption(
            "AI-assisted draft. Verify all statistics, "
            "review evidence, requirements, and recommendations "
            "before making product decisions."
        )

        with st.container(border=True):
            st.markdown(report)

        st.download_button(
            "Download Markdown Report",
            data=report,
            file_name="voice2roadmap_ai_report.md",
            mime="text/markdown"
        )


st.divider()
st.caption("Voice2Roadmap AI · Demonstration / portfolio project · Human review required before implementation")
