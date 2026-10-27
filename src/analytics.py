"""Issue ranking, requirement traceability, and deterministic summary."""
from __future__ import annotations
import pandas as pd

RECOMMENDATIONS = {
    "Login & Access": ("Improve account sign-in reliability", "Users can sign in using a valid account without unexpected errors in repeated test runs."),
    "Performance": ("Improve responsiveness and crash resilience", "Core screens load within the agreed performance target and critical workflows do not crash in regression tests."),
    "Payments & Pricing": ("Clarify pricing and stabilize payment flow", "Total fees appear before confirmation, and payment failures return actionable messages without duplicate charges."),
    "Notifications": ("Make notifications more reliable and configurable", "Users can choose alert preferences, and test notifications arrive in the defined delivery window."),
    "Usability & Design": ("Simplify high-friction navigation", "Target users can complete primary tasks in usability testing without help."),
    "Customer Support": ("Improve support response workflow", "Every support ticket receives a tracking ID and an explicit service-level target."),
    "Accessibility": ("Strengthen accessible interactions", "Key workflows pass agreed keyboard and screen-reader acceptance checks."),

    "Other / General": ("Investigate uncategorized feedback", "A human analyst reviews a representative sample and records root causes before implementation."),

    "Connectivity & Integration": (
        "Improve device connectivity and third-party integrations",
        "Devices and connected services sync successfully under agreed test conditions."
    ),
    "Content & Data": (
        "Improve content availability and data completeness",
        "Priority content and data fields meet an agreed completeness checklist."
    ),
    "Feature Requests": (
        "Evaluate and prioritize requested product features",
        "A product owner validates feature scope and acceptance criteria with users."
    ),

}


BACKLOG_COLUMNS = [
    "theme", "mentions", "negative_reviews", "negative_rate",
    "signal_score", "requirement_id", "proposed_requirement",
    "example_acceptance_criterion", "evidence_ids", "status",
]

def build_backlog(reviews: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty:
        return pd.DataFrame(columns=BACKLOG_COLUMNS)
    records = []
    for theme, chunk in reviews.groupby("theme"):
        # Uncategorized reviews are not actionable requirement themes.
        if theme == "Other / General":
            continue
        negatives = chunk[chunk["sentiment"] == "Negative"]
        total = len(chunk)
        neg_share = len(negatives) / total
        # A transparent heuristic, NOT RICE and NOT a validated estimate of business impact.
        reach_proxy = total / len(reviews)
        score = round(100 * (0.45 * min(1.0, reach_proxy * 5) + 0.55 * neg_share), 1)
        title, acceptance = RECOMMENDATIONS.get(
    theme,
    (
        f"Investigate feedback related to {theme}",
        "Define measurable acceptance criteria after human review."
    )
)
        samples = negatives if not negatives.empty else chunk
        evidence = samples.sort_values("sentiment_score").head(3)["review_id"].astype(str).tolist()
        records.append({
            "theme": theme, "mentions": total, "negative_reviews": len(negatives),
            "negative_rate": round(neg_share * 100, 1), "signal_score": score,
            "requirement_id": "", "proposed_requirement": title,
            "example_acceptance_criterion": acceptance, "evidence_ids": ", ".join(evidence),
            "status": "Proposed / Needs human review",
        })
    if not records:
        return pd.DataFrame(columns=BACKLOG_COLUMNS)
    output = pd.DataFrame(records).sort_values(["signal_score", "negative_reviews"], ascending=False).reset_index(drop=True)
    output["requirement_id"] = [f"REQ-{i+1:03d}" for i in range(len(output))]
    return output


def make_summary(reviews: pd.DataFrame, backlog: pd.DataFrame) -> str:
    if reviews.empty:
        return "There is no data in the current selection."
    negative = int((reviews["sentiment"] == "Negative").sum())
    headline = f"Analyzed {len(reviews):,} reviews. {negative:,} ({negative/len(reviews):.1%}) were classified as negative."
    if backlog.empty:
        return headline
    top = backlog.iloc[0]
    return headline + f" The highest-ranked signal is {top['theme']} ({int(top['mentions'])} mentions, {top['negative_rate']:.0f}% negative within this theme). This is a feedback signal, not proof of business impact."
