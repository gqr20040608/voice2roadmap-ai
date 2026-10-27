"""Conservative embedding-based customer-feedback theme classifier.

Usage from the project root:
    python -m src.semantic_classifier

This script is intended for evaluating classification on English reviews.
It does NOT modify the Streamlit app's existing theme pipeline.

Important: cosine similarity scores are NOT calibrated probabilities.
A decrease in 'Other / General' is NOT proof of increased accuracy.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.pipeline import prepare_reviews


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OTHER = "Other / General"

# The new themes give genuine feature requests and integration complaints a home.
# These are seed phrases, not labeled training examples.
THEME_EXAMPLES = {
    "Login & Access": [
        "I cannot sign in or log in to my account.",
        "My password reset fails and I cannot access my account.",
        "The authentication and verification process is broken.",
        "I am locked out of my account after entering my password.",
    ],
    "Performance": [
        "The app crashes whenever I try to open it.",
        "The screen freezes and the application becomes unresponsive.",
        "The app is laggy and loads very slowly.",
        "The application has glitches and crashes during use.",
    ],
    "Connectivity & Integration": [
        "The app will not connect to my car or smart device.",
        "Bluetooth pairing to my headphones or wearable device keeps failing.",
        "My account or device will not sync with the connected service.",
        "The integration between the app and my device stopped working.",
    ],
    "Payments & Pricing": [
        "I was charged twice or billed incorrectly.",
        "My subscription is expensive and premium features cost too much.",
        "The checkout and payment process does not work.",
        "I paid for something and cannot receive my refund.",
    ],
    "Notifications": [
        "I keep receiving duplicate push notifications and alerts.",
        "Important notifications never arrive on my phone.",
        "I want to change notification and reminder settings.",
        "The app sends too many alerts at the wrong time.",
    ],
    "Usability & Design": [
        "The app interface is confusing and difficult to navigate.",
        "The menu and buttons are hard to find.",
        "The new screen layout makes simple actions difficult.",
        "The app design and user experience need improvement.",
    ],
    "Customer Support": [
        "Customer service never replies to my support request.",
        "The support team and help desk did not solve my problem.",
        "My ticket has been ignored by the support agent.",
        "I cannot contact a customer service representative.",
    ],
    "Accessibility": [
        "The screen reader is broken for visually impaired users.",
        "Low contrast and tiny fonts make the app hard to read.",
        "The app is difficult to use with accessibility tools.",
        "VoiceOver and accessibility settings do not work properly.",
    ],
    "Feature Requests": [
        "Please add new features and more customization options.",
        "I wish the app had an additional feature for my workflow.",
        "It would be helpful to include a new chart or capability.",
        "The app is missing an option I need to complete my tasks.",
    ],
    "Content & Data": [
        "The app needs more articles, stories, and useful content.",
        "Financial data and stock market information are incomplete.",
        "Search results and news feeds are missing important information.",
        "The video and media library lacks the content I need.",
    ],
}

# A few categories are especially prone to false positives. A concrete
# phrase must be present before the model can assign these labels.
EVIDENCE_PATTERNS = {
    "Performance": re.compile(
        r"\b(?:crash\w*|freez\w*|frozen|lag\w*|slow\w*|buggy|"
        r"glitch\w*|unresponsive|stuck|hang\w*|won.t (?:even )?open|"
        r"can.t open|cannot open|won.t load|takes? forever to (?:open|load))\b",
        re.I,
    ),
    "Customer Support": re.compile(
        r"\b(?:customer service|customer support|support team|help ?desk|"
        r"support tickets?|service agents?|representative|contact support)\b",
        re.I,
    ),
    "Accessibility": re.compile(
        r"\b(?:screen ?reader|voiceover|accessib\w*|visual\w* impair\w*|"
        r"low contrast|font size|small font|text size|color contrast)\b",
        re.I,
    ),
    "Connectivity & Integration": re.compile(
        r"\b(?:connect\w*|disconnect\w*|pair\w*|sync\w*|bluetooth|"
        r"integration|linked? (?:my |the )?(?:account|device)|"
        r"linking? (?:my |the )?(?:account|device))\b",
        re.I,
    ),
    "Feature Requests": re.compile(
        r"\b(?:wish|please add|should add|could (?:you|we) add|"
        r"would (?:be helpful|love|like)|need (?:a |an |more )?feature|"
        r"missing (?:a |the )?(?:feature|option)|add (?:a |an )?(?:feature|option)|"
        r"more data|new feature|feature request|would help|could use)\b",
        re.I,
    ),
}

NEGATED_PERFORMANCE = re.compile(
    r"\b(?:no|without|never) (?:lags?|glitches?|crashes?|freezing)"
    r"(?:\s+(?:or|and)\s+(?:lags?|glitches?|crashes?|freezing))*\b|"
    r"\b(?:does not|doesn.t|is not|isn.t) (?:crash\w*|lag\w*|freez\w*)\b",
    re.I,
)
GENERIC_REVIEW = re.compile(
    r"^\s*(?:please fix it|fix it please|thank you|great app|bad app|"
    r"good app|terrible app|useless app|love it|hate it)[.!\s]*$",
    re.I,
)


def _has_category_evidence(text: str, theme: str) -> bool:
    pattern = EVIDENCE_PATTERNS.get(theme)
    if pattern is None:
        return True
    if pattern.search(text) is None:
        return False
    # Don't classify positive "no lag or crashes" as a performance issue.
    if theme == "Performance" and NEGATED_PERFORMANCE.search(text):
        without_negations = NEGATED_PERFORMANCE.sub("", text)
        return pattern.search(without_negations) is not None
    return True


def classify_themes(
    texts,
    threshold: float = 0.47,
    min_margin: float = 0.06,
) -> pd.DataFrame:
    """Predict themes conservatively, abstaining on ambiguous cases.

    Thresholds are initial engineering defaults, NOT validated metrics.
    Cosine similarity is not the probability that a label is correct.
    """
    if not (0 <= threshold <= 1 and 0 <= min_margin <= 1):
        raise ValueError("threshold and min_margin must be in [0, 1]")

    reviews = ["" if pd.isna(t) else str(t) for t in texts]
    if not reviews:
        return pd.DataFrame(columns=[
            "semantic_theme", "similarity_score", "score_margin", "review_needed"
        ])

    model = SentenceTransformer(MODEL_NAME)
    theme_names = list(THEME_EXAMPLES)

    examples = []
    groups = []
    for theme, phrases in THEME_EXAMPLES.items():
        for phrase in phrases:
            examples.append(phrase)
            groups.append(theme)

    review_vectors = model.encode(
        reviews,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    example_vectors = model.encode(examples, normalize_embeddings=True)
    similarities = review_vectors @ example_vectors.T

    theme_scores = np.column_stack([
        similarities[:, [i for i, group in enumerate(groups) if group == theme]].max(axis=1)
        for theme in theme_names
    ])

    outputs = []
    for text, row in zip(reviews, theme_scores):
        # Avoid low-information text and allow the model to abstain.
        if len(text.strip()) < 12 or GENERIC_REVIEW.fullmatch(text.strip()):
            outputs.append((OTHER, float(row.max()), 0.0, True))
            continue

        allowed = [
            i for i, theme in enumerate(theme_names)
            if _has_category_evidence(text, theme)
        ]
        ranked = sorted(allowed, key=lambda i: float(row[i]), reverse=True)
        if not ranked:
            outputs.append((OTHER, float(row.max()), 0.0, True))
            continue

        best = ranked[0]
        best_score = float(row[best])
        second_score = float(row[ranked[1]]) if len(ranked) > 1 else 0.0
        margin = best_score - second_score

        if best_score < threshold or margin < min_margin:
            outputs.append((OTHER, best_score, margin, True))
        else:
            outputs.append((theme_names[best], best_score, margin, False))

    return pd.DataFrame(
        outputs,
        columns=["semantic_theme", "similarity_score", "score_margin", "review_needed"],
    ).assign(
        similarity_score=lambda d: d["similarity_score"].round(3),
        score_margin=lambda d: d["score_margin"].round(3),
    )


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "english_reviews_sample.csv"
    output = root / "data" / "semantic_comparison_v2.csv"
    eval_output = root / "data" / "manual_label_sample.csv"

    raw = pd.read_csv(source)
    baseline = prepare_reviews(raw).reset_index(drop=True)
    predictions = classify_themes(baseline["review"])

    result = baseline.copy()
    result["baseline_theme"] = result["theme"]
    for col in predictions.columns:
        result[col] = predictions[col].to_numpy()

    print("\n=== Old Keyword Classification ===")
    print(result["baseline_theme"].value_counts())
    print("\n=== Improved Semantic Classification ===")
    print(result["semantic_theme"].value_counts())
    print("\n=== Predictions That Changed ===")
    changed = result[result["baseline_theme"] != result["semantic_theme"]]
    for _, item in changed.head(8).iterrows():
        print("\nReview:", str(item["review"])[:160])
        print("Old:", item["baseline_theme"])
        print("New:", item["semantic_theme"])

    result.to_csv(output, index=False)

    # Randomly sampled review queue. The blank human_label MUST be assigned
    # by a person before evaluating model accuracy. This sample is not ground truth.
    sample = result.sample(n=min(60, len(result)), random_state=42)
    sample = sample[[
        "review_id", "rating", "review", "baseline_theme",
        "semantic_theme", "similarity_score", "score_margin",
    ]].copy()
    sample["human_label"] = ""
    sample["annotation_notes"] = ""
    sample.to_csv(eval_output, index=False)

    print(f"\nSaved predictions: {output}")
    print(f"Saved manual annotation queue: {eval_output}")
    print("Note: true accuracy cannot be measured without human labels.")


if __name__ == "__main__":
    main()
