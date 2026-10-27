"""Voice2Roadmap AI - safely separate uncategorized reviews from ranked requirements.

Put this file in voice2roadmap-ai/scripts and run:
    python scripts/fix_backlog_priorities.py

The script does not alter any review CSV or local AI model files.
"""

from datetime import datetime
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_PATH = ROOT / "src" / "analytics.py"
APP_PATH = ROOT / "app.py"

MARKER = "# Uncategorized reviews are not actionable requirement themes."
EMPTY_COLUMNS = '''BACKLOG_COLUMNS = [
    "theme", "mentions", "negative_reviews", "negative_rate",
    "signal_score", "requirement_id", "proposed_requirement",
    "example_acceptance_criterion", "evidence_ids", "status",
]

'''
QUEUE_SECTION = '''    # Keep unclassified feedback visible, without ranking it as a requirement.
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

'''


def replace_exact(text: str, old: str, new: str, description: str) -> str:
    matches = text.count(old)
    if matches != 1:
        raise RuntimeError(
            f"Expected exactly one '{description}' location, found {matches}. "
            "No project files changed. Please share this error."
        )
    return text.replace(old, new, 1)


def main() -> None:
    if not ANALYTICS_PATH.is_file() or not APP_PATH.is_file():
        raise FileNotFoundError(
            "Place this script inside voice2roadmap-ai/scripts/. "
            "The project must contain app.py and src/analytics.py."
        )

    analytics = ANALYTICS_PATH.read_text(encoding="utf-8")
    app = APP_PATH.read_text(encoding="utf-8")

    if MARKER in analytics and "Uncategorized review queue" in app:
        print("Priority fix was already applied. No changes needed.")
        return
    if MARKER in analytics or "Uncategorized review queue" in app:
        raise RuntimeError(
            "A partial fix was detected. No files changed. "
            "Please share this message before proceeding."
        )

    # No change to score calculation, evidence IDs, model choices, or Qwen logic.
    analytics_new = replace_exact(
        analytics,
        "\ndef build_backlog(reviews: pd.DataFrame) -> pd.DataFrame:\n",
        "\n" + EMPTY_COLUMNS + "def build_backlog(reviews: pd.DataFrame) -> pd.DataFrame:\n",
        "build_backlog definition",
    )
    analytics_new = replace_exact(
        analytics_new,
        "    if reviews.empty:\n        return pd.DataFrame()\n",
        "    if reviews.empty:\n        return pd.DataFrame(columns=BACKLOG_COLUMNS)\n",
        "empty reviews branch",
    )
    analytics_new = replace_exact(
        analytics_new,
        '    for theme, chunk in reviews.groupby("theme"):\n',
        '    for theme, chunk in reviews.groupby("theme"):\n'
        f'        {MARKER}\n'
        '        if theme == "Other / General":\n'
        '            continue\n',
        "theme iteration",
    )
    analytics_new = replace_exact(
        analytics_new,
        "    output = pd.DataFrame(records).sort_values(",
        "    if not records:\n"
        "        return pd.DataFrame(columns=BACKLOG_COLUMNS)\n"
        "    output = pd.DataFrame(records).sort_values(",
        "backlog output",
    )

    # Add the manual-review queue to the requirements tab.
    app_new = replace_exact(
        app,
        '    st.markdown("#### Evidence traceability")\n',
        QUEUE_SECTION + '    st.markdown("#### Evidence traceability")\n',
        "evidence traceability heading",
    )

    # Validate syntax of BOTH modified files before changing either.
    compile(analytics_new, str(ANALYTICS_PATH), "exec")
    compile(app_new, str(APP_PATH), "exec")

    backup_root = ROOT.parent / "voice2roadmap-backups"
    backup_dir = backup_root / ("backlog-fix-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ANALYTICS_PATH, backup_dir / "analytics.py")
    shutil.copy2(APP_PATH, backup_dir / "app.py")

    ANALYTICS_PATH.write_text(analytics_new, encoding="utf-8")
    APP_PATH.write_text(app_new, encoding="utf-8")

    print("Backups created:", backup_dir)
    print("Priority ranking fix applied successfully.")
    print("Other / General stays in the dataset and its own review queue.")
    print("Ranked requirements now exclude Other / General.")
    print("No AI model or uploaded data files were modified.")


if __name__ == "__main__":
    main()
