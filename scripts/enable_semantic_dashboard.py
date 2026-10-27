"""Enable optional local embedding theme classification in Voice2Roadmap AI.

Run from the project root:
    python scripts/enable_semantic_dashboard.py

- Preserves the existing local-Qwen AI Decision Brief section.
- Adds an optional semantic mode; baseline remains the default.
- Adds backlog templates for the three new theme categories.
- Creates backups outside the GitHub project, next to voice2roadmap-ai.
"""

from datetime import datetime
from pathlib import Path
import py_compile
import shutil

ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "app.py"
ANALYTICS_FILE = ROOT / "src" / "analytics.py"

HELPER_MARKER = "def apply_embedding_themes(reviews: pd.DataFrame) -> pd.DataFrame:"
MODE_MARKER = '    theme_mode = st.radio("Theme classification"'

HELPER = '''@st.cache_data(show_spinner=False)
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


'''

MODE = '''    theme_mode = st.radio("Theme classification", [
        "Keyword baseline", "Embedding AI (experimental)"
    ], index=0, help="Experimental English-language theme predictions. Not verified accuracy.")
    if theme_mode == "Embedding AI (experimental)":
        try:
            with st.spinner("Analyzing themes with the local embedding model..."):
                reviews = apply_embedding_themes(reviews)
        except Exception as exc:
            st.error(f"Embedding classification failed: {exc}")
            st.stop()
'''

THEMES = '''    "Connectivity & Integration": (
        "Improve device connectivity and third-party integrations",
        "Target devices and connected services sync successfully under agreed test conditions."
    ),
    "Content & Data": (
        "Improve content availability and data completeness",
        "Priority content and data fields meet an agreed completeness checklist in validation tests."
    ),
    "Feature Requests": (
        "Evaluate and prioritize requested product capabilities",
        "A product owner records the feature scope and validates agreed acceptance criteria with users."
    ),
'''


def insert_once(text: str, anchor: str, insertion: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} anchor, found {count}. No files changed.")
    return text.replace(anchor, insertion + anchor, 1)


def main() -> None:
    if not APP_FILE.is_file() or not ANALYTICS_FILE.is_file():
        raise FileNotFoundError(
            "Run this script from the correct Voice2Roadmap AI project: "
            "app.py and src/analytics.py must exist."
        )
    if not (ROOT / "src" / "semantic_classifier.py").is_file():
        raise FileNotFoundError("Missing src/semantic_classifier.py (V2 model).")

    app = APP_FILE.read_text(encoding="utf-8")
    analytics = ANALYTICS_FILE.read_text(encoding="utf-8")

    installed = HELPER_MARKER in app and MODE_MARKER in app
    theme_names = [
        '"Connectivity & Integration":', '"Content & Data":', '"Feature Requests":'
    ]
    new_themes_installed = all(name in analytics for name in theme_names)
    if installed and new_themes_installed:
        print("Already enabled. No changes needed.")
        return
    if installed or new_themes_installed or HELPER_MARKER in app or MODE_MARKER in app:
        raise RuntimeError(
            "Detected partially modified files. No changes made. "
            "Please share the message before trying again."
        )

    app_new = insert_once(app, "with st.sidebar:\n", HELPER, "sidebar")
    app_new = insert_once(
        app_new,
        '    products = ["All products"] + sorted(reviews["product"].unique().tolist())\n',
        MODE,
        "product filter",
    )
    app_new = insert_once(
        app_new,
        'st.caption("Product intelligence · NLP analytics · evidence-backed requirements")\n',
        'st.caption("Select the baseline or experimental embedding model in the sidebar. "\n'
        '           "Embedding similarity is not a calibrated confidence probability.")\n',
        "dashboard heading",
    )
    analytics_new = insert_once(
        analytics,
        '    "Other / General": (',
        THEMES,
        "backlog themes",
    )

    # Check syntax before touching the user's original files.
    compile(app_new, str(APP_FILE), "exec")
    compile(analytics_new, str(ANALYTICS_FILE), "exec")

    # Backups are deliberately kept OUTSIDE the project/GitHub repository.
    backup_folder = ROOT.parent / "voice2roadmap-backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(APP_FILE, backup_folder / "app.py")
    shutil.copy2(ANALYTICS_FILE, backup_folder / "analytics.py")
    print(f"Backups saved: {backup_folder}")

    APP_FILE.write_text(app_new, encoding="utf-8")
    ANALYTICS_FILE.write_text(analytics_new, encoding="utf-8")
    py_compile.compile(str(APP_FILE), doraise=True)
    py_compile.compile(str(ANALYTICS_FILE), doraise=True)
    print("Semantic mode enabled successfully!")
    print("Your local Qwen report code has NOT been replaced.")
    print("Run: python -m streamlit run app.py --server.port 8502")
    print("Sidebar: Theme classification -> Embedding AI (experimental)")


if __name__ == "__main__":
    main()
