# Voice2Roadmap AI — Product Requirements Document (MVP)

## 1. Problem and audience

**Users:** Product Analysts, Business Analysts, and Product Managers who need to turn feedback into transparent improvement proposals. **Problem:** Scattered comments do not directly map to prioritized, verifiable software requirements.

## 2. Goal

Reduce the manual steps required to go from a sample of customer reviews to a first-pass backlog, without obscuring the source evidence or presenting invented business metrics.

## 3. In scope

- Import CSV feedback and validate text rows; support optional date/product/rating columns.
- Compute baseline sentiment and single-theme labels.
- Display aggregate metrics, a time series, and a ranked issue table.
- Propose a textual improvement item and sample acceptance criterion for each theme.
- Show the source review IDs used as supporting evidence.
- Provide TF-IDF evidence retrieval and CSV export.
- If API credentials exist, generate an AI brief with bounded reviewed evidence.

## 4. Out of scope

Live app-store scraping, private customer account integrations, PII processing, operational ticket creation, and verified financial ROI forecasts.

## 5. Representative user stories

| ID | Story | Acceptance check |
| --- | --- | --- |
| US-01 | As an analyst, I can upload a CSV so I can explore real review data. | Required review column loads; empty/missing schema produces a useful error. |
| US-02 | As a PM, I can filter by product and sentiment. | All KPI and backlog values reflect the selected subset. |
| US-03 | As a systems analyst, I can trace a proposed requirement to customer statements. | Every row contains an evidence review ID present in the dataset. |
| US-04 | As a product lead, I can export a proposed backlog. | A CSV download includes requirement IDs, scores, status, evidence IDs and acceptance criteria. |
| US-05 | As an AI analyst, I can prepare a first-draft brief. | Generation is optional, requires credentials and instructs the model not to invent statistics. |

## 6. Non-functional requirements

- **Explainability:** Publish scoring formula and limitations.
- **Privacy:** Do not log/store customer keys; never commit secrets to source control.
- **Portability:** Deploy from a GitHub repo on Streamlit Community Cloud.
- **Failure handling:** Upload errors do not terminate the entire server process.
- **Human in the loop:** All proposed improvements must be reviewed prior to implementation.

## 7. Risks and mitigations

- **Unrepresentative data:** Reviews may overrepresent unhappy users; do not generalize to the full customer base.
- **Model error:** Lexicon errors and keyword drift; planned manual-label evaluation.
- **False confidence:** Heuristic priority is not true business impact; present it explicitly as a signal.
- **Prompt injection:** Customer reviews are untrusted; the LLM receives instructions to ignore embedded commands.
- **Dataset licensing:** Only use datasets with appropriate permissions and attribution.

## 8. Success criteria for the MVP

The app imports a valid file, generates traceable candidate requirements, returns deterministic search evidence, exports a backlog, and passes automated tests. Future evaluation will quantify label quality on a real annotated dataset.
