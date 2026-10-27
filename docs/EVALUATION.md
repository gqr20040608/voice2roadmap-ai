# Evaluation Plan

## Current automated checks

`python -m pytest -q` validates five areas: ingestion with aliases, sentiment sanity, missing columns, backlog traceability, TF-IDF retrieval, and topic fallback (these are grouped into five test functions).

## Before making any model-performance claims

1. Source a real review dataset with a permissible license, recorded provenance, and documented collection period.
2. Draw a random sample of at least 150–300 reviews; have human annotators mark sentiment and primary topic using a written codebook.
3. Resolve disagreements by a second reviewer; preserve a separate hold-out set.
4. Report class distribution, precision, recall, macro-F1, and a confusion matrix.
5. Evaluate the relationship of model errors to negation, software-specific vocabulary, and short-text noise.
6. For 25 realistic user questions, review top-5 evidence retrieval relevance; report precision@5 and failure examples.
7. For LLM briefs, manually check each cited review ID, factual statement, requirement interpretation, and implied business conclusion.
8. Compare requirements before/after human review, recording edit effort and discarded candidates.

**No benchmark accuracy or ROI has been measured for this MVP.** The synthetic demo is not a valid benchmark for model quality.

## Reliability / threat considerations

- Injection strings embedded inside reviews should not be followed by the assistant.
- API calls must require opt-in, and provider usage may have a cost.
- Avoid loading confidential customer comments on a public demo instance.
- Validate uploaded data types and skip blank review text.
