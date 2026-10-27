# Preliminary model comparison (AI-assisted labels)

**Important:** All reference labels in this file were produced with AI assistance. They have **not** been independently reviewed by a human. The results below are model-to-AI-label agreement diagnostics, **not validated accuracy, precision, or F1 against human ground truth**.

- Reviews in evaluation file: **60**
- Baseline exact agreement: **20/60 (33.3%)**
- Semantic V2 exact agreement: **25/60 (41.7%)**
- Baseline macro F1 vs AI-assisted labels: **0.127**
- Semantic V2 macro F1 vs AI-assisted labels: **0.227**

## Interpretation

These are exploratory diagnostics only. The sample contains 60 comments and is too small to establish reliable per-category performance. The `Other / General` label is broad, and taxonomy mismatches can affect scores. The same AI that suggested labels may share biases with the classifier.

## Next validation steps

1. Independently human-review a larger stratified sample; record labeling rules and ambiguous cases.
2. Compare class-wise precision, recall, and F1 using confirmed labels.
3. Tune thresholds on a separate development set and evaluate once on a held-out set.

Files: `data/ai_assisted_label_comparison.csv`, `data/ai_assisted_v2_confusion_matrix.csv`.
