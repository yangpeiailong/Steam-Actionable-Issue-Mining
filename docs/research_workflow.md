# Research workflow and script map

## Stage 1 — actionable-issue detection

| Script | Role |
|---|---|
| 01 | Prepare a candidate annotation sample |
| 02 | Optional LLM-assisted pre-screening (not Gold) |
| 03 | Prepare a human-review workbook |
| 04 | Consolidate human labels / Gold |
| 05 | Train the initial binary classifier |
| 06 | Apply the initial classifier |
| 07 | Prepare disagreement audit |
| 08 | Mine and screen hard negatives |
| 09 | Train revised DeBERTa with human-confirmed hard negatives |
| 10 | Apply the revised Stage-1 model |

## Stage 2 — multi-label issue-domain classification

| Script | Role |
|---|---|
| 07 | Prepare Stage-2 annotation sample |
| 08 / 11 | Optional LLM pre-annotation |
| 13_prepare | Retrieve teacher-labelling candidates |
| 14 | Teacher-label the initial candidate pool |
| 15 | Build a non-overlapping expansion pool |
| 16 | Teacher-label the expansion pool |
| 13_train | Train Gold-only, hard, or soft configurations with compatible encoders |
| 17 test | Prespecified DeBERTa held-out test comparison |
| 17 architecture extension | Prespecified RoBERTa / ModernBERT held-out test comparison |
| 19 | Full-corpus inference with frozen RoBERTa T2 and DeBERTa T3 |
| 21 | Population summaries and descriptive analysis |
| 22 | Cross-game inferential comparisons and model robustness |
| 23 | Targeted qualitative case extraction |
| 24 | Bounded event-window diagnostic |

## Reproducibility guardrails

1. Split membership is defined by review IDs, not row order.
2. Test labels are not used for training, pseudo-labelling, threshold tuning, or model selection.
3. Thresholds are selected on validation data and frozen before final test evaluation.
4. Teacher labels remain distinct from human Gold labels.
5. Full-corpus predictions are not treated as individually verified annotations.
