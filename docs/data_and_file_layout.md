# Data and local file layout

This repository deliberately excludes review-level Steam data and annotation records.

Set `STEAM_DATA_ROOT` to a directory containing `Paper1_Actionable_Issue_Mining/`. The local tree is expected to contain:

```text
Paper1_Actionable_Issue_Mining/
└── data/
    └── analysis/
        ├── all_reviews.csv
        ├── stage1_gold/
        ├── stage1_inference_hardneg_v2/
        │   ├── stage1_all_review_predictions.csv
        │   └── stage2_candidate_reviews.csv
        └── stage2_annotation/
            ├── Stage2_Gold_Release_v1/
            │   ├── stage2_gold_all_2000.csv
            │   ├── stage2_gold_train_1400.csv
            │   ├── stage2_gold_val_300.csv
            │   └── stage2_gold_test_300.csv
            ├── distillation_candidates_v1/
            └── distillation_candidates_v2_expansion/
```

The final inference script writes generated files under `Paper1_Actionable_Issue_Mining/results/`. Keep review text, labels, predictions, logs, and model checkpoints outside Git.

## Separation of data roles

- Human Gold labels are ground truth for the manually labelled sample.
- Teacher labels are provisional pseudo-labels and are not human annotations.
- The locked test split must not be used for training, pseudo-labelling, threshold tuning, or model selection.
- The 400 Stage-1-negative Gold audit records are a separate audit sample; they are not included in the primary Stage-1-positive Stage-2 denominator.
