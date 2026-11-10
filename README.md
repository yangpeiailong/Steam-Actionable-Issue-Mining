# Steam Actionable Issue Mining

Research code accompanying a study of actionable issues reported in Steam game reviews.

The project uses a two-stage NLP pipeline:

1. **Stage 1 — actionable-issue detection:** identify reviews containing at least one actionable, product-addressable issue.
2. **Stage 2 — issue-domain classification:** assign one or more of 13 issue domains, or `NO_ISSUE`, to Stage-1-positive reviews.

This repository includes data-preparation and annotation-support scripts, model training and distillation, locked-test evaluation, full-corpus inference, cross-game analysis, qualitative case extraction, and a bounded event-window diagnostic.

> **Code-only release.** Steam review text, review-level metadata, annotation workbooks, Gold/test labels, full-corpus predictions, model checkpoints, and API credentials are not included. Reproducing the reported numerical results requires the corresponding locally held data and checkpoints.

## Repository structure

```text
.
├── config_local.py
├── requirements.txt
├── requirements-ml.txt
├── requirements-llm.txt
├── paper1_issue_mining/
│   ├── 03_issue_detection/          # Stage 1
│   └── 04_stage2_classification/    # Stage 2 and downstream analyses
├── data/                            # data policy / layout only
├── docs/
└── scripts/
```

Historical duplicate scripts, unsuccessful minority-enrichment experiments, temporary diagnostics, local outputs, and data files are intentionally excluded from this release.

## Environment

The code was developed on Windows. The research environment used Python 3.13, PyTorch 2.10.0, Transformers 5.17.0, and SentencePiece 0.2.2. Install a PyTorch build that matches your operating system, GPU, and CUDA runtime.

```powershell
conda create -n steamresearch python=3.13 -y
conda activate steamresearch
python -m pip install -r requirements.txt
```

For model training and inference, install the appropriate PyTorch build first, then:

```powershell
python -m pip install -r requirements-ml.txt
```

For optional LLM-assisted pre-annotation and teacher labelling:

```powershell
python -m pip install -r requirements-llm.txt
```

## Configure local paths

No author-specific local paths are stored in the repository. Set these environment variables in PowerShell:

```powershell
$env:STEAM_DATA_ROOT = "D:\path\to\Steam_Research"
$env:STEAM_MODELS_DIR = "D:\path\to\AI_Models\Steam_Research"
```

`STEAM_DATA_ROOT` must contain `Paper1_Actionable_Issue_Mining/data/` and `Paper1_Actionable_Issue_Mining/results/`.

`STEAM_MODELS_DIR` must contain `base_models/` and `stage2_runs/`. The final full-corpus analysis uses these frozen checkpoints by default:

```text
stage2_runs/
├── roberta_targeted_T2_arch_v1/best_model/
└── targeted_T3_v1/best_model/
```

Additional model paths used by the one-time test scripts can be overridden through the environment variables defined in `config_local.py`. Keep model weights outside the Git repository.

## Main workflow

Run commands from the repository root. Use `--help` on individual scripts for all supported options.

### Stage 1 — actionable-issue detection

The numbered scripts in `paper1_issue_mining/03_issue_detection/` follow the research workflow:

- `01_prepare_stage1_sample.py`: prepare a candidate annotation sample.
- `02_llm_prelabel_stage1.py`: optional LLM-assisted screening; these outputs are not human Gold labels.
- `03_prepare_stage1_manual_review.py` and `04_prepare_stage1_gold.py`: prepare and consolidate human annotations.
- `05_train_stage1_classifier.py` and `06_apply_stage1_classifier.py`: train and apply the initial classifier.
- `07_prepare_stage1_disagreement_audit.py` and `08_mine_stage1_hard_negatives.py`: prepare disagreement audits and hard-negative review.
- `09_train_stage1_deberta_hardneg.py` and `10_apply_stage1_deberta_hardneg_v2.py`: train and apply the revised Stage-1 model.

### Stage 2 — Gold and teacher data

Stage 2 uses a fixed 2,000-review Gold set partitioned into 1,400 training, 300 validation, and 300 held-out test records. The test split must remain locked until model and threshold choices have been made using training/validation data.

Candidate preparation and optional teacher labelling scripts are:

- `07_prepare_stage2_sample.py`
- `08_llm_prelabel_stage2.py`
- `11_llm_prelabel_stage2_prompt_v2.py`
- `13_prepare_stage2_distillation_candidates.py`
- `14_teacher_label_stage2_distillation.py`
- `15_build_stage2_distillation_expansion_pool.py`
- `16_teacher_label_stage2_expansion_1000.py`

Teacher labels are provisional pseudo-labels, not human Gold annotations. They must remain separate from the Gold data.

For DeepSeek-based steps, set the key in the process environment; never place it in source code:

```powershell
$env:DEEPSEEK_API_KEY = "YOUR_KEY"
```

### Stage 2 — training

`13_train_stage2_distillation.py` is the architecture-compatible training script. It supports:

- `gold_only`: manually labelled training rows only;
- `hard`: Gold labels plus hard pseudo-labels;
- `soft`: Gold labels plus teacher scores as soft targets.

Example preflight (adapt paths and base encoder to the local setup):

```powershell
python .\paper1_issue_mining\04_stage2_classification\13_train_stage2_distillation.py `
  --mode hard `
  --teacher-csv "D:\path\to\stage2_distillation_provisional_pool_expanded_1289.csv" `
  --base-model-dir "D:\path\to\base_models\roberta" `
  --model-output-dir "D:\path\to\stage2_runs\roberta_targeted_T2_arch_v1" `
  --results-dir "D:\path\to\results\roberta_targeted_T2_arch_v1" `
  --dry-run
```

Remove `--dry-run` to train. Use independent output directories for each architecture and configuration. Threshold selection must use validation data only.

### Locked test evaluation

Two scripts evaluate prespecified checkpoints:

- `17_evaluate_stage2_test_once.py`
- `17_evaluate_stage2_test_once_architecture_extension.py`

Run their `--help` and `--dry-run` modes first. The held-out test set is a one-time final evaluation resource. Do not use test results to tune thresholds, choose new configurations, or iterate model design.

### Full-corpus inference and analysis

The final full-corpus inference script is:

```powershell
python .\paper1_issue_mining\04_stage2_classification\19_infer_stage2_full_corpus.py --models both --dry-run
python .\paper1_issue_mining\04_stage2_classification\19_infer_stage2_full_corpus.py --models both --run
```

It retains manual Gold labels, does not send Gold records through inference models, and keeps the 400 Stage-1-negative Gold audit records separate from the primary Stage-1-positive analysis population.

After inference, the downstream analysis scripts are:

```powershell
python .\paper1_issue_mining\04_stage2_classification\21_analyze_stage2_population_v1.py --dry-run
python .\paper1_issue_mining\04_stage2_classification\22_test_stage2_game_differences.py --dry-run
python .\paper1_issue_mining\04_stage2_classification\23_extract_stage2_case_examples_v1.py --dry-run
python .\paper1_issue_mining\04_stage2_classification\24_analyze_event_window_v1.py --dry-run
```

Run with `--run` after each preflight succeeds. These scripts consume previously generated analysis files; they do not retrain the models.

## Stage-2 label taxonomy

Stage 2 is multi-label. Its 13 actionable issue domains are:

`BUG_STABILITY`, `PERFORMANCE`, `COMPATIBILITY`, `UI_UX`, `MULTIPLAYER_NETWORK`, `GAMEPLAY_BALANCE`, `AI_SYSTEM`, `CONTENT`, `MOD_SUPPORT`, `MONETIZATION`, `PRIVACY_POLICY`, `COMMUNITY_MODERATION`, and `OTHER_ISSUE`.

`NO_ISSUE` is mutually exclusive with the 13 issue labels. A review with no output label above the selected threshold is marked unassigned, not automatically `NO_ISSUE`.

## Interpretation and limitations

- Stage-2 estimates are conditional on Stage-1-positive candidate reviews; they are not unconditional issue prevalence estimates for all Steam reviews.
- Stage-2 F1 is approximately 0.7 in the current experiments. Full-corpus predictions are noisy model-based estimates, not individually verified labels.
- Gold labels and model-inferred labels are distinguished in the generated files.
- Multi-label percentages need not sum to 100%.
- Agreement between two models is a robustness signal, not proof that an individual prediction is correct.
- The event-window analysis is descriptive and does not establish that an external event caused a review burst or label shift.
- Findings are tied to the sampled games, collection period, annotation protocol, model checkpoints, and thresholds.

## Data and model availability

No raw Steam review text, review-level metadata, annotation files, Gold/test labels, full-corpus predictions, model checkpoints, or API credentials are distributed. Users must obtain data through permitted means and supply the expected local files. See `docs/data_and_file_layout.md`.

## Citation

Please cite the associated paper after its bibliographic details become available. A formal citation file can be added after publication.
