# -*- coding: utf-8 -*-
"""
Steam Stage 2: controlled comparison of Gold-only, hard pseudo-label, and soft distillation.

Gold train/validation/test split remains unchanged. The test split is read only for
review_id overlap checks; this script does NOT evaluate the test set.

Modes:
  gold_only : train only on the original 1,400 manually labelled Gold rows
  hard      : Gold + provisional high-confidence teacher labels as 0/1 targets
  soft      : Gold + per-label teacher scores as soft targets

Recommended first command:
  python paper1_issue_mining/04_stage2_classification/13_train_stage2_distillation.py --mode hard --dry-run

Then train variants separately:
  python paper1_issue_mining/04_stage2_classification/13_train_stage2_distillation.py --mode gold_only
  python paper1_issue_mining/04_stage2_classification/13_train_stage2_distillation.py --mode hard
  python paper1_issue_mining/04_stage2_classification/13_train_stage2_distillation.py --mode soft

All output folders are independent; existing models/results are never overwritten unless
--overwrite-output is explicitly passed.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config_local import PAPER1_ROOT as PAPER_ROOT, MODELS_ROOT, DEBERTA_BASE_MODEL_DIR

GOLD_DIR = PAPER_ROOT / "data" / "analysis" / "stage2_annotation" / "Stage2_Gold_Release_v1"
TEACHER_DIR = PAPER_ROOT / "data" / "analysis" / "stage2_annotation" / "distillation_candidates_v1"
DEFAULT_TEACHER_CSV = TEACHER_DIR / "stage2_distillation_provisional_pool_467.csv"
FALLBACK_TEACHER_CSV = TEACHER_DIR / "stage2_distillation_teacher_annotations_v1.csv"
BASE_MODEL_DIR = DEBERTA_BASE_MODEL_DIR
MODEL_RUNS_ROOT = MODELS_ROOT / "stage2_runs"
RESULTS_ROOT = PAPER_ROOT / "results" / "stage2_models"

ISSUE_LABELS = [
    "BUG_STABILITY",
    "PERFORMANCE",
    "COMPATIBILITY",
    "UI_UX",
    "MULTIPLAYER_NETWORK",
    "GAMEPLAY_BALANCE",
    "AI_SYSTEM",
    "CONTENT",
    "MOD_SUPPORT",
    "MONETIZATION",
    "PRIVACY_POLICY",
    "COMMUNITY_MODERATION",
    "OTHER_ISSUE",
]
LABELS = ISSUE_LABELS + ["NO_ISSUE"]
LABEL_COLS = [f"label_{x}" for x in LABELS]


def parse_args():
    p = argparse.ArgumentParser(description="Train/evaluate Stage-2 distillation variants using validation only.")
    p.add_argument("--mode", choices=["gold_only", "hard", "soft"], required=True,
                   help="gold_only baseline, hard pseudo-label training, or soft teacher-score distillation")
    p.add_argument("--train-csv", type=Path, default=GOLD_DIR / "stage2_gold_train_1400.csv")
    p.add_argument("--val-csv", type=Path, default=GOLD_DIR / "stage2_gold_val_300.csv")
    p.add_argument("--test-csv", type=Path, default=GOLD_DIR / "stage2_gold_test_300.csv",
                   help="Read only review_id for leakage checks; test labels are never evaluated.")
    p.add_argument("--teacher-csv", type=Path, default=DEFAULT_TEACHER_CSV)
    p.add_argument("--base-model-dir", type=Path, default=BASE_MODEL_DIR)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--pseudo-weight", type=float, default=0.5,
                   help="Relative per-example loss weight for each pseudo-labelled row; ignored in gold_only mode.")
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--model-output-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--overwrite-output", action="store_true",
                   help="Delete this run's model/results directories before starting.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate paths, IDs, labels, and candidate filtering without loading the model.")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def load_local_tokenizer(model_dir: Path, prefer_fast: bool = False):
    """Architecture-compatible local tokenizer loader."""
    if prefer_fast:
        print("Tokenizer mode: fast (ModernBERT)")
        return AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, use_fast=True
        )
    try:
        return AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, use_fast=False
        )
    except Exception as exc:
        print(f"Slow tokenizer unavailable ({type(exc).__name__}); trying fast.")
        return AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, use_fast=True
        )


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def load_gold(path: Path, split_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Gold split not found: {path}\n"
            "Confirm Stage2_Gold_Release_v1 is extracted into the documented folder."
        )
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"stage2_id", "review_id", "review_text", "manual_domains", *LABEL_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{split_name} is missing columns: {sorted(missing)}")
    df["review_id"] = df["review_id"].map(normalize_id)
    if df["review_id"].eq("").any() or df["review_id"].duplicated().any():
        raise ValueError(f"{split_name}: review_id is empty or duplicated.")
    text = df["review_text"].fillna("").astype(str).str.strip()
    if text.eq("").any():
        raise ValueError(f"{split_name}: empty review_text exists.")
    for col in LABEL_COLS:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"{split_name}: {col} must contain only 0/1.")
        df[col] = values.astype(int)
    issue_sum = df[[f"label_{x}" for x in ISSUE_LABELS]].sum(axis=1)
    no_issue = df["label_NO_ISSUE"]
    if ((no_issue == 1) & (issue_sum > 0)).any():
        raise ValueError(f"{split_name}: NO_ISSUE co-occurs with issue labels.")
    if ((no_issue == 0) & (issue_sum == 0)).any():
        raise ValueError(f"{split_name}: row has neither issue label nor NO_ISSUE.")
    df["review_text"] = text
    print(f"{split_name}: {len(df)} rows | unique games={df['game'].nunique() if 'game' in df else 'n/a'}")
    return df


def load_test_ids(path: Path) -> set:
    if not path.is_file():
        raise FileNotFoundError(
            f"Test split not found: {path}. The script needs its IDs only to prevent leakage."
        )
    ids_df = pd.read_csv(path, usecols=["review_id"], dtype={"review_id": "string"}, encoding="utf-8-sig")
    ids = ids_df["review_id"].map(normalize_id)
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Test split review_id is empty or duplicated.")
    print(f"test: {len(ids)} IDs loaded for overlap checking only; no test metrics will be calculated.")
    return set(ids)


def resolve_teacher_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path == DEFAULT_TEACHER_CSV and FALLBACK_TEACHER_CSV.is_file():
        print(f"Provisional pool not found; using full teacher file and applying the same filters: {FALLBACK_TEACHER_CSV}")
        return FALLBACK_TEACHER_CSV
    raise FileNotFoundError(
        f"Teacher pseudo-label file not found: {path}\n"
        "Extract Steam_Stage2_distillation_data_package_v1.zip into the distillation_candidates_v1 folder, "
        "or pass --teacher-csv with the correct path."
    )


def truthy_series(s: pd.Series, field_name: str = "flag") -> pd.Series:
    """
    Parse boolean-like CSV columns robustly.

    Teacher annotation files may contain either numeric 0/1 values or native
    boolean values. A mixed pandas object column can turn False into NaN under
    pd.to_numeric(errors="coerce"), which previously excluded valid rows.
    Unknown/missing values are rejected rather than silently treated as False.
    """
    normalized = s.astype("string").str.strip().str.lower()
    text_map = {
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
    }
    parsed_text = normalized.map(text_map)
    parsed_numeric = pd.to_numeric(s, errors="coerce")
    parsed = parsed_text.where(parsed_text.notna(), parsed_numeric)

    invalid = parsed.isna() | ~parsed.isin([0, 1])
    if invalid.any():
        examples = s.loc[invalid].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{field_name} contains missing or invalid boolean values. "
            f"Expected True/False or 1/0; examples: {examples}"
        )
    return parsed.astype(int).eq(1)


def parse_domains(value) -> List[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


def load_teacher_candidates(path: Path, excluded_ids: set) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = resolve_teacher_path(path)
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"review_id": "string"}, low_memory=False)
    required = {
        "review_id", "review_text", "teacher_domains", "teacher_overall_confidence",
        "teacher_needs_human_review", "teacher_status", "gold_or_test_data"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Teacher file is missing required columns: {sorted(missing)}")

    df["review_id"] = df["review_id"].map(normalize_id)
    if df["review_id"].eq("").any():
        raise ValueError("Teacher file contains empty review_id.")
    if df["review_id"].duplicated().any():
        raise ValueError("Teacher file contains duplicated review_id.")
    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
    if df["review_text"].eq("").any():
        raise ValueError("Teacher file contains empty review_text.")

    conf = pd.to_numeric(df["teacher_overall_confidence"], errors="coerce")
    needs_review = truthy_series(
        df["teacher_needs_human_review"], "teacher_needs_human_review"
    )
    is_gold_test = truthy_series(df["gold_or_test_data"], "gold_or_test_data")
    success = df["teacher_status"].astype(str).str.strip().str.lower().eq("success")
    confidence_ok = conf.ge(0.70)
    mask = success & ~needs_review & confidence_ok & ~is_gold_test
    candidates = df.loc[mask].copy().reset_index(drop=True)

    overlap = set(candidates["review_id"]) & excluded_ids
    if overlap:
        sample = sorted(overlap)[:10]
        raise ValueError(
            f"Teacher candidates overlap Gold train/validation/test IDs: {len(overlap)}. "
            f"Examples: {sample}. Do not train until resolved."
        )
    if candidates.empty:
        raise ValueError("No eligible pseudo-label candidates remain after quality and leakage filters.")

    hard_rows = []
    for _, row in candidates.iterrows():
        domains = parse_domains(row["teacher_domains"])
        if not domains or any(label not in LABELS for label in domains):
            raise ValueError(f"Invalid teacher_domains for review_id={row['review_id']}: {domains}")
        if "NO_ISSUE" in domains and len(domains) != 1:
            raise ValueError(f"NO_ISSUE co-occurs with issue label for review_id={row['review_id']}")
        target = np.zeros(len(LABELS), dtype=np.float32)
        for label in domains:
            target[LABELS.index(label)] = 1.0
        hard_rows.append(target)
    hard_y = np.stack(hard_rows)

    soft_rows = []
    for _, row in candidates.iterrows():
        scores = {}
        valid_score_cols = all(f"teacher_score_{label}" in candidates.columns for label in LABELS)
        if valid_score_cols:
            for label in LABELS:
                val = pd.to_numeric(pd.Series([row[f"teacher_score_{label}"]]), errors="coerce").iloc[0]
                if pd.isna(val):
                    raise ValueError(f"Missing soft score {label} for review_id={row['review_id']}")
                scores[label] = float(val)
        elif "teacher_domain_scores_json" in candidates.columns:
            try:
                scores = json.loads(row["teacher_domain_scores_json"])
            except Exception as exc:
                raise ValueError(f"Invalid teacher_domain_scores_json for review_id={row['review_id']}: {exc}")
        else:
            raise ValueError("Teacher file has neither teacher_score_* columns nor teacher_domain_scores_json.")

        if set(scores) != set(LABELS):
            raise ValueError(f"Soft score keys do not match the 14-label schema for review_id={row['review_id']}")
        vals = np.array([float(scores[label]) for label in LABELS], dtype=np.float32)
        if not np.isfinite(vals).all() or (vals < 0).any() or (vals > 1).any():
            raise ValueError(f"Soft scores outside [0,1] for review_id={row['review_id']}")

        # Preserve the task's exclusive NO_ISSUE rule in the soft targets:
        # issue-labelled examples have NO_ISSUE target zero; NO_ISSUE examples have
        # all issue targets zero. The selected domain scores remain soft targets.
        domains = parse_domains(row["teacher_domains"])
        if "NO_ISSUE" in domains:
            vals[:len(ISSUE_LABELS)] = 0.0
        else:
            vals[LABELS.index("NO_ISSUE")] = 0.0
        soft_rows.append(vals)
    soft_y = np.stack(soft_rows)

    counts = pd.Series([x for row in candidates["teacher_domains"].map(parse_domains) for x in row]).value_counts()
    print(f"Teacher source: {path}")
    print(f"Eligible teacher candidates: {len(candidates)}")
    print(f"Excluded from source by review/confidence/status/Gold-test flags: {len(df) - len(candidates)}")
    print(
        "Filter diagnostics:",
        {
            "status_not_success": int((~success).sum()),
            "needs_human_review": int(needs_review.sum()),
            "confidence_below_0.70_or_missing": int((~confidence_ok).sum()),
            "gold_or_test_flagged": int(is_gold_test.sum()),
        },
    )
    print(f"Teacher labels (multi-label counts): {counts.to_dict()}")
    print(f"Teacher candidate game coverage: {candidates['game'].nunique() if 'game' in candidates else 'n/a'} games")
    return candidates, hard_y, soft_y


class ReviewDataset(Dataset):
    def __init__(self, texts: List[str], targets: np.ndarray, is_pseudo: np.ndarray):
        self.texts = texts
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.is_pseudo = torch.tensor(is_pseudo, dtype=torch.bool)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "labels": self.targets[idx],
            "is_pseudo": self.is_pseudo[idx],
        }


def make_loader(texts, targets, is_pseudo, tokenizer, batch_size, max_length, shuffle):
    dataset = ReviewDataset(texts, targets, is_pseudo)
    pad_multiple = 8 if torch.cuda.is_available() else None

    def collate(items):
        encoded = tokenizer(
            [x["text"] for x in items],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        batch = tokenizer.pad(
            encoded, padding=True, pad_to_multiple_of=pad_multiple, return_tensors="pt"
        )
        batch["labels"] = torch.stack([x["labels"] for x in items])
        batch["is_pseudo"] = torch.stack([x["is_pseudo"] for x in items])
        return batch

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=torch.cuda.is_available(), collate_fn=collate
    )


def apply_thresholds(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    issue_pred = probabilities[:, :len(ISSUE_LABELS)] >= threshold
    no_issue_pred = (
        (probabilities[:, LABELS.index("NO_ISSUE")] >= threshold)
        & (~issue_pred.any(axis=1))
    )
    return np.concatenate([issue_pred.astype(int), no_issue_pred[:, None].astype(int)], axis=1)


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    issue_true = y_true[:, :len(ISSUE_LABELS)]
    issue_pred = y_pred[:, :len(ISSUE_LABELS)]
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        issue_true, issue_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        issue_true, issue_pred, average="macro", zero_division=0
    )
    full_issue_true = 1 - y_true[:, LABELS.index("NO_ISSUE")]
    full_issue_pred = issue_pred.any(axis=1).astype(int)
    return {
        "issue_domain_micro_precision": float(p_micro),
        "issue_domain_micro_recall": float(r_micro),
        "issue_domain_micro_f1": float(f_micro),
        "issue_domain_macro_precision": float(p_macro),
        "issue_domain_macro_recall": float(r_macro),
        "issue_domain_macro_f1": float(f_macro),
        "all_14_label_micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "all_14_label_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_set_match": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "issue_vs_no_issue_accuracy": float(accuracy_score(full_issue_true, full_issue_pred)),
        "issue_vs_no_issue_f1": float(f1_score(full_issue_true, full_issue_pred, zero_division=0)),
    }


def best_global_threshold(probs: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    candidates = [round(x, 2) for x in np.arange(0.20, 0.801, 0.05)]
    scored = []
    for threshold in candidates:
        pred_issue = (probs[:, :len(ISSUE_LABELS)] >= threshold).astype(int)
        score = f1_score(
            y_true[:, :len(ISSUE_LABELS)],
            pred_issue,
            average="macro",
            zero_division=0,
        )
        scored.append((float(score), abs(threshold - 0.5), threshold))
    score, _, threshold = sorted(scored, key=lambda x: (-x[0], x[1]))[0]
    return float(threshold), float(score)


@torch.no_grad()
def predict_probabilities(model, loader, device, use_amp):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, total_n = 0.0, 0
    for batch in loader:
        labels = batch.pop("labels").to(device, non_blocking=True)
        batch.pop("is_pseudo")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(**batch).logits
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels)
        total_loss += float(loss.item()) * len(labels)
        total_n += len(labels)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0).astype(int)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    return probs, y_true, total_loss / max(total_n, 1)


def write_val_predictions(path: Path, val_df: pd.DataFrame, probs: np.ndarray,
                          y_true: np.ndarray, threshold: float):
    pred = apply_thresholds(probs, threshold)
    out = pd.DataFrame({
        "stage2_id": val_df["stage2_id"].astype(str).values,
        "game": val_df["game"].astype(str).values if "game" in val_df else "",
        "review_id": val_df["review_id"].values,
        "review_text": val_df["review_text"].astype(str).values,
        "manual_domains": val_df["manual_domains"].astype(str).values,
        "predicted_domains": [
            "|".join([LABELS[j] for j, val in enumerate(row) if val == 1]) for row in pred
        ],
        "threshold": threshold,
    })
    for j, label in enumerate(LABELS):
        out[f"prob_{label}"] = probs[:, j]
        out[f"pred_{label}"] = pred[:, j]
        out[f"true_{label}"] = y_true[:, j]
    out.to_csv(path, index=False, encoding="utf-8-sig")


def per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, label in enumerate(LABELS):
        p, r, f, support = precision_recall_fscore_support(
            y_true[:, j], y_pred[:, j], average="binary", zero_division=0
        )
        rows.append({
            "label": label, "precision": float(p), "recall": float(r),
            "f1": float(f), "support": int(y_true[:, j].sum()),
            "predicted_positive": int(y_pred[:, j].sum()),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 64:
        raise ValueError("--epochs/batch-size must be positive and --max-length >= 64.")
    if not 0.0 < args.pseudo_weight <= 1.0:
        raise ValueError("--pseudo-weight must be in (0,1].")
    if not args.base_model_dir.is_dir() and not args.dry_run:
        raise FileNotFoundError(f"Local base model not found: {args.base_model_dir}")

    run_tag = {
        "gold_only": "deberta_stage2_gold_only_repro_v1",
        "hard": "deberta_stage2_distill_hard_v1",
        "soft": "deberta_stage2_distill_soft_v1",
    }[args.mode]
    model_dir = args.model_output_dir or (MODEL_RUNS_ROOT / run_tag)
    results_dir = args.results_dir or (RESULTS_ROOT / run_tag)

    train_df = load_gold(args.train_csv, "train")
    val_df = load_gold(args.val_csv, "validation")
    train_ids = set(train_df["review_id"])
    val_ids = set(val_df["review_id"])
    if train_ids & val_ids:
        raise ValueError("Gold train and validation review IDs overlap.")

    test_ids = load_test_ids(args.test_csv)
    if train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Gold train/validation/test review IDs overlap.")

    teacher_df = None
    hard_y = soft_y = None
    if args.mode in {"hard", "soft"}:
        teacher_df, hard_y, soft_y = load_teacher_candidates(
            args.teacher_csv, train_ids | val_ids | test_ids
        )
        if set(teacher_df["review_id"]) & (train_ids | val_ids | test_ids):
            raise ValueError("Pseudo-label candidate overlaps a Gold split.")

    if args.mode == "gold_only":
        combined_texts = train_df["review_text"].astype(str).tolist()
        combined_y = train_df[LABEL_COLS].to_numpy(dtype=np.float32)
        is_pseudo = np.zeros(len(train_df), dtype=bool)
        gold_count, pseudo_count = len(train_df), 0
    else:
        pseudo_y = hard_y if args.mode == "hard" else soft_y
        combined_texts = (
            train_df["review_text"].astype(str).tolist()
            + teacher_df["review_text"].astype(str).tolist()
        )
        combined_y = np.concatenate([
            train_df[LABEL_COLS].to_numpy(dtype=np.float32), pseudo_y
        ], axis=0)
        is_pseudo = np.concatenate([
            np.zeros(len(train_df), dtype=bool),
            np.ones(len(teacher_df), dtype=bool),
        ])
        gold_count, pseudo_count = len(train_df), len(teacher_df)

    print("\nRUN CONFIGURATION")
    print(f"Mode: {args.mode}")
    print(f"Gold training rows: {gold_count}")
    print(f"Teacher pseudo-label rows: {pseudo_count}")
    print(f"Combined training rows: {len(combined_texts)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Labels: {', '.join(LABELS)}")
    print(f"Pseudo loss weight: {args.pseudo_weight if args.mode != 'gold_only' else 'not applicable'}")
    print(f"Model output: {model_dir}")
    print(f"Results output: {results_dir}")
    print("Test split: IDs used for leakage checks only; no test evaluation.")

    if args.dry_run:
        print("\nDRY RUN PASSED: input schemas, labels, exclusivity, and split/candidate overlap checks passed.")
        if teacher_df is not None:
            print("Eligible pseudo candidates:", len(teacher_df))
            print("Hard teacher labels:", pd.Series([
                lab for row in teacher_df["teacher_domains"].map(parse_domains) for lab in row
            ]).value_counts().to_dict())
        return

    for path in [model_dir, results_dir]:
        if path.exists() and any(path.iterdir()):
            if not args.overwrite_output:
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {path}\n"
                    "Choose a new --model-output-dir/--results-dir or pass --overwrite-output."
                )
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP fp16: {use_amp}")
    print(f"Base model: {args.base_model_dir}")

    prefer_fast_tokenizer = "modernbert" in args.base_model_dir.name.lower()
    tokenizer = load_local_tokenizer(
        args.base_model_dir, prefer_fast=prefer_fast_tokenizer
    )
    train_loader = make_loader(
        combined_texts, combined_y, is_pseudo, tokenizer,
        args.batch_size, args.max_length, shuffle=True
    )
    val_y = val_df[LABEL_COLS].to_numpy(dtype=np.float32)
    val_loader = make_loader(
        val_df["review_text"].astype(str).tolist(), val_y,
        np.zeros(len(val_df), dtype=bool), tokenizer,
        args.batch_size, args.max_length, shuffle=False
    )

    id2label = {i: label for i, label in enumerate(LABELS)}
    label2id = {label: i for i, label in enumerate(LABELS)}
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.base_model_dir), local_files_only=True,
        num_labels=len(LABELS), problem_type="multi_label_classification",
        id2label=id2label, label2id=label2id
    )
    model.config.problem_type = "multi_label_classification"
    model = model.float().to(device)

    # Compute class weights only from Gold training labels, never from pseudo-label counts.
    y_gold = train_df[LABEL_COLS].to_numpy(dtype=np.float32)
    positives = y_gold.sum(axis=0)
    negatives = len(y_gold) - positives
    pos_weight = torch.tensor(
        np.clip(negatives / np.maximum(positives, 1.0), 1.0, 5.0),
        dtype=torch.float32, device=device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.10))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_macro = -1.0
    best_epoch = 0
    stale = 0
    history = []
    best_model_dir = model_dir / "best_model"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            targets = batch.pop("labels").to(device, non_blocking=True)
            pseudo_mask = batch.pop("is_pseudo").to(device, non_blocking=True)
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**inputs).logits
                logits32 = logits.float()
                # Gold receives bounded class-weighted BCE; teacher rows receive
                # unweighted BCE so subjective teacher scores are not amplified by pos_weight.
                gold_loss = F.binary_cross_entropy_with_logits(
                    logits32, targets, pos_weight=pos_weight, reduction="none"
                ).mean(dim=1)
                pseudo_loss = F.binary_cross_entropy_with_logits(
                    logits32, targets, reduction="none"
                ).mean(dim=1)
                row_loss = torch.where(pseudo_mask, pseudo_loss, gold_loss)
                row_weight = torch.where(
                    pseudo_mask,
                    torch.full_like(row_loss, float(args.pseudo_weight)),
                    torch.ones_like(row_loss)
                )
                loss = (row_loss * row_weight).sum() / row_weight.sum().clamp_min(1e-8)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.item()) * len(targets)
            seen += len(targets)

        val_probs, val_true, val_loss = predict_probabilities(model, val_loader, device, use_amp)
        val_pred_05 = apply_thresholds(val_probs, 0.5)
        val_metrics_05 = metric_summary(val_true, val_pred_05)
        val_macro = val_metrics_05["issue_domain_macro_f1"]
        train_loss = running_loss / max(seen, 1)
        row = {
            "epoch": epoch,
            "train_loss_weighted_bce": train_loss,
            "val_loss_bce": val_loss,
            "val_issue_domain_macro_f1_at_0.5": val_macro,
            "val_issue_domain_micro_f1_at_0.5": val_metrics_05["issue_domain_micro_f1"],
            "val_all_14_label_macro_f1_at_0.5": val_metrics_05["all_14_label_macro_f1"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_issue_macro_f1@0.5={val_macro:.4f} "
            f"val_issue_micro_f1@0.5={val_metrics_05['issue_domain_micro_f1']:.4f}",
            flush=True
        )

        if val_macro > best_macro + 1e-6:
            best_macro = val_macro
            best_epoch = epoch
            stale = 0
            if best_model_dir.exists():
                shutil.rmtree(best_model_dir)
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_model_dir)
            print(f"  Saved new best model (epoch {epoch})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping: no validation Macro-F1 improvement for {args.patience} epochs.")
                break

    pd.DataFrame(history).to_csv(
        results_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    if not best_model_dir.exists():
        raise RuntimeError("Training finished without a saved best model.")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    best_tokenizer = load_local_tokenizer(
        best_model_dir, prefer_fast=prefer_fast_tokenizer
    )
    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(best_model_dir), local_files_only=True
    ).float().to(device)
    val_loader = make_loader(
        val_df["review_text"].astype(str).tolist(), val_y,
        np.zeros(len(val_df), dtype=bool), best_tokenizer,
        args.batch_size, args.max_length, shuffle=False
    )
    val_probs, val_true, val_loss = predict_probabilities(
        best_model, val_loader, device, use_amp
    )
    threshold, tuned_macro = best_global_threshold(val_probs, val_true)
    val_pred = apply_thresholds(val_probs, threshold)
    val_metrics = metric_summary(val_true, val_pred)

    write_val_predictions(
        results_dir / "validation_predictions.csv", val_df, val_probs, val_true, threshold
    )
    per_label_metrics(val_true, val_pred).to_csv(
        results_dir / "validation_per_label_metrics.csv", index=False, encoding="utf-8-sig"
    )

    report = {
        "task": "Steam Stage 2 multi-label issue-domain classification",
        "mode": args.mode,
        "base_model": str(args.base_model_dir),
        "best_model_dir": str(best_model_dir),
        "gold_train_rows": int(len(train_df)),
        "teacher_pseudo_rows": int(pseudo_count),
        "validation_rows": int(len(val_df)),
        "label_order": LABELS,
        "best_epoch_by_validation_issue_macro_f1_at_0.5": int(best_epoch),
        "best_validation_issue_macro_f1_at_0.5": float(best_macro),
        "selected_global_threshold_from_validation": float(threshold),
        "validation_issue_macro_f1_at_selected_threshold": float(tuned_macro),
        "validation_loss_bce": float(val_loss),
        "validation_metrics": val_metrics,
        "training_settings": {
            "epochs_requested": int(args.epochs),
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "seed": int(args.seed),
            "amp_fp16": bool(use_amp),
            "gold_positive_weight_cap": 5.0,
            "pseudo_loss_weight": float(args.pseudo_weight) if args.mode != "gold_only" else None,
        },
        "pseudo_label_policy": (
            "teacher labels are provisional pseudo-labels, not Gold; eligible rows require "
            "teacher_status=success, needs_human_review=0, confidence>=0.70, and gold_or_test_data=False"
        ),
        "soft_target_policy": (
            "soft mode uses per-label teacher scores; NO_ISSUE is made exclusive with issue targets"
            if args.mode == "soft" else None
        ),
        "test_policy": "test IDs used only to check leakage; test labels/metrics were not evaluated",
        "gold_modified": False,
    }
    (results_dir / "metrics_summary_validation_only.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTRAINING COMPLETE")
    print(f"Mode: {args.mode}")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation-selected global threshold: {threshold:.2f}")
    print(f"Validation issue-domain Macro-F1: {val_metrics['issue_domain_macro_f1']:.4f}")
    print(f"Validation issue-domain Micro-F1: {val_metrics['issue_domain_micro_f1']:.4f}")
    print(f"Best model: {best_model_dir}")
    print(f"Metrics/predictions: {results_dir}")
    print("Test set was not evaluated. Do not use test data for tuning.")


if __name__ == "__main__":
    main()
