# -*- coding: utf-8 -*-
"""Full-corpus Stage-2 inference for the Steam Actionable Issue Mining paper.

Safeguards
----------
1. Uses the frozen Stage-1 full-review predictions and Stage-1-positive candidate file.
2. Keeps all 2,000 Stage-2 Gold reviews out of model inference and retains their human
   labels, including Gold IDs now predicted negative by the newer Stage-1 model.
3. The analysis population is the union of Stage-1-positive candidates and all Gold IDs.
3. Applies validation-fixed thresholds only: RoBERTa T2=0.45; DeBERTa T3=0.65.
4. Does not train, tune thresholds, or read Gold test labels for model selection.
5. Resumable chunk inference; input/model/config hashes prevent stale checkpoint reuse.

Run from Steam_Research_Code:
  python paper1_issue_mining/04_stage2_classification/19_infer_stage2_full_corpus.py --models both --dry-run
  python paper1_issue_mining/04_stage2_classification/19_infer_stage2_full_corpus.py --models both --run

Use --models main for RoBERTa T2 only, or --models robust for DeBERTa T3 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config_local import PAPER1_ROOT as PAPER_ROOT, MODELS_ROOT, MODEL_RUNS_ROOT, DEBERTA_BASE_MODEL_DIR, PAPER1_ANALYSIS_ROOT, PAPER1_RESULTS_ROOT, STAGE2_ROBERTA_T2_DIR, STAGE2_DEBERTA_T3_DIR, DEBERTA_GOLD_ONLY_DIR, DEBERTA_HARD_EXPANDED_1289_DIR, ROBERTA_GOLD_ONLY_DIR, MODERNBERT_GOLD_ONLY_DIR, MODERNBERT_TARGETED_T1_DIR
ANALYSIS_DIR = PAPER_ROOT / "data" / "analysis"
STAGE1_DIR = ANALYSIS_DIR / "stage1_inference_hardneg_v2"
STAGE1_ALL = STAGE1_DIR / "stage1_all_review_predictions.csv"
STAGE1_CANDIDATES = STAGE1_DIR / "stage2_candidate_reviews.csv"
GOLD_DIR = ANALYSIS_DIR / "stage2_annotation" / "Stage2_Gold_Release_v1"
GOLD_ALL = GOLD_DIR / "stage2_gold_all_2000.csv"
GOLD_SPLITS = [
    GOLD_DIR / "stage2_gold_train_1400.csv",
    GOLD_DIR / "stage2_gold_val_300.csv",
    GOLD_DIR / "stage2_gold_test_300.csv",
]
DEFAULT_OUTPUT_ROOT = PAPER_ROOT / "results" / "stage2_models" / "full_corpus_inference_20261031_goldoverride_v1"

ISSUE_LABELS = [
    "BUG_STABILITY", "PERFORMANCE", "COMPATIBILITY", "UI_UX",
    "MULTIPLAYER_NETWORK", "GAMEPLAY_BALANCE", "AI_SYSTEM", "CONTENT",
    "MOD_SUPPORT", "MONETIZATION", "PRIVACY_POLICY", "COMMUNITY_MODERATION",
    "OTHER_ISSUE",
]
LABELS = ISSUE_LABELS + ["NO_ISSUE"]
LABEL_COLS = [f"label_{x}" for x in LABELS]

MODEL_SPECS = {
    "main": {
        "name": "roberta_targeted_T2",
        "role": "Primary model selected using validation results",
        "path": STAGE2_ROBERTA_T2_DIR,
        "threshold": 0.45,
        "use_fast": False,
    },
    "robust": {
        "name": "deberta_targeted_T3",
        "role": "Robustness model selected using validation results",
        "path": STAGE2_DEBERTA_T3_DIR,
        "threshold": 0.65,
        "use_fast": False,
    },
}


def normalize_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_model_weights(model_dir: Path) -> Dict[str, str]:
    files = sorted(list(model_dir.glob("*.safetensors")) + list(model_dir.glob("pytorch_model*.bin")))
    if not files:
        raise FileNotFoundError(f"No model weight files found in {model_dir}")
    return {f.name: sha256_file(f) for f in files}


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_source_files():
    for p in (STAGE1_ALL, STAGE1_CANDIDATES):
        if not p.is_file():
            raise FileNotFoundError(f"Required Stage-1 file not found: {p}")
    all_df = pd.read_csv(STAGE1_ALL, dtype={"review_id": "string"}, low_memory=False, encoding="utf-8-sig")
    cand_df = pd.read_csv(STAGE1_CANDIDATES, dtype={"review_id": "string"}, low_memory=False, encoding="utf-8-sig")
    required_all = {"review_id", "game", "review_text", "stage1_pred", "stage1_score"}
    required_cand = {"review_id", "game", "review_text", "stage1_pred", "stage1_score"}
    if missing := required_all - set(all_df.columns):
        raise ValueError(f"Stage-1 full predictions missing columns: {sorted(missing)}")
    if missing := required_cand - set(cand_df.columns):
        raise ValueError(f"Stage-1 candidate file missing columns: {sorted(missing)}")
    for name, df in (("Stage-1 full", all_df), ("Stage-1 candidates", cand_df)):
        df["review_id"] = df["review_id"].map(normalize_id)
        if df["review_id"].eq("").any() or df["review_id"].duplicated().any():
            raise ValueError(f"{name}: review_id is empty or duplicated")
        if df["game"].isna().any() or df["review_text"].isna().any():
            raise ValueError(f"{name}: missing game or review_text")
    all_df["stage1_pred"] = pd.to_numeric(all_df["stage1_pred"], errors="raise").astype(int)
    cand_df["stage1_pred"] = pd.to_numeric(cand_df["stage1_pred"], errors="raise").astype(int)
    if not all_df["stage1_pred"].isin([0, 1]).all() or not cand_df["stage1_pred"].eq(1).all():
        raise ValueError("Unexpected Stage-1 prediction values or non-positive rows in candidate file")
    expected = set(all_df.loc[all_df.stage1_pred.eq(1), "review_id"])
    observed = set(cand_df["review_id"])
    if expected != observed:
        raise ValueError(
            f"Candidate file does not exactly match Stage-1 positives: "
            f"missing={len(expected-observed)}, extra={len(observed-expected)}"
        )
    return all_df, cand_df


def read_gold() -> pd.DataFrame:
    if GOLD_ALL.is_file():
        gold = pd.read_csv(GOLD_ALL, dtype={"review_id": "string"}, low_memory=False, encoding="utf-8-sig")
    else:
        parts = []
        for p in GOLD_SPLITS:
            if not p.is_file():
                raise FileNotFoundError(f"Gold all-2000 file and split fallback are incomplete: {p}")
            parts.append(pd.read_csv(p, dtype={"review_id": "string"}, low_memory=False, encoding="utf-8-sig"))
        gold = pd.concat(parts, ignore_index=True)
    required = {"review_id", "game", "review_text", *LABEL_COLS}
    if missing := required - set(gold.columns):
        raise ValueError(f"Stage-2 Gold file missing columns: {sorted(missing)}")
    gold["review_id"] = gold["review_id"].map(normalize_id)
    if gold["review_id"].eq("").any() or gold["review_id"].duplicated().any():
        raise ValueError("Gold review_id is empty or duplicated")
    for c in LABEL_COLS:
        v = pd.to_numeric(gold[c], errors="coerce")
        if v.isna().any() or not v.isin([0, 1]).all():
            raise ValueError(f"Gold label column {c} must contain only 0/1")
        gold[c] = v.astype(int)
    issue_sum = gold[[f"label_{x}" for x in ISSUE_LABELS]].sum(axis=1)
    if ((gold["label_NO_ISSUE"].eq(1)) & issue_sum.gt(0)).any():
        raise ValueError("Gold violates NO_ISSUE exclusivity")
    if ((gold["label_NO_ISSUE"].eq(0)) & issue_sum.eq(0)).any():
        raise ValueError("Gold rows must have at least one issue label or NO_ISSUE")
    if len(gold) != 2000:
        raise ValueError(f"Expected 2,000 Stage-2 Gold rows, found {len(gold)}")
    return gold


def validate_model(spec: Dict) -> Dict:
    model_dir = spec["path"]
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model checkpoint not found: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Transformers often serializes id2label/label2id but omits a literal
    # "num_labels" key. AutoConfig reconstructs config.num_labels from the
    # mapping length, whereas reading raw config.json does not. Infer the
    # dimension from the serialized mappings when the explicit key is absent.
    id2label = config.get("id2label") or {}
    label2id = config.get("label2id") or {}
    raw_num_labels = config.get("num_labels")
    if raw_num_labels is not None:
        configured_num_labels = int(raw_num_labels)
    elif id2label:
        configured_num_labels = len(id2label)
    elif label2id:
        configured_num_labels = len(label2id)
    else:
        raise ValueError(
            f"{spec['name']}: config.json contains neither num_labels nor "
            "id2label/label2id; cannot verify classifier output dimension safely."
        )

    if configured_num_labels != len(LABELS):
        raise ValueError(
            f"{spec['name']}: checkpoint has {configured_num_labels} configured outputs; "
            f"expected {len(LABELS)}."
        )

    if id2label:
        try:
            actual = [id2label.get(str(i), id2label.get(i)) for i in range(len(LABELS))]
        except Exception as exc:
            raise ValueError(f"{spec['name']}: cannot parse id2label mapping: {exc}") from exc
        if actual != LABELS:
            raise ValueError(f"{spec['name']}: checkpoint label order mismatch: {actual}")

    if label2id:
        expected_label2id = {label: i for i, label in enumerate(LABELS)}
        normalized_label2id = {str(k): int(v) for k, v in label2id.items()}
        if normalized_label2id != expected_label2id:
            raise ValueError(
                f"{spec['name']}: checkpoint label2id mapping mismatch: {normalized_label2id}"
            )

    weights = sha256_model_weights(model_dir)
    return {"config": config, "weight_hashes": weights,
            "configured_num_labels": configured_num_labels}


def make_manifest(spec, non_gold, batch_size, chunk_size, max_length):
    return {
        "task": "Stage-2 inference on non-Gold Stage-1 positives; retain all Gold IDs as manual labels, including Stage-1-negative Gold overrides",
        "model_name": spec["name"],
        "model_dir": str(spec["path"]),
        "model_weight_sha256": sha256_model_weights(spec["path"]),
        "stage1_full_sha256": sha256_file(STAGE1_ALL),
        "stage1_candidates_sha256": sha256_file(STAGE1_CANDIDATES),
        "gold_all_sha256": sha256_file(GOLD_ALL) if GOLD_ALL.is_file() else "gold_splits_used",
        "threshold_fixed_from_validation": float(spec["threshold"]),
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "max_length": max_length,
        "non_gold_candidate_rows": int(len(non_gold)),
        "gold_rows_excluded_from_model_inference": 2000,
        "label_order": LABELS,
    }


def apply_thresholds(probs: np.ndarray, threshold: float) -> np.ndarray:
    issue_pred = probs[:, :len(ISSUE_LABELS)] >= threshold
    no_issue_pred = (probs[:, len(ISSUE_LABELS)] >= threshold) & (~issue_pred.any(axis=1))
    return np.concatenate([issue_pred.astype(np.int8), no_issue_pred[:, None].astype(np.int8)], axis=1)


def infer_model(spec: Dict, non_gold: pd.DataFrame, out_root: Path,
                batch_size: int, chunk_size: int, max_length: int, cpu: bool) -> pd.DataFrame:
    model_dir = spec["path"]
    run_dir = out_root / spec["name"]
    chunks_dir = run_dir / "checkpoints"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    final_path = run_dir / "model_predictions_non_gold.csv"
    signature = make_manifest(spec, non_gold, batch_size, chunk_size, max_length)

    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != signature:
            raise RuntimeError(
                f"Existing run manifest differs for {spec['name']}. Use a new --output-root; "
                "do not reuse stale chunks."
            )
    else:
        if any(chunks_dir.glob("chunk_*.csv")):
            raise RuntimeError(f"Chunk files exist without manifest: {chunks_dir}")
        atomic_json(signature, manifest_path)

    total_chunks = (len(non_gold) + chunk_size - 1) // chunk_size
    pending = [i for i in range(total_chunks) if not (chunks_dir / f"chunk_{i:05d}.csv").is_file()]
    print(f"\n[{spec['name']}] rows={len(non_gold)} chunks={total_chunks} completed={total_chunks-len(pending)} pending={len(pending)}")

    if pending:
        device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
        use_amp = device.type == "cuda"
        print(f"Loading {model_dir} | device={device} | AMP fp16={use_amp}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=spec["use_fast"])
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True).float().to(device)
        model.eval()
        if int(model.config.num_labels) != len(LABELS):
            raise ValueError(f"{spec['name']}: model output dimension is not 14")

        for chunk_idx in pending:
            start = chunk_idx * chunk_size
            part = non_gold.iloc[start:start + chunk_size].copy().reset_index(drop=True)
            prob_batches = []
            texts = part["review_text"].astype(str).tolist()
            with torch.inference_mode():
                for bstart in range(0, len(texts), batch_size):
                    batch_texts = texts[bstart:bstart + batch_size]
                    enc = tokenizer(
                        batch_texts, truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt"
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        logits = model(**enc).logits
                    logits = logits.float().cpu().numpy()
                    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
                    prob_batches.append(probs)
            probs = np.concatenate(prob_batches, axis=0)
            if probs.shape != (len(part), len(LABELS)):
                raise RuntimeError(f"Unexpected output shape in chunk {chunk_idx}: {probs.shape}")
            pred = apply_thresholds(probs, float(spec["threshold"]))
            for j, label in enumerate(LABELS):
                part[f"prob_{label}"] = probs[:, j]
                part[f"pred_{label}"] = pred[:, j]
            part["stage2_predicted_domains"] = [
                "|".join(LABELS[j] for j, v in enumerate(row) if v == 1) or "UNASSIGNED"
                for row in pred
            ]
            issue_any = pred[:, :len(ISSUE_LABELS)].any(axis=1)
            no_issue = pred[:, len(ISSUE_LABELS)].astype(bool)
            part["stage2_status"] = np.where(issue_any, "issue", np.where(no_issue, "no_issue", "unassigned"))
            part["stage2_source"] = f"model_{spec['name']}"
            part["stage2_threshold"] = float(spec["threshold"])
            atomic_csv(part, chunks_dir / f"chunk_{chunk_idx:05d}.csv")
            print(f"  chunk {chunk_idx+1}/{total_chunks}: {len(part)} reviews", flush=True)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pieces = []
    for i in range(total_chunks):
        f = chunks_dir / f"chunk_{i:05d}.csv"
        if not f.is_file():
            raise RuntimeError(f"Missing inference chunk: {f}")
        pieces.append(pd.read_csv(f, dtype={"review_id": "string"}, low_memory=False, encoding="utf-8-sig"))
    result = pd.concat(pieces, ignore_index=True)
    if len(result) != len(non_gold) or result["review_id"].map(normalize_id).tolist() != non_gold["review_id"].map(normalize_id).tolist():
        raise RuntimeError("Chunk reconstruction failed row/order verification")
    atomic_csv(result, final_path)
    print(f"Saved model-only predictions: {final_path}")
    return result


def make_gold_predictions(gold: pd.DataFrame) -> pd.DataFrame:
    out = gold.copy()
    for label in LABELS:
        out[f"pred_{label}"] = out[f"label_{label}"].astype(int)
        out[f"prob_{label}"] = np.nan
    issue_any = out[[f"pred_{x}" for x in ISSUE_LABELS]].sum(axis=1).gt(0)
    no_issue = out["pred_NO_ISSUE"].eq(1)
    out["stage2_predicted_domains"] = [
        "|".join(label for label in LABELS if int(row[f"pred_{label}"]) == 1)
        for _, row in out.iterrows()
    ]
    out["stage2_status"] = np.where(issue_any, "issue", np.where(no_issue, "no_issue", "unassigned"))
    out["stage2_source"] = "gold_manual"
    out["stage2_threshold"] = np.nan
    return out


def merge_full_corpus(all_df: pd.DataFrame, candidates: pd.DataFrame,
                      gold: pd.DataFrame, model_preds: pd.DataFrame,
                      spec: Dict, out_root: Path) -> Dict:
    """Merge Gold manual labels with model predictions over the union population.

    Population = all current Stage-1 positives UNION all 2,000 Gold IDs.
    Gold rows are retained even when the current Stage-1 model predicts 0.
    """
    gold_pred = make_gold_predictions(gold)
    keep_cols = ["review_id", "stage2_predicted_domains", "stage2_status", "stage2_source", "stage2_threshold"]
    keep_cols += [f"pred_{x}" for x in LABELS] + [f"prob_{x}" for x in LABELS]
    gold_part = gold_pred[keep_cols].copy()
    model_part = model_preds[keep_cols].copy()
    labels = pd.concat([gold_part, model_part], ignore_index=True)
    labels["review_id"] = labels["review_id"].map(normalize_id)
    if labels["review_id"].duplicated().any():
        raise ValueError("Gold/model prediction IDs overlap")

    candidate_ids = set(candidates["review_id"].map(normalize_id))
    gold_ids = set(gold["review_id"].map(normalize_id))
    population_ids = candidate_ids | gold_ids
    if set(labels["review_id"]) != population_ids:
        missing = population_ids - set(labels["review_id"])
        extra = set(labels["review_id"]) - population_ids
        raise ValueError(
            "Gold plus model predictions do not cover the union population: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    # Build the Stage-2 analysis population from the authoritative full Stage-1 table.
    population = all_df.copy()
    population["review_id"] = population["review_id"].map(normalize_id)
    population = population.loc[population["review_id"].isin(population_ids)].copy()
    if len(population) != len(population_ids):
        raise ValueError(
            f"Population row count mismatch: rows={len(population)}, unique IDs={len(population_ids)}"
        )
    candidate_out = population.merge(labels, on="review_id", how="left", validate="one_to_one", sort=False)
    if candidate_out["stage2_status"].isna().any():
        raise ValueError("Missing Stage-2 labels after union-population merge")
    candidate_path = out_root / spec["name"] / "stage2_analysis_population_with_gold_and_model.csv"
    atomic_csv(candidate_out, candidate_path)

    full = all_df.copy()
    full["review_id"] = full["review_id"].map(normalize_id)
    full = full.merge(labels, on="review_id", how="left", validate="one_to_one", sort=False)
    in_population = full["review_id"].isin(population_ids)
    if full.loc[in_population, "stage2_status"].isna().any():
        raise ValueError("At least one union-population review has no Stage-2 result")
    if full.loc[~in_population, "stage2_status"].notna().any():
        raise ValueError("A review outside the union population unexpectedly received a Stage-2 label")

    # Distinguish Stage-1 negatives with Gold overrides from unclassified negatives.
    full.loc[~in_population, "stage2_status"] = "not_stage2_population"
    full.loc[~in_population, "stage2_source"] = "not_applicable"
    full_path = out_root / spec["name"] / f"full_corpus_labeled_{spec['name']}.csv"
    atomic_csv(full, full_path)

    gold_stage1_negative = len(gold_ids - candidate_ids)
    stage1_negative_total = int(full["stage1_pred"].eq(0).sum())
    summary = {
        "model_name": spec["name"],
        "model_role": spec["role"],
        "threshold_fixed_from_validation": float(spec["threshold"]),
        "all_reviews": int(len(full)),
        "stage1_positive_candidates": int(len(candidates)),
        "gold_manual_rows": int(len(gold)),
        "gold_rows_also_in_current_stage1_positive_candidates": int(len(gold_ids & candidate_ids)),
        "gold_rows_overriding_current_stage1_negative": int(gold_stage1_negative),
        "stage2_analysis_population_rows": int(len(population)),
        "model_inference_rows": int(len(model_preds)),
        "stage2_issue_rows": int(full["stage2_status"].eq("issue").sum()),
        "stage2_no_issue_rows": int(full["stage2_status"].eq("no_issue").sum()),
        "stage2_unassigned_rows": int(full["stage2_status"].eq("unassigned").sum()),
        "stage1_negative_rows_not_stage2_classified": int(stage1_negative_total - gold_stage1_negative),
        "candidate_output": str(candidate_path),
        "full_corpus_output": str(full_path),
        "gold_labels_used_as_manual_labels_not_model_predictions": True,
        "population_definition": "current Stage-1 positives UNION all 2,000 Stage-2 Gold IDs",
    }
    atomic_json(summary, out_root / spec["name"] / "full_corpus_summary.json")
    return summary

def main():
    p = argparse.ArgumentParser(description="Run frozen Stage-2 model(s) on the unlabeled Stage-1-positive corpus.")
    p.add_argument("--models", choices=["main", "robust", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.dry_run == args.run:
        p.error("Specify exactly one of --dry-run or --run")
    if min(args.batch_size, args.chunk_size) <= 0 or args.max_length < 64:
        p.error("batch-size/chunk-size must be positive and max-length >= 64")

    all_df, candidates = read_source_files()
    gold = read_gold()
    candidate_ids = set(candidates["review_id"].map(normalize_id))
    gold_ids = set(gold["review_id"].map(normalize_id))
    all_ids = set(all_df["review_id"].map(normalize_id))

    # Gold IDs must exist in the full corpus, but need not remain Stage-1-positive
    # after the newer Stage-1 model is applied.
    missing_gold_from_full = gold_ids - all_ids
    if missing_gold_from_full:
        raise ValueError(f"Gold IDs missing from Stage-1 full corpus: {len(missing_gold_from_full)}")

    # Verify exact record identity against the frozen full-corpus table.
    gold_check = gold[["review_id", "game", "review_text"]].merge(
        all_df[["review_id", "game", "review_text"]],
        on="review_id", how="left", suffixes=("_gold", "_stage1"), validate="one_to_one"
    )
    game_match = gold_check["game_gold"].astype(str).eq(gold_check["game_stage1"].astype(str))
    text_match = gold_check["review_text_gold"].fillna("").astype(str).eq(
        gold_check["review_text_stage1"].fillna("").astype(str)
    )
    if not game_match.all() or not text_match.all():
        raise ValueError(
            "Gold/full-corpus record mismatch: "
            f"game_mismatch={(~game_match).sum()}, text_mismatch={(~text_match).sum()}"
        )

    gold_stage1_negative = gold_ids - candidate_ids
    non_gold = candidates.loc[
        ~candidates["review_id"].map(normalize_id).isin(gold_ids)
    ].copy().reset_index(drop=True)
    expected_non_gold = len(candidate_ids - gold_ids)
    if len(non_gold) != expected_non_gold:
        raise RuntimeError("Non-Gold Stage-1-positive partition arithmetic failed")
    population_ids = candidate_ids | gold_ids
    print(f"Gold rows now outside Stage-1-positive candidates (manual override): {len(gold_stage1_negative):,}")

    selected = [MODEL_SPECS["main"], MODEL_SPECS["robust"]] if args.models == "both" else [MODEL_SPECS[args.models]]
    model_checks = {}
    for spec in selected:
        model_checks[spec["name"]] = validate_model(spec)

    print("PRE-FLIGHT CHECK PASSED")
    print(f"Stage-1 full review rows: {len(all_df):,}")
    print(f"Stage-1 positive candidates: {len(candidates):,}")
    print(f"Stage-2 Gold rows retained as manual labels: {len(gold):,}")
    print(f"Gold rows overriding current Stage-1 negatives: {len(gold_stage1_negative):,}")
    print(f"Unlabeled Stage-1-positive candidates for model inference: {len(non_gold):,}")
    print(f"Union analysis population (Stage-1 positives + all Gold): {len(population_ids):,}")
    print(f"Games: {all_df['game'].nunique()}")
    print("No Gold IDs will be sent through the inference model.")
    for spec in selected:
        print(f"  {spec['name']}: threshold={spec['threshold']:.2f}; checkpoint={spec['path']}")
    print(f"Output root: {args.output_root}")
    if args.dry_run:
        print("DRY RUN ONLY: no model inference or output files were generated.")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in selected:
        model_preds = infer_model(
            spec, non_gold, args.output_root,
            args.batch_size, args.chunk_size, args.max_length, args.cpu
        )
        summaries.append(merge_full_corpus(all_df, candidates, gold, model_preds, spec, args.output_root))
    atomic_json({
        "task": "Steam Stage-2 full corpus inference",
        "created_on": "2026-10-31",
        "stage1_all_rows": int(len(all_df)),
        "stage1_positive_candidates": int(len(candidates)),
        "gold_manual_rows": int(len(gold)),
        "gold_rows_overriding_current_stage1_negative": int(len(gold_stage1_negative)),
        "union_analysis_population_rows": int(len(population_ids)),
        "model_inference_rows_per_model": int(len(non_gold)),
        "models": summaries,
        "interpretation_note": (
            "Stage-1-negative reviews were not passed to Stage 2. The 2,000 manually labelled "
            "Stage-2 Gold reviews are retained as human labels even if current Stage-1 predicts negative; "
            "models are applied only to non-Gold Stage-1-positive candidates. Other Stage-1 negatives "
            "remain outside Stage-2. Category counts are multi-label and need not sum to population totals."
        ),
    }, args.output_root / "full_corpus_inference_summary.json")
    print("\nFULL-CORPUS STAGE-2 INFERENCE COMPLETE")
    for s in summaries:
        print(f"{s['model_name']}: {s['full_corpus_output']}")
    print(f"Summary: {args.output_root / 'full_corpus_inference_summary.json'}")


if __name__ == "__main__":
    main()