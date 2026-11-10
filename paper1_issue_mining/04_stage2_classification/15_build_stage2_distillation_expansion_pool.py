# -*- coding: utf-8 -*-
"""
Build a new, non-overlapping Stage-2 teacher-label expansion pool.

Default target: 1,000 additional reviews, sampled from Stage-1 positive reviews.
Sampling is balanced across games and approximately stratified by review length
and Stage-1 score. Gold/train/validation/test IDs and the old 527-row candidate
pool are excluded. This script does not call an LLM and does not modify Gold.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from config_local import PAPER1_ANALYSIS_ROOT  # noqa: E402

ANALYSIS = PAPER1_ANALYSIS_ROOT
SOURCE = ANALYSIS / "stage1_inference_hardneg_v2" / "stage2_candidate_reviews.csv"
OLD_POOL = ANALYSIS / "stage2_annotation" / "distillation_candidates_v1" / "stage2_distillation_candidate_pool_v1.csv"
GOLD_DIR = ANALYSIS / "stage2_annotation" / "Stage2_Gold_Release_v1"
OUT_DIR = ANALYSIS / "stage2_annotation" / "distillation_candidates_v2_expansion"
OUT_CSV = OUT_DIR / "stage2_distillation_candidate_pool_v2_expansion_1000.csv"
OUT_SUMMARY = OUT_DIR / "stage2_distillation_candidate_pool_v2_expansion_1000_summary.json"

def norm_id(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s

def proportional_quota(counts: pd.Series, n: int) -> dict:
    """Integer quotas proportional to stratum sizes, respecting capacities."""
    counts = counts.astype(int)
    total = int(counts.sum())
    if n <= 0 or total <= 0:
        return {k: 0 for k in counts.index}
    n = min(n, total)
    exact = counts / total * n
    quota = exact.apply(int).to_dict()
    left = n - sum(quota.values())
    order = sorted(counts.index, key=lambda k: (-(exact[k] - quota[k]), str(k)))
    for key in order:
        if left <= 0:
            break
        if quota[key] < counts[key]:
            quota[key] += 1
            left -= 1
    # In case a cap left residual slots, fill any remaining capacity.
    while left > 0:
        progressed = False
        for key in counts.sort_values(ascending=False).index:
            if quota[key] < counts[key]:
                quota[key] += 1
                left -= 1
                progressed = True
                if left == 0:
                    break
        if not progressed:
            break
    return quota

def read_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Gold split file not found: {path}")
    df = pd.read_csv(path, usecols=["review_id"], dtype={"review_id": "string"}, low_memory=False)
    if "review_id" not in df.columns:
        raise ValueError(f"No review_id column in {path}")
    return set(df["review_id"].map(norm_id)) - {""}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-n", type=int, default=1000, help="Number of NEW reviews to sample (default 1000)")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--dry-run", action="store_true", help="Check inputs and report expected selection; do not save output")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing expansion pool (normally avoid)")
    args = ap.parse_args()
    if args.target_n < 1:
        ap.error("--target-n must be positive")
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Stage-1 candidate file not found: {SOURCE}")
    if not OLD_POOL.is_file():
        raise FileNotFoundError(f"Original candidate pool not found: {OLD_POOL}")
    if OUT_CSV.exists() and not args.overwrite:
        raise FileExistsError(f"Expansion pool already exists: {OUT_CSV}. Use --overwrite only if you intend to replace it.")

    source = pd.read_csv(SOURCE, dtype={"review_id": "string"}, low_memory=False)
    required = {"review_id", "game", "review_text", "stage1_pred", "stage1_score"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Stage-1 source missing columns: {sorted(missing)}")
    source["review_id"] = source["review_id"].map(norm_id)
    source["review_text"] = source["review_text"].fillna("").astype(str)
    source["game"] = source["game"].fillna("UNKNOWN_GAME").astype(str)
    source["stage1_pred"] = pd.to_numeric(source["stage1_pred"], errors="coerce").fillna(0).astype(int)
    source["stage1_score"] = pd.to_numeric(source["stage1_score"], errors="coerce")
    source = source.loc[
        source["stage1_pred"].eq(1)
        & source["review_id"].ne("")
        & source["review_text"].str.strip().ne("")
    ].copy()
    if source["review_id"].duplicated().any():
        raise ValueError("Duplicate review_id found in Stage-1 candidates; stop and inspect the source.")

    old = pd.read_csv(OLD_POOL, dtype={"review_id": "string"}, low_memory=False)
    if "review_id" not in old.columns:
        raise ValueError(f"Old pool has no review_id column: {OLD_POOL}")
    old_ids = set(old["review_id"].map(norm_id)) - {""}

    gold_files = [
        GOLD_DIR / "stage2_gold_train_1400.csv",
        GOLD_DIR / "stage2_gold_val_300.csv",
        GOLD_DIR / "stage2_gold_test_300.csv",
    ]
    gold_ids = set()
    for path in gold_files:
        gold_ids |= read_ids(path)

    n_source = len(source)
    n_old_overlap = int(source["review_id"].isin(old_ids).sum())
    n_gold_overlap = int(source["review_id"].isin(gold_ids).sum())
    eligible = source.loc[
        ~source["review_id"].isin(old_ids | gold_ids)
    ].copy()
    eligible = eligible.drop_duplicates("review_id")
    if len(eligible) < args.target_n:
        raise ValueError(f"Only {len(eligible)} eligible reviews remain, fewer than requested {args.target_n}.")

    # Build broad, reproducible strata. Game coverage is prioritized; within each
    # game, sample proportionally across length quartiles and Stage-1 score tertiles.
    eligible["review_length"] = eligible["review_text"].str.len()
    eligible["length_bin"] = eligible.groupby("game")["review_length"].transform(
        lambda x: pd.qcut(x.rank(method="first"), q=4, labels=False, duplicates="drop")
    )
    eligible["score_bin"] = eligible.groupby("game")["stage1_score"].transform(
        lambda x: pd.qcut(x.rank(method="first"), q=3, labels=False, duplicates="drop")
    )
    eligible["length_bin"] = eligible["length_bin"].fillna(0).astype(int)
    eligible["score_bin"] = eligible["score_bin"].fillna(0).astype(int)

    rng = random.Random(args.seed)
    games = sorted(eligible["game"].unique())
    # Equal per-game targets where possible; redistribute any unused slots.
    game_counts = eligible["game"].value_counts().to_dict()
    game_quota = {g: min(args.target_n // len(games), game_counts[g]) for g in games}
    remaining = args.target_n - sum(game_quota.values())
    while remaining > 0:
        available = [g for g in games if game_quota[g] < game_counts[g]]
        if not available:
            break
        available.sort(key=lambda g: (-(game_counts[g] - game_quota[g]), g))
        for g in available:
            if remaining <= 0:
                break
            game_quota[g] += 1
            remaining -= 1

    sampled_parts = []
    for game in games:
        n_game = game_quota[game]
        if n_game <= 0:
            continue
        gdf = eligible.loc[eligible["game"].eq(game)].copy()
        gdf["_stratum"] = gdf["length_bin"].astype(str) + "_" + gdf["score_bin"].astype(str)
        counts = gdf["_stratum"].value_counts()
        quotas = proportional_quota(counts, n_game)
        for stratum, n_take in quotas.items():
            if n_take:
                part = gdf.loc[gdf["_stratum"].eq(stratum)].sample(
                    n=n_take, random_state=rng.randrange(1, 2**31 - 1)
                )
                sampled_parts.append(part)
    selected = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else eligible.head(0).copy()
    if len(selected) != args.target_n:
        raise RuntimeError(f"Sampler produced {len(selected)} rows, expected {args.target_n}.")
    if selected["review_id"].duplicated().any():
        raise RuntimeError("Sampler produced duplicate review_id values.")
    if selected["review_id"].isin(old_ids | gold_ids).any():
        raise RuntimeError("Overlap check failed after sampling.")

    selected = selected.drop(columns=["length_bin", "score_bin", "_stratum"], errors="ignore")
    selected["candidate_group"] = "EXPANSION_STRATIFIED_RANDOM"
    selected["selection_reasons"] = "New sample stratified by game, review length, and Stage-1 score; excludes prior pool and Gold splits"
    selected["selected_for_teacher_review"] = True
    selected["gold_or_test_data"] = False
    # Add a common length field for compatibility with the existing teacher script.
    if "review_length" not in selected.columns:
        selected["review_length"] = selected["review_text"].str.len()

    report = {
        "purpose": "Additional Stage-2 teacher-label pool; not Gold",
        "source": str(SOURCE),
        "old_pool": str(OLD_POOL),
        "gold_split_files": [str(p) for p in gold_files],
        "source_stage1_positive_rows": int(n_source),
        "old_pool_overlap_excluded": int(n_old_overlap),
        "gold_train_val_test_overlap_excluded": int(n_gold_overlap),
        "eligible_after_exclusions": int(len(eligible)),
        "target_new_rows": int(args.target_n),
        "selected_rows": int(len(selected)),
        "selected_by_game": {str(k): int(v) for k, v in selected["game"].value_counts().sort_index().items()},
        "seed": int(args.seed),
        "gold_modified": False,
        "test_labels_not_read": True,
        "note": "Only IDs from split files are used for exclusion; label columns are not used. Teacher labels remain provisional pseudo-labels."
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("\nDRY RUN: no output file was written.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_SUMMARY.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved pool:", OUT_CSV)
    print("Saved summary:", OUT_SUMMARY)

if __name__ == "__main__":
    main()
