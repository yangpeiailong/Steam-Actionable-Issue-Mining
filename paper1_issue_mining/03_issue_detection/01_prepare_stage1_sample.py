"""
Prepare the Stage-1 human-review sample for review-level actionable-issue detection.

Task:
    Full Steam review -> contains at least one actionable issue? (0/1)

Input:
    data/analysis/all_reviews.csv

Output:
    data/analysis/stage1_review_sample_2000.csv
    results/statistics/stage1_sample_report.csv

Sampling design:
- 1,500 core reviews: simple random sample from all reviews.
- 500 boundary-enriched reviews from the remaining pool:
    * long recommended reviews (likely mixed/qualified positives)
    * short non-recommended reviews (likely vague/general negatives)
    * random remainder fill
- No issue keyword filter is used.
- The sample is for annotation/training, NOT prevalence estimation.

The script is deterministic with seed=2026.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config_local import PAPER_ROOT


INPUT_FILE = (
    PAPER_ROOT
    / "data"
    / "analysis"
    / "all_reviews.csv"
)

OUTPUT_FILE = (
    PAPER_ROOT
    / "data"
    / "analysis"
    / "stage1_review_sample_2000.csv"
)

REPORT_FILE = (
    PAPER_ROOT
    / "results"
    / "statistics"
    / "stage1_sample_report.csv"
)


# =========================================================
# Configuration
# =========================================================

RANDOM_SEED = 2026

TOTAL_SAMPLE = 2000
CORE_RANDOM_N = 1500
BOUNDARY_N = TOTAL_SAMPLE - CORE_RANDOM_N

LONG_POSITIVE_TARGET = 250
SHORT_NEGATIVE_TARGET = 150
RANDOM_FILL_TARGET = 100


# =========================================================
# Helpers
# =========================================================

def normalize_bool(value):
    """
    Convert common CSV boolean representations to True/False/NaN.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    return np.nan


def sample_without_overlap(
    pool,
    n,
    seed,
):
    if n <= 0 or len(pool) == 0:
        return pool.iloc[0:0].copy()

    n = min(n, len(pool))

    return pool.sample(
        n=n,
        random_state=seed
    ).copy()


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("Preparing Stage-1 review sample")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    required = {
        "game",
        "review_id",
        "review_text",
        "voted_up",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"all_reviews.csv missing columns: {sorted(missing)}"
        )

    # -----------------------------------------------------
    # Basic cleaning
    # -----------------------------------------------------

    df = df.copy()

    df["review_text"] = (
        df["review_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["review_text"].ne("")
    ].copy()

    df["voted_up_bool"] = (
        df["voted_up"]
        .apply(normalize_bool)
    )

    # Keep rows with known recommendation status.
    df = df[
        df["voted_up_bool"].notna()
    ].copy()

    df["review_length_chars"] = (
        df["review_text"]
        .str.len()
    )

    # Stable identity check.
    dup_n = int(
        df["review_id"]
        .astype(str)
        .duplicated()
        .sum()
    )

    if dup_n > 0:
        print(
            f"WARNING: duplicate review_id rows found: {dup_n}"
        )

        df = (
            df
            .drop_duplicates(
                subset=["review_id"],
                keep="first"
            )
            .copy()
        )

    if len(df) < TOTAL_SAMPLE:
        raise ValueError(
            f"Only {len(df)} usable reviews are available; "
            f"need at least {TOTAL_SAMPLE}."
        )

    print(f"Usable reviews: {len(df)}")

    # -----------------------------------------------------
    # 1. Core random sample
    # -----------------------------------------------------

    core = df.sample(
        n=CORE_RANDOM_N,
        random_state=RANDOM_SEED
    ).copy()

    core["sample_source"] = "core_random"

    used_ids = set(
        core["review_id"]
        .astype(str)
    )

    remaining = df[
        ~df["review_id"]
        .astype(str)
        .isin(used_ids)
    ].copy()

    # -----------------------------------------------------
    # 2. Boundary-enriched sample
    #
    # No issue keywords are used.
    # We use review length + recommendation polarity only:
    # - long recommended reviews often contain mixed praise/criticism;
    # - short negative reviews often contain vague/general dislike.
    # -----------------------------------------------------

    pos_pool = remaining[
        remaining["voted_up_bool"] == True
    ].copy()

    neg_pool = remaining[
        remaining["voted_up_bool"] == False
    ].copy()

    # Within-polarity quantiles, so "long" and "short" are
    # defined relative to each polarity group.
    pos_long_cut = (
        pos_pool["review_length_chars"]
        .quantile(0.60)
        if len(pos_pool) > 0
        else np.nan
    )

    neg_short_cut = (
        neg_pool["review_length_chars"]
        .quantile(0.40)
        if len(neg_pool) > 0
        else np.nan
    )

    long_positive_pool = pos_pool[
        pos_pool["review_length_chars"]
        >= pos_long_cut
    ].copy()

    short_negative_pool = neg_pool[
        neg_pool["review_length_chars"]
        <= neg_short_cut
    ].copy()

    long_positive = sample_without_overlap(
        long_positive_pool,
        LONG_POSITIVE_TARGET,
        RANDOM_SEED + 1
    )

    long_positive["sample_source"] = (
        "boundary_long_recommended"
    )

    used_ids |= set(
        long_positive["review_id"]
        .astype(str)
    )

    short_negative_pool = short_negative_pool[
        ~short_negative_pool["review_id"]
        .astype(str)
        .isin(used_ids)
    ].copy()

    short_negative = sample_without_overlap(
        short_negative_pool,
        SHORT_NEGATIVE_TARGET,
        RANDOM_SEED + 2
    )

    short_negative["sample_source"] = (
        "boundary_short_nonrecommended"
    )

    used_ids |= set(
        short_negative["review_id"]
        .astype(str)
    )

    # Random fill from all remaining rows.
    fill_needed = (
        BOUNDARY_N
        - len(long_positive)
        - len(short_negative)
    )

    fill_pool = df[
        ~df["review_id"]
        .astype(str)
        .isin(used_ids)
    ].copy()

    random_fill = sample_without_overlap(
        fill_pool,
        fill_needed,
        RANDOM_SEED + 3
    )

    random_fill["sample_source"] = (
        "boundary_random_fill"
    )

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    sample = pd.concat(
        [
            core,
            long_positive,
            short_negative,
            random_fill,
        ],
        ignore_index=True
    )

    if len(sample) != TOTAL_SAMPLE:
        raise RuntimeError(
            f"Expected {TOTAL_SAMPLE} rows, got {len(sample)}."
        )

    if (
        sample["review_id"]
        .astype(str)
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate review_id found in final sample."
        )

    # Randomize display order after sampling.
    sample = (
        sample
        .sample(
            frac=1.0,
            random_state=RANDOM_SEED + 10
        )
        .reset_index(drop=True)
    )

    sample.insert(
        0,
        "stage1_id",
        range(1, len(sample) + 1)
    )

    # -----------------------------------------------------
    # Annotation columns
    # -----------------------------------------------------

    sample["llm_is_actionable_issue"] = pd.NA
    sample["llm_evidence"] = ""
    sample["llm_confidence"] = pd.NA
    sample["llm_annotation_status"] = ""

    sample["manual_is_actionable_issue"] = pd.NA
    sample["manual_notes"] = ""
    sample["manual_reviewed"] = 0

    # -----------------------------------------------------
    # Keep useful columns only
    # -----------------------------------------------------

    preferred_columns = [
        "stage1_id",
        "game",
        "review_id",
        "date",
        "voted_up",
        "review_length_chars",
        "sample_source",
        "review_text",
        "llm_is_actionable_issue",
        "llm_evidence",
        "llm_confidence",
        "llm_annotation_status",
        "manual_is_actionable_issue",
        "manual_notes",
        "manual_reviewed",
    ]

    keep_columns = [
        c
        for c in preferred_columns
        if c in sample.columns
    ]

    sample = sample[
        keep_columns
    ].copy()

    # -----------------------------------------------------
    # Save sample
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sample.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    report_rows = []

    # Overall
    report_rows.append({
        "section": "overall",
        "group": "total",
        "n": len(sample),
        "share": 1.0,
        "mean_review_length_chars": (
            sample["review_length_chars"].mean()
        ),
    })

    # By source
    for source, group in sample.groupby(
        "sample_source"
    ):
        report_rows.append({
            "section": "sample_source",
            "group": source,
            "n": len(group),
            "share": len(group) / len(sample),
            "mean_review_length_chars": (
                group["review_length_chars"].mean()
            ),
        })

    # By recommendation
    for voted, group in sample.groupby(
        "voted_up",
        dropna=False
    ):
        report_rows.append({
            "section": "voted_up",
            "group": str(voted),
            "n": len(group),
            "share": len(group) / len(sample),
            "mean_review_length_chars": (
                group["review_length_chars"].mean()
            ),
        })

    # By game
    for game, group in sample.groupby(
        "game"
    ):
        report_rows.append({
            "section": "game",
            "group": str(game),
            "n": len(group),
            "share": len(group) / len(sample),
            "mean_review_length_chars": (
                group["review_length_chars"].mean()
            ),
        })

    report_df = pd.DataFrame(
        report_rows
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report_df.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("Stage-1 sample prepared")
    print("=" * 70)

    print(f"Final sample: {len(sample)}")

    print()
    print("Sample source:")
    print(
        sample["sample_source"]
        .value_counts()
        .to_string()
    )

    print()
    print("Recommendation distribution:")
    print(
        sample["voted_up"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("Game distribution:")
    print(
        sample["game"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Review length (chars): "
        f"mean={sample['review_length_chars'].mean():.1f}, "
        f"median={sample['review_length_chars'].median():.1f}"
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)
    print(REPORT_FILE)


if __name__ == "__main__":
    main()
