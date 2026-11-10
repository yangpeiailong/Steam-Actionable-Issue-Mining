# -*- coding: utf-8 -*-
"""
Steam Paper 1: targeted qualitative case extraction for game-specific issue patterns.

Purpose
-------
Extract real Steam review examples for a small, theory-oriented qualitative
inspection. This is NOT a new annotation round and does NOT create labels.

Primary model: RoBERTa T2
Robustness model: DeBERTa T3

The script:
1. Reads the two 8,034-row full-corpus analysis files.
2. Verifies review IDs, game assignment, Stage-1 inclusion and Gold labels.
3. Restricts case extraction to the 7,634 Stage-1-positive primary population.
4. Samples per prespecified game/category stratum:
   - manually labeled Gold examples;
   - examples where both models agree;
   - model-disagreement examples for diagnostic inspection.
5. Includes two NO_ISSUE comparison strata (Factorio and Cities: Skylines).
6. Exports review text, metadata, both models' predictions/probabilities,
   selection reasons, and a sampling audit.

No API calls, no model inference, no synthetic text, and no modification of
original data are performed.

Run from Steam_Research_Code:
  python -u .\\paper1_issue_mining\\04_stage2_classification\\23_extract_stage2_case_examples_v1.py --dry-run
  python -u .\\paper1_issue_mining\\04_stage2_classification\\23_extract_stage2_case_examples_v1.py --run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config_local import PAPER1_ROOT as PAPER_ROOT, MODELS_ROOT, MODEL_RUNS_ROOT, DEBERTA_BASE_MODEL_DIR, PAPER1_ANALYSIS_ROOT, PAPER1_RESULTS_ROOT, STAGE2_ROBERTA_T2_DIR, STAGE2_DEBERTA_T3_DIR, DEBERTA_GOLD_ONLY_DIR, DEBERTA_HARD_EXPANDED_1289_DIR, ROBERTA_GOLD_ONLY_DIR, MODERNBERT_GOLD_ONLY_DIR, MODERNBERT_TARGETED_T1_DIR
INFERENCE_ROOT = (
    PAPER_ROOT / "results" / "stage2_models"
    / "full_corpus_inference_20261031_goldoverride_v1"
)
DEFAULT_PRIMARY = (
    INFERENCE_ROOT / "roberta_targeted_T2"
    / "stage2_analysis_population_with_gold_and_model.csv"
)
DEFAULT_ROBUST = (
    INFERENCE_ROOT / "deberta_targeted_T3"
    / "stage2_analysis_population_with_gold_and_model.csv"
)
DEFAULT_OUTPUT = (
    PAPER_ROOT / "results" / "paper_analysis"
    / "stage2_case_review_20261031_v1"
)

ISSUE_LABELS = [
    "BUG_STABILITY", "PERFORMANCE", "COMPATIBILITY", "UI_UX",
    "MULTIPLAYER_NETWORK", "GAMEPLAY_BALANCE", "AI_SYSTEM", "CONTENT",
    "MOD_SUPPORT", "MONETIZATION", "PRIVACY_POLICY",
    "COMMUNITY_MODERATION", "OTHER_ISSUE", "NO_ISSUE",
]

GAME_NAMES = {
    "105600_Terraria": "Terraria",
    "1086940_BaldursGate3": "Baldur's Gate 3",
    "1091500_Cyberpunk2077": "Cyberpunk 2077",
    "1145360_Hades": "Hades",
    "255710_CitiesSkylines": "Cities: Skylines",
    "275850_NoMansSky": "No Man's Sky",
    "289070_CivilizationVI": "Civilization VI",
    "367520_HollowKnight": "Hollow Knight",
    "413150_StardewValley": "Stardew Valley",
    "427520_Factorio": "Factorio",
}

# Prespecified from the completed distribution and inferential analyses.
# These are case-study strata, not an exhaustive sample of all categories.
ISSUE_STRATA = [
    ("255710_CitiesSkylines", "COMMUNITY_MODERATION"),
    ("255710_CitiesSkylines", "MONETIZATION"),
    ("1091500_Cyberpunk2077", "BUG_STABILITY"),
    ("1145360_Hades", "GAMEPLAY_BALANCE"),
    ("1145360_Hades", "CONTENT"),
    ("427520_Factorio", "GAMEPLAY_BALANCE"),
    ("427520_Factorio", "MONETIZATION"),
    ("427520_Factorio", "UI_UX"),
]
CONTROL_STRATA = [
    ("427520_Factorio", "NO_ISSUE"),
    ("255710_CitiesSkylines", "NO_ISSUE"),
]

BASE_META = [
    "game", "review_id", "review_text", "voted_up", "negative_flag",
    "votes_up", "playtime_forever", "timestamp", "date", "week", "month",
    "review_length", "stage1_score", "stage1_pred",
]


def read_population(path: Path, model_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{model_name} input not found: {path}")
    df = pd.read_csv(
        path, dtype={"review_id": "string"}, encoding="utf-8-sig",
        low_memory=False,
    )
    required = {
        "review_id", "game", "review_text", "stage1_pred", "stage2_source",
        "stage2_status", "stage2_threshold",
        *[f"pred_{x}" for x in ISSUE_LABELS],
        *[f"prob_{x}" for x in ISSUE_LABELS],
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{model_name} input missing columns: {missing}")
    if df["review_id"].isna().any() or df["review_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{model_name}: empty review_id")
    if df["review_id"].duplicated().any():
        raise ValueError(f"{model_name}: duplicate review_id")
    df["review_id"] = df["review_id"].astype(str)
    df["stage1_pred"] = pd.to_numeric(df["stage1_pred"], errors="raise").astype(int)
    if not set(df["stage1_pred"].unique()).issubset({0, 1}):
        raise ValueError(f"{model_name}: stage1_pred must be binary")
    for label in ISSUE_LABELS:
        pc, qc = f"pred_{label}", f"prob_{label}"
        df[pc] = pd.to_numeric(df[pc], errors="raise").astype(int)
        if not set(df[pc].unique()).issubset({0, 1}):
            raise ValueError(f"{model_name}: {pc} must be binary")
        df[qc] = pd.to_numeric(df[qc], errors="coerce")
    df["stage2_threshold"] = pd.to_numeric(df["stage2_threshold"], errors="coerce")
    return df


def validate_and_merge(primary: pd.DataFrame, robust: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    if len(primary) != 8034 or len(robust) != 8034:
        raise ValueError(
            f"Expected 8,034 rows in each population file; "
            f"got {len(primary)} and {len(robust)}"
        )
    for col in ["review_id", "game", "stage1_pred", "review_text"]:
        if not primary[col].astype(str).reset_index(drop=True).equals(
            robust[col].astype(str).reset_index(drop=True)
        ):
            raise ValueError(f"RoBERTa and DeBERTa differ in {col}")

    psrc = primary["stage2_source"].astype(str)
    rsrc = robust["stage2_source"].astype(str)
    p_gold = psrc.eq("gold_manual")
    r_gold = rsrc.eq("gold_manual")
    p_model = psrc.str.startswith("model_")
    r_model = rsrc.str.startswith("model_")
    if not p_gold.equals(r_gold) or not p_model.equals(r_model):
        raise ValueError("Gold/model source membership differs between model files")
    if not (p_gold | p_model).all() or not (r_gold | r_model).all():
        raise ValueError("Unexpected stage2_source values found")

    # The 400 Stage-1-negative records are retained in the source files for audit,
    # but excluded from the primary case-study population.
    pos = primary["stage1_pred"].eq(1)
    if int(pos.sum()) != 7634 or int((~pos).sum()) != 400:
        raise ValueError("Unexpected Stage-1 positive/negative population counts")
    if int((pos & p_gold).sum()) != 1600 or int((pos & p_model).sum()) != 6034:
        raise ValueError("Unexpected Gold/model counts within the primary population")
    if not primary.loc[~pos, "stage2_source"].eq("gold_manual").all():
        raise ValueError("Stage-1-negative audit rows must all be Gold")

    # Gold labels are shared ground-truth labels in both files and must match.
    for label in ISSUE_LABELS:
        c = f"pred_{label}"
        if not primary.loc[p_gold, c].reset_index(drop=True).equals(
            robust.loc[r_gold, c].reset_index(drop=True)
        ):
            raise ValueError(f"Gold labels differ between files: {label}")

    # Build a paired analysis frame while keeping both models' fields explicit.
    keep_robust = ["review_id", "stage2_source", "stage2_status", "stage2_threshold"]
    for label in ISSUE_LABELS:
        keep_robust += [f"pred_{label}", f"prob_{label}"]
    r = robust[keep_robust].copy()
    rename = {
        "stage2_source": "stage2_source_deberta",
        "stage2_status": "stage2_status_deberta",
        "stage2_threshold": "stage2_threshold_deberta",
    }
    for label in ISSUE_LABELS:
        rename[f"pred_{label}"] = f"deberta_pred_{label}"
        rename[f"prob_{label}"] = f"deberta_prob_{label}"
    r = r.rename(columns=rename)

    merged = primary.merge(r, on="review_id", how="left", validate="one_to_one")
    if merged.filter(regex=r"^deberta_pred_").isna().any().any():
        raise ValueError("Missing DeBERTa predictions after ID merge")

    merged = merged.loc[merged["stage1_pred"].eq(1)].copy()
    merged["game_name"] = merged["game"].map(GAME_NAMES).fillna(merged["game"])
    merged["source_role"] = np.where(
        merged["stage2_source"].eq("gold_manual"), "Gold manual",
        "Model inference",
    )
    checks = {
        "rows_in_each_input": 8034,
        "stage1_positive_primary_population": int(len(merged)),
        "stage1_negative_gold_audit_excluded": 400,
        "gold_manual_in_primary_population": int(merged["stage2_source"].eq("gold_manual").sum()),
        "model_inferred_in_primary_population": int(merged["stage2_source"].astype(str).str.startswith("model_").sum()),
        "unique_games": int(merged["game"].nunique()),
        "review_text_identical_across_model_files": True,
        "review_ids_and_game_assignments_paired": True,
        "gold_labels_identical_across_model_files": True,
    }
    return merged, checks


def _sample_random(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    return df.sample(n=min(n, len(df)), random_state=seed).copy()


def sample_gold(g: pd.DataFrame, label: str, n: int, seed: int) -> pd.DataFrame:
    """Sample Gold positives across review-length bands for text diversity."""
    x = g.loc[
        g["stage2_source"].eq("gold_manual") & g[f"pred_{label}"].eq(1)
    ].copy()
    if x.empty or n <= 0:
        return x.iloc[0:0].copy()
    if "review_length" in x.columns and x["review_length"].notna().sum() >= 3:
        x["_length_band"] = pd.qcut(
            pd.to_numeric(x["review_length"], errors="coerce").rank(method="first"),
            q=min(3, len(x)), labels=False, duplicates="drop",
        )
        chosen = []
        bands = sorted(x["_length_band"].dropna().unique().tolist())
        # Allocate nearly equally across short, medium and long comments.
        quotas = [n // len(bands)] * len(bands)
        for i in range(n % len(bands)):
            quotas[i] += 1
        for i, band in enumerate(bands):
            part = x.loc[x["_length_band"].eq(band)]
            chosen.append(_sample_random(part, quotas[i], seed + i))
        out = pd.concat(chosen, ignore_index=False) if chosen else x.iloc[0:0]
        if len(out) < min(n, len(x)):
            remaining = x.loc[~x.index.isin(out.index)]
            out = pd.concat([out, _sample_random(remaining, min(n, len(x))-len(out), seed+99)])
        return out.drop(columns=["_length_band"], errors="ignore")
    return _sample_random(x, n, seed)


def sample_agreement(g: pd.DataFrame, label: str, n: int, seed: int) -> pd.DataFrame:
    """Sample model-inferred cases predicted positive by both models."""
    x = g.loc[
        g["stage2_source"].astype(str).str.startswith("model_")
        & g[f"pred_{label}"].eq(1)
        & g[f"deberta_pred_{label}"].eq(1)
    ].copy()
    if x.empty or n <= 0:
        return x.iloc[0:0].copy()

    p_margin = (
        pd.to_numeric(x[f"prob_{label}"], errors="coerce")
        - pd.to_numeric(x["stage2_threshold"], errors="coerce")
    )
    d_margin = (
        pd.to_numeric(x[f"deberta_prob_{label}"], errors="coerce")
        - pd.to_numeric(x["stage2_threshold_deberta"], errors="coerce")
    )
    x["_joint_margin"] = pd.concat([p_margin, d_margin], axis=1).min(axis=1)
    x = x.sort_values("_joint_margin")
    # Avoid selecting only extreme-confidence examples: sample across the
    # lower and upper halves of the joint confidence-margin distribution.
    mid = max(1, len(x) // 2)
    lower = x.iloc[:mid]
    upper = x.iloc[mid:]
    n_low = n // 2
    n_high = n - n_low
    a = _sample_random(lower, n_low, seed)
    b = _sample_random(upper, n_high, seed + 1)
    out = pd.concat([a, b])
    if len(out) < min(n, len(x)):
        remain = x.loc[~x.index.isin(out.index)]
        out = pd.concat([out, _sample_random(remain, min(n, len(x))-len(out), seed+2)])
    out["joint_confidence_margin"] = out["_joint_margin"]
    return out.drop(columns=["_joint_margin"], errors="ignore")


def sample_disagreements(
    g: pd.DataFrame, label: str, n: int, seed: int
) -> pd.DataFrame:
    """Sample model-inferred disagreements, balancing the two directions."""
    model = g.loc[g["stage2_source"].astype(str).str.startswith("model_")].copy()
    p = model[f"pred_{label}"].eq(1)
    d = model[f"deberta_pred_{label}"].eq(1)
    p_only = model.loc[p & ~d].copy()
    d_only = model.loc[~p & d].copy()

    n_each = n // 2
    a = _sample_random(p_only, n_each, seed)
    b = _sample_random(d_only, n - n_each, seed + 1)
    out = pd.concat([a, b])
    if len(out) < min(n, len(p_only) + len(d_only)):
        remaining = pd.concat([
            p_only.loc[~p_only.index.isin(out.index)],
            d_only.loc[~d_only.index.isin(out.index)],
        ])
        out = pd.concat([out, _sample_random(remaining, min(n, len(p_only)+len(d_only))-len(out), seed+2)])
    if out.empty:
        return out
    out["disagreement_direction"] = np.where(
        out[f"pred_{label}"].eq(1), "RoBERTa-only positive", "DeBERTa-only positive"
    )
    return out


def select_cases(
    df: pd.DataFrame,
    issue_gold_n: int,
    issue_agree_n: int,
    issue_disagree_n: int,
    control_gold_n: int,
    control_agree_n: int,
    control_disagree_n: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selections = []
    audit = []
    strata = [(g, l, "issue") for g, l in ISSUE_STRATA] + [
        (g, l, "control") for g, l in CONTROL_STRATA
    ]

    for idx, (game, label, kind) in enumerate(strata):
        g = df.loc[df["game"].eq(game)].copy()
        is_control = kind == "control"
        q_gold = control_gold_n if is_control else issue_gold_n
        q_agree = control_agree_n if is_control else issue_agree_n
        q_disagree = control_disagree_n if is_control else issue_disagree_n

        gold_pool = g.loc[
            g["stage2_source"].eq("gold_manual") & g[f"pred_{label}"].eq(1)
        ]
        model = g.loc[g["stage2_source"].astype(str).str.startswith("model_")]
        agree_pool = model.loc[
            model[f"pred_{label}"].eq(1) & model[f"deberta_pred_{label}"].eq(1)
        ]
        disagree_pool = model.loc[
            model[f"pred_{label}"].ne(model[f"deberta_pred_{label}"])
        ]

        gold = sample_gold(g, label, q_gold, seed + idx * 101)
        gold["selection_type"] = "Gold-confirmed example"
        gold["selection_reason"] = "Stage-2 manually labeled Gold positive for target category"

        agree = sample_agreement(g, label, q_agree, seed + idx * 101 + 10)
        agree["selection_type"] = "Both-model agreement"
        agree["selection_reason"] = "Both RoBERTa T2 and DeBERTa T3 predict target category positive"

        disagree = sample_disagreements(g, label, q_disagree, seed + idx * 101 + 20)
        disagree["selection_type"] = "Model disagreement audit"
        disagree["selection_reason"] = "One model predicts target category positive and the other does not"

        selected = pd.concat([gold, agree, disagree], ignore_index=False)
        selected["case_stratum"] = f"{GAME_NAMES.get(game, game)} | {label}"
        selected["target_game"] = game
        selected["target_label"] = label
        selected["stratum_kind"] = kind
        selected["requested_gold_n"] = q_gold
        selected["requested_agreement_n"] = q_agree
        selected["requested_disagreement_n"] = q_disagree
        selections.append(selected)

        audit.append({
            "game": game,
            "game_name": GAME_NAMES.get(game, game),
            "target_label": label,
            "stratum_kind": kind,
            "eligible_gold_positive": int(len(gold_pool)),
            "eligible_model_agreement": int(len(agree_pool)),
            "eligible_model_disagreement": int(len(disagree_pool)),
            "selected_gold": int(len(gold)),
            "selected_model_agreement": int(len(agree)),
            "selected_model_disagreement": int(len(disagree)),
            "selected_total": int(len(selected)),
        })

    long = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    if long.empty:
        raise ValueError("No case examples selected; check input labels and game IDs")

    # Deduplicate the master sheet while preserving every stratum assignment.
    agg = (
        long.groupby("review_id", as_index=False)
        .agg(
            selected_for_strata=("case_stratum", lambda s: " || ".join(sorted(set(s)))),
            selected_for_labels=("target_label", lambda s: "; ".join(sorted(set(s)))),
            selection_types=("selection_type", lambda s: "; ".join(sorted(set(s)))),
            selection_reasons=("selection_reason", lambda s: " || ".join(sorted(set(s)))),
        )
    )
    base_cols = [c for c in (["game_name"] + BASE_META) if c in long.columns]
    extra_cols = [
        "stage2_source", "stage2_source_deberta", "stage2_status",
        "stage2_status_deberta", "stage2_threshold", "stage2_threshold_deberta",
        "source_role",
    ]
    for label in ISSUE_LABELS:
        extra_cols += [
            f"pred_{label}", f"prob_{label}",
            f"deberta_pred_{label}", f"deberta_prob_{label}",
        ]
    keep = list(dict.fromkeys(["review_id"] + base_cols + extra_cols))
    master = long[keep].drop_duplicates("review_id").merge(
        agg, on="review_id", how="left", validate="one_to_one"
    )
    master = master.sort_values(["game_name", "review_id"]).reset_index(drop=True)
    return master, long, pd.DataFrame(audit)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract real Steam review cases for qualitative inspection."
    )
    p.add_argument("--primary-input", type=Path, default=DEFAULT_PRIMARY)
    p.add_argument("--robust-input", type=Path, default=DEFAULT_ROBUST)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--issue-gold-n", type=int, default=4)
    p.add_argument("--issue-agreement-n", type=int, default=4)
    p.add_argument("--issue-disagreement-n", type=int, default=2)
    p.add_argument("--control-gold-n", type=int, default=3)
    p.add_argument("--control-agreement-n", type=int, default=3)
    p.add_argument("--control-disagreement-n", type=int, default=2)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if min(
        args.issue_gold_n, args.issue_agreement_n, args.issue_disagreement_n,
        args.control_gold_n, args.control_agreement_n, args.control_disagreement_n,
    ) < 0:
        raise ValueError("Sampling quotas must be non-negative")

    print("Loading RoBERTa T2 and DeBERTa T3 population files...")
    primary = read_population(args.primary_input, "RoBERTa T2")
    robust = read_population(args.robust_input, "DeBERTa T3")
    df, checks = validate_and_merge(primary, robust)

    print("\nDATA VALIDATION")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print("\nPRESPECIFIED CASE STRATA")
    _, long, audit = select_cases(
        df,
        args.issue_gold_n, args.issue_agreement_n, args.issue_disagreement_n,
        args.control_gold_n, args.control_agreement_n, args.control_disagreement_n,
        args.seed,
    )
    print(audit.to_string(index=False))
    print(f"\nSelected case assignments (duplicates across multi-label strata allowed): {len(long)}")
    print(f"Unique review IDs selected: {long['review_id'].nunique()}")
    print("Original review text is retained; no synthetic text or new labels are generated.")

    if args.dry_run:
        print("\nDRY RUN PASSED: input files, population counts, paired IDs, Gold labels, and case strata validated.")
        print("No output files were written.")
        return

    master, long, audit = select_cases(
        df,
        args.issue_gold_n, args.issue_agreement_n, args.issue_disagreement_n,
        args.control_gold_n, args.control_agreement_n, args.control_disagreement_n,
        args.seed,
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Export both a unique-review master and a long table retaining stratum-level
    # assignments (a multi-label review can legitimately appear in multiple strata).
    master.to_csv(out / "01_case_candidates_unique.csv", index=False, encoding="utf-8-sig")
    long.to_csv(out / "02_case_candidates_by_stratum.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "03_sampling_audit_by_stratum.csv", index=False, encoding="utf-8-sig")

    disagreements = long.loc[long["selection_type"].eq("Model disagreement audit")].copy()
    gold_cases = long.loc[long["selection_type"].eq("Gold-confirmed example")].copy()
    agreement_cases = long.loc[long["selection_type"].eq("Both-model agreement")].copy()
    controls = long.loc[long["stratum_kind"].eq("control")].copy()
    disagreements.to_csv(out / "04_model_disagreement_cases.csv", index=False, encoding="utf-8-sig")
    gold_cases.to_csv(out / "05_gold_confirmed_cases.csv", index=False, encoding="utf-8-sig")
    agreement_cases.to_csv(out / "06_two_model_agreement_cases.csv", index=False, encoding="utf-8-sig")
    controls.to_csv(out / "07_no_issue_control_cases.csv", index=False, encoding="utf-8-sig")

    readme = [
        "Steam Paper 1 - qualitative case extraction",
        "",
        "Purpose: select real Steam review examples for focused qualitative inspection.",
        "This is not a new annotation round. No new labels were assigned.",
        "No API calls, model inference, synthetic text generation, or source-file modification occurred.",
        "",
        "Primary model: RoBERTa T2. Robustness model: DeBERTa T3.",
        "Primary population: Stage-1-positive reviews only (7,634 records).",
        "The 400 Stage-1-negative Gold audit records were excluded from case selection.",
        "",
        "Sampling strata:",
        "Cities: COMMUNITY_MODERATION and MONETIZATION",
        "Cyberpunk 2077: BUG_STABILITY",
        "Hades: GAMEPLAY_BALANCE and CONTENT",
        "Factorio: GAMEPLAY_BALANCE, MONETIZATION, UI_UX",
        "NO_ISSUE controls: Factorio and Cities: Skylines",
        "",
        "Selection types:",
        "- Gold-confirmed: target label is positive in the manually labeled Gold record.",
        "- Both-model agreement: both RoBERTa T2 and DeBERTa T3 predict the target label positive.",
        "- Model disagreement audit: exactly one of the two models predicts the target label positive.",
        "",
        "The same review may appear in multiple strata because Stage 2 is multi-label.",
        "Use 01_case_candidates_unique.csv for a deduplicated review list.",
        "Use 02_case_candidates_by_stratum.csv to retain the reason each review was selected.",
        "Do not treat model-inferred examples as human-verified labels.",
        "For manuscript quotations, manually verify the original wording and context.",
        "",
        f"Random seed: {args.seed}",
    ]
    (out / "00_README.txt").write_text("\n".join(readme), encoding="utf-8")

    summary = {
        "purpose": "Targeted qualitative case extraction from real Steam reviews",
        "primary_model": "RoBERTa T2",
        "robustness_model": "DeBERTa T3",
        "validation": checks,
        "selected_unique_reviews": int(master["review_id"].nunique()),
        "selected_stratum_assignments": int(len(long)),
        "selected_by_type": {
            str(k): int(v) for k, v in long["selection_type"].value_counts().items()
        },
        "sampling_audit": audit.to_dict(orient="records"),
        "seed": int(args.seed),
        "no_new_labels_assigned": True,
        "no_synthetic_text_generated": True,
        "stage1_negative_gold_audit_excluded": 400,
    }
    (out / "08_case_extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Excel workbook for convenient reading and filtering.
    xlsx = out / "stage2_qualitative_case_examples.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        master.to_excel(writer, sheet_name="Unique cases", index=False)
        long.to_excel(writer, sheet_name="By stratum", index=False)
        audit.to_excel(writer, sheet_name="Sampling audit", index=False)
        gold_cases.to_excel(writer, sheet_name="Gold examples", index=False)
        agreement_cases.to_excel(writer, sheet_name="Model agreement", index=False)
        disagreements.to_excel(writer, sheet_name="Disagreement audit", index=False)
        controls.to_excel(writer, sheet_name="NO_ISSUE controls", index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells[: min(len(col_cells), 150)]:
                    value = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, min(len(value), 60))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 48)
            # Keep long review text readable without making the workbook enormous.
            header = {cell.value: cell.column for cell in ws[1]}
            if "review_text" in header:
                ws.column_dimensions[ws.cell(1, header["review_text"]).column_letter].width = 72

    print("\nCASE EXTRACTION COMPLETE")
    print(f"Unique review cases: {len(master)}")
    print(f"Stratum-level assignments: {len(long)}")
    print(f"Output directory: {out}")
    print(f"Excel workbook: {xlsx}")
    print("Please send the ZIP of this output folder for case-content analysis.")


if __name__ == "__main__":
    main()