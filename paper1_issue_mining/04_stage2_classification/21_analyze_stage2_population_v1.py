# -*- coding: utf-8 -*-
"""
Steam Stage-2 population analysis (local execution only).

Primary analysis:
  7,634 Stage-1-positive candidate reviews
  = 1,600 Stage-2 Gold reviews + 6,034 model-inferred reviews.

Separate audit:
  400 Stage-1-negative reviews with manual Gold labels.
  These are NOT mixed into the primary distribution denominator.

Models:
  Primary: RoBERTa T2
  Robustness: DeBERTa T3

Run from Steam_Research_Code:
  python paper1_issue_mining/04_stage2_classification/21_analyze_stage2_population_v1.py --dry-run
  python paper1_issue_mining/04_stage2_classification/21_analyze_stage2_population_v1.py --run

Optional:
  --primary-input "<analysis-root>\\roberta_targeted_T2\\stage2_analysis_population_with_gold_and_model.csv"
  --robust-input "<analysis-root>\\deberta_targeted_T3\\stage2_analysis_population_with_gold_and_model.csv"
  --output-dir "D:\\...\\stage2_population_analysis_20261031_v1"
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    / "stage2_population_analysis_20261031_v1"
)

ISSUE_LABELS = [
    "BUG_STABILITY", "PERFORMANCE", "COMPATIBILITY", "UI_UX",
    "MULTIPLAYER_NETWORK", "GAMEPLAY_BALANCE", "AI_SYSTEM", "CONTENT",
    "MOD_SUPPORT", "MONETIZATION", "PRIVACY_POLICY",
    "COMMUNITY_MODERATION", "OTHER_ISSUE",
]
ALL_LABELS = ISSUE_LABELS + ["NO_ISSUE"]
PRED_COLS = [f"pred_{x}" for x in ALL_LABELS]
PROB_COLS = [f"prob_{x}" for x in ALL_LABELS]


def read_data(path: Path, model_tag: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{model_tag} input not found: {path}")
    df = pd.read_csv(
        path, dtype={"review_id": "string"}, low_memory=False, encoding="utf-8-sig"
    )
    required = {
        "review_id", "game", "stage1_pred", "stage2_status", "stage2_source",
        "stage2_threshold", *PRED_COLS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{model_tag} missing required columns: {sorted(missing)}")
    if df["review_id"].isna().any() or df["review_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{model_tag}: empty review_id found")
    if df["review_id"].duplicated().any():
        raise ValueError(f"{model_tag}: duplicated review_id found")
    df["stage1_pred"] = pd.to_numeric(df["stage1_pred"], errors="raise").astype(int)
    if not set(df["stage1_pred"].unique()).issubset({0, 1}):
        raise ValueError(f"{model_tag}: stage1_pred must contain only 0/1")
    for c in PRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            raise ValueError(f"{model_tag}: missing/non-numeric values in {c}")
        if not set(df[c].unique()).issubset({0, 1}):
            raise ValueError(f"{model_tag}: {c} must be binary 0/1")
        df[c] = df[c].astype(int)
    return df


def issue_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[[f"pred_{x}" for x in ISSUE_LABELS]]


def add_issue_flags(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["issue_any"] = issue_matrix(x).any(axis=1)
    x["no_issue"] = x["pred_NO_ISSUE"].eq(1)
    x["unassigned"] = x["stage2_status"].astype(str).eq("unassigned")
    return x


def validate_pair(main: pd.DataFrame, robust: pd.DataFrame) -> dict:
    if len(main) != 8034 or len(robust) != 8034:
        raise ValueError(
            f"Expected 8,034 rows in each input; got {len(main)} and {len(robust)}"
        )
    if not main["review_id"].astype(str).equals(robust["review_id"].astype(str)):
        raise ValueError("The two model files do not have identical review_id order")
    if not main["game"].astype(str).equals(robust["game"].astype(str)):
        raise ValueError("The two model files have different game assignments")
    if not main["stage1_pred"].equals(robust["stage1_pred"]):
        raise ValueError("The two model files have different Stage-1 inclusion flags")

    for tag, df in [("RoBERTa T2", main), ("DeBERTa T3", robust)]:
        n_pos = int(df["stage1_pred"].eq(1).sum())
        n_neg = int(df["stage1_pred"].eq(0).sum())
        n_gold = int(df["stage2_source"].eq("gold_manual").sum())
        n_model = int(df["stage2_source"].astype(str).str.startswith("model_").sum())
        if (n_pos, n_neg, n_gold, n_model) != (7634, 400, 2000, 6034):
            raise ValueError(
                f"{tag}: unexpected population counts "
                f"stage1+={n_pos}, stage1-={n_neg}, gold={n_gold}, model={n_model}"
            )
        if not df.loc[df["stage1_pred"].eq(0), "stage2_source"].eq("gold_manual").all():
            raise ValueError(f"{tag}: Stage-1-negative rows include non-Gold records")

    gold_main = main.loc[main["stage2_source"].eq("gold_manual")].copy()
    gold_robust = robust.loc[robust["stage2_source"].eq("gold_manual")].copy()
    if not gold_main["review_id"].astype(str).equals(gold_robust["review_id"].astype(str)):
        raise ValueError("Gold IDs differ between model files")
    for c in PRED_COLS:
        if not gold_main[c].equals(gold_robust[c]):
            raise ValueError(f"Gold labels differ between model files: {c}")
    if not gold_main["stage2_status"].equals(gold_robust["stage2_status"]):
        raise ValueError("Gold stage2_status differs between model files")

    return {
        "rows_each_file": int(len(main)),
        "unique_games": int(main["game"].nunique()),
        "stage1_positive_primary_population": int(main["stage1_pred"].eq(1).sum()),
        "stage1_negative_gold_audit": int(main["stage1_pred"].eq(0).sum()),
        "gold_total": int(main["stage2_source"].eq("gold_manual").sum()),
        "gold_in_primary_population": int(
            (main["stage1_pred"].eq(1) & main["stage2_source"].eq("gold_manual")).sum()
        ),
        "model_inference_rows": int(
            main["stage2_source"].astype(str).str.startswith("model_").sum()
        ),
        "review_id_order_identical": True,
        "gold_labels_identical_across_models": True,
    }


def distribution_tables(df: pd.DataFrame, model_name: str, scope_name: str):
    """Tables for a defined population; multi-label counts are non-exclusive."""
    x = add_issue_flags(df)
    n = len(x)
    issue_n = int(x["issue_any"].sum())

    overall_rows = []
    for label in ISSUE_LABELS:
        count = int(x[f"pred_{label}"].sum())
        overall_rows.append({
            "model": model_name,
            "scope": scope_name,
            "issue_domain": label,
            "n_reviews_with_domain": count,
            "share_of_population": count / n if n else np.nan,
            "share_of_issue_bearing_reviews": count / issue_n if issue_n else np.nan,
        })
    overall = pd.DataFrame(overall_rows).sort_values(
        "n_reviews_with_domain", ascending=False
    )

    games = sorted(x["game"].astype(str).unique())
    game_overview_rows, game_long_rows = [], []
    for game in games:
        g = x.loc[x["game"].astype(str).eq(game)]
        ng = len(g)
        gi = int(g["issue_any"].sum())
        game_overview_rows.append({
            "model": model_name,
            "game": game,
            "n_primary_reviews": ng,
            "n_gold_manual": int(g["stage2_source"].eq("gold_manual").sum()),
            "n_model_inferred": int(
                g["stage2_source"].astype(str).str.startswith("model_").sum()
            ),
            "n_issue_bearing_reviews": gi,
            "issue_bearing_share": gi / ng if ng else np.nan,
            "n_no_issue": int(g["no_issue"].sum()),
            "no_issue_share": float(g["no_issue"].mean()) if ng else np.nan,
            "n_unassigned": int(g["unassigned"].sum()),
            "unassigned_share": float(g["unassigned"].mean()) if ng else np.nan,
            "mean_issue_labels_per_review": float(issue_matrix(g).sum(axis=1).mean()) if ng else np.nan,
        })
        for label in ISSUE_LABELS:
            count = int(g[f"pred_{label}"].sum())
            game_long_rows.append({
                "model": model_name,
                "game": game,
                "issue_domain": label,
                "n_reviews_with_domain": count,
                "share_of_game_primary_population": count / ng if ng else np.nan,
                "share_of_game_issue_bearing_reviews": count / gi if gi else np.nan,
            })

    game_overview = pd.DataFrame(game_overview_rows)
    game_long = pd.DataFrame(game_long_rows)
    game_wide = game_long.pivot(
        index="game", columns="issue_domain",
        values="share_of_game_primary_population"
    ).reset_index()
    count_wide = game_long.pivot(
        index="game", columns="issue_domain",
        values="n_reviews_with_domain"
    ).reset_index()

    # Multi-label co-occurrence. Counts are not mutually exclusive.
    pair_rows = []
    for a, b in combinations(ISSUE_LABELS, 2):
        both = int((x[f"pred_{a}"].eq(1) & x[f"pred_{b}"].eq(1)).sum())
        pair_rows.append({
            "model": model_name, "scope": scope_name,
            "issue_a": a, "issue_b": b, "cooccurrence_n": both,
            "share_of_population": both / n if n else np.nan,
            "share_of_issue_bearing_reviews": both / issue_n if issue_n else np.nan,
        })
    cooccurrence = pd.DataFrame(pair_rows).sort_values(
        "cooccurrence_n", ascending=False
    )

    def combo(row):
        labs = [lab for lab in ISSUE_LABELS if row[f"pred_{lab}"] == 1]
        if labs:
            return "|".join(labs)
        if row["pred_NO_ISSUE"] == 1:
            return "NO_ISSUE"
        return "UNASSIGNED"

    x["issue_combination"] = x.apply(combo, axis=1)
    combo = (
        x["issue_combination"].value_counts().head(50)
        .rename_axis("issue_combination").reset_index(name="n_reviews")
    )
    combo["model"] = model_name
    combo["scope"] = scope_name
    combo["share_of_population"] = combo["n_reviews"] / n

    # Distribution separated by Gold/manual and model-inferred source.
    source_rows = []
    for source, g in x.groupby("stage2_source", dropna=False):
        for label in ISSUE_LABELS:
            count = int(g[f"pred_{label}"].sum())
            source_rows.append({
                "model": model_name,
                "source": str(source),
                "n_source_reviews": len(g),
                "issue_domain": label,
                "n_reviews_with_domain": count,
                "share_within_source": count / len(g) if len(g) else np.nan,
            })
    by_source = pd.DataFrame(source_rows)

    return {
        "overall": overall,
        "game_overview": game_overview,
        "game_long": game_long,
        "game_wide": game_wide,
        "count_wide": count_wide,
        "cooccurrence": cooccurrence,
        "combinations": combo,
        "by_source": by_source,
        "population": x,
    }


def audit_400(df: pd.DataFrame):
    audit = df.loc[df["stage1_pred"].eq(0)].copy()
    if len(audit) != 400:
        raise ValueError(f"Expected 400 Stage-1-negative Gold audit rows, got {len(audit)}")
    audit = add_issue_flags(audit)
    rows = []
    for game, g in audit.groupby("game"):
        issue_n = int(g["issue_any"].sum())
        rows.append({
            "game": game,
            "n_stage1_negative_gold_audit": len(g),
            "n_manual_issue": issue_n,
            "manual_issue_share_in_audit": issue_n / len(g) if len(g) else np.nan,
            "n_manual_no_issue": int(g["no_issue"].sum()),
        })
    by_game = pd.DataFrame(rows).sort_values("n_stage1_negative_gold_audit", ascending=False)

    label_rows = []
    for label in ISSUE_LABELS:
        count = int(audit[f"pred_{label}"].sum())
        label_rows.append({
            "issue_domain": label,
            "n_reviews_with_domain": count,
            "share_of_400_audit_rows": count / len(audit),
        })
    by_label = pd.DataFrame(label_rows).sort_values("n_reviews_with_domain", ascending=False)
    summary = {
        "audit_rows": int(len(audit)),
        "manual_issue_rows": int(audit["issue_any"].sum()),
        "manual_no_issue_rows": int(audit["no_issue"].sum()),
        "issue_share_in_this_audit_only": float(audit["issue_any"].mean()),
        "interpretation": (
            "This is a 400-row manually labelled Stage-1-negative audit subset. "
            "Do not interpret its issue share as Stage-1 recall or extrapolate it "
            "to all Stage-1-negative reviews unless the sampling design supports that."
        ),
    }
    return audit, by_game, by_label, summary


def agreement_tables(main: pd.DataFrame, robust: pd.DataFrame):
    a = main.loc[
        main["stage1_pred"].eq(1)
        & main["stage2_source"].astype(str).str.startswith("model_")
    ].copy()
    b = robust.loc[
        robust["stage1_pred"].eq(1)
        & robust["stage2_source"].astype(str).str.startswith("model_")
    ].copy()
    a = a.set_index(a["review_id"].astype(str)).sort_index()
    b = b.set_index(b["review_id"].astype(str)).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Model-inferred review IDs differ; cannot compare agreement")

    issue_a = issue_matrix(a).any(axis=1)
    issue_b = issue_matrix(b).any(axis=1)
    status_a = a["stage2_status"].astype(str)
    status_b = b["stage2_status"].astype(str)

    summary = {
        "common_model_inferred_reviews": int(len(a)),
        "exact_13_issue_label_vector_agreement": float(
            (issue_matrix(a).to_numpy() == issue_matrix(b).to_numpy()).all(axis=1).mean()
        ),
        "issue_any_binary_agreement": float((issue_a.to_numpy() == issue_b.to_numpy()).mean()),
        "three_state_status_agreement": float((status_a.to_numpy() == status_b.to_numpy()).mean()),
        "roberta_issue_bearing_n": int(issue_a.sum()),
        "deberta_issue_bearing_n": int(issue_b.sum()),
        "roberta_no_issue_n": int(a["pred_NO_ISSUE"].sum()),
        "deberta_no_issue_n": int(b["pred_NO_ISSUE"].sum()),
        "roberta_unassigned_n": int(status_a.eq("unassigned").sum()),
        "deberta_unassigned_n": int(status_b.eq("unassigned").sum()),
    }

    label_rows = []
    for label in ISSUE_LABELS + ["NO_ISSUE"]:
        ca, cb = f"pred_{label}", f"pred_{label}"
        va, vb = a[ca].astype(int), b[cb].astype(int)
        observed = float((va.to_numpy() == vb.to_numpy()).mean())
        p_a, p_b = float(va.mean()), float(vb.mean())
        p_yes = (p_a + p_b) / 2
        p_no = 1 - p_yes
        expected = p_yes ** 2 + p_no ** 2
        kappa = (observed - expected) / (1 - expected) if expected < 1 else np.nan
        label_rows.append({
            "issue_domain": label,
            "roberta_positive_n": int(va.sum()),
            "deberta_positive_n": int(vb.sum()),
            "roberta_positive_share": p_a,
            "deberta_positive_share": p_b,
            "absolute_share_difference": abs(p_a - p_b),
            "binary_agreement": observed,
            "cohen_kappa": kappa,
        })
    per_label = pd.DataFrame(label_rows).sort_values(
        "absolute_share_difference", ascending=False
    )

    # Compare game-level prevalence on the same 6,034 model-only reviews.
    game_rows = []
    for game in sorted(a["game"].astype(str).unique()):
        ga = a.loc[a["game"].astype(str).eq(game)]
        gb = b.loc[b["game"].astype(str).eq(game)]
        for label in ISSUE_LABELS:
            pa, pb = float(ga[f"pred_{label}"].mean()), float(gb[f"pred_{label}"].mean())
            game_rows.append({
                "game": game, "issue_domain": label,
                "roberta_T2_share": pa, "deberta_T3_share": pb,
                "difference_T2_minus_T3": pa - pb,
                "absolute_difference": abs(pa - pb),
                "n_reviews": len(ga),
            })
    game_compare = pd.DataFrame(game_rows)

    corr_rows = []
    for label in ISSUE_LABELS:
        z = game_compare.loc[game_compare["issue_domain"].eq(label)]
        corr_rows.append({
            "issue_domain": label,
            "pearson_correlation_across_10_games": z["roberta_T2_share"].corr(z["deberta_T3_share"]),
            "mean_absolute_game_share_difference": float(z["absolute_difference"].mean()),
        })
    corr = pd.DataFrame(corr_rows).sort_values(
        "mean_absolute_game_share_difference", ascending=False
    )

    # Game-level overall issue rate agreement.
    game_issue_rows = []
    for game in sorted(a["game"].astype(str).unique()):
        ga, gb = a.loc[a["game"].astype(str).eq(game)], b.loc[b["game"].astype(str).eq(game)]
        pa = float(issue_matrix(ga).any(axis=1).mean())
        pb = float(issue_matrix(gb).any(axis=1).mean())
        game_issue_rows.append({
            "game": game, "n_model_inferred_reviews": len(ga),
            "roberta_issue_share": pa, "deberta_issue_share": pb,
            "absolute_difference": abs(pa - pb),
        })
    game_issue = pd.DataFrame(game_issue_rows)
    game_issue["rank_roberta"] = game_issue["roberta_issue_share"].rank(ascending=False, method="min").astype(int)
    game_issue["rank_deberta"] = game_issue["deberta_issue_share"].rank(ascending=False, method="min").astype(int)
    return summary, per_label, game_compare, corr, game_issue


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def make_figures(primary: dict, robustness: dict, audit_by_game: pd.DataFrame, out: Path):
    # Figure 1: overall issue-domain prevalence in the primary 7,634-row population.
    overall = primary["overall"].sort_values("share_of_population", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.barh(overall["issue_domain"], overall["share_of_population"] * 100)
    ax.set_xlabel("% of Stage-1-positive primary population (n=7,634)")
    ax.set_title("Overall prevalence of actionable issue domains")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig01_overall_issue_domain_prevalence.png", dpi=300)
    plt.close(fig)

    # Figure 2: game x issue heatmap, denominator = all Stage-1-positive rows in each game.
    wide = primary["game_wide"].set_index("game")[ISSUE_LABELS].fillna(0) * 100
    fig, ax = plt.subplots(figsize=(16, 7.5))
    im = ax.imshow(wide.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(ISSUE_LABELS)))
    ax.set_xticklabels(ISSUE_LABELS, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(wide.index)))
    ax.set_yticklabels(wide.index, fontsize=9)
    ax.set_title("Issue-domain prevalence by game")
    ax.set_xlabel("Issue domain")
    ax.set_ylabel("Game")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("% of Stage-1-positive reviews within each game")
    fig.tight_layout()
    fig.savefig(out / "fig02_game_issue_prevalence_heatmap.png", dpi=300)
    plt.close(fig)

    # Figure 3: robustness scatter, game-level issue-bearing share on common model-only reviews.
    gi = robustness["game_issue"]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(gi["roberta_issue_share"] * 100, gi["deberta_issue_share"] * 100)
    for _, row in gi.iterrows():
        short = str(row["game"]).split("_", 1)[-1]
        ax.annotate(short, (row["roberta_issue_share"] * 100, row["deberta_issue_share"] * 100),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    lo = min(gi["roberta_issue_share"].min(), gi["deberta_issue_share"].min()) * 100 - 3
    hi = max(gi["roberta_issue_share"].max(), gi["deberta_issue_share"].max()) * 100 + 3
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("RoBERTa T2 issue-bearing share (%)")
    ax.set_ylabel("DeBERTa T3 issue-bearing share (%)")
    ax.set_title("Cross-model robustness of game-level issue prevalence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig03_model_robustness_game_issue_share.png", dpi=300)
    plt.close(fig)

    # Figure 4: 400 manually labelled Stage-1-negative audit rows by game.
    ag = audit_by_game.sort_values("manual_issue_share_in_audit", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(ag["game"], ag["manual_issue_share_in_audit"] * 100)
    ax.set_xlabel("% manually labelled as issue among Stage-1-negative audit rows")
    ax.set_title("Manual audit of 400 Stage-1-negative Gold reviews")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig04_stage1_negative_gold_audit_by_game.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-input", type=Path, default=DEFAULT_PRIMARY)
    ap.add_argument("--robust-input", type=Path, default=DEFAULT_ROBUST)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.dry_run == args.run:
        ap.error("Specify exactly one of --dry-run or --run")

    main_df = read_data(args.primary_input, "RoBERTa T2")
    robust_df = read_data(args.robust_input, "DeBERTa T3")
    checks = validate_pair(main_df, robust_df)

    primary_pop = main_df.loc[main_df["stage1_pred"].eq(1)].copy()
    robust_pop = robust_df.loc[robust_df["stage1_pred"].eq(1)].copy()
    primary = distribution_tables(primary_pop, "RoBERTa_T2", "Stage1_positive_7634")
    robust_primary = distribution_tables(robust_pop, "DeBERTa_T3", "Stage1_positive_7634")

    audit_df, audit_by_game, audit_by_label, audit_summary = audit_400(main_df)
    agree_summary, agree_label, agree_game_label, agree_corr, agree_game = agreement_tables(
        main_df, robust_df
    )

    print("PRE-FLIGHT CHECK PASSED")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print("\nPRIMARY POPULATION (Stage-1 positive only):")
    print(f"  RoBERTa T2: {len(primary_pop):,}")
    print(f"  Gold/manual within primary: {int(primary_pop['stage2_source'].eq('gold_manual').sum()):,}")
    print(f"  Model-inferred within primary: {int(primary_pop['stage2_source'].astype(str).str.startswith('model_').sum()):,}")
    print(f"  DeBERTa T3 parallel population: {len(robust_pop):,}")
    print("\nSEPARATE STAGE-1-NEGATIVE GOLD AUDIT:")
    print(json.dumps(audit_summary, ensure_ascii=False, indent=2))
    print("\nMODEL AGREEMENT ON THE COMMON 6,034 MODEL-INFERRED REVIEWS:")
    print(json.dumps(agree_summary, ensure_ascii=False, indent=2))
    print(f"\nOutput directory: {args.output_dir}")

    if args.dry_run:
        print("\nDRY RUN ONLY: no tables, workbooks, or figures were written.")
        return

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. "
            "Choose a new --output-dir to avoid overwriting previous results."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Main model tables.
    for key, filename in [
        ("overall", "01_overall_issue_distribution_roberta_T2.csv"),
        ("game_overview", "02_game_level_overview_roberta_T2.csv"),
        ("game_long", "03_issue_distribution_by_game_long_roberta_T2.csv"),
        ("game_wide", "04_issue_prevalence_by_game_wide_roberta_T2.csv"),
        ("count_wide", "05_issue_counts_by_game_wide_roberta_T2.csv"),
        ("cooccurrence", "06_issue_pairwise_cooccurrence_roberta_T2.csv"),
        ("combinations", "07_issue_combinations_top50_roberta_T2.csv"),
        ("by_source", "08_issue_distribution_by_source_roberta_T2.csv"),
    ]:
        save_csv(primary[key], args.output_dir / filename)

    # Parallel robustness model distribution.
    for key, filename in [
        ("overall", "09_overall_issue_distribution_deberta_T3.csv"),
        ("game_overview", "10_game_level_overview_deberta_T3.csv"),
        ("game_long", "11_issue_distribution_by_game_long_deberta_T3.csv"),
        ("game_wide", "12_issue_prevalence_by_game_wide_deberta_T3.csv"),
    ]:
        save_csv(robust_primary[key], args.output_dir / filename)

    # 400-row Gold audit, deliberately separate from the primary denominator.
    save_csv(audit_by_game, args.output_dir / "13_stage1_negative_gold_audit_by_game.csv")
    save_csv(audit_by_label, args.output_dir / "14_stage1_negative_gold_audit_by_label.csv")
    save_csv(audit_df, args.output_dir / "15_stage1_negative_gold_audit_400_reviews.csv")

    # Model robustness comparison on the same 6,034 non-Gold rows.
    save_csv(agree_label, args.output_dir / "16_model_agreement_by_label.csv")
    save_csv(agree_game_label, args.output_dir / "17_model_comparison_by_game_and_label.csv")
    save_csv(agree_corr, args.output_dir / "18_model_robustness_label_correlations_across_games.csv")
    save_csv(agree_game, args.output_dir / "19_model_robustness_game_issue_rates.csv")

    make_figures(primary, {"game_issue": agree_game}, audit_by_game, args.output_dir)

    summary = {
        "task": "Steam Stage-2 issue distribution and game-level analysis",
        "primary_model": "RoBERTa T2",
        "robustness_model": "DeBERTa T3",
        "primary_population_definition": (
            "All 7,634 Stage-1-positive candidates; includes 1,600 manual Gold labels "
            "and 6,034 RoBERTa/DeBERTa model-inferred reviews."
        ),
        "separate_audit_definition": (
            "400 Stage-1-negative records with manual Gold labels; excluded from the "
            "primary issue-distribution denominator."
        ),
        "input_primary": str(args.primary_input),
        "input_robustness": str(args.robust_input),
        "output_dir": str(args.output_dir),
        "data_checks": checks,
        "stage1_negative_gold_audit": audit_summary,
        "model_agreement_on_6034_model_rows": agree_summary,
        "multi_label_note": (
            "Issue-domain counts overlap because a review may receive multiple labels. "
            "Domain percentages therefore need not sum to 100%."
        ),
        "denominator_note": (
            "Primary game/domain percentages use Stage-1-positive candidate reviews "
            "within each game. They are not prevalence estimates for all Steam reviews."
        ),
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        with pd.ExcelWriter(args.output_dir / "stage2_population_analysis.xlsx", engine="openpyxl") as writer:
            primary["overall"].to_excel(writer, sheet_name="Overall T2", index=False)
            primary["game_overview"].to_excel(writer, sheet_name="Game overview T2", index=False)
            primary["game_long"].to_excel(writer, sheet_name="Game x issue T2", index=False)
            primary["cooccurrence"].to_excel(writer, sheet_name="Co-occurrence T2", index=False)
            primary["combinations"].to_excel(writer, sheet_name="Combinations T2", index=False)
            robust_primary["overall"].to_excel(writer, sheet_name="Overall T3", index=False)
            robust_primary["game_overview"].to_excel(writer, sheet_name="Game overview T3", index=False)
            agree_label.to_excel(writer, sheet_name="Model agreement", index=False)
            agree_game.to_excel(writer, sheet_name="Game robustness", index=False)
            audit_by_game.to_excel(writer, sheet_name="400 Gold audit", index=False)
    except Exception as exc:
        print(f"Excel workbook skipped ({type(exc).__name__}: {exc})")

    print("\nANALYSIS COMPLETE")
    print(f"Output: {args.output_dir}")
    print("Important files:")
    for fn in [
        "stage2_population_analysis.xlsx",
        "01_overall_issue_distribution_roberta_T2.csv",
        "02_game_level_overview_roberta_T2.csv",
        "03_issue_distribution_by_game_long_roberta_T2.csv",
        "16_model_agreement_by_label.csv",
        "19_model_robustness_game_issue_rates.csv",
        "analysis_summary.json",
        "fig01_overall_issue_domain_prevalence.png",
        "fig02_game_issue_prevalence_heatmap.png",
        "fig03_model_robustness_game_issue_share.png",
        "fig04_stage1_negative_gold_audit_by_game.png",
    ]:
        print(f"  {args.output_dir / fn}")


if __name__ == "__main__":
    main()