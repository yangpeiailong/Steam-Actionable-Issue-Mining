# -*- coding: utf-8 -*-
"""
Steam Paper 1: inferential tests for cross-game issue-structure differences.\nVersion 2 fixes validation of model-specific stage2_source values.

Reads the two already-created full-corpus population files (RoBERTa T2 and
DeBERTa T3). It does NOT train models, change labels, or evaluate the held-out
Gold test set.

Analyses:
  1) Omnibus game differences for each issue label and issue_any.
     - conditional Monte-Carlo permutation test (fixed game sizes and total
       positives), robust when expected cell counts are small;
     - Cramer's V effect size;
     - Holm correction across outcomes.
  2) Pairwise game comparisons using Fisher's exact test, with Holm correction
     within each outcome; Wilson intervals and Newcombe risk-difference CIs.
  3) Separate analyses for the combined primary population, Gold-only rows,
     and model-inferred rows to expose source-composition effects.
  4) Paired McNemar exact tests comparing RoBERTa and DeBERTa on the same
     model-inferred reviews, overall and by game.
  5) Kruskal-Wallis tests for the number of issue labels per review.

Important interpretation:
  - Main population is conditional on Stage-1-positive candidates; it is not
    the prevalence of issues among all Steam reviews.
  - Unassigned rows are excluded from issue-label prevalence tests rather than
    being treated as negative labels. Their rate is reported separately.
  - The 400 Stage-1-negative Gold audit records are not mixed into the primary
    population. Their sampling design must be considered separately.

Run from Steam_Research_Code:
  python -u .\\paper1_issue_mining\\04_stage2_classification\\22_test_stage2_game_differences_v1.py --dry-run
  python -u .\\paper1_issue_mining\\04_stage2_classification\\22_test_stage2_game_differences_v1.py --run

Optional:
  --permutations 20000 --seed 20261031
  --primary-input "<analysis-root>\\roberta_targeted_T2\\stage2_analysis_population_with_gold_and_model.csv"
  --robust-input "<analysis-root>\\deberta_targeted_T3\\stage2_analysis_population_with_gold_and_model.csv"
  --output-dir "<output-root>\\stage2_game_difference_tests"
"""
from __future__ import annotations

import argparse
import json
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, chi2, kruskal, binomtest

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
    / "stage2_game_difference_tests_20261031_v1"
)

ISSUE_LABELS = [
    "BUG_STABILITY", "PERFORMANCE", "COMPATIBILITY", "UI_UX",
    "MULTIPLAYER_NETWORK", "GAMEPLAY_BALANCE", "AI_SYSTEM", "CONTENT",
    "MOD_SUPPORT", "MONETIZATION", "PRIVACY_POLICY",
    "COMMUNITY_MODERATION", "OTHER_ISSUE",
]
PRED_COLS = [f"pred_{x}" for x in ISSUE_LABELS + ["NO_ISSUE"]]


def read_data(path: Path, tag: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{tag} input not found: {path}")
    df = pd.read_csv(path, dtype={"review_id": "string"}, low_memory=False, encoding="utf-8-sig")
    required = {
        "review_id", "game", "stage1_pred", "stage2_status", "stage2_source", *PRED_COLS
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{tag}: missing columns: {sorted(missing)}")
    if df["review_id"].isna().any() or df["review_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{tag}: empty review_id")
    if df["review_id"].duplicated().any():
        raise ValueError(f"{tag}: duplicate review_id")
    df["stage1_pred"] = pd.to_numeric(df["stage1_pred"], errors="raise").astype(int)
    if not set(df["stage1_pred"].unique()).issubset({0, 1}):
        raise ValueError(f"{tag}: stage1_pred must be 0/1")
    df["game"] = df["game"].astype(str)
    df["stage2_source"] = df["stage2_source"].astype(str)
    df["stage2_status"] = df["stage2_status"].astype(str).str.strip().str.lower()
    for c in PRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="raise")
        if df[c].isna().any() or not set(df[c].unique()).issubset({0, 1}):
            raise ValueError(f"{tag}: {c} must be complete binary 0/1")
        df[c] = df[c].astype(int)
    return df


def validate_pair(main: pd.DataFrame, robust: pd.DataFrame) -> dict:
    if len(main) != 8034 or len(robust) != 8034:
        raise ValueError(f"Expected 8,034 rows in each file; got {len(main)} and {len(robust)}")
    # The source field is expected to differ in its model-specific value:
    # e.g. model_roberta_targeted_T2 vs model_deberta_targeted_T3.
    # Validate source *roles* (Gold vs model-inferred), not exact model names.
    for c in ["review_id", "game", "stage1_pred"]:
        if not main[c].astype(str).reset_index(drop=True).equals(robust[c].astype(str).reset_index(drop=True)):
            raise ValueError(f"RoBERTa and DeBERTa differ in {c}; stop before testing")

    main_source = main["stage2_source"].astype(str).str.strip()
    robust_source = robust["stage2_source"].astype(str).str.strip()
    main_gold = main_source.eq("gold_manual")
    robust_gold = robust_source.eq("gold_manual")
    main_model = main_source.str.startswith("model_")
    robust_model = robust_source.str.startswith("model_")

    if not main_gold.equals(robust_gold):
        raise ValueError("Gold/manual source membership differs between RoBERTa and DeBERTa files")
    if not main_model.equals(robust_model):
        raise ValueError("Model-inferred source membership differs between RoBERTa and DeBERTa files")
    if not (main_gold | main_model).all() or not (robust_gold | robust_model).all():
        bad_main = sorted(main_source.loc[~(main_gold | main_model)].unique().tolist())
        bad_robust = sorted(robust_source.loc[~(robust_gold | robust_model)].unique().tolist())
        raise ValueError(
            "Unexpected stage2_source values. "
            f"RoBERTa={bad_main}; DeBERTa={bad_robust}"
        )

    n_pos = int(main.stage1_pred.eq(1).sum())
    n_neg = int(main.stage1_pred.eq(0).sum())
    n_gold = int(main.stage2_source.eq("gold_manual").sum())
    n_model = int(main.stage2_source.str.startswith("model_").sum())
    if (n_pos, n_neg, n_gold, n_model) != (7634, 400, 2000, 6034):
        raise ValueError(
            f"Unexpected source counts: stage1+={n_pos}, stage1-={n_neg}, "
            f"gold={n_gold}, model={n_model}"
        )
    gold_m = main.loc[main.stage2_source.eq("gold_manual")]
    gold_d = robust.loc[robust.stage2_source.eq("gold_manual")]
    for c in PRED_COLS + ["stage2_status"]:
        if not gold_m[c].reset_index(drop=True).equals(gold_d[c].reset_index(drop=True)):
            raise ValueError(f"Gold manual labels/status differ between model files: {c}")
    # The model-inferred rows must be paired on identical review IDs and game assignments.
    mi = main.loc[main.stage2_source.str.startswith("model_")]
    di = robust.loc[robust.stage2_source.str.startswith("model_")]
    if not mi.review_id.astype(str).reset_index(drop=True).equals(di.review_id.astype(str).reset_index(drop=True)):
        raise ValueError("Model-inferred review IDs are not paired")
    return {
        "rows_each_file": len(main), "stage1_positive": n_pos, "stage1_negative_gold_audit": n_neg,
        "gold_total": n_gold, "model_inferred": n_model, "unique_games": int(main.game.nunique()),
        "gold_labels_identical": True, "model_inferred_ids_paired": True,
        "source_role_membership_identical": True,
        "roberta_source_values": sorted(main_source.unique().tolist()),
        "deberta_source_values": sorted(robust_source.unique().tolist()),
        "note": "Model-specific stage2_source strings may differ; only Gold/model source roles are required to match.",
    }


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (n - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def newcombe_diff_ci(k1, n1, k2, n2):
    p1 = k1 / n1 if n1 else np.nan
    p2 = k2 / n2 if n2 else np.nan
    if not n1 or not n2:
        return np.nan, np.nan, np.nan
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return diff, lo, hi


def conditional_permutation_test(success_counts, group_sizes, permutations, seed):
    """Monte-Carlo conditional test under equal prevalence across groups.

    Conditional on the observed total number of positives, group positive
    counts follow a multivariate hypergeometric distribution under H0.
    This avoids relying on large expected cell counts.
    """
    sizes = np.asarray(group_sizes, dtype=int)
    obs = np.asarray(success_counts, dtype=int)
    n = int(sizes.sum())
    k = int(obs.sum())
    g = len(sizes)
    if n == 0 or k == 0 or k == n or g < 2:
        return np.nan, np.nan, np.nan
    exp_pos = sizes * (k / n)
    exp_neg = sizes - exp_pos
    if np.any(exp_pos <= 0) or np.any(exp_neg <= 0):
        return np.nan, np.nan, np.nan
    obs_stat = float(np.sum((obs - exp_pos) ** 2 / exp_pos + ((sizes - obs) - exp_neg) ** 2 / exp_neg))

    rng = np.random.default_rng(seed)
    sims = np.zeros((permutations, g), dtype=np.int32)
    rem_pop = n
    rem_good = k
    for j in range(g - 1):
        draw = rng.hypergeometric(rem_good, rem_pop - rem_good, int(sizes[j]), size=permutations)
        sims[:, j] = draw
        rem_good -= draw
        rem_pop -= int(sizes[j])
    sims[:, -1] = rem_good
    sim_stat = np.sum(
        (sims - exp_pos[None, :]) ** 2 / exp_pos[None, :]
        + ((sizes[None, :] - sims) - exp_neg[None, :]) ** 2 / exp_neg[None, :],
        axis=1,
    )
    p_mc = (1 + int(np.count_nonzero(sim_stat >= obs_stat - 1e-12))) / (permutations + 1)
    p_asym = float(chi2.sf(obs_stat, df=g - 1))
    cramers_v = math.sqrt(max(0.0, obs_stat / n))
    return obs_stat, p_mc, p_asym, cramers_v


def make_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all_primary":
        x = df.loc[df.stage1_pred.eq(1)].copy()
    elif scope == "gold_manual":
        x = df.loc[df.stage1_pred.eq(1) & df.stage2_source.eq("gold_manual")].copy()
    elif scope == "model_inferred":
        x = df.loc[df.stage1_pred.eq(1) & df.stage2_source.str.startswith("model_")].copy()
    else:
        raise ValueError(scope)
    x["unassigned_flag"] = x.stage2_status.eq("unassigned")
    x["issue_any"] = x[[f"pred_{z}" for z in ISSUE_LABELS]].any(axis=1).astype(int)
    x["n_issue_labels"] = x[[f"pred_{z}" for z in ISSUE_LABELS]].sum(axis=1)
    return x


def run_scope_tests(x: pd.DataFrame, scope: str, permutations: int, seed: int):
    games = sorted(x.game.unique())
    outcomes = ISSUE_LABELS + ["issue_any"]
    omnibus_rows, prevalence_rows, pair_rows = [], [], []

    # For issue outcomes, exclude unassigned rows: unknown is not treated as a negative.
    valid = x.loc[~x.unassigned_flag].copy()
    for oi, outcome in enumerate(outcomes):
        col = "issue_any" if outcome == "issue_any" else f"pred_{outcome}"
        group = valid.groupby("game", sort=False)[col]
        sizes = np.array([int((valid.game == g).sum()) for g in games], dtype=int)
        positives = np.array([int(group.get_group(g).sum()) if g in group.groups else 0 for g in games], dtype=int)
        result = conditional_permutation_test(positives, sizes, permutations, seed + oi * 101)
        if len(result) == 3:
            stat, p_mc, p_asym = result
            v = np.nan
        else:
            stat, p_mc, p_asym, v = result
        expected_min = min(
            (sizes * (positives.sum() / sizes.sum())).min() if sizes.sum() and positives.sum() else np.nan,
            (sizes * (1 - positives.sum() / sizes.sum())).min() if sizes.sum() else np.nan,
        ) if sizes.sum() else np.nan
        omnibus_rows.append({
            "scope": scope, "outcome": outcome, "n_valid": int(len(valid)),
            "n_positive": int(positives.sum()), "overall_prevalence": float(positives.sum() / sizes.sum()) if sizes.sum() else np.nan,
            "pearson_chi2_stat": stat, "df": len(games) - 1,
            "p_permutation_mc": p_mc, "p_chi2_asymptotic": p_asym,
            "cramers_v": v, "min_expected_cell": expected_min,
            "permutations": permutations,
        })
        for g, n_g, k_g in zip(games, sizes, positives):
            lo, hi = wilson(int(k_g), int(n_g))
            prevalence_rows.append({
                "scope": scope, "outcome": outcome, "game": g,
                "n_valid": int(n_g), "n_positive": int(k_g),
                "prevalence": float(k_g / n_g) if n_g else np.nan,
                "wilson_95_low": lo, "wilson_95_high": hi,
            })
        for i in range(len(games)):
            for j in range(i + 1, len(games)):
                g1, g2 = games[i], games[j]
                n1, n2 = int(sizes[i]), int(sizes[j])
                k1, k2 = int(positives[i]), int(positives[j])
                table = [[k1, n1 - k1], [k2, n2 - k2]]
                odds, p = fisher_exact(table, alternative="two-sided")
                diff, lo_d, hi_d = newcombe_diff_ci(k1, n1, k2, n2)
                pair_rows.append({
                    "scope": scope, "outcome": outcome, "game_1": g1, "game_2": g2,
                    "n_1": n1, "positive_1": k1, "prevalence_1": k1 / n1 if n1 else np.nan,
                    "n_2": n2, "positive_2": k2, "prevalence_2": k2 / n2 if n2 else np.nan,
                    "risk_difference_1_minus_2": diff,
                    "risk_difference_95_low": lo_d, "risk_difference_95_high": hi_d,
                    "odds_ratio": odds, "p_fisher": p,
                })

    om = pd.DataFrame(omnibus_rows)
    om["p_holm_within_scope"] = holm_adjust(om.p_permutation_mc.fillna(1).values)
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs["p_holm_within_outcome"] = np.nan
        for outcome in pairs.outcome.unique():
            idx = pairs.index[pairs.outcome.eq(outcome)]
            pairs.loc[idx, "p_holm_within_outcome"] = holm_adjust(pairs.loc[idx, "p_fisher"].values)
    prev = pd.DataFrame(prevalence_rows)

    # Unassigned rate is a separate outcome using the full scope denominator.
    un_rows = []
    for g in games:
        gg = x.loc[x.game.eq(g)]
        k, n = int(gg.unassigned_flag.sum()), len(gg)
        lo, hi = wilson(k, n)
        un_rows.append({"scope": scope, "outcome": "unassigned", "game": g,
                        "n_valid": n, "n_positive": k, "prevalence": k/n if n else np.nan,
                        "wilson_95_low": lo, "wilson_95_high": hi})
    unprev = pd.DataFrame(un_rows)
    un_sizes = np.array([int((x.game == g).sum()) for g in games], dtype=int)
    un_pos = np.array([int(x.loc[x.game.eq(g), "unassigned_flag"].sum()) for g in games], dtype=int)
    un_result = conditional_permutation_test(un_pos, un_sizes, permutations, seed + 99991)
    if len(un_result) == 3:
        stat, p_mc, p_asym = un_result; v = np.nan
    else:
        stat, p_mc, p_asym, v = un_result
    un_om = pd.DataFrame([{
        "scope": scope, "outcome": "unassigned", "n_valid": int(len(x)),
        "n_positive": int(un_pos.sum()), "overall_prevalence": float(un_pos.sum()/len(x)) if len(x) else np.nan,
        "pearson_chi2_stat": stat, "df": len(games)-1, "p_permutation_mc": p_mc,
        "p_chi2_asymptotic": p_asym, "cramers_v": v,
        "min_expected_cell": np.nan, "permutations": permutations,
    }])
    return om, prev, pairs, un_om, unprev


def kruskal_tests(scopes):
    rows = []
    for scope, x in scopes.items():
        valid = x.loc[~x.unassigned_flag]
        groups = [g.n_issue_labels.to_numpy() for _, g in valid.groupby("game") if len(g)]
        if len(groups) >= 2:
            h, p = kruskal(*groups)
            n = len(valid); k = len(groups)
            eps2 = max(0.0, min(1.0, (float(h) - k + 1) / (n - k))) if n > k else np.nan
        else:
            h, p, eps2, n = np.nan, np.nan, np.nan, len(valid)
        rows.append({"scope": scope, "n_valid": n, "kruskal_h": h,
                     "df": max(0, len(groups)-1), "p_value": p, "epsilon_squared": eps2})
    out = pd.DataFrame(rows)
    if len(out): out["p_holm_across_scopes"] = holm_adjust(out.p_value.fillna(1).values)
    return out


def paired_mcnemar(main: pd.DataFrame, robust: pd.DataFrame):
    m = main.loc[main.stage1_pred.eq(1) & main.stage2_source.str.startswith("model_")].copy()
    d = robust.loc[robust.stage1_pred.eq(1) & robust.stage2_source.str.startswith("model_")].copy()
    d = d.set_index("review_id").loc[m.review_id].reset_index()
    m = m.reset_index(drop=True)
    d = d.reset_index(drop=True)
    m_valid = ~m.stage2_status.eq("unassigned")
    d_valid = ~d.stage2_status.eq("unassigned")
    both_valid = m_valid & d_valid
    outcomes = ISSUE_LABELS + ["issue_any"]
    rows = []
    def binary(df, outcome):
        if outcome == "issue_any":
            return df[[f"pred_{z}" for z in ISSUE_LABELS]].any(axis=1).astype(int).to_numpy()
        return df[f"pred_{outcome}"].to_numpy(dtype=int)
    for outcome in outcomes:
        a, b = binary(m, outcome), binary(d, outcome)
        aa, bb = a[both_valid.to_numpy()], b[both_valid.to_numpy()]
        discord_r = int(((aa == 1) & (bb == 0)).sum())
        discord_d = int(((aa == 0) & (bb == 1)).sum())
        n_disc = discord_r + discord_d
        p = float(binomtest(min(discord_r, discord_d), n=n_disc, p=0.5, alternative="two-sided").pvalue) if n_disc else 1.0
        rows.append({"scope": "all_model_inferred_paired", "outcome": outcome,
                     "n_paired_assigned": int(both_valid.sum()),
                     "roberta_only_positive": discord_r, "deberta_only_positive": discord_d,
                     "discordant_n": n_disc, "p_mcnemar_exact": p})
        for game in sorted(m.game.unique()):
            gm = (m.game.eq(game).to_numpy()) & both_valid.to_numpy()
            ag, bg = a[gm], b[gm]
            r_only = int(((ag == 1) & (bg == 0)).sum())
            d_only = int(((ag == 0) & (bg == 1)).sum())
            nd = r_only + d_only
            pg = float(binomtest(min(r_only, d_only), n=nd, p=0.5, alternative="two-sided").pvalue) if nd else 1.0
            rows.append({"scope": f"game:{game}", "outcome": outcome,
                         "n_paired_assigned": int(gm.sum()),
                         "roberta_only_positive": r_only, "deberta_only_positive": d_only,
                         "discordant_n": nd, "p_mcnemar_exact": pg})
    out = pd.DataFrame(rows)
    overall = out.scope.eq("all_model_inferred_paired")
    out["p_holm_family"] = np.nan
    out.loc[overall, "p_holm_family"] = holm_adjust(out.loc[overall, "p_mcnemar_exact"].values)
    idx = out.index[~overall]
    out.loc[idx, "p_holm_family"] = holm_adjust(out.loc[idx, "p_mcnemar_exact"].values)
    # Separate paired comparison of unassigned status on all 6,034 inferred reviews.
    un_r = m.stage2_status.eq("unassigned").astype(int).to_numpy()
    un_d = d.stage2_status.eq("unassigned").astype(int).to_numpy()
    r_only = int(((un_r == 1) & (un_d == 0)).sum())
    d_only = int(((un_r == 0) & (un_d == 1)).sum())
    nd = r_only + d_only
    pu = float(binomtest(min(r_only, d_only), n=nd, p=0.5).pvalue) if nd else 1.0
    unassigned = pd.DataFrame([{"n_paired_model_reviews": len(m), "roberta_unassigned_only": r_only,
                                "deberta_unassigned_only": d_only, "discordant_n": nd,
                                "p_mcnemar_exact": pu}])
    return out, unassigned


def source_mix_table(x):
    rows=[]
    for game, g in x.loc[x.stage1_pred.eq(1)].groupby("game"):
        n_gold=int(g.stage2_source.eq("gold_manual").sum())
        n_model=int(g.stage2_source.str.startswith("model_").sum())
        rows.append({"game":game,"n_primary":len(g),"n_gold_manual":n_gold,
                     "gold_share":n_gold/len(g) if len(g) else np.nan,
                     "n_model_inferred":n_model,"model_share":n_model/len(g) if len(g) else np.nan,
                     "n_unassigned":int(g.stage2_status.eq("unassigned").sum())})
    return pd.DataFrame(rows).sort_values("game")


def save_figures(prev_all: pd.DataFrame, source_mix: pd.DataFrame, outdir: Path):
    # Figure 1: issue-any prevalence by game, combined primary vs model-only.
    fig, ax = plt.subplots(figsize=(11, 6))
    sub = prev_all.loc[prev_all.outcome.eq("issue_any")]
    for scope, marker in [("all_primary", "o"), ("model_inferred", "s"), ("gold_manual", "^")]:
        z = sub.loc[sub.scope.eq(scope)].sort_values("game")
        if len(z):
            ax.errorbar(np.arange(len(z)), z.prevalence * 100,
                        yerr=[(z.prevalence-z.wilson_95_low)*100, (z.wilson_95_high-z.prevalence)*100],
                        marker=marker, capsize=3, linewidth=1.4, label=scope)
    games = sorted(sub.game.unique())
    ax.set_xticks(np.arange(len(games))); ax.set_xticklabels([g.split("_",1)[-1] for g in games], rotation=35, ha="right")
    ax.set_ylabel("Issue-bearing reviews (%)"); ax.set_title("Issue-bearing review share by game and data source")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(outdir/"fig05_issue_any_by_game_source.png", dpi=220); plt.close(fig)

    # Figure 2: source composition by game.
    sm=source_mix.sort_values("gold_share")
    fig, ax=plt.subplots(figsize=(10,6))
    y=np.arange(len(sm))
    ax.barh(y, sm.gold_share*100, label="Gold manual")
    ax.barh(y, sm.model_share*100, left=sm.gold_share*100, label="Model inferred")
    ax.set_yticks(y); ax.set_yticklabels([g.split("_",1)[-1] for g in sm.game])
    ax.set_xlabel("Share of primary population (%)"); ax.set_title("Gold vs model-inferred composition by game")
    ax.legend(frameon=False); ax.grid(axis="x", alpha=.2); fig.tight_layout()
    fig.savefig(outdir/"fig06_source_composition_by_game.png", dpi=220); plt.close(fig)

    # Figure 3: heatmap of main-model game prevalence for 13 labels.
    z=prev_all.loc[(prev_all.scope.eq("all_primary")) & prev_all.outcome.isin(ISSUE_LABELS)].copy()
    if not z.empty:
        mat=z.pivot(index="outcome",columns="game",values="prevalence")
        fig,ax=plt.subplots(figsize=(13,7))
        im=ax.imshow(mat.values*100, aspect="auto")
        ax.set_xticks(np.arange(len(mat.columns))); ax.set_xticklabels([g.split("_",1)[-1] for g in mat.columns],rotation=35,ha="right")
        ax.set_yticks(np.arange(len(mat.index))); ax.set_yticklabels(mat.index)
        ax.set_title("Issue-domain prevalence by game (valid assigned rows, %) ")
        fig.colorbar(im,ax=ax,label="Prevalence (%)"); fig.tight_layout()
        fig.savefig(outdir/"fig07_issue_domain_game_heatmap_inferential.png",dpi=220);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--primary-input",type=Path,default=DEFAULT_PRIMARY)
    ap.add_argument("--robust-input",type=Path,default=DEFAULT_ROBUST)
    ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument("--permutations",type=int,default=20000)
    ap.add_argument("--seed",type=int,default=20261031)
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--run",action="store_true")
    args=ap.parse_args()
    if args.dry_run == args.run:
        ap.error("Specify exactly one of --dry-run or --run")
    if args.permutations < 1000:
        ap.error("Use at least 1,000 permutations; default is 20,000")

    print("Loading population files...")
    main_df=read_data(args.primary_input,"RoBERTa T2")
    robust_df=read_data(args.robust_input,"DeBERTa T3")
    checks=validate_pair(main_df,robust_df)
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    print("\nPopulation scopes:")
    for scope in ["all_primary","gold_manual","model_inferred"]:
        x=make_scope(main_df,scope)
        n_un=int(x.unassigned_flag.sum())
        print(f"  {scope}: rows={len(x)}, assigned={len(x)-n_un}, unassigned={n_un}, games={x.game.nunique()}")
    print("\nImportant: all game-level issue rates are conditional on Stage-1-positive candidates.")
    print("Unassigned rows will be excluded from issue-label tests, and reported separately.")
    if args.dry_run:
        print("\nDRY RUN PASSED: input schemas, population sizes, paired IDs, and Gold consistency validated.")
        return

    args.output_dir.mkdir(parents=True,exist_ok=True)
    scopes={s:make_scope(main_df,s) for s in ["all_primary","gold_manual","model_inferred"]}
    all_om=[]; all_prev=[]; all_pairs=[]; all_unom=[]; all_unprev=[]
    for i,(scope,x) in enumerate(scopes.items()):
        print(f"Running {scope} tests ({len(x):,} rows)...")
        om,prev,pairs,unom,unprev=run_scope_tests(x,scope,args.permutations,args.seed+i*100000)
        all_om.append(om);all_prev.append(prev);all_pairs.append(pairs);all_unom.append(unom);all_unprev.append(unprev)
    omnibus=pd.concat(all_om+all_unom,ignore_index=True)
    prevalence=pd.concat(all_prev+all_unprev,ignore_index=True)
    pairwise=pd.concat(all_pairs,ignore_index=True)
    kw=kruskal_tests(scopes)
    mcnemar,unassigned_mcnemar=paired_mcnemar(main_df,robust_df)
    sm=source_mix_table(main_df)

    # Add a single family-wise Holm correction across the 3 scopes' 14 main outcomes
    # as a conservative sensitivity table, while retaining within-scope adjustment.
    main_mask=omnibus.outcome.ne("unassigned")
    omnibus["p_holm_across_all_scopes"] = np.nan
    omnibus.loc[main_mask,"p_holm_across_all_scopes"] = holm_adjust(
        omnibus.loc[main_mask,"p_permutation_mc"].fillna(1).values
    )

    # Descriptive source-specific counts and unassigned rates.
    source_rows=[]
    for scope,x in scopes.items():
        for game,g in x.groupby("game"):
            source_rows.append({"scope":scope,"game":game,"n_rows":len(g),
                                "n_assigned":int((~g.unassigned_flag).sum()),
                                "n_unassigned":int(g.unassigned_flag.sum()),
                                "unassigned_share":float(g.unassigned_flag.mean())})
    source_status=pd.DataFrame(source_rows)

    summary={
        "task":"Inferential tests of game-level issue differences",
        "primary_model":"RoBERTa T2",
        "robustness_model":"DeBERTa T3",
        "population_note":"Primary conclusions apply to Stage-1-positive candidate reviews, not all Steam reviews.",
        "unassigned_handling":"Unassigned rows excluded from issue-label and issue_any prevalence tests; separately tested as an outcome.",
        "scopes":{s:{"n":len(x),"n_assigned":int((~x.unassigned_flag).sum()),"n_unassigned":int(x.unassigned_flag.sum()),"games":int(x.game.nunique())} for s,x in scopes.items()},
        "permutations_per_omnibus_test":args.permutations,
        "seed":args.seed,
        "multiple_testing":"Holm correction across outcomes within each scope; conservative cross-scope Holm also provided; pairwise Fisher tests Holm-adjusted within outcome.",
        "paired_model_test":"Exact McNemar tests on the same model-inferred review IDs; only rows assigned by both models are included for issue-label comparisons.",
        "data_checks":checks,
    }
    (args.output_dir/"00_statistical_test_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    tables={
        "01_omnibus_game_tests.csv":omnibus,
        "02_game_prevalence_wilson.csv":prevalence,
        "03_pairwise_game_fisher_holm.csv":pairwise,
        "04_source_mix_by_game.csv":sm,
        "05_source_status_by_game.csv":source_status,
        "06_kruskal_issue_count.csv":kw,
        "07_model_mcnemar_paired.csv":mcnemar,
        "08_model_unassigned_mcnemar.csv":unassigned_mcnemar,
    }
    for name,df in tables.items():
        df.to_csv(args.output_dir/name,index=False,encoding="utf-8-sig")
    save_figures(prevalence,sm,args.output_dir)
    xlsx=args.output_dir/"stage2_game_difference_tests.xlsx"
    with pd.ExcelWriter(xlsx,engine="openpyxl") as writer:
        for name,df in tables.items():
            sheet=name.replace(".csv","")[:31]
            df.to_excel(writer,sheet_name=sheet,index=False)
    print("\nSTATISTICAL TESTING COMPLETE")
    print(f"Output: {args.output_dir}")
    print("Key files:")
    for name in ["00_statistical_test_summary.json",*tables.keys(),"stage2_game_difference_tests.xlsx",
                 "fig05_issue_any_by_game_source.png","fig06_source_composition_by_game.png","fig07_issue_domain_game_heatmap_inferential.png"]:
        print(f"  {args.output_dir/name}")
    sig=omnibus.loc[(omnibus.outcome.ne("unassigned")) & (omnibus.p_holm_within_scope<0.05),
                    ["scope","outcome","n_valid","n_positive","cramers_v","p_permutation_mc","p_holm_within_scope"]]
    print("\nOmnibus outcomes significant after within-scope Holm correction (first 30):")
    if sig.empty: print("  None at alpha=.05")
    else: print(sig.sort_values(["scope","p_holm_within_scope"]).head(30).to_string(index=False))

if __name__=="__main__":
    main()