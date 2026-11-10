# -*- coding: utf-8 -*-
"""
Steam Paper 1: compact event-window diagnostic (descriptive only).

Question:
Did Cities: Skylines show an unusual short-term concentration of reviews and
Stage-2 issue labels around 2026-09-23 to 2026-09-25, and was anything similar
visible in the other nine games?

Design:
- Primary: RoBERTa T2 predictions (Gold rows retain their manual labels).
- Robustness: DeBERTa T3 predictions (Gold rows retain the same manual labels).
- Analysis population: 7,634 Stage-1-positive reviews only.
- Excludes 400 Stage-1-negative Gold audit records from temporal rates.
- Event window: 2026-09-23 through 2026-09-25.
- Comparison windows: 14 calendar days immediately before and after.
- Descriptive comparisons only; no p-values and no claims of causality.

Run from Steam_Research_Code:
  python -u .\\paper1_issue_mining\\04_stage2_classification\\24_analyze_event_window_v1.py --dry-run
  python -u .\\paper1_issue_mining\\04_stage2_classification\\24_analyze_event_window_v1.py --run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
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
    / "event_window_analysis_20261031_v1"
)

EVENT_START = pd.Timestamp("2026-09-23")
EVENT_END = pd.Timestamp("2026-09-25")
PRE_START = EVENT_START - pd.Timedelta(days=14)
PRE_END = EVENT_START - pd.Timedelta(days=1)
POST_START = EVENT_END + pd.Timedelta(days=1)
POST_END = EVENT_END + pd.Timedelta(days=14)
PLOT_START = EVENT_START - pd.Timedelta(days=21)
PLOT_END = EVENT_END + pd.Timedelta(days=21)

LABELS = [
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
CITIES_ID = "255710_CitiesSkylines"


def read_model(path: Path, model: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{model} input not found: {path}")
    df = pd.read_csv(
        path, dtype={"review_id": "string"}, encoding="utf-8-sig",
        low_memory=False,
    )
    required = {
        "review_id", "game", "review_text", "date", "stage1_pred",
        "stage2_source", "negative_flag",
        *[f"pred_{x}" for x in LABELS],
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{model} file missing columns: {missing}")
    if df["review_id"].isna().any() or df["review_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{model}: missing review_id")
    df["review_id"] = df["review_id"].astype(str)
    if df["review_id"].duplicated().any():
        raise ValueError(f"{model}: duplicate review_id")
    df["stage1_pred"] = pd.to_numeric(df["stage1_pred"], errors="raise").astype(int)
    if not set(df["stage1_pred"].unique()).issubset({0, 1}):
        raise ValueError(f"{model}: stage1_pred must be binary")
    for label in LABELS:
        c = f"pred_{label}"
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
        if not set(df[c].unique()).issubset({0, 1}):
            raise ValueError(f"{model}: {c} must be binary")
    df["negative_flag"] = pd.to_numeric(df["negative_flag"], errors="coerce")
    df["_parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["_parsed_date"].notna().mean() < 0.98:
        # Fallback only for rows where the supplied date field is unavailable.
        if "timestamp" in df.columns:
            ts = pd.to_numeric(df["timestamp"], errors="coerce")
            fallback = pd.to_datetime(ts, unit="s", errors="coerce", utc=True)
            fallback = fallback.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
            df["_parsed_date"] = df["_parsed_date"].fillna(fallback)
    if df["_parsed_date"].notna().mean() < 0.98:
        raise ValueError(f"{model}: fewer than 98% of rows have parseable dates")
    df["_day"] = df["_parsed_date"].dt.normalize()
    return df


def validate_pair(p: pd.DataFrame, r: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if len(p) != 8034 or len(r) != 8034:
        raise ValueError(f"Expected 8,034 rows per file; got {len(p)} and {len(r)}")
    for c in ["review_id", "game", "review_text", "date", "stage1_pred"]:
        if not p[c].astype(str).reset_index(drop=True).equals(
            r[c].astype(str).reset_index(drop=True)
        ):
            raise ValueError(f"Paired model files differ in {c}")
    if not p["stage2_source"].astype(str).eq("gold_manual").equals(
        r["stage2_source"].astype(str).eq("gold_manual")
    ):
        raise ValueError("Gold/manual source membership differs between models")

    pos = p["stage1_pred"].eq(1)
    neg = ~pos
    if int(pos.sum()) != 7634 or int(neg.sum()) != 400:
        raise ValueError(
            f"Expected 7,634 Stage-1 positives and 400 audit rows; "
            f"got {int(pos.sum())} and {int(neg.sum())}"
        )
    if not p.loc[neg, "stage2_source"].astype(str).eq("gold_manual").all():
        raise ValueError("Stage-1-negative audit rows are not all manual Gold")

    gold = p["stage2_source"].astype(str).eq("gold_manual")
    for label in LABELS:
        c = f"pred_{label}"
        if not p.loc[gold, c].reset_index(drop=True).equals(
            r.loc[gold, c].reset_index(drop=True)
        ):
            raise ValueError(f"Manual Gold labels differ for {label}")

    # Keep the primary predictions and append paired robustness predictions.
    rob_cols = ["review_id"] + [f"pred_{x}" for x in LABELS]
    rob = r[rob_cols].rename(
        columns={f"pred_{x}": f"deberta_pred_{x}" for x in LABELS}
    )
    merged = p.loc[pos].copy().merge(
        rob, on="review_id", how="left", validate="one_to_one"
    )
    if len(merged) != 7634:
        raise ValueError("Unexpected row count after paired merge")
    if merged[[f"deberta_pred_{x}" for x in LABELS]].isna().any().any():
        raise ValueError("Missing robustness predictions after merge")
    merged["game_name"] = merged["game"].map(GAME_NAMES).fillna(merged["game"])
    merged["day"] = merged["_day"].dt.normalize()

    checks = {
        "rows_per_model_file": 8034,
        "stage1_positive_analysis_population": int(len(merged)),
        "stage1_negative_gold_audit_rows_excluded": 400,
        "unique_games": int(merged["game"].nunique()),
        "review_ids_text_game_date_paired": True,
        "gold_labels_match_between_models": True,
        "analysis_primary_model": "RoBERTa T2",
        "robustness_model": "DeBERTa T3",
        "date_min": str(merged["day"].min().date()),
        "date_max": str(merged["day"].max().date()),
        "event_window": f"{EVENT_START.date()} to {EVENT_END.date()}",
        "pre_window": f"{PRE_START.date()} to {PRE_END.date()}",
        "post_window": f"{POST_START.date()} to {POST_END.date()}",
    }
    return merged, checks


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game, g in df.groupby("game", sort=True):
        first = min(g["day"].min(), PLOT_START)
        last = max(g["day"].max(), PLOT_END)
        calendar = pd.date_range(first, last, freq="D")
        indexed = g.set_index("day")
        for day in calendar:
            if day in indexed.index:
                d = indexed.loc[[day]]
            else:
                d = indexed.iloc[0:0]
            n = len(d)
            neg_share = (
                float(d["negative_flag"].mean())
                if n and d["negative_flag"].notna().any() else np.nan
            )
            rec = {
                "game": game,
                "game_name": GAME_NAMES.get(game, game),
                "day": day.date().isoformat(),
                "n_stage1_positive_reviews": n,
                "negative_review_share": neg_share,
            }
            for label in LABELS:
                rec[f"roberta_n_{label}"] = int(d[f"pred_{label}"].sum()) if n else 0
                rec[f"roberta_share_{label}"] = (
                    float(d[f"pred_{label}"].mean()) if n else np.nan
                )
                rec[f"deberta_n_{label}"] = (
                    int(d[f"deberta_pred_{label}"].sum()) if n else 0
                )
                rec[f"deberta_share_{label}"] = (
                    float(d[f"deberta_pred_{label}"].mean()) if n else np.nan
                )
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize_window(g: pd.DataFrame, model_prefix: str, label: str) -> dict:
    col = f"pred_{label}" if model_prefix == "roberta" else f"deberta_pred_{label}"

    def get(start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int, float]:
        x = g.loc[g["day"].between(start, end)]
        n = len(x)
        positive = int(x[col].sum()) if n else 0
        share = positive / n if n else np.nan
        return n, positive, share

    pre_n, pre_pos, pre_share = get(PRE_START, PRE_END)
    event_n, event_pos, event_share = get(EVENT_START, EVENT_END)
    post_n, post_pos, post_share = get(POST_START, POST_END)

    pre_days = int(g.loc[g["day"].between(PRE_START, PRE_END), "day"].nunique())
    event_days = int(g.loc[g["day"].between(EVENT_START, EVENT_END), "day"].nunique())
    post_days = int(g.loc[g["day"].between(POST_START, POST_END), "day"].nunique())

    pre_daily = pre_n / 14
    event_daily = event_n / 3
    post_daily = post_n / 14
    return {
        "model": "RoBERTa T2" if model_prefix == "roberta" else "DeBERTa T3",
        "label": label,
        "pre_reviews_14d": pre_n,
        "event_reviews_3d": event_n,
        "post_reviews_14d": post_n,
        "pre_days_with_data": pre_days,
        "event_days_with_data": event_days,
        "post_days_with_data": post_days,
        "pre_avg_reviews_per_calendar_day": pre_daily,
        "event_avg_reviews_per_calendar_day": event_daily,
        "post_avg_reviews_per_calendar_day": post_daily,
        "event_volume_ratio_vs_pre": event_daily / pre_daily if pre_daily else np.nan,
        "pre_label_positive_n": pre_pos,
        "event_label_positive_n": event_pos,
        "post_label_positive_n": post_pos,
        "pre_label_share": pre_share,
        "event_label_share": event_share,
        "post_label_share": post_share,
        "event_minus_pre_percentage_points": (
            (event_share - pre_share) * 100
            if pd.notna(event_share) and pd.notna(pre_share) else np.nan
        ),
        "event_minus_post_percentage_points": (
            (event_share - post_share) * 100
            if pd.notna(event_share) and pd.notna(post_share) else np.nan
        ),
    }


def build_event_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game, g in df.groupby("game", sort=True):
        for label in LABELS:
            for model in ["roberta", "deberta"]:
                rec = summarize_window(g, model, label)
                rec.update({"game": game, "game_name": GAME_NAMES.get(game, game)})
                rows.append(rec)
    return pd.DataFrame(rows)


def plot_cities(daily: pd.DataFrame, out: Path) -> None:
    x = daily.loc[
        (daily["game"] == CITIES_ID)
        & pd.to_datetime(daily["day"]).between(PLOT_START, PLOT_END)
    ].copy()
    if x.empty:
        print("WARNING: no Cities: Skylines observations in plotting window")
        return
    x["day"] = pd.to_datetime(x["day"])

    # Figure 1: daily volume.
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(x["day"], x["n_stage1_positive_reviews"], marker="o", linewidth=1.3)
    ax.axvspan(EVENT_START, EVENT_END + pd.Timedelta(days=1), alpha=0.15)
    ax.set_title("Cities: Skylines — daily Stage-1-positive review volume")
    ax.set_ylabel("Review count")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "01_cities_skylines_daily_volume.png", dpi=220)
    plt.close(fig)

    # Figure 2: target-category shares, both models.
    fig, ax = plt.subplots(figsize=(11, 5.2))
    series = [
        ("roberta_share_COMMUNITY_MODERATION", "RoBERTa — Community moderation"),
        ("deberta_share_COMMUNITY_MODERATION", "DeBERTa — Community moderation"),
        ("roberta_share_MONETIZATION", "RoBERTa — Monetization"),
        ("deberta_share_MONETIZATION", "DeBERTa — Monetization"),
    ]
    for col, label in series:
        ax.plot(x["day"], x[col] * 100, marker="o", linewidth=1.2, label=label)
    ax.axvspan(EVENT_START, EVENT_END + pd.Timedelta(days=1), alpha=0.15, label="Event window")
    ax.set_title("Cities: Skylines — daily predicted issue prevalence")
    ax.set_ylabel("Share of Stage-1-positive reviews (%)")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "02_cities_skylines_daily_issue_share.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-input", type=Path, default=DEFAULT_PRIMARY)
    ap.add_argument("--robust-input", type=Path, default=DEFAULT_ROBUST)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = ap.parse_args()

    print("Loading paired full-corpus analysis files...")
    p = read_model(args.primary_input, "RoBERTa T2")
    r = read_model(args.robust_input, "DeBERTa T3")
    df, checks = validate_pair(p, r)

    print("\nVALIDATION")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print("\nCITIES: SKYLINES EVENT-WINDOW COUNTS")
    city = df.loc[df["game"].eq(CITIES_ID)].copy()
    for name, start, end in [
        ("PRE", PRE_START, PRE_END),
        ("EVENT", EVENT_START, EVENT_END),
        ("POST", POST_START, POST_END),
    ]:
        x = city.loc[city["day"].between(start, end)]
        print(
            f"{name}: calendar_days={(end-start).days+1}, "
            f"observed_reviews={len(x)}, "
            f"days_with_reviews={x['day'].nunique()}"
        )
    print("\nNo individual review is being manually adjudicated in this step.")
    print("This is a bounded descriptive diagnostic, not a new validation study.")

    if args.dry_run:
        print("\nDRY RUN PASSED. No output files written.")
        return

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    daily = build_daily(df)
    summary = build_event_summary(df)

    daily.to_csv(out / "01_daily_game_issue_series.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "02_event_window_comparison_all_games.csv", index=False, encoding="utf-8-sig")

    city_summary = summary.loc[summary["game"].eq(CITIES_ID)].copy()
    city_summary.to_csv(
        out / "03_cities_skylines_event_window_comparison.csv",
        index=False, encoding="utf-8-sig",
    )

    # Compact volume-only summary, independent of issue labels.
    volume_rows = []
    for game, g in df.groupby("game", sort=True):
        for period, start, end, days in [
            ("pre", PRE_START, PRE_END, 14),
            ("event", EVENT_START, EVENT_END, 3),
            ("post", POST_START, POST_END, 14),
        ]:
            x = g.loc[g["day"].between(start, end)]
            volume_rows.append({
                "game": game,
                "game_name": GAME_NAMES.get(game, game),
                "period": period,
                "calendar_days": days,
                "observed_reviews": len(x),
                "days_with_reviews": int(x["day"].nunique()),
                "avg_reviews_per_calendar_day": len(x) / days,
                "negative_review_share": (
                    float(x["negative_flag"].mean())
                    if len(x) and x["negative_flag"].notna().any() else np.nan
                ),
            })
    volume = pd.DataFrame(volume_rows)
    volume.to_csv(out / "04_review_volume_comparison.csv", index=False, encoding="utf-8-sig")

    plot_cities(daily, out)

    xlsx = out / "event_window_diagnostic.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        volume.to_excel(writer, sheet_name="Volume", index=False)
        city_summary.to_excel(writer, sheet_name="Cities issue comparison", index=False)
        summary.to_excel(writer, sheet_name="All games all labels", index=False)
        daily.to_excel(writer, sheet_name="Daily series", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cells in ws.columns:
                letter = cells[0].column_letter
                max_len = max(
                    (min(len(str(c.value or "")), 45) for c in cells[:min(len(cells), 120)]),
                    default=12,
                )
                ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 42)

    readme = f"""Steam Paper 1 — event-window diagnostic

Purpose:
A bounded descriptive check of whether Cities: Skylines review volume and
predicted issue prevalence changed around 2026-09-23 to 2026-09-25.

Models:
Primary: RoBERTa T2.
Robustness: DeBERTa T3.
Gold rows retain manual labels in both files.

Population:
7,634 Stage-1-positive reviews only.
400 Stage-1-negative Gold audit records are excluded from temporal rates.

Windows:
Pre: {PRE_START.date()} to {PRE_END.date()} (14 calendar days)
Event: {EVENT_START.date()} to {EVENT_END.date()} (3 calendar days)
Post: {POST_START.date()} to {POST_END.date()} (14 calendar days)

Interpretation limits:
- These are descriptive patterns in the collected corpus, not population-level
  incidence estimates unless the source corpus is known to be exhaustive.
- Stage-2 F1 is around 0.7; daily category rates are therefore approximate.
- No individual review is assumed to be correctly classified.
- Agreement between models is robustness evidence, not proof of correctness.
- No p-values or causal claims are produced.
- If post-window data coverage is incomplete, interpret post comparisons cautiously.

Files:
01_daily_game_issue_series.csv: daily counts and category shares for all games.
02_event_window_comparison_all_games.csv: all games and all labels, both models.
03_cities_skylines_event_window_comparison.csv: focused Cities: Skylines table.
04_review_volume_comparison.csv: review volume and negative-review share.
event_window_diagnostic.xlsx: convenient workbook.
01/02 PNG files: local diagnostic figures.

Validation:
{json.dumps(checks, ensure_ascii=False, indent=2)}
"""
    (out / "00_README.txt").write_text(readme, encoding="utf-8")
    summary_json = {
        "validation": checks,
        "population": "Stage-1-positive only",
        "primary_model": "RoBERTa T2",
        "robustness_model": "DeBERTa T3",
        "event_window": [str(EVENT_START.date()), str(EVENT_END.date())],
        "pre_window": [str(PRE_START.date()), str(PRE_END.date())],
        "post_window": [str(POST_START.date()), str(POST_END.date())],
        "outputs": [
            "01_daily_game_issue_series.csv",
            "02_event_window_comparison_all_games.csv",
            "03_cities_skylines_event_window_comparison.csv",
            "04_review_volume_comparison.csv",
            "event_window_diagnostic.xlsx",
            "01_cities_skylines_daily_volume.png",
            "02_cities_skylines_daily_issue_share.png",
        ],
        "no_new_annotation": True,
        "descriptive_only": True,
        "no_causal_claim": True,
    }
    (out / "05_run_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nEVENT-WINDOW ANALYSIS COMPLETE")
    print(f"Output directory: {out}")
    print(f"Workbook: {xlsx}")
    print("Please send the ZIP of this output folder; no need to send individual files.")


if __name__ == "__main__":
    main()