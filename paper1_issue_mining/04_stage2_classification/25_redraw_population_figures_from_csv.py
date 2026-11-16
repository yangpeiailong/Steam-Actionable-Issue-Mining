"""Redraw the four population figures from existing CSVs; no model inference."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication typography: use the installed Times New Roman font.
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
import pandas as pd

ISSUE_LABELS = ['BUG_STABILITY', 'PERFORMANCE', 'COMPATIBILITY', 'UI_UX', 'MULTIPLAYER_NETWORK', 'GAMEPLAY_BALANCE', 'AI_SYSTEM', 'CONTENT', 'MOD_SUPPORT', 'MONETIZATION', 'PRIVACY_POLICY', 'COMMUNITY_MODERATION', 'OTHER_ISSUE']

def make_figures(primary: dict, robustness: dict, audit_by_game: pd.DataFrame, out: Path):
    # Figure 1: overall issue-domain prevalence in the primary 7,634-row population.
    overall = primary["overall"].sort_values("share_of_population", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.barh(overall["issue_domain"], overall["share_of_population"] * 100)
    ax.set_xlabel("% of Stage-1-positive primary population (n=7,634)")
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
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig03_model_robustness_game_issue_share.png", dpi=300)
    plt.close(fig)

    # Figure 4: 400 manually labelled Stage-1-negative audit rows by game.
    ag = audit_by_game.sort_values("manual_issue_share_in_audit", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(ag["game"], ag["manual_issue_share_in_audit"] * 100)
    ax.set_xlabel("% manually labelled as issue among Stage-1-negative audit rows")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig04_stage1_negative_gold_audit_by_game.png", dpi=300)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    src = args.results_dir
    out = args.output_dir or src / "figures_no_titles"
    out.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(src / name)
    primary = {"overall": read("01_overall_issue_distribution_roberta_T2.csv"),
               "game_wide": read("04_issue_prevalence_by_game_wide_roberta_T2.csv")}
    robust = {"game_issue": read("19_model_robustness_game_issue_rates.csv")}
    audit = read("13_stage1_negative_gold_audit_by_game.csv")
    make_figures(primary, robust, audit, out)
    print(f"Figures saved to: {out.resolve()}")
if __name__ == "__main__":
    main()
