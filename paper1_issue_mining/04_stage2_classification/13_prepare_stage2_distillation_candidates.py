# -*- coding: utf-8 -*-
"""
Prepare Stage-2 distillation candidates from the UNLABELED Steam review pool.

Important safeguards:
- Does not change any Gold labels.
- Excludes the complete 2,000-review Stage-2 Gold sample and any discovered
  train/validation/test Gold split files by Steam review_id.
- Uses keyword patterns ONLY for candidate retrieval, never as labels.
- Produces a candidate pool for subsequent LLM teacher scoring and human audit.

Run from Steam_Research_Code:
  python paper1_issue_mining/04_stage2_classification/13_prepare_stage2_distillation_candidates.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from config_local import PAPER1_ANALYSIS_ROOT  # noqa: E402

SOURCE = PAPER1_ANALYSIS_ROOT / "stage1_inference" / "stage2_candidate_reviews.csv"
ANNOTATION_DIR = PAPER1_ANALYSIS_ROOT / "stage2_annotation"
GOLD_SAMPLE = ANNOTATION_DIR / "stage2_review_sample_2000.csv"
OUTPUT_DIR = ANNOTATION_DIR / "distillation_candidates_v1"

SEED = 20261029
PER_GROUP = 80
RANDOM_RESIDUAL_N = 120

# Candidate retrieval only. Matching a keyword does NOT mean the label is present.
PATTERNS = {
    "PRIVACY_POLICY": [
        (r"\bprivacy\b", "privacy"),
        (r"\bpersonal data\b", "personal data"),
        (r"\bdata collection\b", "data collection"),
        (r"\bcollect(?:s|ed|ing)? (?:my |your |user )?data\b", "collect data"),
        (r"\btelemetry\b", "telemetry"),
        (r"\btrack(?:ing|s)? (?:players|users|me|you)\b", "tracking users"),
        (r"\btracking\b", "tracking"),
        (r"\bip address\b", "IP address"),
        (r"\blocation data\b", "location data"),
        (r"\buser data\b", "user data"),
        (r"\bsell(?:s|ing)? (?:my |your |user )?data\b", "sell data"),
        (r"\bgdpr\b", "GDPR"),
        (r"\bconsent\b", "consent"),
        (r"\bsurveillance\b", "surveillance"),
        (r"\bdata sharing\b", "data sharing"),
        (r"\baccount information\b", "account information"),
    ],
    "COMPATIBILITY": [
        (r"\bcross[- ]?play\b", "crossplay"),
        (r"\bcross[- ]platform\b", "cross-platform"),
        (r"\blinux\b", "Linux"),
        (r"\bmac(?:os)?\b", "Mac/macOS"),
        (r"\bsteam deck\b", "Steam Deck"),
        (r"\bcontroller support\b", "controller support"),
        (r"\bcontroller (?:doesn't|does not|won't|will not) work\b", "controller not working"),
        (r"\bcompatib(?:le|ility)\b", "compatibility"),
        (r"\boperating system\b", "operating system"),
        (r"\bplatform[- ]specific\b", "platform-specific"),
        (r"\bunsupported platform\b", "unsupported platform"),
        (r"\bwon't launch on\b", "won't launch on"),
        (r"\bdoesn't run on\b", "doesn't run on"),
    ],
    "MOD_SUPPORT": [
        (r"\bmods?\b", "mod/mods"),
        (r"\bmodding\b", "modding"),
        (r"\bsteam workshop\b", "Steam Workshop"),
        (r"\bworkshop\b", "Workshop"),
        (r"\bmod manager\b", "mod manager"),
        (r"\bmod support\b", "mod support"),
        (r"\bmod compatibility\b", "mod compatibility"),
        (r"\bload order\b", "load order"),
        (r"\basset packs?\b", "asset pack"),
        (r"\bmod(?:s)? (?:break|broke|broken|crash|crashes|conflict|conflicts)\b", "mod failure/conflict"),
    ],
    "PERFORMANCE": [
        (r"\bfps\b", "FPS"),
        (r"\bframe[- ]?rate\b", "frame rate"),
        (r"\blag(?:gy|ging)?\b", "lag"),
        (r"\bstutter(?:ing|s)?\b", "stutter"),
        (r"\bperformance\b", "performance"),
        (r"\bpoorly optimized\b", "poor optimization"),
        (r"\boptimization\b", "optimization"),
        (r"\bslow loading\b", "slow loading"),
        (r"\bloading times?\b", "loading time"),
        (r"\bmemory leak\b", "memory leak"),
        (r"\bhigh cpu\b", "high CPU"),
        (r"\bhigh gpu\b", "high GPU"),
        (r"\bframe drops?\b", "frame drop"),
        (r"\bslow(?:er)? (?:after|during|when|in|with)\b", "contextual slowdown"),
    ],
    "MULTIPLAYER_NETWORK": [
        (r"\bmultiplayer\b", "multiplayer"),
        (r"\bonline (?:mode|play|match|matches)\b", "online play"),
        (r"\bservers?\b", "server"),
        (r"\bdisconnect(?:ed|s|ing)?\b", "disconnect"),
        (r"\bmatchmaking\b", "matchmaking"),
        (r"\bping\b", "ping"),
        (r"\blatency\b", "latency"),
        (r"\bconnection\b", "connection"),
        (r"\bco[- ]?op\b", "co-op"),
        (r"\blobbies\b|\blobby\b", "lobby"),
        (r"\bnetwork(?:ing)?\b", "network"),
    ],
    "AI_SYSTEM": [
        (r"\bai\b", "AI"),
        (r"\bnpcs?\b", "NPC"),
        (r"\bpathfinding\b", "pathfinding"),
        (r"\benemy (?:behavior|behaviour|ai)\b", "enemy behavior"),
        (r"\bbot behavior\b|\bbot behaviour\b", "bot behavior"),
        (r"\bartificial intelligence\b", "artificial intelligence"),
    ],
}

LABELS = list(PATTERNS)


def norm_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def read_csv_ids(path: Path, id_col: str = "review_id") -> set[str]:
    try:
        df = pd.read_csv(path, dtype={id_col: "string"}, low_memory=False)
    except Exception as exc:
        print(f"WARNING: Could not read {path}: {exc}")
        return set()
    if id_col not in df.columns:
        print(f"WARNING: {path.name} has no {id_col}; skipped for exclusion.")
        return set()
    return {norm_id(x) for x in df[id_col] if norm_id(x)}


def discover_gold_ids() -> tuple[set[str], list[str]]:
    if not GOLD_SAMPLE.is_file():
        raise FileNotFoundError(
            f"Required leakage-protection file missing: {GOLD_SAMPLE}\n"
            "Do not continue until stage2_review_sample_2000.csv is available."
        )

    paths = [GOLD_SAMPLE]
    # Also exclude released split files or a combined Gold file if present.
    for p in ANNOTATION_DIR.rglob("*.csv"):
        name = p.name.lower()
        if p == GOLD_SAMPLE:
            continue
        if (
            name.startswith("stage2_gold_train_")
            or name.startswith("stage2_gold_val_")
            or name.startswith("stage2_gold_test_")
            or name == "stage2_gold_all_2000.csv"
        ):
            paths.append(p)

    excluded = set()
    used = []
    for path in sorted(set(paths)):
        ids = read_csv_ids(path, "review_id")
        excluded.update(ids)
        used.append(str(path))
        print(f"Gold exclusion: {path.name}: {len(ids)} review IDs")
    return excluded, used


def matched_labels_and_terms(text: str) -> tuple[list[str], dict[str, list[str]]]:
    labels = []
    terms_by_label = {}
    low = text.lower()
    for label, rules in PATTERNS.items():
        terms = []
        for pattern, term in rules:
            if re.search(pattern, low, flags=re.I):
                terms.append(term)
        if terms:
            labels.append(label)
            terms_by_label[label] = sorted(set(terms))
    return labels, terms_by_label


def stratified_sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample approximately proportionally by game, then length group if present."""
    n = min(int(n), len(pool))
    if n <= 0:
        return pool.iloc[0:0].copy()
    work = pool.copy()
    if "length_group" not in work.columns:
        work["length_group"] = pd.cut(
            work["review_text"].astype(str).str.len(),
            bins=[-1, 100, 300, float("inf")],
            labels=["short", "medium", "long"],
        ).astype(str)
    # Use pandas proportional sampling by game/length strata.
    strata = work.groupby(["game", "length_group"], dropna=False, observed=False).size()
    ideal = strata.astype(float) * n / len(work)
    quota = ideal.apply(int)
    remainder = n - int(quota.sum())
    frac_order = (ideal - quota).sort_values(ascending=False, kind="stable")
    for key in frac_order.index[:remainder]:
        quota.loc[key] += 1

    parts = []
    rng = seed
    for (game, length_group), count in quota.items():
        if count <= 0:
            continue
        part = work[(work["game"] == game) & (work["length_group"].astype(str) == str(length_group))]
        take = min(int(count), len(part))
        if take:
            parts.append(part.sample(n=take, random_state=rng))
            rng += 1
    result = pd.concat(parts, ignore_index=False) if parts else work.iloc[0:0].copy()
    if len(result) < n:
        missing = n - len(result)
        rest = work.loc[~work.index.isin(result.index)]
        if len(rest):
            result = pd.concat([result, rest.sample(n=min(missing, len(rest)), random_state=seed)], axis=0)
    return result.drop_duplicates(subset=["review_id"]).head(n)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Candidate source missing: {SOURCE}")

    source = pd.read_csv(SOURCE, dtype={"review_id": "string"}, low_memory=False)
    required = {"review_id", "game", "review_text"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Candidate source missing required columns: {sorted(missing)}")

    source["review_id"] = source["review_id"].map(norm_id)
    source["review_text"] = source["review_text"].fillna("").astype(str).str.strip()
    source = source[
        source["review_id"].ne("") &
        source["game"].notna() &
        source["review_text"].ne("")
    ].copy()

    if "stage1_pred" in source.columns:
        source = source[pd.to_numeric(source["stage1_pred"], errors="coerce").eq(1)].copy()

    gold_ids, exclusion_files = discover_gold_ids()
    before_exclusion = len(source)
    source = source[~source["review_id"].isin(gold_ids)].copy()
    excluded_count = before_exclusion - len(source)
    source = source.drop_duplicates(subset=["review_id"]).copy()

    if source.empty:
        raise RuntimeError("No unlabeled candidates remain after excluding Gold IDs.")

    source["length_group"] = pd.cut(
        source["review_text"].str.len(),
        bins=[-1, 100, 300, float("inf")],
        labels=["short", "medium", "long"],
    ).astype(str)

    labels_and_terms = source["review_text"].map(matched_labels_and_terms)
    source["matched_candidate_labels"] = labels_and_terms.map(lambda x: "|".join(x[0]))
    source["matched_terms_json"] = labels_and_terms.map(lambda x: json.dumps(x[1], ensure_ascii=False))
    source["candidate_group"] = source["matched_candidate_labels"].replace("", "GENERAL_RESIDUAL")

    selected_indices = set()
    selections = []
    group_pool_report = []

    # Each target category contributes up to PER_GROUP candidates.
    for i, label in enumerate(LABELS):
        pool = source[source["matched_candidate_labels"].str.split("|").map(lambda xs: label in xs)].copy()
        group_pool_report.append({
            "candidate_group": label,
            "available_unlabeled_pool": len(pool),
            "target_n": PER_GROUP,
        })
        if pool.empty:
            continue
        sampled = stratified_sample(pool, PER_GROUP, SEED + i * 17)
        for idx, row in sampled.iterrows():
            selected_indices.add(idx)
            selections.append((idx, label))

    # Random residual examples help discover less keyword-obvious issues and possible OTHER_ISSUE cases.
    residual = source.loc[~source.index.isin(selected_indices)].copy()
    random_sample = stratified_sample(residual, RANDOM_RESIDUAL_N, SEED + 999)
    for idx, row in random_sample.iterrows():
        selected_indices.add(idx)
        selections.append((idx, "GENERAL_RESIDUAL_RANDOM"))
    group_pool_report.append({
        "candidate_group": "GENERAL_RESIDUAL_RANDOM",
        "available_unlabeled_pool": len(residual),
        "target_n": RANDOM_RESIDUAL_N,
    })

    if not selected_indices:
        raise RuntimeError("No candidates selected; inspect candidate source and exclusion files.")

    out = source.loc[sorted(selected_indices)].copy()
    selected_reason = {}
    for idx, group in selections:
        selected_reason.setdefault(idx, [])
        if group not in selected_reason[idx]:
            selected_reason[idx].append(group)
    out["selection_reasons"] = out.index.map(lambda idx: "|".join(selected_reason.get(idx, [])))
    out["selected_for_teacher_review"] = True
    out["gold_or_test_data"] = False
    out = out.reset_index(drop=True)

    # Ensure no selected review ID overlaps any known Gold row.
    overlap = set(out["review_id"]) & gold_ids
    if overlap:
        raise RuntimeError(f"Leakage guard failed: {len(overlap)} selected IDs overlap Gold.")

    pool_path = OUTPUT_DIR / "stage2_distillation_candidate_pool_v1.csv"
    summary_path = OUTPUT_DIR / "stage2_distillation_candidate_pool_summary.csv"
    manifest_path = OUTPUT_DIR / "stage2_distillation_candidate_pool_manifest.json"

    out.to_csv(pool_path, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(group_pool_report)
    summary["selected_n"] = summary["candidate_group"].map(
        out["selection_reasons"].fillna("").str.split("|").explode().value_counts().to_dict()
    ).fillna(0).astype(int)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    manifest = {
        "purpose": "Unlabeled candidate retrieval for Stage-2 teacher scoring/distillation",
        "source_file": str(SOURCE),
        "gold_exclusion_files": exclusion_files,
        "output_file": str(pool_path),
        "random_seed": SEED,
        "per_keyword_group_target": PER_GROUP,
        "random_residual_target": RANDOM_RESIDUAL_N,
        "source_rows_after_cleaning_before_gold_exclusion": before_exclusion,
        "gold_ids_excluded_count_from_source": excluded_count,
        "unlabeled_source_rows_after_exclusion_and_dedup": int(len(source)),
        "selected_unique_candidates": int(len(out)),
        "gold_overlap_after_selection": int(len(overlap)),
        "groups": LABELS + ["GENERAL_RESIDUAL_RANDOM"],
        "note": "Keyword hits are retrieval hints only, not labels. No Gold labels were changed. Do not use the 32 validation-error cases as training data."
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nCANDIDATE MINING COMPLETE")
    print(f"Source: {SOURCE}")
    print(f"Gold IDs excluded from source: {excluded_count}")
    print(f"Remaining unlabeled pool: {len(source)}")
    print(f"Selected unique candidates: {len(out)}")
    print(f"Gold overlap check: {len(overlap)} (must be 0)")
    print("\nSelection reasons:")
    print(out["selection_reasons"].str.split("|").explode().value_counts().to_string())
    print(f"\nCandidate CSV: {pool_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Manifest: {manifest_path}")
    print("\nReminder: candidate groups are keyword-based retrieval cues only; they are NOT Gold labels.")


if __name__ == "__main__":
    main()
