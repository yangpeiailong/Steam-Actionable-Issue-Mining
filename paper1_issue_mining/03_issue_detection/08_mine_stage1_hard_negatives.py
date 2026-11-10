"""
Stage 1 hard-negative mining (two commands, one script).

1) Prepare a new pool of 650 DeBERTa-positive reviews that were not used in
   Stage 1 gold or the current Stage 2 annotation sample.
2) Ask DeepSeek Flash to screen those reviews.
3) Export only LLM-negative predictions for HUMAN confirmation.

No LLM judgment is ever treated as a human gold label.
The manually reviewed 40-row audit remains untouched.

Run from Steam_Research_Code:
  python paper1_issue_mining/03_issue_detection/08_mine_stage1_hard_negatives.py --prepare
  python paper1_issue_mining/03_issue_detection/08_mine_stage1_hard_negatives.py --screen --limit 30
  python paper1_issue_mining/03_issue_detection/08_mine_stage1_hard_negatives.py --screen
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config_local import PAPER1_ANALYSIS_ROOT, PAPER1_RESULTS_ROOT

SOURCE = PAPER1_ANALYSIS_ROOT / "stage1_inference" / "stage2_candidate_reviews.csv"
STAGE2_SAMPLE = PAPER1_ANALYSIS_ROOT / "stage2_annotation" / "stage2_review_sample_2000.csv"
STAGE1_GOLD_DIR = PAPER1_ANALYSIS_ROOT / "stage1_gold"
STAGE1_PRELABEL = PAPER1_ANALYSIS_ROOT / "stage1_review_sample_balanced_llm.csv"
AUDIT = PAPER1_ANALYSIS_ROOT / "stage1_error_audit" / "stage1_disagreement_audit.csv"

OUT_DIR = PAPER1_ANALYSIS_ROOT / "stage1_hard_negative_mining"
POOL = OUT_DIR / "hard_negative_screening_pool.csv"
SCREENED = OUT_DIR / "hard_negative_screened.csv"
REVIEW = OUT_DIR / "hard_negative_manual_review.csv"
RAW = OUT_DIR / "llm_raw.jsonl"
REPORT = OUT_DIR / "mining_report.json"

MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
SEED = 2026

PROMPT = """
You are an expert annotator of Steam user reviews.

TASK
Determine whether the FULL review contains at least one
ACTIONABLE USER-REPORTED ISSUE.

An actionable issue is any identifiable problem, limitation,
deficiency, dissatisfaction with a specific product aspect,
or improvement request that a developer, publisher, or
product team could potentially address.

IMPORTANT:
An issue does NOT have to be a technical bug.
An issue does NOT have to be severe.
An issue does NOT require an explicit request for a fix.
A positive review can still contain actionable issues.

ISSUE CATEGORIES

1. BUG_STABILITY:
Crashes, freezes, broken mechanics, errors, glitches.

2. PERFORMANCE:
Low FPS, lag, stuttering, slow loading, poor optimization.

3. COMPATIBILITY:
Hardware, OS, controller, device compatibility problems.

4. UI_UX:
Confusing interfaces, inconvenient controls,
poor usability, unclear guidance.

5. MULTIPLAYER_NETWORK:
Connectivity, matchmaking, server or multiplayer issues.

6. GAMEPLAY_BALANCE:
Unfair or unsatisfying mechanics, ineffective character
progression, unreasonable difficulty, poor balancing,
frustrating game systems.

7. AI_SYSTEM:
Poor NPC behavior, pathfinding, enemy AI or system logic.

8. CONTENT:
Weak storytelling, repetitive content, insufficient
content, poorly designed missions, historical inaccuracies,
unsatisfactory writing or narrative design.

9. MOD_SUPPORT:
Insufficient or broken mod support.

10. MONETIZATION:
Expensive DLC, unfair pricing, excessive monetization,
paywalls, questionable commercial practices.

11. PRIVACY_POLICY:
Privacy, accounts, access restrictions, product policies.

12. COMMUNITY_MODERATION:
Unfair moderation, discriminatory treatment,
toxic official communities, poor developer communication.

13. OTHER_ISSUE:
Other identifiable developer-addressable deficiencies.

DECISION RULES

A. Label TRUE if the review identifies at least one
specific problem or deficiency in any category above.

B. Criticism of a concrete game feature or design choice
may be actionable even if the feature works as intended.

C. Mild criticism counts. For example:
"The DLC is too expensive" -> TRUE.

D. Positive reviews can contain issues. For example:
"Great game, but the UI is confusing" -> TRUE.

E. Sarcasm, humor, exaggeration or personal preference
do not automatically make a criticism non-actionable.
Identify the underlying criticism when it is clear.

F. Do NOT require explicit suggestions such as
"developers should fix this".

G. Label FALSE for pure praise, vague dislike, jokes
without an identifiable issue, or descriptions of
normal gameplay challenges without criticizing their
design or implementation.

H. Label FALSE when a player merely encounters an
intended game challenge, such as traffic congestion
in a city-building simulation, without identifying
a deficiency in the game mechanics.

I. Review the entire text. If even ONE actionable
issue exists, label TRUE.

J. Complaints about the developer or publisher count
when they concern an identifiable product, commercial,
policy or community-management practice.

OUTPUT
Return JSON only:

{
  "is_actionable_issue": true,
  "evidence": "exact substring copied from the review"
}

If false, return:
{
  "is_actionable_issue": false,
  "evidence": ""
}

FULL REVIEW:
<review>
{review}
</review>
"""


def strings(series):
    return series.fillna("").astype(str).str.strip()


def load_keys(path, id_column="review_id"):
    if not path.exists():
        return set()
    d = pd.read_csv(path, dtype={id_column: "string"}, low_memory=False)
    if id_column not in d:
        raise ValueError(f"{path} lacks {id_column}")
    return set(strings(d[id_column])) - {""}


def prepare(n):
    if POOL.exists() or SCREENED.exists():
        raise FileExistsError(
            f"Mining files already exist in {OUT_DIR}. "
            "To resume use --screen; do not re-prepare/change the candidate pool."
        )
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not STAGE2_SAMPLE.exists():
        raise FileNotFoundError(STAGE2_SAMPLE)
    if not STAGE1_PRELABEL.exists():
        raise FileNotFoundError(
            f"Original Stage-1 balanced sample not found: {STAGE1_PRELABEL}"
        )

    df = pd.read_csv(SOURCE, dtype={"review_id": "string"}, low_memory=False)
    needed = {"review_id", "game", "review_text", "stage1_score"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{SOURCE} missing columns {needed - set(df.columns)}")
    df["review_id"] = strings(df["review_id"])
    df["review_text"] = strings(df["review_text"])
    df["stage1_score"] = pd.to_numeric(df["stage1_score"], errors="coerce")
    if df["review_id"].duplicated().any():
        raise ValueError("Duplicate review IDs in source")

    exclusion = load_keys(STAGE2_SAMPLE)
    exclusion |= load_keys(STAGE1_PRELABEL)
    exclusion |= load_keys(AUDIT)
    # Also exclude all Stage-1 gold if the gold contains an exported full CSV.
    for p in STAGE1_GOLD_DIR.glob("*.csv"):
        dhead = pd.read_csv(p, nrows=0)
        if "review_id" in dhead.columns:
            exclusion |= load_keys(p)

    before = len(df)
    df = df[
        (~df["review_id"].isin(exclusion))
        & df["review_text"].ne("")
        & df["stage1_score"].ge(0.80)
    ].copy()

    if len(df) < n:
        raise ValueError(f"Only {len(df)} eligible candidates; requested {n}")

    # Across games, sample proportional to remaining counts. Within each game,
    # balance three approximate confidence bands; these are all Stage 1 positives.
    df["score_band"] = pd.cut(
        df["stage1_score"],
        bins=[0.8, 0.95, 0.99, 1.0000001],
        include_lowest=True,
        labels=["0.80-0.95", "0.95-0.99", "0.99-1.00"],
    ).astype(str)
    # Randomized allocation, stratified on game and score band.
    rng = np.random.default_rng(SEED)
    df["_rank"] = rng.random(len(df))
    groups = df.groupby(["game", "score_band"], dropna=False)
    raw = groups.size()
    alloc = np.floor(raw / len(df) * n).astype(int)
    remaining = n - int(alloc.sum())
    residual = (raw / len(df) * n - alloc).sort_values(ascending=False)
    for idx in residual.index[:remaining]:
        alloc.loc[idx] += 1
    pieces = []
    for key, amt in alloc.items():
        if amt > 0:
            pieces.append(
                df[(df["game"] == key[0]) & (df["score_band"] == key[1])]
                .nsmallest(int(amt), "_rank")
            )
    selected = pd.concat(pieces, ignore_index=True)
    selected = selected.sample(frac=1, random_state=SEED).reset_index(drop=True)
    selected.insert(0, "mining_id", [f"HN-{i:04d}" for i in range(1, len(selected) + 1)])
    cols = ["mining_id", "game", "review_id", "review_text", "stage1_score", "score_band"]
    if "voted_up" in selected:
        cols.append("voted_up")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected[cols].to_csv(POOL, index=False, encoding="utf-8-sig")
    print(f"Stage-1 predicted-positive source: {before}")
    print(f"Excluded Stage-1 gold, Stage-2 sample and prior audit IDs: {len(exclusion)} unique")
    print(f"Eligible new candidates with score >= 0.80: {len(df)}")
    print(f"New screening pool: {len(selected)}")
    print(f"Saved: {POOL}")


def parse_response(s, review):
    obj = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.I))
    if not isinstance(obj, dict) or not isinstance(obj.get("is_actionable_issue"), bool):
        raise ValueError("Missing valid boolean is_actionable_issue")
    evidence = obj.get("evidence", "")
    if not isinstance(evidence, str):
        raise ValueError("Evidence must be string")
    if evidence and evidence not in review:
        raise ValueError("Evidence is not an exact review substring")
    if obj["is_actionable_issue"] and not evidence:
        raise ValueError("Positive without evidence")
    if not obj["is_actionable_issue"] and evidence:
        raise ValueError("Negative with evidence")
    return int(obj["is_actionable_issue"]), evidence


async def one(client, semaphore, row):
    review = row["review_text"]
    async with semaphore:
        error = ""
        for attempt in range(1, 4):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": PROMPT.replace("{review}", review)}],
                    temperature=0,
                    timeout=120,
                )
                txt = resp.choices[0].message.content or ""
                label, evidence = parse_response(txt, review)
                return {
                    "mining_id": row["mining_id"], "llm_status": "success",
                    "llm_is_actionable_issue": label, "llm_evidence": evidence,
                    "llm_error": "", "raw_response": txt,
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < 3:
                    await asyncio.sleep(min(2 ** attempt, 8))
        return {
            "mining_id": row["mining_id"], "llm_status": "failed",
            "llm_is_actionable_issue": None, "llm_evidence": "",
            "llm_error": error, "raw_response": "",
        }


def export_review(pool, results, review_path=REVIEW, report_path=REPORT):
    joined = pool.merge(results.drop(columns=["raw_response"], errors="ignore"),
                        on="mining_id", how="left", validate="one_to_one")
    negative = joined[
        joined["llm_status"].eq("success")
        & pd.to_numeric(joined["llm_is_actionable_issue"], errors="coerce").eq(0)
    ].copy()
    negative = negative.sort_values(["stage1_score", "mining_id"], ascending=[False, True])
    negative.insert(0, "audit_id", [f"HNA-{i:04d}" for i in range(1, len(negative) + 1)])
    negative["manual_is_actionable_issue"] = pd.NA
    negative["manual_notes"] = ""
    negative["manual_reviewed"] = 0
    negative.to_csv(review_path, index=False, encoding="utf-8-sig")
    counts = {
        "screening_pool": len(pool),
        "screened": int(results["llm_status"].eq("success").sum()),
        "failed": int(results["llm_status"].eq("failed").sum()),
        "llm_negative_to_review": len(negative),
        "note": "Human labels required. LLM-negative is NOT gold.",
    }
    report_path.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


async def audit_test(args):
    """Re-evaluate ONLY previously human-reviewed records; never overwrite prior files."""
    if args.prompt_version != "v2":
        raise ValueError("--audit-test requires --prompt-version v2")
    if not REVIEW.exists():
        raise FileNotFoundError(f"Original human review CSV missing: {REVIEW}")
    human = pd.read_csv(REVIEW, dtype={"review_id": "string"}, low_memory=False)
    required = {"mining_id", "review_text", "manual_reviewed", "manual_is_actionable_issue"}
    if not required.issubset(human.columns):
        raise ValueError(f"Human review CSV missing fields: {required - set(human.columns)}")
    flag = pd.to_numeric(human["manual_reviewed"], errors="coerce").eq(1)
    labels = pd.to_numeric(human["manual_is_actionable_issue"], errors="coerce")
    target = human[flag & labels.isin([0, 1])].copy()
    if args.limit is not None:
        target = target.head(args.limit)
    if target.empty:
        raise ValueError("No completed (manual_reviewed=1 and label 0/1) records found")
    if target["mining_id"].duplicated().any():
        raise ValueError("Duplicate mining IDs in manual audit")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / "audit_test_prompt_v2.csv"
    response_cols = ["mining_id", "llm_status", "llm_is_actionable_issue", "llm_evidence", "llm_error", "raw_response"]
    if result_path.exists():
        old = pd.read_csv(result_path, low_memory=False)
        # Repair the previous v2 run: merge of old LLM columns with new LLM
        # columns created suffixes (_x = v1, _y = v2).
        for col in response_cols[1:]:
            if col not in old.columns and f"{col}_y" in old.columns:
                old[col] = old[f"{col}_y"]
        old = old[response_cols].copy()
        completed = set(old.loc[old["llm_status"].eq("success"), "mining_id"].astype(str))
    else:
        old = pd.DataFrame()
        completed = set()
    todo = target[~target["mining_id"].isin(completed)]
    print(f"Human-labelled test set: {len(target)} | completed v2: {len(completed)} | pending: {len(todo)}")
    if len(todo):
        from openai import AsyncOpenAI
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing in current PowerShell session")
        client = AsyncOpenAI(api_key=key, base_url=BASE_URL)
        try:
            sem = asyncio.Semaphore(args.concurrency)
            for start in range(0, len(todo), args.batch_size):
                batch = todo.iloc[start:start + args.batch_size]
                response = await asyncio.gather(*[one(client, sem, r) for r in batch.to_dict("records")])
                old = pd.concat([old, pd.DataFrame(response)], ignore_index=True)
                old = old.drop_duplicates("mining_id", keep="last")
                # Checkpoint includes human labels for inspection, but no source is mutated.
                # Target contains old v1 LLM fields. Remove them before
                # joining v2 responses, otherwise pandas creates _x/_y names.
                clean_target = target.drop(columns=[c for c in response_cols[1:] if c in target.columns])
                joined = clean_target.merge(old, on="mining_id", how="left", validate="one_to_one")
                joined.to_csv(result_path, index=False, encoding="utf-8-sig")
                print(f"Processed {min(start + len(batch), len(todo))}/{len(todo)}")
        finally:
            await client.close()
    # Normalize an already saved checkpoint too (no extra API calls).
    clean_target = target.drop(columns=[c for c in response_cols[1:] if c in target.columns])
    final = clean_target.merge(old[response_cols], on="mining_id", how="left", validate="one_to_one")
    final.to_csv(result_path, index=False, encoding="utf-8-sig")
    ok = final["llm_status"].eq("success")
    truth = pd.to_numeric(final["manual_is_actionable_issue"], errors="coerce")
    pred = pd.to_numeric(final["llm_is_actionable_issue"], errors="coerce")
    correct = int((truth[ok] == pred[ok]).sum())
    print(f"v2 agreement with human: {correct}/{int(ok.sum())} (descriptive, selected v1 negatives)")
    print("v2 positive / negative:", int((pred[ok] == 1).sum()), "/", int((pred[ok] == 0).sum()))
    print("Failures:", int((~ok).sum()))
    print("Saved:", result_path)


async def screen(args):
    if not POOL.exists():
        raise FileNotFoundError(f"Run --prepare first. Missing {POOL}")
    from openai import AsyncOpenAI
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing in current PowerShell session.")
    pool = pd.read_csv(POOL, dtype={"review_id": "string"}, low_memory=False)
    if args.prompt_version == "v1":
        screened_path, raw_path, review_path, report_path = SCREENED, RAW, REVIEW, REPORT
    else:
        screened_path = OUT_DIR / f"hard_negative_screened_prompt_{args.prompt_version}.csv"
        raw_path = OUT_DIR / f"llm_raw_prompt_{args.prompt_version}.jsonl"
        review_path = OUT_DIR / f"hard_negative_manual_review_prompt_{args.prompt_version}.csv"
        report_path = OUT_DIR / f"mining_report_prompt_{args.prompt_version}.json"
    if screened_path.exists():
        results = pd.read_csv(screened_path, low_memory=False)
        if results["mining_id"].duplicated().any():
            raise ValueError("Duplicate mining IDs in checkpoint")
    else:
        results = pd.DataFrame(columns=["mining_id", "llm_status",
                                         "llm_is_actionable_issue", "llm_evidence",
                                         "llm_error", "raw_response"])
    done = set(results.loc[results["llm_status"].eq("success"), "mining_id"].astype(str))
    pending = pool[~pool["mining_id"].isin(done)].copy()
    if args.limit is not None:
        pending = pending.head(args.limit)
    print(f"Pool: {len(pool)} | completed: {len(done)} | pending this run: {len(pending)}")
    sem = asyncio.Semaphore(args.concurrency)
    client = AsyncOpenAI(api_key=key, base_url=BASE_URL)
    try:
        for start in range(0, len(pending), args.batch_size):
            batch = pending.iloc[start:start + args.batch_size]
            fresh = await asyncio.gather(*[
                one(client, sem, row)
                for row in batch.to_dict("records")
            ])
            with raw_path.open("a", encoding="utf-8") as f:
                for entry in fresh:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            results = pd.concat([results, pd.DataFrame(fresh)], ignore_index=True)
            results = results.drop_duplicates("mining_id", keep="last")
            # Write all fields as strings to avoid pandas float-vs-str resume errors.
            results.to_csv(screened_path, index=False, encoding="utf-8-sig")
            print(f"Processed {min(start+len(batch),len(pending))}/{len(pending)} | "
                  f"total success={results['llm_status'].eq('success').sum()}")
    finally:
        await client.close()
    report = export_review(pool, results, review_path, report_path)
    print("Report:", report)
    print("Human-review candidates:", review_path)
    print("Checkpoint:", screened_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    act = parser.add_mutually_exclusive_group(required=True)
    act.add_argument("--prepare", action="store_true")
    act.add_argument("--screen", action="store_true")
    act.add_argument("--audit-test", action="store_true")
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--pool-size", type=int, default=650)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()
    if args.pool_size <= 0 or args.batch_size <= 0 or args.concurrency <= 0:
        parser.error("Positive pool-size, batch-size and concurrency required")
    if args.prepare:
        prepare(args.pool_size)
    elif args.audit_test:
        asyncio.run(audit_test(args))
    else:
        if args.prompt_version == "v1":
            parser.error("Existing v1 screening is preserved; use --prompt-version v2")
        asyncio.run(screen(args))
