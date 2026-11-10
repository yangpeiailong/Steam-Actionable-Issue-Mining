# -*- coding: utf-8 -*-
"""
Stage-2 teacher annotation for an unlabeled distillation candidate pool.

This script:
- Reads stage2_distillation_candidate_pool_v1.csv.
- Uses DeepSeek to label actionable issue domains with evidence and per-label
  confidence scores.
- Saves resumable annotations separately from all Gold data.
- Does NOT modify Gold labels or train a student model.

Run from Steam_Research_Code:
  python paper1_issue_mining/04_stage2_classification/14_teacher_label_stage2_distillation.py --limit 5

After checking the first five:
  python paper1_issue_mining/04_stage2_classification/14_teacher_label_stage2_distillation.py --limit 50
  python paper1_issue_mining/04_stage2_classification/14_teacher_label_stage2_distillation.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from config_local import PAPER1_ANALYSIS_ROOT  # noqa: E402

BASE_DIR = PAPER1_ANALYSIS_ROOT / "stage2_annotation" / "distillation_candidates_v2_expansion"
SOURCE = BASE_DIR / "stage2_distillation_candidate_pool_v2_expansion_1000.csv"
OUTPUT = BASE_DIR / "stage2_distillation_teacher_annotations_v2_expansion_1000.csv"
RAW_LOG = BASE_DIR / "stage2_distillation_teacher_raw_v2_expansion_1000.jsonl"
FAILURES = BASE_DIR / "stage2_distillation_teacher_failures_v2_expansion_1000.csv"
MANIFEST = BASE_DIR / "stage2_distillation_teacher_manifest_v2_expansion_1000.json"

MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
PROMPT_VERSION = "teacher_v1_rules_confirmed_20261029_expansion1000"

ISSUE_LABELS = [
    "BUG_STABILITY",
    "PERFORMANCE",
    "COMPATIBILITY",
    "UI_UX",
    "MULTIPLAYER_NETWORK",
    "GAMEPLAY_BALANCE",
    "AI_SYSTEM",
    "CONTENT",
    "MOD_SUPPORT",
    "MONETIZATION",
    "PRIVACY_POLICY",
    "COMMUNITY_MODERATION",
    "OTHER_ISSUE",
]
ALL_LABELS = ISSUE_LABELS + ["NO_ISSUE"]

LABEL_DEFINITIONS = """
LABEL DEFINITIONS (assign all that genuinely apply; multi-label is allowed):
- BUG_STABILITY: crashes, freezes, bugs, errors, glitches, broken features or mechanics.
- PERFORMANCE: low FPS, lag, stuttering, frame drops, slow loading, poor optimization, excessive resource use.
- COMPATIBILITY: a problem or explicit missing support tied to an OS, hardware, device, controller, or platform.
  A platform-specific performance problem may receive both COMPATIBILITY and PERFORMANCE.
  A general performance complaint without a platform-specific link is PERFORMANCE only.
- UI_UX: confusing or inconvenient interface, controls, navigation, usability, unclear guidance/tutorials.
- MULTIPLAYER_NETWORK: multiplayer connectivity, servers, matchmaking, disconnects, network latency or online-service failures.
  Merely requesting cross-platform play without a network failure is COMPATIBILITY, not automatically MULTIPLAYER_NETWORK.
- GAMEPLAY_BALANCE: concrete issues with difficulty, balancing, progression, unfair mechanics, grind, or frustrating game systems.
- AI_SYSTEM: concrete problems with NPC/enemy/companion AI, behavior, decision logic or pathfinding.
- CONTENT: concrete issues with story, writing, missions, maps, historical accuracy, repetitive/insufficient content or narrative design.
- MOD_SUPPORT: problems with mod installation, management, support, loading, workshop tools, or explicit mod compatibility/conflicts.
  Mentioning or recommending a mod alone is NOT enough. If mods merely provide context for a performance problem,
  label PERFORMANCE but do not automatically label MOD_SUPPORT.
- MONETIZATION: explicit criticism of price, paid content, DLC paywalls, microtransactions, monetization or unfair payment design.
  A bare suggestion to discount a game without clear price/payment criticism is not enough.
- PRIVACY_POLICY: an expressed concern/criticism about privacy, telemetry, tracking, data collection/sharing, surveillance or consent.
  Neutral listing of data collected, without concern, criticism, risk or improvement request, is NOT enough.
- COMMUNITY_MODERATION: toxicity, harassment, abusive community behavior, bans, reporting or inadequate moderation.
- OTHER_ISSUE: a concrete actionable issue that does not fit any other issue category. Do not use as a catch-all for vague dislike.
- NO_ISSUE: no actionable issue is expressed. Use this only when none of the 13 issue labels applies.
  NO_ISSUE is mutually exclusive with all issue labels.
"""

PROMPT = r"""
You are an expert annotator of FULL Steam game reviews. Label concrete, actionable
issues expressed in the review, not merely topics mentioned.

{label_definitions}

GENERAL RULES
1. Read the entire review and preserve context, negation, irony, and contrast.
2. A review may express several distinct issues; assign multiple issue labels when justified.
3. Positive reviews can contain actionable issues; negative sentiment alone is not enough.
4. Do not infer an issue just because a related word appears. A topic mention, recommendation,
   joke, neutral description, or praise does not establish a problem.
5. Use exact evidence substrings copied verbatim from the review for every selected issue label.
   Do not paraphrase evidence. Each evidence string must be a contiguous substring.
6. The agreed boundary rules above are mandatory, especially for COMPATIBILITY,
   MULTIPLAYER_NETWORK, MOD_SUPPORT, MONETIZATION, and PRIVACY_POLICY.
7. Do not force a rare category to be present. If the review does not express that issue, leave it out.
8. Confidence scores are your subjective evidence-strength estimates, NOT statistically calibrated probabilities.
   Use 0.00 for no evidence, 0.10-0.39 for weak/ambiguous evidence, 0.40-0.69 for plausible but uncertain,
   and 0.70-1.00 for clear evidence. The explicit `domains` list is your final label decision.
9. If `domains` contains NO_ISSUE, it must be the only label. If any actionable issue is present,
   do not include NO_ISSUE.
10. Flag `needs_human_review=true` when the case is genuinely ambiguous or the boundary rules do not settle it.
    Do not flag routine clear cases just because they contain several labels.

Return one valid JSON object and no Markdown. Required schema:
{
  "domains": ["PERFORMANCE"],
  "domain_scores": {
    "BUG_STABILITY": 0.0,
    "PERFORMANCE": 0.95,
    "COMPATIBILITY": 0.0,
    "UI_UX": 0.0,
    "MULTIPLAYER_NETWORK": 0.0,
    "GAMEPLAY_BALANCE": 0.0,
    "AI_SYSTEM": 0.0,
    "CONTENT": 0.0,
    "MOD_SUPPORT": 0.0,
    "MONETIZATION": 0.0,
    "PRIVACY_POLICY": 0.0,
    "COMMUNITY_MODERATION": 0.0,
    "OTHER_ISSUE": 0.0,
    "NO_ISSUE": 0.0
  },
  "evidence": {
    "PERFORMANCE": ["exact substring from review"]
  },
  "rationales": {
    "PERFORMANCE": "Short reason tied to the evidence."
  },
  "overall_confidence": 0.95,
  "needs_human_review": false,
  "human_review_reason": ""
}

For every label in `domains` except NO_ISSUE, provide at least one exact evidence substring and a concise rationale.
Evidence and rationale keys should only be provided for selected issue labels. For NO_ISSUE, evidence and rationales must be empty objects.
`domain_scores` must contain every label exactly once, with numeric values from 0 to 1.
Use NO_ISSUE as the only domain when no actionable issue is present. Set its score high in that case.
Do not use the candidate-retrieval keywords or candidate group as labels. They are retrieval hints only.

REVIEW:
<review>
{review}
</review>
"""


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    return value


def parse_json_response(content: str) -> dict:
    content = (content or "").strip()
    content = re.sub(r"^\s*```(?:json)?\s*", "", content, flags=re.I)
    content = re.sub(r"\s*```\s*$", "", content)
    obj = json.loads(content)
    if not isinstance(obj, dict):
        raise ValueError("Response must be a JSON object")
    return obj


def validate_response(obj: dict, review: str) -> dict:
    domains = obj.get("domains")
    scores = obj.get("domain_scores")
    evidence = obj.get("evidence")
    rationales = obj.get("rationales")
    if not isinstance(domains, list) or not domains:
        raise ValueError("domains must be a non-empty list")
    if any(not isinstance(x, str) or x not in ALL_LABELS for x in domains):
        raise ValueError("domains contains an unknown label")
    if len(domains) != len(set(domains)):
        raise ValueError("domains contains duplicate labels")
    if "NO_ISSUE" in domains and len(domains) != 1:
        raise ValueError("NO_ISSUE cannot co-occur with issue labels")
    if not isinstance(scores, dict) or set(scores) != set(ALL_LABELS):
        raise ValueError("domain_scores must contain every allowed label exactly once")
    clean_scores = {}
    for label in ALL_LABELS:
        val = scores[label]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not 0 <= float(val) <= 1:
            raise ValueError(f"Invalid score for {label}: expected number in [0,1]")
        clean_scores[label] = round(float(val), 4)
    if not isinstance(evidence, dict) or not isinstance(rationales, dict):
        raise ValueError("evidence and rationales must be objects")

    expected_issue_keys = set(domains) - {"NO_ISSUE"}
    if set(evidence) != expected_issue_keys:
        raise ValueError("evidence keys must match selected issue labels exactly")
    if set(rationales) != expected_issue_keys:
        raise ValueError("rationale keys must match selected issue labels exactly")

    clean_evidence = {}
    clean_rationales = {}
    for label in expected_issue_keys:
        spans = evidence[label]
        if isinstance(spans, str):
            spans = [spans]
        if not isinstance(spans, list) or not spans:
            raise ValueError(f"{label} must have at least one evidence substring")
        spans_clean = []
        for span in spans:
            if not isinstance(span, str) or not span.strip():
                raise ValueError(f"{label} evidence must be non-empty strings")
            if span not in review:
                raise ValueError(f"{label} evidence is not an exact substring of the review: {span[:100]!r}")
            spans_clean.append(span)
        clean_evidence[label] = spans_clean
        rationale = rationales[label]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"{label} rationale must be a non-empty string")
        clean_rationales[label] = rationale.strip()

    if "NO_ISSUE" in domains:
        if evidence or rationales:
            raise ValueError("NO_ISSUE response must have empty evidence/rationales")
        if clean_scores["NO_ISSUE"] < 0.5:
            raise ValueError("NO_ISSUE selected with score below 0.5")
    else:
        if clean_scores["NO_ISSUE"] >= 0.8:
            raise ValueError("Issue labels selected but NO_ISSUE score is contradictory (>=0.8)")

    overall = obj.get("overall_confidence")
    if isinstance(overall, bool) or not isinstance(overall, (int, float)) or not 0 <= float(overall) <= 1:
        raise ValueError("overall_confidence must be numeric in [0,1]")
    needs_review = obj.get("needs_human_review")
    review_reason = obj.get("human_review_reason", "")
    if not isinstance(needs_review, bool) or not isinstance(review_reason, str):
        raise ValueError("needs_human_review must be boolean and human_review_reason a string")
    if needs_review and not review_reason.strip():
        raise ValueError("human_review_reason is required when needs_human_review=true")

    return {
        "domains": domains,
        "domain_scores": clean_scores,
        "evidence": clean_evidence,
        "rationales": clean_rationales,
        "overall_confidence": round(float(overall), 4),
        "needs_human_review": needs_review,
        "human_review_reason": review_reason.strip(),
    }


def annotation_columns() -> list[str]:
    cols = [
        "teacher_domains", "teacher_domain_scores_json", "teacher_evidence_json",
        "teacher_rationales_json", "teacher_overall_confidence",
        "teacher_needs_human_review", "teacher_human_review_reason",
        "teacher_prompt_version", "teacher_model", "teacher_status", "teacher_error",
        "teacher_attempts",
    ]
    cols += [f"teacher_score_{label}" for label in ALL_LABELS]
    cols += [f"teacher_evidence_{label}" for label in ISSUE_LABELS]
    return cols


def to_result(review_id: str, parsed: dict, attempts: int) -> dict:
    out = {
        "review_id": review_id,
        "teacher_domains": "|".join(parsed["domains"]),
        "teacher_domain_scores_json": json.dumps(parsed["domain_scores"], ensure_ascii=False),
        "teacher_evidence_json": json.dumps(parsed["evidence"], ensure_ascii=False),
        "teacher_rationales_json": json.dumps(parsed["rationales"], ensure_ascii=False),
        "teacher_overall_confidence": parsed["overall_confidence"],
        "teacher_needs_human_review": int(parsed["needs_human_review"]),
        "teacher_human_review_reason": parsed["human_review_reason"],
        "teacher_prompt_version": PROMPT_VERSION,
        "teacher_model": MODEL,
        "teacher_status": "success",
        "teacher_error": "",
        "teacher_attempts": attempts,
    }
    for label in ALL_LABELS:
        out[f"teacher_score_{label}"] = parsed["domain_scores"][label]
    for label in ISSUE_LABELS:
        out[f"teacher_evidence_{label}"] = json.dumps(parsed["evidence"].get(label, []), ensure_ascii=False)
    return out


async def annotate_one(client, semaphore: asyncio.Semaphore, row: dict, retries: int) -> tuple[dict, dict]:
    review_id = normalize_id(row["review_id"])
    review = str(row["review_text"])
    prompt = PROMPT.replace("{label_definitions}", LABEL_DEFINITIONS).replace("{review}", review)
    last_error = ""
    raw_content = ""
    attempts = 0
    async with semaphore:
        for attempt in range(1, retries + 1):
            attempts = attempt
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                raw_content = response.choices[0].message.content or ""
                parsed = validate_response(parse_json_response(raw_content), review)
                result = to_result(review_id, parsed, attempts)
                raw_record = {
                    "review_id": review_id, "status": "success", "attempts": attempts,
                    "model": MODEL, "prompt_version": PROMPT_VERSION,
                    "parsed": parsed, "raw": raw_content,
                }
                return result, raw_record
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:800]}"
                if attempt < retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 8) + random.random() * 0.25)

    failed = {c: "" for c in annotation_columns()}
    failed.update({
        "review_id": review_id,
        "teacher_domains": "",
        "teacher_domain_scores_json": "",
        "teacher_evidence_json": "",
        "teacher_rationales_json": "",
        "teacher_overall_confidence": None,
        "teacher_needs_human_review": None,
        "teacher_human_review_reason": "",
        "teacher_prompt_version": PROMPT_VERSION,
        "teacher_model": MODEL,
        "teacher_status": "failed",
        "teacher_error": last_error,
        "teacher_attempts": attempts,
    })
    for label in ALL_LABELS:
        failed[f"teacher_score_{label}"] = None
    for label in ISSUE_LABELS:
        failed[f"teacher_evidence_{label}"] = ""
    raw_record = {
        "review_id": review_id, "status": "failed", "attempts": attempts,
        "model": MODEL, "prompt_version": PROMPT_VERSION,
        "error": last_error, "raw": raw_content,
    }
    return failed, raw_record


def atomic_save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def load_source() -> pd.DataFrame:
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Candidate pool not found: {SOURCE}\n"
            "Copy the generated stage2_distillation_candidate_pool_v1.csv into this directory first."
        )
    df = pd.read_csv(SOURCE, dtype={"review_id": "string"}, low_memory=False)
    required = {"review_id", "game", "review_text", "gold_or_test_data"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidate source is missing columns: {sorted(missing)}")
    df["review_id"] = df["review_id"].map(normalize_id)
    df["review_text"] = df["review_text"].fillna("").astype(str)
    if df["review_id"].eq("").any() or df["review_id"].duplicated().any():
        raise ValueError("review_id must be non-empty and unique in candidate pool")
    if df["review_text"].str.strip().eq("").any():
        raise ValueError("Candidate pool contains empty review_text")
    if df["gold_or_test_data"].astype(str).str.lower().isin(["true", "1", "yes"]).any():
        raise ValueError("Gold/test rows detected in candidate pool; refusing to annotate")
    return df


async def run(args) -> None:
    from openai import AsyncOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing in this PowerShell session.")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    source = load_source()
    ann_cols = annotation_columns()
    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT, dtype={"review_id": "string"}, low_memory=False)
        if existing["review_id"].duplicated().any():
            raise ValueError("Duplicate review_id in existing output; inspect before resuming.")
        unknown = set(existing["review_id"].map(normalize_id)) - set(source["review_id"])
        if unknown:
            raise ValueError(f"Existing output contains {len(unknown)} IDs not in source; refusing resume.")
        old_text = existing.set_index("review_id")["review_text"].astype(str).to_dict()
        for _, row in source[source["review_id"].isin(old_text)].iterrows():
            if old_text[row["review_id"]] != str(row["review_text"]):
                raise ValueError(f"Review text changed for review_id={row['review_id']}; refusing resume.")
        existing = existing.set_index("review_id")
        data = source.copy()
        for col in ann_cols:
            if col in existing.columns:
                mapping = existing[col].to_dict()
                # Existing CSV columns may be inferred as float64 when their
                # first completed rows contain only missing values. Force object
                # dtype so later string/number results can be assigned safely.
                data[col] = data["review_id"].map(mapping).astype("object")
            else:
                data[col] = pd.Series(pd.NA, index=data.index, dtype="object")
    else:
        data = source.copy()
        for col in ann_cols:
            data[col] = pd.Series(pd.NA, index=data.index, dtype="object")

    done = set(data.loc[data["teacher_status"].eq("success"), "review_id"].astype(str))
    pending = source[~source["review_id"].isin(done)].copy()
    if args.limit is not None:
        pending = pending.head(args.limit)

    print(f"Candidate pool: {len(source)} | completed: {len(done)} | pending this run: {len(pending)}")
    print(f"Model: {MODEL} | batch size: {args.batch_size} | concurrency: {args.concurrency}")
    print(f"Output: {OUTPUT}")
    if pending.empty:
        print("No pending rows. Existing annotations are complete for the requested pool.")
        return

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=0,
    )
    try:
        for start in range(0, len(pending), args.batch_size):
            batch = pending.iloc[start:start + args.batch_size]
            pairs = await asyncio.gather(*[
                annotate_one(client, sem, row, args.retries)
                for row in batch.to_dict("records")
            ])
            results = [p[0] for p in pairs]
            raw_records = [p[1] for p in pairs]
            with RAW_LOG.open("a", encoding="utf-8") as f:
                for record in raw_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Defensive dtype normalization for resumable annotation columns.
            # This prevents pandas' strict upcast error when a previously empty
            # column (e.g. a textual rationale) was inferred as float64.
            for col in ann_cols:
                if col in data.columns and data[col].dtype != "object":
                    data[col] = data[col].astype("object")

            data = data.set_index("review_id", drop=False)
            for result in results:
                rid = result["review_id"]
                for col in ann_cols:
                    if col in result:
                        data.at[rid, col] = result[col]
            data = data.reset_index(drop=True)
            atomic_save(data, OUTPUT)

            failures = data.loc[data["teacher_status"].eq("failed"),
                                ["review_id", "game", "teacher_error"]].copy()
            atomic_save(failures, FAILURES)

            success_n = int(data["teacher_status"].eq("success").sum())
            fail_n = int(data["teacher_status"].eq("failed").sum())
            print(
                f"Processed {min(start + len(batch), len(pending))}/{len(pending)} this run "
                f"| total success={success_n} | failed={fail_n}",
                flush=True,
            )
    finally:
        await client.close()

    counts = data.loc[data["teacher_status"].eq("success"), "teacher_domains"].fillna("").str.split("|").explode().value_counts().to_dict()
    manifest = {
        "purpose": "Teacher annotation for candidate generation and distillation research; not Gold data",
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "raw_log": str(RAW_LOG),
        "model": MODEL,
        "base_url": args.base_url,
        "prompt_version": PROMPT_VERSION,
        "source_rows": int(len(source)),
        "completed_success": int(data["teacher_status"].eq("success").sum()),
        "failed": int(data["teacher_status"].eq("failed").sum()),
        "teacher_domain_counts": counts,
        "gold_modified": False,
        "note": "Teacher confidence scores are subjective, not calibrated probabilities. Verify teacher labels before using them for student distillation. Validation-error cases must remain diagnostic-only."
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nTEACHER ANNOTATION RUN COMPLETE")
    print("Saved:", OUTPUT)
    print("Raw responses:", RAW_LOG)
    print("Failures:", FAILURES)
    print("Manifest:", MANIFEST)
    print("Teacher label counts:")
    print(pd.Series(counts).sort_values(ascending=False).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek teacher labels for Stage-2 distillation candidates")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N pending reviews")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.batch_size < 1 or args.concurrency < 1 or args.retries < 1 or args.timeout < 1:
        parser.error("batch-size, concurrency, retries, and timeout must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
