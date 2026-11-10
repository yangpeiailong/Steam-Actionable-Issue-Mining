"""
Stage 1 LLM screening with quota-based sampling.

Goal:
    Build a roughly balanced HUMAN-REVIEW dataset for the binary task:
        Full Steam review -> contains at least one actionable issue? (0/1)

Sampling logic:
- Randomly shuffle all usable reviews with a fixed seed.
- Send reviews to DeepSeek Flash in batches.
- Keep LLM-positive reviews until POSITIVE_TARGET is reached.
- Keep LLM-negative reviews until NEGATIVE_TARGET is reached.
- Once a quota is full, additional reviews of that class are still recorded
  in the screening log but are NOT added to the selected annotation set.
- Stop when both quotas are reached.

Inputs:
    data/analysis/all_reviews.csv

Outputs:
    data/analysis/stage1_review_sample_balanced_llm.csv
        -> selected quota-based sample for human review

    results/llm_stage1/stage1_screened_pool.csv
        -> every review queried by the LLM, including unselected overflow rows

    results/llm_stage1/deepseek_flash_stage1_raw.jsonl
        -> raw API responses

    results/llm_stage1/deepseek_flash_stage1_failures.csv
        -> non-success rows

Important:
- Only review_text is sent to the LLM.
- voted_up, game, date, and all metadata are NOT sent.
- This selected sample is for TRAINING / annotation efficiency.
- It must NOT be used to estimate the natural prevalence of actionable issues.
"""

from pathlib import Path
import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

import pandas as pd
from langchain_openai import ChatOpenAI
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio


try:
    from openai import (
        APIError,
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        AuthenticationError,
        BadRequestError,
    )

    OPENAI_API_EXCEPTIONS = (
        APIError,
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        AuthenticationError,
        BadRequestError,
    )

except Exception:
    OPENAI_API_EXCEPTIONS = tuple()


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

SELECTED_FILE = (
    PAPER_ROOT
    / "data"
    / "analysis"
    / "stage1_review_sample_balanced_llm.csv"
)

RUN_DIR = (
    PAPER_ROOT
    / "results"
    / "llm_stage1"
)

SCREENED_FILE = (
    RUN_DIR
    / "stage1_screened_pool.csv"
)

RAW_FILE = (
    RUN_DIR
    / "deepseek_flash_stage1_raw.jsonl"
)

FAILURE_FILE = (
    RUN_DIR
    / "deepseek_flash_stage1_failures.csv"
)


# =========================================================
# Runtime configuration
# =========================================================

MODEL_NAME = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"

ASYNC_TIMEOUT = 300
RETRY_ATTEMPTS = 3

DEFAULT_CONCURRENCY = 5
DEFAULT_BATCH_SIZE = 100

RANDOM_SEED = 2026

POSITIVE_TARGET = 1000
NEGATIVE_TARGET = 1000


# =========================================================
# Prompt
# =========================================================

PROMPT_TEMPLATE = r"""
You are an expert annotator of Steam user reviews for software-quality research.

Your task is to decide whether the FULL REVIEW contains at least one ACTIONABLE USER-REPORTED ISSUE.

DEFINITION
An actionable issue is a concrete problem, defect, deficiency, request, or concern that a developer or product team could understand and potentially act on.

Examples include:
- crashes, bugs, freezes, broken features
- poor performance, lag, stuttering
- compatibility or controller problems
- UI/UX or control problems
- multiplayer/network problems
- gameplay balance or progression problems
- problematic AI/system behavior
- insufficient, repetitive, or missing content
- mod-support problems
- monetization/pricing/DLC concerns
- privacy/account/policy concerns
- community/moderation problems
- explicit feature requests or improvement requests

BOUNDARY RULES
1. Judge ONLY from the review text.
2. A positive review may still contain an actionable issue.
3. A negative review is NOT automatically actionable.
4. General dislike, preference, sarcasm, praise, or vague dissatisfaction is not enough unless a concrete issue is stated.
5. A comparison to another game/version is not actionable unless the review clearly identifies a current deficiency/problem/request.
6. A feature wish/request counts as actionable only when the requested change is explicit enough for a developer to understand.
7. If the review contains both praise and a concrete issue, label it actionable.
8. If there are several issues, you still return only the binary Stage-1 decision.
9. Evidence must be copied exactly from the review as one or more continuous substrings.
10. If there is no actionable issue, evidence must be [].
11. Return JSON only. Do not output markdown, reasoning, or extra text.

OUTPUT SCHEMA
{
  "is_actionable_issue": true or false,
  "evidence_spans": ["exact substring 1", "exact substring 2"],
  "confidence": 0.0
}

CONFIDENCE
Use a number from 0.0 to 1.0 reflecting certainty in the binary label.

FULL REVIEW
__REVIEW__
""".strip()


# =========================================================
# Helpers
# =========================================================

def build_llm():

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            'Missing DEEPSEEK_API_KEY. Run:\n'
            '$env:DEEPSEEK_API_KEY="your_key"'
        )

    print(
        f"Model: {MODEL_NAME} | base_url={BASE_URL}"
    )

    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=0.0,
        timeout=ASYNC_TIMEOUT,
    )


def build_prompt(review):

    return PROMPT_TEMPLATE.replace(
        "__REVIEW__",
        review
    )


def strip_code_fences(text):

    text = str(text).strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


def parse_json(text):

    return json.loads(
        strip_code_fences(text)
    )


def validate_annotation(
    obj: Dict[str, Any],
    review: str,
) -> List[str]:

    errors = []

    if not isinstance(obj, dict):
        return ["top_level_not_object"]

    label = obj.get(
        "is_actionable_issue"
    )

    evidence = obj.get(
        "evidence_spans"
    )

    confidence = obj.get(
        "confidence"
    )

    if not isinstance(label, bool):
        errors.append(
            "is_actionable_issue_not_boolean"
        )

    if not isinstance(evidence, list):

        errors.append(
            "evidence_spans_not_list"
        )

        evidence = []

    for span in evidence:

        if (
            not isinstance(span, str)
            or not span.strip()
        ):
            errors.append(
                "empty_evidence_span"
            )

        elif span not in review:

            errors.append(
                f"evidence_not_exact:{span}"
            )

    if isinstance(label, bool):

        if (
            label
            and len(evidence) == 0
        ):
            errors.append(
                "actionable_without_evidence"
            )

        if (
            not label
            and len(evidence) > 0
        ):
            errors.append(
                "nonactionable_with_evidence"
            )

    if not isinstance(
        confidence,
        (int, float)
    ):

        errors.append(
            "confidence_not_numeric"
        )

    else:

        if not (
            0.0
            <= float(confidence)
            <= 1.0
        ):
            errors.append(
                "confidence_out_of_range"
            )

    return errors


def classify_exception(exc):

    if isinstance(
        exc,
        asyncio.TimeoutError
    ):
        return "timeout"

    if isinstance(
        exc,
        json.JSONDecodeError
    ):
        return "parse_error"

    if (
        isinstance(exc, ValueError)
        and str(exc).startswith(
            "VALIDATION_ERROR"
        )
    ):
        return "validation_error"

    if (
        OPENAI_API_EXCEPTIONS
        and isinstance(
            exc,
            OPENAI_API_EXCEPTIONS
        )
    ):
        return "api_error"

    return "local_error"


async def annotate_one(
    llm,
    semaphore,
    screening_order,
    review_id,
    review,
):

    start = time.time()

    raw_response = ""
    last_error = ""
    last_status = "local_error"
    attempts_used = 0

    async with semaphore:

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(
                RETRY_ATTEMPTS
            ),
            wait=wait_exponential(
                multiplier=1,
                min=1,
                max=8,
            ),
            reraise=False,
        ):

            attempts_used = (
                attempt
                .retry_state
                .attempt_number
            )

            with attempt:

                try:

                    prompt = build_prompt(
                        review
                    )

                    response = (
                        await asyncio.wait_for(
                            llm.ainvoke(
                                prompt
                            ),
                            timeout=ASYNC_TIMEOUT,
                        )
                    )

                    raw_response = (
                        response.content
                        or ""
                    )

                    annotation = (
                        parse_json(
                            raw_response
                        )
                    )

                    validation_errors = (
                        validate_annotation(
                            annotation,
                            review,
                        )
                    )

                    if validation_errors:

                        raise ValueError(
                            "VALIDATION_ERROR | "
                            + " | ".join(
                                validation_errors
                            )
                        )

                    return {
                        "screening_order":
                            screening_order,

                        "review_id":
                            review_id,

                        "annotation_status":
                            "success",

                        "annotation":
                            annotation,

                        "raw_response":
                            raw_response,

                        "elapsed_seconds":
                            round(
                                time.time()
                                - start,
                                3,
                            ),

                        "attempts_used":
                            attempts_used,

                        "error_type":
                            "",

                        "error":
                            "",
                    }

                except Exception as exc:

                    last_status = (
                        classify_exception(
                            exc
                        )
                    )

                    last_error = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    if (
                        attempts_used
                        >= RETRY_ATTEMPTS
                    ):
                        break

                    raise

    return {
        "screening_order":
            screening_order,

        "review_id":
            review_id,

        "annotation_status":
            last_status,

        "annotation":
            None,

        "raw_response":
            raw_response,

        "elapsed_seconds":
            round(
                time.time()
                - start,
                3,
            ),

        "attempts_used":
            attempts_used,

        "error_type":
            last_status,

        "error":
            last_error,
    }


def flatten_result(
    row,
    result,
):

    out = row.to_dict()

    out[
        "screening_order"
    ] = result[
        "screening_order"
    ]

    out[
        "llm_model"
    ] = MODEL_NAME

    out[
        "llm_annotation_status"
    ] = result[
        "annotation_status"
    ]

    out[
        "llm_is_actionable_issue"
    ] = pd.NA

    out[
        "llm_evidence"
    ] = ""

    out[
        "llm_confidence"
    ] = pd.NA

    out[
        "llm_elapsed_seconds"
    ] = result[
        "elapsed_seconds"
    ]

    out[
        "llm_attempts_used"
    ] = result[
        "attempts_used"
    ]

    out[
        "llm_error_type"
    ] = result[
        "error_type"
    ]

    out[
        "llm_error"
    ] = result[
        "error"
    ]

    if (
        result[
            "annotation_status"
        ]
        != "success"
    ):
        return out

    ann = result[
        "annotation"
    ]

    out[
        "llm_is_actionable_issue"
    ] = int(
        ann[
            "is_actionable_issue"
        ]
    )

    out[
        "llm_evidence"
    ] = " || ".join(
        ann[
            "evidence_spans"
        ]
    )

    out[
        "llm_confidence"
    ] = float(
        ann[
            "confidence"
        ]
    )

    return out


def append_raw_records(
    results
):

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RAW_FILE.open(
        "a",
        encoding="utf-8",
    ) as f:

        for item in results:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_screened(
    screened_df
):

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    screened_df.to_csv(
        SCREENED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    failures = screened_df[
        screened_df[
            "llm_annotation_status"
        ]
        != "success"
    ].copy()

    failures.to_csv(
        FAILURE_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def build_selected_sample(
    screened_df,
    positive_target,
    negative_target,
):

    success = screened_df[
        screened_df[
            "llm_annotation_status"
        ]
        == "success"
    ].copy()

    success[
        "llm_is_actionable_issue"
    ] = pd.to_numeric(
        success[
            "llm_is_actionable_issue"
        ],
        errors="coerce",
    )

    positives = (
        success[
            success[
                "llm_is_actionable_issue"
            ]
            == 1
        ]
        .sort_values(
            "screening_order"
        )
        .head(
            positive_target
        )
        .copy()
    )

    negatives = (
        success[
            success[
                "llm_is_actionable_issue"
            ]
            == 0
        ]
        .sort_values(
            "screening_order"
        )
        .head(
            negative_target
        )
        .copy()
    )

    positives[
        "quota_class"
    ] = "llm_positive"

    negatives[
        "quota_class"
    ] = "llm_negative"

    selected = pd.concat(
        [
            positives,
            negatives,
        ],
        ignore_index=True,
    )

    selected = (
        selected
        .sort_values(
            "screening_order"
        )
        .reset_index(
            drop=True
        )
    )

    selected.insert(
        0,
        "stage1_id",
        range(
            1,
            len(selected) + 1
        )
    )

    selected[
        "manual_is_actionable_issue"
    ] = pd.NA

    selected[
        "manual_notes"
    ] = ""

    selected[
        "manual_reviewed"
    ] = 0

    return selected


# =========================================================
# Main
# =========================================================

async def main_async(args):

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required = {
        "game",
        "review_id",
        "review_text",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"all_reviews.csv missing columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df[
        "review_text"
    ] = (
        df[
            "review_text"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df[
            "review_text"
        ]
        .ne("")
    ].copy()

    df = (
        df
        .drop_duplicates(
            subset=[
                "review_id"
            ],
            keep="first",
        )
        .copy()
    )

    # Deterministic random screening order.
    df = (
        df
        .sample(
            frac=1.0,
            random_state=RANDOM_SEED,
        )
        .reset_index(
            drop=True
        )
    )

    df[
        "_screening_order"
    ] = range(
        1,
        len(df) + 1
    )

    # -----------------------------------------------------
    # Resume support
    # -----------------------------------------------------

    if SCREENED_FILE.exists():

        screened = pd.read_csv(
            SCREENED_FILE,
            low_memory=False,
        )

    else:

        screened = pd.DataFrame()

    screened_ids = set()

    if (
        not screened.empty
        and "review_id"
        in screened.columns
    ):

        screened_ids = set(
            screened[
                "review_id"
            ]
            .astype(str)
            .tolist()
        )

    def current_counts():

        if screened.empty:
            return 0, 0

        ok = screened[
            screened[
                "llm_annotation_status"
            ]
            == "success"
        ].copy()

        labels = pd.to_numeric(
            ok[
                "llm_is_actionable_issue"
            ],
            errors="coerce",
        )

        pos = int(
            (labels == 1).sum()
        )

        neg = int(
            (labels == 0).sum()
        )

        return pos, neg


    pos_count, neg_count = (
        current_counts()
    )

    print("=" * 72)
    print(
        "Stage-1 quota-based LLM screening"
    )
    print("=" * 72)

    print(
        f"Usable reviews: "
        f"{len(df)}"
    )

    print(
        f"Already screened: "
        f"{len(screened_ids)}"
    )

    print(
        f"Current LLM positives: "
        f"{pos_count}"
    )

    print(
        f"Current LLM negatives: "
        f"{neg_count}"
    )

    print(
        f"Targets: "
        f"positive={args.positive_target}, "
        f"negative={args.negative_target}"
    )

    if (
        pos_count
        >= args.positive_target
        and neg_count
        >= args.negative_target
    ):

        print(
            "Both quotas already reached."
        )

        selected = (
            build_selected_sample(
                screened,
                args.positive_target,
                args.negative_target,
            )
        )

        SELECTED_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected.to_csv(
            SELECTED_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Selected sample saved: "
            f"{SELECTED_FILE}"
        )

        return


    pending = df[
        ~df[
            "review_id"
        ]
        .astype(str)
        .isin(
            screened_ids
        )
    ].copy()

    llm = build_llm()

    semaphore = asyncio.Semaphore(
        args.concurrency
    )

    batch_number = 0


    while (
        (
            pos_count
            < args.positive_target
            or neg_count
            < args.negative_target
        )
        and len(pending) > 0
    ):

        batch_number += 1

        batch_df = (
            pending
            .head(
                args.batch_size
            )
            .copy()
        )

        if len(batch_df) == 0:
            break

        print()
        print(
            f"Batch {batch_number} "
            f"| rows={len(batch_df)} "
            f"| pos={pos_count}/{args.positive_target} "
            f"| neg={neg_count}/{args.negative_target}"
        )

        tasks = [
            annotate_one(
                llm=llm,
                semaphore=semaphore,
                screening_order=int(
                    row[
                        "_screening_order"
                    ]
                ),
                review_id=str(
                    row[
                        "review_id"
                    ]
                ),
                review=str(
                    row[
                        "review_text"
                    ]
                ),
            )
            for _, row
            in batch_df.iterrows()
        ]

        results = (
            await tqdm_asyncio.gather(
                *tasks
            )
        )

        flat_rows = [
            flatten_result(
                row.drop(
                    labels=[
                        "_screening_order"
                    ],
                    errors="ignore",
                ),
                result,
            )
            for (
                (_, row),
                result,
            )
            in zip(
                batch_df.iterrows(),
                results,
            )
        ]

        new_df = pd.DataFrame(
            flat_rows
        )

        if screened.empty:

            screened = (
                new_df.copy()
            )

        else:

            screened = pd.concat(
                [
                    screened,
                    new_df,
                ],
                ignore_index=True,
            )

        screened = (
            screened
            .drop_duplicates(
                subset=[
                    "review_id"
                ],
                keep="last",
            )
        )

        append_raw_records(
            results
        )

        save_screened(
            screened
        )

        labels = pd.to_numeric(
            new_df[
                "llm_is_actionable_issue"
            ],
            errors="coerce",
        )

        pos_count = int(
            (
                pd.to_numeric(
                    screened[
                        "llm_is_actionable_issue"
                    ],
                    errors="coerce",
                )
                == 1
            ).sum()
        )

        neg_count = int(
            (
                pd.to_numeric(
                    screened[
                        "llm_is_actionable_issue"
                    ],
                    errors="coerce",
                )
                == 0
            ).sum()
        )

        print(
            "Batch status: "
            f"{new_df['llm_annotation_status'].value_counts(dropna=False).to_dict()}"
        )

        print(
            f"Accumulated: "
            f"pos={pos_count}, "
            f"neg={neg_count}"
        )

        processed_batch_ids = set(
            batch_df[
                "review_id"
            ]
            .astype(str)
        )

        pending = pending[
            ~pending[
                "review_id"
            ]
            .astype(str)
            .isin(
                processed_batch_ids
            )
        ].copy()


    # -----------------------------------------------------
    # Build selected quota sample
    # -----------------------------------------------------

    selected = build_selected_sample(
        screened,
        args.positive_target,
        args.negative_target,
    )

    SELECTED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        SELECTED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    final_pos = int(
        (
            pd.to_numeric(
                selected[
                    "llm_is_actionable_issue"
                ],
                errors="coerce",
            )
            == 1
        ).sum()
    )

    final_neg = int(
        (
            pd.to_numeric(
                selected[
                    "llm_is_actionable_issue"
                ],
                errors="coerce",
            )
            == 0
        ).sum()
    )

    print()
    print("=" * 72)
    print(
        "Quota-based screening completed"
    )
    print("=" * 72)

    print(
        f"Reviews screened by LLM: "
        f"{len(screened)}"
    )

    print(
        f"Selected rows: "
        f"{len(selected)}"
    )

    print(
        f"Selected LLM positives: "
        f"{final_pos}"
    )

    print(
        f"Selected LLM negatives: "
        f"{final_neg}"
    )

    print()
    print(
        f"Selected sample:\n"
        f"{SELECTED_FILE}"
    )

    print(
        f"Screening log:\n"
        f"{SCREENED_FILE}"
    )

    print(
        f"Raw responses:\n"
        f"{RAW_FILE}"
    )

    print(
        f"Failures:\n"
        f"{FAILURE_FILE}"
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Quota-based Stage-1 LLM screening."
        )
    )

    parser.add_argument(
        "--positive-target",
        type=int,
        default=POSITIVE_TARGET,
    )

    parser.add_argument(
        "--negative-target",
        type=int,
        default=NEGATIVE_TARGET,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if os.name == "nt":

        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )

    asyncio.run(
        main_async(args)
    )
