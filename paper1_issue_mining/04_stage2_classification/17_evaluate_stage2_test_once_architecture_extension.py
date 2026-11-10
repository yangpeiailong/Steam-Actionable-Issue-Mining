# -*- coding: utf-8 -*-
"""One-time held-out test evaluation for the additional RoBERTa/ModernBERT models.

The DeBERTa Gold-only and targeted-T3 models were already evaluated on this
locked test split in the earlier final_test_once_targeted_T3_20261030 run.
This script evaluates only the four additional, validation-prespecified models.
No threshold tuning or model selection is performed on the test set.

Place this script beside 13_train_stage2_distillation.py and run --dry-run first.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config_local import (
    PAPER1_ANALYSIS_ROOT, PAPER1_RESULTS_ROOT,
    DEBERTA_GOLD_ONLY_DIR, DEBERTA_HARD_EXPANDED_1289_DIR,
    STAGE2_DEBERTA_T3_DIR, ROBERTA_GOLD_ONLY_DIR,
    STAGE2_ROBERTA_T2_DIR, MODERNBERT_GOLD_ONLY_DIR,
    MODERNBERT_TARGETED_T1_DIR,
)

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CODE_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = CODE_DIR / "13_train_stage2_distillation.py"

TEST_CSV = PAPER1_ANALYSIS_ROOT / "stage2_annotation" / "Stage2_Gold_Release_v1" / "stage2_gold_test_300.csv"
RESULTS_ROOT = PAPER1_RESULTS_ROOT / "stage2_models"
DEFAULT_OUTPUT = RESULTS_ROOT / "final_test_architecture_extension_20261030"

# All four checkpoints and thresholds were fixed using validation results only.
MODELS = [
    {
        "name": "roberta_gold_only",
        "architecture": "RoBERTa",
        "role": "Gold-only baseline",
        "path": ROBERTA_GOLD_ONLY_DIR,
        "threshold": 0.50,
        "use_fast_tokenizer": False,
    },
    {
        "name": "roberta_targeted_T2",
        "architecture": "RoBERTa",
        "role": "Best validation-selected distillation configuration (T2)",
        "path": STAGE2_ROBERTA_T2_DIR,
        "threshold": 0.45,
        "use_fast_tokenizer": False,
    },
    {
        "name": "modernbert_gold_only",
        "architecture": "ModernBERT",
        "role": "Gold-only baseline",
        "path": MODERNBERT_GOLD_ONLY_DIR,
        "threshold": 0.50,
        "use_fast_tokenizer": True,
    },
    {
        "name": "modernbert_targeted_T1",
        "architecture": "ModernBERT",
        "role": "Best validation-selected distillation configuration (T1)",
        "path": MODERNBERT_TARGETED_T1_DIR,
        "threshold": 0.25,
        "use_fast_tokenizer": True,
    },
]


def load_helpers():
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Training helper script not found: {TRAIN_SCRIPT}\n"
            "Place this evaluation script beside 13_train_stage2_distillation.py."
        )
    spec = importlib.util.spec_from_file_location("stage2_train_helpers", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper functions from {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(
        path, usecols=["review_id"], dtype={"review_id": "string"},
        encoding="utf-8-sig"
    )
    ids = df["review_id"].fillna("").astype(str).str.strip()
    ids = ids.map(lambda x: x[:-2] if x.endswith(".0") and x[:-2].isdigit() else x)
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"Empty or duplicated review_id in {path}")
    return set(ids)


def parse_args():
    p = argparse.ArgumentParser(
        description="One-time test evaluation for RoBERTa and ModernBERT; no test tuning."
    )
    p.add_argument("--test-csv", type=Path, default=TEST_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm-final-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.dry_run and args.confirm_final_test:
        raise SystemExit("Use either --dry-run or --confirm-final-test, not both.")
    if not args.dry_run and not args.confirm_final_test:
        raise SystemExit(
            "No test evaluation was run. First use --dry-run, then explicitly "
            "pass --confirm-final-test."
        )

    helpers = load_helpers()
    gold_dir = helpers.GOLD_DIR
    train_ids = read_ids(gold_dir / "stage2_gold_train_1400.csv")
    val_ids = read_ids(gold_dir / "stage2_gold_val_300.csv")
    test_ids = read_ids(args.test_csv)

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Gold train/validation/test review_id overlap detected.")
    if len(test_ids) != 300:
        raise ValueError(f"Expected 300 locked test IDs, found {len(test_ids)}.")

    missing = [f"{m['name']}: {m['path']}" for m in MODELS if not m["path"].is_dir()]
    if missing:
        raise FileNotFoundError("Missing model directories:\n" + "\n".join(missing))
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}\n"
            "Choose a new output directory; existing results will not be overwritten."
        )

    print("PRE-FLIGHT CHECK PASSED")
    print(f"Test file: {args.test_csv}")
    print(f"Train/validation/test ID counts: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print("No split overlap detected.")
    print("Four prespecified models and validation-selected thresholds:")
    for m in MODELS:
        print(
            f"  {m['name']}: architecture={m['architecture']}; "
            f"threshold={m['threshold']:.2f}; model={m['path']}"
        )
    print(f"Output directory: {args.output_dir}")

    if args.dry_run:
        print("DRY RUN ONLY: test labels were not loaded; no inference or metrics were generated.")
        return

    # Load test labels only after explicit final-test confirmation.
    test_df = helpers.load_gold(args.test_csv, "test")
    if set(test_df["review_id"].astype(str)) != test_ids:
        raise ValueError("Test IDs changed during loading; refusing evaluation.")
    y_true = test_df[helpers.LABEL_COLS].to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP fp16: {use_amp}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    comparison_rows = []
    all_reports = {}

    for spec in MODELS:
        print(f"\nEvaluating: {spec['name']}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec["path"]),
            local_files_only=True,
            use_fast=spec["use_fast_tokenizer"],
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(spec["path"]), local_files_only=True
        ).float().to(device)

        if getattr(model.config, "num_labels", None) != len(helpers.LABELS):
            raise ValueError(
                f"{spec['name']} has {model.config.num_labels} outputs; "
                f"expected {len(helpers.LABELS)}."
            )

        loader = helpers.make_loader(
            test_df["review_text"].astype(str).tolist(),
            y_true,
            np.zeros(len(test_df), dtype=bool),
            tokenizer,
            batch_size=8,
            max_length=384,
            shuffle=False,
        )
        probs, inferred_true, test_loss = helpers.predict_probabilities(
            model, loader, device, use_amp
        )
        if not np.array_equal(inferred_true, y_true.astype(int)):
            raise RuntimeError(f"{spec['name']}: labels changed during inference.")

        threshold = float(spec["threshold"])
        pred = helpers.apply_thresholds(probs, threshold)
        metrics = helpers.metric_summary(y_true.astype(int), pred)

        model_out = args.output_dir / spec["name"]
        model_out.mkdir(parents=True, exist_ok=False)
        helpers.write_val_predictions(
            model_out / "test_predictions.csv",
            test_df,
            probs,
            y_true.astype(int),
            threshold,
        )
        helpers.per_label_metrics(y_true.astype(int), pred).to_csv(
            model_out / "test_per_label_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        report = {
            "model_name": spec["name"],
            "architecture": spec["architecture"],
            "role": spec["role"],
            "model_dir": str(spec["path"]),
            "threshold_fixed_from_validation": threshold,
            "test_rows": int(len(test_df)),
            "test_bce_loss": float(test_loss),
            "test_metrics": metrics,
            "test_labels_used_only_for_final_evaluation": True,
            "threshold_tuned_on_test": False,
            "model_selected_using_test": False,
        }
        (model_out / "test_metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_reports[spec["name"]] = report
        comparison_rows.append({
            "model_name": spec["name"],
            "architecture": spec["architecture"],
            "role": spec["role"],
            "threshold_fixed_from_validation": threshold,
            **metrics,
            "test_bce_loss": float(test_loss),
        })

        print(
            f"{spec['name']}: issue Macro-F1={metrics['issue_domain_macro_f1']:.4f}; "
            f"issue Micro-F1={metrics['issue_domain_micro_f1']:.4f}; "
            f"exact-set={metrics['exact_set_match']:.4f}"
        )

        del model, tokenizer, loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        args.output_dir / "final_test_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "task": "Steam Stage 2 multi-label issue-domain classification",
        "evaluation_type": "prespecified held-out test evaluation for additional architectures",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_rows": int(len(test_df)),
        "train_validation_test_id_overlap": False,
        "test_threshold_tuning": False,
        "test_based_model_selection": False,
        "gold_data_modified": False,
        "thresholds_fixed_before_test": {
            m["name"]: float(m["threshold"]) for m in MODELS
        },
        "models": all_reports,
        "note": (
            "DeBERTa Gold-only and targeted-T3 test results were produced in the "
            "earlier final_test_once_targeted_T3_20261030 run. This run evaluates "
            "only the four additional RoBERTa/ModernBERT checkpoints. Do not use "
            "these test results to tune thresholds or change model configurations."
        ),
    }
    (args.output_dir / "final_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nADDITIONAL ARCHITECTURE TEST EVALUATION COMPLETE")
    print(f"Summary: {args.output_dir / 'final_test_summary.json'}")
    print(f"Comparison: {args.output_dir / 'final_test_model_comparison.csv'}")
    print("Do not use test results for further tuning or model selection.")


if __name__ == "__main__":
    main()
