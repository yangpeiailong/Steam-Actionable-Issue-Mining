# -*- coding: utf-8 -*-
"""One-shot final test evaluation for Steam Stage 2.

This script evaluates prespecified, already-selected models on the locked 300-row
Gold test split. Thresholds are fixed from validation results and are NOT tuned on
this test set. It does not train models or modify Gold data.

Run --dry-run first. Only run with --confirm-final-test when ready to perform the
final test evaluation. Existing output folders are never overwritten.
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

DEFAULT_TEST_CSV = PAPER1_ANALYSIS_ROOT / "stage2_annotation" / "Stage2_Gold_Release_v1" / "stage2_gold_test_300.csv"
DEFAULT_RESULTS_ROOT = PAPER1_RESULTS_ROOT / "stage2_models"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "final_test_once_20261029"

DEFAULT_MODELS = [
    {
        "name": "gold_only",
        "display_name": "Gold-only baseline",
        "path": DEBERTA_GOLD_ONLY_DIR,
        "threshold": 0.50,
    },
    {
        "name": "hard_expanded_1289",
        "display_name": "Hard distillation, 1289 pseudo-labels",
        "path": DEBERTA_HARD_EXPANDED_1289_DIR,
        "threshold": 0.50,
    },
    {
        "name": "targeted_T3_1511",
        "display_name": "Targeted minority enrichment, T3",
        "path": STAGE2_DEBERTA_T3_DIR,
        "threshold": 0.65,
    },
]


def load_training_module():
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Cannot find training script: {TRAIN_SCRIPT}\n"
            "Place this evaluation script beside 13_train_stage2_distillation.py."
        )
    spec = importlib.util.spec_from_file_location("stage2_train_helpers", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper functions from {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description="One-shot final test evaluation; no test-set tuning.")
    p.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dry-run", action="store_true", help="Check files/IDs/models only; do not load test labels or run inference.")
    p.add_argument("--confirm-final-test", action="store_true", help="Explicitly authorize the final evaluation on the locked test set.")
    p.add_argument("--expanded-only", action="store_true", help="Evaluate only the expanded 1,289-pseudo-label model if baseline model folders are unavailable.")
    p.add_argument("--gold-model-dir", type=Path, default=DEFAULT_MODELS[0]["path"])
    p.add_argument("--hard467-model-dir", type=Path, default=DEFAULT_MODELS[1]["path"])
    p.add_argument("--expanded-model-dir", type=Path, default=DEFAULT_MODELS[2]["path"])
    p.add_argument("--gold-threshold", type=float, default=0.50)
    p.add_argument("--hard467-threshold", type=float, default=0.55)
    p.add_argument("--expanded-threshold", type=float, default=0.50)
    return p.parse_args()


def read_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path, usecols=["review_id"], dtype={"review_id": "string"}, encoding="utf-8-sig")
    ids = df["review_id"].map(lambda x: "" if pd.isna(x) else str(x).strip())
    ids = ids.map(lambda s: s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s)
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"Empty or duplicated review_id in {path}")
    return set(ids)


def get_models(args):
    if args.expanded_only:
        return [{
            **DEFAULT_MODELS[2],
            "path": args.expanded_model_dir,
            "threshold": args.expanded_threshold,
        }]
    return [
        {**DEFAULT_MODELS[0], "path": args.gold_model_dir, "threshold": args.gold_threshold},
        {**DEFAULT_MODELS[1], "path": args.hard467_model_dir, "threshold": args.hard467_threshold},
        {**DEFAULT_MODELS[2], "path": args.expanded_model_dir, "threshold": args.expanded_threshold},
    ]


def main():
    args = parse_args()
    if args.dry_run and args.confirm_final_test:
        raise SystemExit("Use either --dry-run or --confirm-final-test, not both.")
    if not args.dry_run and not args.confirm_final_test:
        raise SystemExit("No test evaluation was run. First use --dry-run, then explicitly pass --confirm-final-test.")

    helpers = load_training_module()
    test_path = args.test_csv
    if not test_path.is_file():
        raise FileNotFoundError(f"Test split not found: {test_path}")

    train_path = helpers.GOLD_DIR / "stage2_gold_train_1400.csv"
    val_path = helpers.GOLD_DIR / "stage2_gold_val_300.csv"
    train_ids = read_ids(train_path)
    val_ids = read_ids(val_path)
    test_ids = read_ids(test_path)
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Gold train/validation/test review_id overlap detected; refusing evaluation.")

    models = get_models(args)
    missing = [f"{m['name']}: {m['path']}" for m in models if not m["path"].is_dir()]
    if missing:
        raise FileNotFoundError("Missing model directory/directories:\n" + "\n".join(missing))
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}\n"
            "Use a new output directory; this script will not overwrite final-test results."
        )

    print("PRE-FLIGHT CHECK PASSED")
    print(f"Test file: {test_path}")
    print(f"Train/validation/test ID counts: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print("No split overlap detected.")
    print("Prespecified models and validation-selected thresholds:")
    for m in models:
        print(f"  {m['name']}: threshold={m['threshold']:.2f}; model={m['path']}")
    print(f"Output directory: {args.output_dir}")
    if args.dry_run:
        print("DRY RUN ONLY: test labels were not loaded; no inference or metrics were generated.")
        return

    # From this point, this is the one explicitly authorized final test evaluation.
    test_df = helpers.load_gold(test_path, "test")
    if set(test_df["review_id"].astype(str)) != test_ids:
        raise ValueError("Test IDs changed during loading; refusing evaluation.")
    y_true = test_df[helpers.LABEL_COLS].to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP fp16: {use_amp}")

    # Create output directory only after all preflight checks and explicit confirmation.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    comparison_rows = []
    all_reports = {}

    for spec in models:
        print(f"\nEvaluating fixed model: {spec['display_name']}")
        tokenizer = AutoTokenizer.from_pretrained(str(spec["path"]), local_files_only=True, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(str(spec["path"]), local_files_only=True).float().to(device)
        if getattr(model.config, "num_labels", None) != len(helpers.LABELS):
            raise ValueError(f"{spec['name']} has {model.config.num_labels} outputs; expected {len(helpers.LABELS)}.")
        loader = helpers.make_loader(
            test_df["review_text"].astype(str).tolist(), y_true,
            np.zeros(len(test_df), dtype=bool), tokenizer,
            batch_size=8, max_length=384, shuffle=False,
        )
        probs, inferred_true, test_loss = helpers.predict_probabilities(model, loader, device, use_amp)
        if not np.array_equal(inferred_true, y_true.astype(int)):
            raise RuntimeError("Labels changed during inference; refusing to save metrics.")
        threshold = float(spec["threshold"])
        pred = helpers.apply_thresholds(probs, threshold)
        metrics = helpers.metric_summary(y_true.astype(int), pred)
        model_out = args.output_dir / spec["name"]
        model_out.mkdir(parents=True, exist_ok=False)
        helpers.write_val_predictions(
            model_out / "test_predictions.csv", test_df, probs, y_true.astype(int), threshold
        )
        helpers.per_label_metrics(y_true.astype(int), pred).to_csv(
            model_out / "test_per_label_metrics.csv", index=False, encoding="utf-8-sig"
        )
        report = {
            "model_name": spec["name"],
            "display_name": spec["display_name"],
            "model_dir": str(spec["path"]),
            "threshold_fixed_from_validation": threshold,
            "test_rows": int(len(test_df)),
            "test_bce_loss": float(test_loss),
            "test_metrics": metrics,
            "test_labels_used_only_for_final_evaluation": True,
            "threshold_tuned_on_test": False,
            "model_selected_using_test": False,
        }
        (model_out / "test_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        all_reports[spec["name"]] = report
        comparison_rows.append({
            "model_name": spec["name"],
            "display_name": spec["display_name"],
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
    comparison.to_csv(args.output_dir / "final_test_model_comparison.csv", index=False, encoding="utf-8-sig")
    summary = {
        "task": "Steam Stage 2 multi-label issue-domain classification",
        "evaluation_type": "one-shot final test evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_rows": int(len(test_df)),
        "train_validation_test_id_overlap": False,
        "test_threshold_tuning": False,
        "test_based_model_selection": False,
        "gold_data_modified": False,
        "thresholds_fixed_before_test": {m["name"]: float(m["threshold"]) for m in models},
        "models": all_reports,
        "interpretation_note": "Do not use these test results to tune thresholds, change training settings, or select a new model. Report this prespecified comparison as the final test evaluation.",
    }
    (args.output_dir / "final_test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nFINAL TEST EVALUATION COMPLETE")
    print(f"Summary: {args.output_dir / 'final_test_summary.json'}")
    print(f"Comparison: {args.output_dir / 'final_test_model_comparison.csv'}")
    print("Do not use test results for further tuning or model selection.")


if __name__ == "__main__":
    main()
