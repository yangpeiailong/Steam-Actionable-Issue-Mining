"""
Stage 1 revised: DeBERTa with HUMAN-CONFIRMED hard negatives.

Original model and original split CSV files are never overwritten.

Examples:
  python paper1_issue_mining/03_issue_detection/05_train_stage1_classifier.py --model roberta --split random
  python paper1_issue_mining/03_issue_detection/05_train_stage1_classifier.py --model deberta --split random
  python paper1_issue_mining/03_issue_detection/05_train_stage1_classifier.py --model modernbert --split random
  python paper1_issue_mining/03_issue_detection/05_train_stage1_classifier.py --model roberta --split game_holdout

No test data is accessed until a validation-best checkpoint has been selected.
The split's test set is evaluated ONCE after training.
"""

import argparse
import os
import json
import math
import random
import sys
from pathlib import Path

# Configure Hugging Face BEFORE importing transformers/huggingface_hub.
# All three downloaded base models stay in a predictable D: drive folder.
DEFAULT_MODELS_DIR = Path(os.environ.get("STEAM_MODELS_DIR", Path(__file__).resolve().parents[2] / "models_local"))
DEFAULT_MIRROR = "https://hf-mirror.com"
MODEL_STORAGE = Path(os.environ.get("STEAM_MODELS_DIR", str(DEFAULT_MODELS_DIR)))
os.environ.setdefault("HF_ENDPOINT", DEFAULT_MIRROR)
os.environ["HF_HOME"] = str(MODEL_STORAGE / "hf_cache")
os.environ["HF_HUB_CACHE"] = str(MODEL_STORAGE / "hf_cache" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_STORAGE / "hf_cache" / "hub")
os.environ["XDG_CACHE_HOME"] = str(MODEL_STORAGE / "other_cache")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config_local import PAPER1_ANALYSIS_ROOT, PAPER1_RESULTS_ROOT  # noqa: E402

MODEL_IDS = {
    "roberta": "FacebookAI/roberta-base",
    "deberta": "microsoft/deberta-v3-base",
    "modernbert": "answerdotai/ModernBERT-base",
}
SPLITS = {
    "random": ("random_train.csv", "random_val.csv", "random_test.csv"),
    "game_holdout": ("game_holdout_train.csv", "game_holdout_val.csv", "game_holdout_test.csv"),
}


class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["review_text"].fillna("").astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = self.labels[i]
        return enc


def make_collate(tokenizer):
    def collate(items):
        labels = torch.tensor([int(x["labels"]) for x in items], dtype=torch.long)
        inputs = [{k: v for k, v in x.items() if k != "labels"} for x in items]
        batch = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch
    return collate


def metrics(labels, preds):
    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "n": int(len(labels)),
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, scores = [], [], []
    total_loss = 0.0
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        probs = torch.softmax(out.logits.float(), dim=-1)[:, 1]
        n = batch["labels"].size(0)
        total_loss += out.loss.item() * n
        count += n
        y_true.extend(batch["labels"].cpu().tolist())
        scores.extend(probs.cpu().tolist())
        y_pred.extend((probs >= 0.5).int().cpu().tolist())
    result = metrics(y_true, y_pred)
    result["loss"] = float(total_loss / max(1, count))
    return result, y_true, y_pred, scores


def read_split(name):
    path = PAPER1_ANALYSIS_ROOT / "stage1_gold" / name
    if not path.is_file():
        raise FileNotFoundError(f"Split not found: {path}\nRun 04_prepare_stage1_gold.py first.")
    df = pd.read_csv(path, low_memory=False, dtype={"stage1_id": str, "review_id": str})
    missing = {"review_text", "label", "stage1_id", "game"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns: {sorted(missing)}")
    if df["review_text"].isna().any() or not df["label"].isin([0, 1]).all():
        raise ValueError(f"{path.name}: empty review text or invalid labels.")
    return df


def get_local_model(model_alias: str, offline: bool = False) -> Path:
    repo_id = MODEL_IDS[model_alias]
    model_dir = MODEL_STORAGE / "base_models" / model_alias
    if (model_dir / "config.json").is_file() and any(
        (model_dir / f).exists()
        for f in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
    ):
        print(f"Using local base model: {model_dir}")
        return model_dir

    if offline:
        raise FileNotFoundError(f"Model not downloaded: {model_dir}. Remove --offline to download from mirror.")

    print(f"Downloading {repo_id} via {os.environ['HF_ENDPOINT']}")
    print(f"Destination: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        allow_patterns=[
            "*.json", "*.txt", "*.safetensors", "*.model", "*.bin",
            "vocab.*", "merges.txt", "spiece.model", "tokenizer.*",
        ],
        ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.ot", "*.ckpt"],
    )
    return model_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["deberta"], default="deberta")
    parser.add_argument("--split", choices=["random"], default="random")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Only use existing local models; no download")
    parser.add_argument("--download-only", action="store_true", help="Download model weights without training")
    parser.add_argument("--omit-audit-negatives", action="store_true", help="Do not add confirmed negative rows from the previous 40-row audit")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1:
        parser.error("epochs, batch-size, grad-accum must be positive.")

    print("Mirror:", os.environ["HF_ENDPOINT"])
    print("Base model directory:", MODEL_STORAGE / "base_models")
    print("Hugging Face cache:", os.environ["HF_HOME"])
    local_model_dir = get_local_model(args.model, offline=args.offline)
    if args.download_only:
        print("Download completed; no training started.")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    files = SPLITS[args.split]
    train, val, test = [read_split(name) for name in files]

    train_ids, val_ids, test_ids = (
        set(x["stage1_id"].tolist()) for x in (train, val, test)
    )
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Data leakage: overlapping stage1_id across splits.")
    if args.split == "game_holdout":
        if (set(train["game"]) | set(val["game"])) & set(test["game"]):
            raise RuntimeError("Game holdout leakage.")

    # Read the human-reviewed v2 file, NOT unverified LLM negatives.
    review_path = PAPER1_ANALYSIS_ROOT / "stage1_hard_negative_mining" / "hard_negative_manual_review_prompt_v2.csv"
    audit_path = PAPER1_ANALYSIS_ROOT / "stage1_error_audit" / "stage1_disagreement_audit.csv"
    if not review_path.is_file():
        raise FileNotFoundError(f"Missing completed human review file: {review_path}")
    hard = pd.read_csv(review_path, dtype={"review_id": str}, low_memory=False)
    must = {"review_id", "game", "review_text", "manual_reviewed", "manual_is_actionable_issue"}
    if must - set(hard.columns):
        raise ValueError(f"Human review file missing columns: {sorted(must - set(hard.columns))}")
    reviewed = pd.to_numeric(hard["manual_reviewed"], errors="coerce").eq(1)
    human = pd.to_numeric(hard["manual_is_actionable_issue"], errors="coerce")
    if not (reviewed & human.isin([0, 1])).all():
        raise ValueError("Some v2 rows lack completed, valid 0/1 human labels; do not train yet.")
    hard = hard.loc[human.eq(0), ["review_id", "game", "review_text"]].copy()
    hard["label"] = 0
    hard["origin"] = "human_v2_hard_negative"
    print("Confirmed v2 hard negatives:", len(hard))

    # Prevent leaks against BOTH published random and game-holdout val/test sets.
    forbidden = set()
    for split_name in ("random_val.csv", "random_test.csv", "game_holdout_val.csv", "game_holdout_test.csv"):
        forbidden.update(read_split(split_name)["review_id"].astype(str))
    existing_train = set(train["review_id"].astype(str))
    hard["review_id"] = hard["review_id"].astype(str)
    if hard["review_id"].duplicated().any():
        raise RuntimeError("Duplicate review_id in human-v2 hard negatives")
    if set(hard["review_id"]) & forbidden:
        raise RuntimeError("HARD STOP: v2 hard negative overlaps original validation/test set")
    hard = hard.loc[~hard["review_id"].isin(existing_train)].copy()
    extras = [hard]

    if not args.omit_audit_negatives:
        if not audit_path.is_file():
            raise FileNotFoundError(f"Missing old human audit: {audit_path}. Use --omit-audit-negatives to skip.")
        audit = pd.read_csv(audit_path, dtype={"review_id": str}, low_memory=False)
        req = {"review_id", "game", "review_text", "manual_reviewed", "manual_is_actionable_issue"}
        if req - set(audit.columns):
            raise ValueError(f"Audit missing columns: {sorted(req - set(audit.columns))}")
        mask = pd.to_numeric(audit["manual_reviewed"], errors="coerce").eq(1) & pd.to_numeric(audit["manual_is_actionable_issue"], errors="coerce").eq(0)
        audit = audit.loc[mask, ["review_id", "game", "review_text"]].copy()
        audit["review_id"] = audit["review_id"].astype(str)
        audit["label"] = 0
        audit["origin"] = "human_prior_audit_negative"
        # Exclude all existing gold (including train); never alter gold labels.
        forbidden_audit = forbidden | existing_train | set(hard["review_id"])
        before = len(audit)
        audit = audit.loc[~audit["review_id"].isin(forbidden_audit)].drop_duplicates("review_id")
        print(f"Earlier audit negatives: {before}, additional safe unique: {len(audit)}")
        extras.append(audit)

    new_train = pd.concat(extras, ignore_index=True)
    if len(new_train) == 0:
        raise RuntimeError("No new human-verified hard negatives available")
    new_train.insert(0, "stage1_id", [f"HARDNEG-{i:04d}" for i in range(1, len(new_train) + 1)])
    if set(new_train["review_id"]) & (forbidden | existing_train):
        raise RuntimeError("Leakage or duplicate review ID detected")
    train = pd.concat([train, new_train], ignore_index=True)
    if train["review_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate review IDs in augmented train")
    print(f"Augmented train: {len(train)} = original 1400 + {len(new_train)} human-confirmed negatives")
    print("Train label counts:", train["label"].value_counts().sort_index().to_dict())

    # Separate directory: NEVER overwrite the original baseline checkpoint/results.
    run_dir = PAPER1_RESULTS_ROOT / "stage1_models" / "random" / "deberta_hardneg_v2"
    if (run_dir / "summary.json").exists() or (run_dir / "best_model" / "config.json").exists():
        raise FileExistsError(f"Run output already exists: {run_dir}. Move/archive it before retraining.")
    run_dir.mkdir(parents=True, exist_ok=True)
    new_train.to_csv(run_dir / "added_human_hard_negatives.csv", index=False, encoding="utf-8-sig")
    (run_dir / "hard_negative_manifest.json").write_text(
        json.dumps({"new_rows": len(new_train), "sources": new_train["origin"].value_counts().to_dict(),
                    "v2_human_review_file": str(review_path), "audit_file": str(audit_path),
                    "validation_and_test_unchanged": True}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Model:", MODEL_IDS[args.model])
    print("Split:", args.split)
    print("Rows:", len(train), len(val), len(test))
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)
    if device.type == "cpu":
        print("WARNING: CPU training may take a long time.")

    tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(local_model_dir),
        dtype=torch.float32,
        local_files_only=True,
        num_labels=2,
        id2label={0: "NO_ISSUE", 1: "ACTIONABLE_ISSUE"},
        label2id={"NO_ISSUE": 0, "ACTIONABLE_ISSUE": 1},
    )
    model.to(device)

    collate = make_collate(tokenizer)
    train_loader = DataLoader(
        ReviewDataset(train, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate
    )
    val_loader = DataLoader(
        ReviewDataset(val, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )
    test_loader = DataLoader(
        ReviewDataset(test, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(0.1 * total_updates)),
        num_training_steps=total_updates
    )
    # Native AMP works on recent NVIDIA GPUs. CPU and unsupported hardware use fp32.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_f1 = -1.0
    stale = 0
    history = []
    best_epoch = None
    best_dir = run_dir / "best_model"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, samples = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                out = model(**batch)
                loss = out.loss
            loss_sum += float(loss.item()) * batch["labels"].size(0)
            samples += batch["labels"].size(0)
            scaler.scale(loss / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_result, *_ = evaluate(model, val_loader, device)
        val_result.update({"epoch": epoch, "train_loss": loss_sum / max(1, samples)})
        history.append(val_result)
        print(f"Epoch {epoch}: train_loss={val_result['train_loss']:.4f}"
              f" val_loss={val_result['loss']:.4f}"
              f" val_precision={val_result['precision']:.4f}"
              f" val_recall={val_result['recall']:.4f}"
              f" val_f1={val_result['f1']:.4f}", flush=True)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8-sig")

        if val_result["f1"] > best_f1 + 1e-8:
            best_f1 = val_result["f1"]
            best_epoch = epoch
            stale = 0
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    test_result, test_labels, test_preds, test_scores = evaluate(model, test_loader, device)
    print("Best epoch:", best_epoch)
    print("TEST (one-time):", json.dumps(test_result, ensure_ascii=False), flush=True)

    out_predictions = test[["stage1_id", "game", "review_id", "label"]].copy().reset_index(drop=True)
    out_predictions["prediction"] = test_preds
    out_predictions["prob_actionable"] = test_scores
    out_predictions.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model_alias": args.model,
        "model_id": MODEL_IDS[args.model],
        "split": args.split,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test": test_result,
        "parameters": vars(args),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "test_games": sorted(test["game"].unique().tolist()),
        "warning": "Original sample is LLM-balanced, and hard negatives are selected nonrandomly; test metrics are not natural-prevalence estimates.",
        "new_human_confirmed_negative_rows": len(new_train),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved:", run_dir)


if __name__ == "__main__":
    main()
