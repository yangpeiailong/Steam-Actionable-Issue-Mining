"""Batch inference with the selected Stage-1 DeBERTa review classifier.

Run (from Steam_Research_Code):
 python paper1_issue_mining/03_issue_detection/06_apply_stage1_classifier.py

Input:  Paper1_Actionable_Issue_Mining/data/analysis/all_reviews.csv
Model:  Paper1_Actionable_Issue_Mining/results/stage1_models/random/deberta/best_model
Output: Paper1_Actionable_Issue_Mining/data/analysis/stage1_inference/

Checkpoint: one atomic CSV per input chunk, resuming without recomputing completed
chunks. Never sends data to any API. Prediction score is not a calibrated probability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Set caches BEFORE importing Hugging Face libraries. Keep caches off C:.
HF_ROOT = Path(os.environ.get("STEAM_MODELS_DIR", Path(__file__).resolve().parents[2] / "models_local"))
os.environ.setdefault("HF_HOME", str(HF_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(HF_ROOT / "hf_cache" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_ROOT / "hf_cache" / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from config_local import PAPER1_ANALYSIS_ROOT, PAPER1_RESULTS_ROOT

SOURCE = PAPER1_ANALYSIS_ROOT / "all_reviews.csv"
MODEL = PAPER1_RESULTS_ROOT / "stage1_models" / "random" / "deberta" / "best_model"
OUT = PAPER1_ANALYSIS_ROOT / "stage1_inference"
CHUNKS = OUT / "checkpoints"


class TextData(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.tokenizer(self.texts[index], truncation=True,
                              max_length=self.max_length, padding=False)


def file_hash(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(df: pd.DataFrame, path: Path):
    temp = path.with_name(path.name + ".tmp")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def atomic_json(payload, path: Path):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.chunk_size <= 0 or args.max_length <= 0:
        parser.error("batch-size, chunk-size, max-length must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing source: {SOURCE}")
    for name in ("config.json",):
        if not (MODEL / name).is_file():
            raise FileNotFoundError(f"Missing trained checkpoint {MODEL / name}")
    if not (MODEL / "model.safetensors").exists() and not (MODEL / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"Missing model weights in {MODEL}")

    df = pd.read_csv(SOURCE, dtype={"review_id": str}, low_memory=False)
    needed = {"review_id", "review_text", "game"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns in {SOURCE.name}: {sorted(needed - set(df.columns))}")
    if df["review_id"].isna().any() or df["review_id"].duplicated().any():
        raise ValueError("review_id must be non-null and unique; fix input before inference")
    # 清理评论文本
    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()

    # 过滤空评论
    empty_count = df["review_text"].eq("").sum()
    if empty_count > 0:
        print(f"Skipping {empty_count} empty reviews.")

    df = df[df["review_text"].ne("")].copy()
    df = df.reset_index(drop=True)

    print(f"Usable reviews for inference: {len(df)}")
    df = df.reset_index(drop=True)
    df.insert(0, "source_row", range(len(df)))

    # Explicit run manifest guards against silently reusing stale chunk files.
    signature = {
        "source": str(SOURCE),
        "source_sha256": file_hash(SOURCE),
        "model_dir": str(MODEL),
        "model_weight_sha256": file_hash(MODEL / "model.safetensors") if (MODEL / "model.safetensors").exists() else file_hash(MODEL / "pytorch_model.bin"),
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "max_length": args.max_length,
        "threshold": args.threshold,
        "rows": len(df),
    }
    CHUNKS.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != signature:
            raise RuntimeError("Existing checkpoint configuration differs. Move/rename stage1_inference directory before starting a new run.\n"
                               f"Existing manifest: {manifest_path}")
    else:
        if any(CHUNKS.glob("chunk_*.csv")):
            raise RuntimeError("Checkpoint files exist without manifest; inspect before rerunning")
        atomic_json(signature, manifest_path)

    total_chunks = (len(df) + args.chunk_size - 1) // args.chunk_size
    print(f"Reviews: {len(df)} | chunks: {total_chunks} | threshold: {args.threshold}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUT}")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)

    pending = [idx for idx in range(total_chunks) if not (CHUNKS / f"chunk_{idx:05d}.csv").exists()]
    print(f"Chunks completed: {total_chunks - len(pending)} | pending: {len(pending)}")
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL), local_files_only=True).to(device)
        model.eval()
        for idx in pending:
            start = idx * args.chunk_size
            part = df.iloc[start: start + args.chunk_size].copy()
            dataset = TextData(part["review_text"].astype(str).tolist(), tokenizer, args.max_length)
            def collate(items):
                return tokenizer.pad(items, padding=True, return_tensors="pt")
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, collate_fn=collate)
            probabilities = []
            with torch.inference_mode():
                for batch in loader:
                    batch = {key: val.to(device) for key, val in batch.items()}
                    logits = model(**batch).logits.float()
                    probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            if len(probabilities) != len(part):
                raise RuntimeError(f"Chunk {idx} has unexpected prediction length")
            part["stage1_score"] = probabilities
            part["stage1_pred"] = (part["stage1_score"] >= args.threshold).astype(int)
            atomic_csv(part, CHUNKS / f"chunk_{idx:05d}.csv")
            print(f"Chunk {idx + 1}/{total_chunks}: {len(part)} reviews | positives={int(part['stage1_pred'].sum())}", flush=True)

    pieces = []
    for idx in range(total_chunks):
        p = CHUNKS / f"chunk_{idx:05d}.csv"
        if not p.exists():
            raise RuntimeError(f"Missing chunk: {p}")
        part = pd.read_csv(p, dtype={"review_id": str}, low_memory=False)
        expected = list(range(idx * args.chunk_size, min((idx + 1) * args.chunk_size, len(df))))
        if part["source_row"].tolist() != expected:
            raise RuntimeError(f"Wrong rows in chunk {idx}")
        pieces.append(part)
    all_pred = pd.concat(pieces, ignore_index=True)
    if len(all_pred) != len(df) or all_pred["review_id"].tolist() != df["review_id"].tolist():
        raise RuntimeError("Final row verification failed")

    all_file = OUT / "stage1_all_review_predictions.csv"
    candidates_file = OUT / "stage2_candidate_reviews.csv"
    atomic_csv(all_pred, all_file)
    candidates = all_pred.loc[all_pred["stage1_pred"] == 1].copy()
    atomic_csv(candidates, candidates_file)
    by_game = all_pred.groupby("game", dropna=False).agg(
        n_reviews=("stage1_pred", "size"),
        n_pred_actionable=("stage1_pred", "sum")
    ).reset_index()
    by_game["pred_actionable_share"] = by_game["n_pred_actionable"] / by_game["n_reviews"]
    atomic_csv(by_game, OUT / "stage1_prediction_summary_by_game.csv")
    print("\nCOMPLETE")
    print(f"Total: {len(all_pred)} | predicted actionable: {len(candidates)} ({len(candidates) / max(1, len(all_pred)):.2%})")
    print(f"All predictions: {all_file}")
    print(f"Stage 2 candidates: {candidates_file}")
    print(f"By-game summary: {OUT / 'stage1_prediction_summary_by_game.csv'}")
    print("Important: predicted positive share is not human-validated issue prevalence.")


if __name__ == "__main__":
    main()
