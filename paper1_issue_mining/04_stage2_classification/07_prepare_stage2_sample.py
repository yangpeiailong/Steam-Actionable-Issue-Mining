"""Prepare 2,000 full-review candidates for Stage-2 multi-label labeling.

Run from Steam_Research_Code:
    python paper1_issue_mining/04_stage2_classification/07_prepare_stage2_sample.py

Input:  Paper1_Actionable_Issue_Mining/data/analysis/stage1_inference/stage2_candidate_reviews.csv
Output: Paper1_Actionable_Issue_Mining/data/analysis/stage2_annotation/

Sampling is reproducible and stratified by game and review length (short/medium/long).
No issue labels, sentiment, LLM calls, or keyword rules are used for selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from config_local import PAPER1_ANALYSIS_ROOT  # noqa: E402

SOURCE = PAPER1_ANALYSIS_ROOT / 'stage1_inference' / 'stage2_candidate_reviews.csv'
OUTPUT_DIR = PAPER1_ANALYSIS_ROOT / 'stage2_annotation'
OUTPUT = OUTPUT_DIR / 'stage2_review_sample_2000.csv'
REPORT = OUTPUT_DIR / 'stage2_sample_distribution.csv'
MANIFEST = OUTPUT_DIR / 'stage2_sample_manifest.json'


def quotas_by_largest_remainder(counts: pd.Series, target: int) -> pd.Series:
    """Allocate exact sample total without exceeding group sizes."""
    if target > int(counts.sum()):
        raise ValueError('Target exceeds available rows')
    ideal = counts.astype(float) * target / counts.sum()
    quotas = np.floor(ideal).astype(int)
    remaining = target - int(quotas.sum())
    # Deterministic allocation on ties: group labels are sorted ahead of time.
    fractional = (ideal - quotas).sort_values(ascending=False, kind='stable')
    for key in fractional.index[:remaining]:
        quotas.loc[key] += 1
    if int(quotas.sum()) != target or (quotas > counts).any():
        raise RuntimeError('Quota allocation failed')
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser(description='Build Stage-2 annotation sample')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output')
    args = parser.parse_args()

    if args.n <= 0:
        parser.error('--n must be positive')
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Missing Stage-1 candidates: {SOURCE}')
    if not args.overwrite and any(p.exists() for p in (OUTPUT, REPORT, MANIFEST)):
        raise FileExistsError('Stage-2 sample already exists. Use --overwrite intentionally.')

    df = pd.read_csv(SOURCE, dtype={'review_id': str}, low_memory=False)
    needed = {'review_id', 'game', 'review_text', 'stage1_pred', 'stage1_score'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df['review_id'].isna().any() or df['review_id'].duplicated().any():
        raise ValueError('review_id must be nonempty and unique')
    if df['game'].isna().any():
        raise ValueError('game must be nonempty')
    if df['review_text'].isna().any() or df['review_text'].astype(str).str.strip().eq('').any():
        raise ValueError('Empty reviews found in Stage-2 candidate input')
    if not (pd.to_numeric(df['stage1_pred'], errors='coerce') == 1).all():
        raise ValueError('Source contains rows not predicted actionable by Stage 1')
    if df['stage1_score'].isna().any():
        raise ValueError('Missing Stage-1 scores')
    if len(df) < args.n:
        raise ValueError(f'Only {len(df)} candidates, cannot sample {args.n}')

    df = df.copy()
    df['review_length_chars'] = df['review_text'].astype(str).str.len()
    # rank(method='first') creates three near-equal strata even when many reviews
    # have exactly the same character length; no reviews are removed.
    length_rank = df['review_length_chars'].rank(method='first', pct=True)
    df['length_group'] = pd.cut(
        length_rank, bins=[0, 1 / 3, 2 / 3, 1],
        labels=['short', 'medium', 'long'], include_lowest=True
    ).astype(str)

    # Exact game-by-length proportional allocation; no class balancing.
    group_cols = ['game', 'length_group']
    counts = df.groupby(group_cols, sort=True, observed=True).size()
    quotas = quotas_by_largest_remainder(counts, args.n)
    selected_parts = []
    for (game, length_group), n in quotas.items():
        if n == 0:
            continue
        pool = df[(df['game'] == game) & (df['length_group'] == length_group)]
        selected_parts.append(pool.sample(n=int(n), random_state=args.seed))

    sample = pd.concat(selected_parts, ignore_index=True)
    if len(sample) != args.n or sample['review_id'].duplicated().any():
        raise RuntimeError('Sample size/uniqueness verification failed')
    sample = sample.sample(frac=1, random_state=args.seed + 1).reset_index(drop=True)
    sample.insert(0, 'stage2_id', [f'S2-{i:05d}' for i in range(1, len(sample) + 1)])
    sample['llm_domains'] = ''
    sample['llm_evidence'] = ''
    sample['llm_is_feature_request'] = pd.NA
    sample['llm_update_related'] = pd.NA
    sample['llm_status'] = ''
    sample['manual_domains'] = ''
    sample['manual_is_feature_request'] = pd.NA
    sample['manual_update_related'] = pd.NA
    sample['manual_notes'] = ''
    sample['manual_reviewed'] = 0

    leading = ['stage2_id', 'game', 'review_id', 'review_text', 'stage1_score',
               'review_length_chars', 'length_group']
    annotation = ['llm_domains', 'llm_evidence', 'llm_is_feature_request',
                  'llm_update_related', 'llm_status', 'manual_domains',
                  'manual_is_feature_request', 'manual_update_related',
                  'manual_notes', 'manual_reviewed']
    sample = sample[leading + [c for c in df.columns if c not in leading and c not in annotation]
                    + annotation]

    source_games = df['game'].value_counts().sort_index()
    sample_games = sample['game'].value_counts().sort_index()
    report = pd.DataFrame({'candidate_n': source_games, 'sample_n': sample_games}).fillna(0)
    report['candidate_share'] = report['candidate_n'] / len(df)
    report['sample_share'] = report['sample_n'] / len(sample)
    report = report.reset_index().rename(columns={'index': 'game'})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    report.to_csv(REPORT, index=False, encoding='utf-8-sig')
    manifest = {
        'source': str(SOURCE), 'candidate_rows': len(df), 'sample_rows': len(sample),
        'seed': args.seed, 'sampling': 'proportional game × length tertile',
        'length_group_counts': sample['length_group'].value_counts().to_dict(),
        'no_issue_allowed_in_future_manual_labels': True,
        'note': 'Stage 1 scores are NOT gold labels. No LLM prelabeling has happened.',
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'Candidates: {len(df)} | selected: {len(sample)} | games: {sample["game"].nunique()}')
    print('Sample by length:', sample['length_group'].value_counts().to_dict())
    print('Sample by game:\n' + report[['game', 'candidate_n', 'sample_n']].to_string(index=False))
    print('Saved:\n' + '\n'.join(map(str, [OUTPUT, REPORT, MANIFEST])))
    print('Next: LLM prelabeling; DO NOT give this unlabelled CSV to students yet.')


if __name__ == '__main__':
    main()
