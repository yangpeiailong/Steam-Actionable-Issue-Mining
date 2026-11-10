"""Prepare reviewed Stage-1 gold dataset and reproducible evaluation splits.

Requirements: pandas, openpyxl, scikit-learn, numpy

Inputs (all inside PAPER1_ANALYSIS_ROOT):
  stage1_manual_review_2000.xlsx
  stage1_review_sample_balanced_llm.csv (for game/review metadata)

Outputs (inside PAPER1_ANALYSIS_ROOT/stage1_gold):
  stage1_gold.csv
  random_train.csv / random_val.csv / random_test.csv
  game_holdout_train.csv / game_holdout_val.csv / game_holdout_test.csv
  split_summary.csv / split_manifest.json

Note: The balanced gold set is selected using LLM labels. Do NOT use its
class proportions to estimate the natural prevalence of actionable issues.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from config_local import PAPER1_ANALYSIS_ROOT

SEED = 2026
INPUT_XLSX = PAPER1_ANALYSIS_ROOT / 'stage1_manual_review_2000.xlsx'
INPUT_META = PAPER1_ANALYSIS_ROOT / 'stage1_review_sample_balanced_llm.csv'
OUT = PAPER1_ANALYSIS_ROOT / 'stage1_gold'


def counts(rows, split_name, strategy):
    return {
        'strategy': strategy,
        'split': split_name,
        'n': len(rows),
        'positive': int(rows['label'].sum()),
        'negative': int(len(rows) - rows['label'].sum()),
        'positive_rate': round(float(rows['label'].mean()), 4),
        'n_games': rows['game'].nunique(),
        'games': ';'.join(sorted(rows['game'].astype(str).unique()))
    }


def save_split(frame, name):
    frame.sort_values('stage1_id').to_csv(OUT / name, index=False, encoding='utf-8-sig')


def validate_disjoint(three_splits, name):
    sets = [set(f['stage1_id']) for f in three_splits]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), f'Overlap: {name}'
    assert sum(len(x) for x in sets) == 2000, f'Incomplete partition: {name}'


def normalize_for_comparison(value):
    """Normalize Excel-escaped carriage returns, not substantive text."""
    if pd.isna(value):
        return ''
    text = str(value)
    return (text.replace('_x000D_', '\r')
                .replace('\r\n', '\n')
                .replace('\r', '\n'))


def audit_text_match(merged):
    """Require exact equality after newline normalization; never mask edits."""
    left = merged['review_text'].map(normalize_for_comparison)
    right = merged['source_review_text'].map(normalize_for_comparison)
    mismatch = left.ne(right)
    if mismatch.any():
        details = merged.loc[mismatch, ['stage1_id', 'review_text', 'source_review_text']].copy()
        print('Substantive text mismatch, first rows:')
        for _, row in details.head(5).iterrows():
            print('  ID:', row['stage1_id'])
            print('  Excel:', repr(str(row['review_text'])[:180]))
            print('  Source:', repr(str(row['source_review_text'])[:180]))
        raise ValueError('Text mismatch after newline normalization for IDs: '
                         + ', '.join(details['stage1_id'].astype(str).tolist()[:15]))
    formatting_differences = merged['review_text'].fillna('').ne(merged['source_review_text'].fillna('')).sum()
    print(f'Excel newline/encoding differences normalized: {formatting_differences}')


def main():
    if not INPUT_XLSX.is_file():
        raise FileNotFoundError(f'Human reviewed workbook not found: {INPUT_XLSX}')
    if not INPUT_META.is_file():
        raise FileNotFoundError(f'Original screened sample not found: {INPUT_META}')

    human = pd.read_excel(INPUT_XLSX, sheet_name='Manual Review', engine='openpyxl', dtype={'stage1_id': str})
    meta = pd.read_csv(INPUT_META, low_memory=False, dtype={'stage1_id': str, 'review_id': str})

    required_human = {'stage1_id', 'review_text', 'manual_is_actionable_issue', 'manual_reviewed'}
    required_meta = {'stage1_id', 'game', 'review_id', 'review_text'}
    assert required_human <= set(human.columns), f'Missing workbook columns: {required_human-set(human.columns)}'
    assert required_meta <= set(meta.columns), f'Missing metadata columns: {required_meta-set(meta.columns)}'

    if len(human) != 2000 or human['stage1_id'].duplicated().any():
        raise ValueError('Workbook must contain exactly 2000 unique stage1_id values.')
    if meta['stage1_id'].duplicated().any():
        raise ValueError('Metadata contains duplicate stage1_id values.')

    human['label'] = pd.to_numeric(human['manual_is_actionable_issue'], errors='coerce')
    human['reviewed'] = pd.to_numeric(human['manual_reviewed'], errors='coerce')
    invalid = human.loc[~human['label'].isin([0, 1]) | human['reviewed'].ne(1)]
    if len(invalid):
        raise ValueError('Invalid or unreviewed stage1_id values: ' + ', '.join(invalid['stage1_id'].tolist()[:20]))

    merged = human.merge(
        meta[['stage1_id', 'game', 'review_id', 'review_text']].rename(columns={'review_text': 'source_review_text'}),
        on='stage1_id', how='left', validate='one_to_one', indicator=True
    )
    if not (merged['_merge'] == 'both').all():
        raise ValueError('Some stage1_ids are missing in the selected metadata CSV.')
    audit_text_match(merged)
    # The original CSV is canonical: do not export Excel-escaped text as gold.
    merged['review_text'] = merged['source_review_text']
    if merged['review_id'].duplicated().any():
        raise ValueError('Duplicate Steam review_id in gold data.')

    merged['label'] = merged['label'].astype(int)
    columns = ['stage1_id', 'game', 'review_id', 'review_text', 'label']
    for optional in ['llm_is_actionable_issue', 'llm_confidence', 'manual_notes']:
        if optional in merged.columns:
            columns.append(optional)
    gold = merged[columns].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    save_split(gold, 'stage1_gold.csv')

    # Split A: class-stratified review-level 70/15/15.
    train, temp = train_test_split(gold, test_size=.30, stratify=gold['label'], random_state=SEED)
    val, test = train_test_split(temp, test_size=.50, stratify=temp['label'], random_state=SEED)
    validate_disjoint((train, val, test), 'random')
    for name, frame in [('random_train.csv', train), ('random_val.csv', val), ('random_test.csv', test)]:
        save_split(frame, name)

    # Split B: fixed 2 games test; train/val from the other 8 games.
    # Game selection is deterministic and independent of labels.
    games = np.array(sorted(gold['game'].unique()), dtype=object)
    if len(games) < 3:
        raise ValueError('At least 3 games needed for game holdout.')
    rng = np.random.default_rng(SEED)
    n_holdout = max(1, round(.20 * len(games)))
    held_games = sorted(rng.choice(games, size=n_holdout, replace=False).tolist())
    game_test = gold.loc[gold['game'].isin(held_games)].copy()
    game_train_val = gold.loc[~gold['game'].isin(held_games)].copy()
    game_train, game_val = train_test_split(
        game_train_val, test_size=.20, stratify=game_train_val['label'], random_state=SEED
    )
    validate_disjoint((game_train, game_val, game_test), 'game_holdout')
    if set(game_train['game']) & set(game_test['game']) or set(game_val['game']) & set(game_test['game']):
        raise RuntimeError('Game leakage in holdout split.')
    for name, frame in [('game_holdout_train.csv', game_train), ('game_holdout_val.csv', game_val), ('game_holdout_test.csv', game_test)]:
        save_split(frame, name)

    rows = []
    for strategy, three in [('random', (train, val, test)), ('game_holdout', (game_train, game_val, game_test))]:
        for split_name, frame in zip(['train', 'val', 'test'], three):
            rows.append(counts(frame, split_name, strategy))
    pd.DataFrame(rows).to_csv(OUT / 'split_summary.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'seed': SEED,
        'source_workbook': str(INPUT_XLSX),
        'source_metadata': str(INPUT_META),
        'gold_rows': len(gold),
        'gold_positive': int(gold['label'].sum()),
        'gold_negative': int(len(gold)-gold['label'].sum()),
        'random_split': '70% train / 15% validation / 15% test, stratified by human label',
        'game_holdout_test_games': held_games,
        'game_holdout': '20% games held out for test; remaining reviews split 80/20 train/val (stratified)',
        'warning': 'LLM-balanced sample; not suitable for estimating natural class prevalence. Do not tune hyperparameters on test sets.'
    }
    (OUT / 'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Validated gold: {len(gold)} | positive={manifest["gold_positive"]} | negative={manifest["gold_negative"]}')
    print('Held-out games: ' + ', '.join(held_games))
    print(pd.DataFrame(rows)[['strategy','split','n','positive','negative','positive_rate']].to_string(index=False))
    print('Saved to:', OUT)


if __name__ == '__main__':
    main()
