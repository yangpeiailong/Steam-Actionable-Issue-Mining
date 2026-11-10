"""Prepare a small human audit of Stage-1 / Stage-2 disagreements.

Input: Paper1/data/analysis/stage2_annotation/stage2_review_sample_2000_llm.csv
Output: Paper1/data/analysis/stage1_error_audit/stage1_disagreement_audit.csv

Select all LLM NO_ISSUE among the first 100 completed rows, plus up to 10
LLM-positive controls. No model labels are treated as human gold automatically.
Do not use this audit file to update training until manual review is finished
and original train/val/test membership is checked.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config_local import PAPER1_ANALYSIS_ROOT

INFILE = PAPER1_ANALYSIS_ROOT / 'stage2_annotation' / 'stage2_review_sample_2000_llm.csv'
OUTDIR = PAPER1_ANALYSIS_ROOT / 'stage1_error_audit'
OUTFILE = OUTDIR / 'stage1_disagreement_audit.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-completed', type=int, default=100)
    ap.add_argument('--positive-controls', type=int, default=10)
    args = ap.parse_args()
    with INFILE.open('r', encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        rows = list(r)
    completed = [r for r in rows if (r.get('llm_status') or '').strip().lower() == 'success']
    completed = completed[:args.max_completed]
    if len(completed) < args.max_completed:
        raise ValueError(f'Only {len(completed)} successful rows; expected {args.max_completed}')
    negatives = [r for r in completed if r.get('llm_domains','').strip() == 'NO_ISSUE']
    controls = [r for r in completed if r.get('llm_domains','').strip() != 'NO_ISSUE']
    rng = random.Random(2026)
    controls = rng.sample(controls, min(args.positive_controls, len(controls)))
    chosen = negatives + controls
    rng.shuffle(chosen)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fields = ['audit_id','stage2_id','game','review_id','review_text',
              'stage1_score','llm_domains','llm_evidence',
              'manual_is_actionable_issue','manual_notes','manual_reviewed']
    with OUTFILE.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i,r in enumerate(chosen,1):
            out = {k:r.get(k,'') for k in fields}
            out['audit_id'] = f'AUD-{i:03d}'
            out['manual_is_actionable_issue'] = ''
            out['manual_notes'] = ''
            out['manual_reviewed'] = 0
            # Prevent spreadsheet programs from treating a text value as a formula.
            for field in ('review_text','llm_evidence','manual_notes'):
                val = out.get(field) or ''
                if val.lstrip().startswith(('=','+','-','@')):
                    out[field] = "'" + val
            w.writerow(out)
    print(f'Successful stage2 rows checked: {len(completed)}')
    print(f'LLM NO_ISSUE audit: {len(negatives)}')
    print(f'Positive controls: {len(controls)}')
    print(f'Human review rows: {len(chosen)}')
    print(f'Saved: {OUTFILE}')
    print('Fill manual_is_actionable_issue (0/1) and manual_reviewed=1; do not change other columns.')
    print('After review, exclude original validation/test IDs before any training update.')

if __name__ == '__main__':
    main()
