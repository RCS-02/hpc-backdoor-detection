"""
detect.py

Classify HPC traces as clean or backdoor. Handles both possible readings of
the competition data format:

  --mode row    (default) every row is an independent inference. Each row gets
                its own score and clean/backdoor label.
  --mode trace  the whole file is ONE inference (rows are samples over time).
                Row scores are averaged into a single file-level verdict.
  --mode auto   1 row -> row mode; many rows -> row labels PLUS a file-level
                verdict printed alongside.

Usage:
    python detect.py --csv new_traces.csv
    python detect.py --csv one_inference.csv --mode trace
    python detect.py --csv new_traces.csv --out predictions.csv
"""
import argparse
import json
import numpy as np
import pandas as pd
import joblib


def load_artifacts(model_dir='.'):
    scaler = joblib.load(f'{model_dir}/scaler.joblib')
    model = joblib.load(f'{model_dir}/isolation_forest.joblib')
    with open(f'{model_dir}/model_meta.json') as f:
        meta = json.load(f)
    clean_scores = np.load(f'{model_dir}/clean_train_scores.npy')
    return scaler, model, meta, clean_scores


def score_rows(df, scaler, model, meta):
    missing = [c for c in meta['feature_names'] if c not in df.columns]
    if missing:
        raise ValueError(f'Missing HPC columns: {missing}')
    X = df[meta['feature_names']].apply(pd.to_numeric, errors='coerce').values
    if not np.isfinite(X).all():
        raise ValueError('Input contains NaN or non-finite values')
    return -model.decision_function(scaler.transform(X))  # higher = more anomalous


def trace_threshold(clean_scores, n_rows, contamination, n_boot=5000, seed=0):
    """Threshold for the MEAN row score of an n-row file: the (1-contamination)
    quantile of means of n_rows clean scores resampled from the training set."""
    rng = np.random.default_rng(seed)
    means = rng.choice(clean_scores, size=(n_boot, n_rows), replace=True).mean(axis=1)
    return float(np.quantile(means, 1 - contamination))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--model-dir', default='.')
    ap.add_argument('--mode', choices=['row', 'trace', 'auto'], default='row')
    ap.add_argument('--out', default=None, help='Write per-row predictions CSV')
    args = ap.parse_args()

    scaler, model, meta, clean_scores = load_artifacts(args.model_dir)
    df = pd.read_csv(args.csv)
    scores = score_rows(df, scaler, model, meta)
    row_preds = (scores > meta['threshold']).astype(int)  # 1 = backdoor

    mode = args.mode
    if mode == 'auto':
        mode = 'row' if len(df) == 1 else 'both'

    if mode in ('row', 'both'):
        print(f'[row-level] {row_preds.sum()}/{len(row_preds)} rows flagged as backdoor '
              f'({100*row_preds.mean():.2f}%)')
    if mode in ('trace', 'both'):
        thr = trace_threshold(clean_scores, len(scores), meta['contamination'])
        agg = float(scores.mean())
        verdict = 'backdoor' if agg > thr else 'clean'
        print(f'[trace-level] mean score {agg:.4f} vs threshold {thr:.4f} -> {verdict}')

    if args.out:
        out = df.copy()
        out['anomaly_score'] = scores
        out['prediction'] = np.where(row_preds == 1, 'backdoor', 'clean')
        out.to_csv(args.out, index=False)
        print(f'Wrote per-row predictions to {args.out}')


if __name__ == '__main__':
    main()
