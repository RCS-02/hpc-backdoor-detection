"""
train_detector.py

Trains an unsupervised anomaly detector on benign HPC (hardware performance
counter) traces, and saves it for use at inference time to flag backdoor
inputs.

Why Isolation Forest?
- We are only given clean/benign traces (strict anomaly-detection setting,
  no labeled backdoor examples to train against).
- Exploratory analysis showed `cache-references` and `LLC-loads` are
  bimodal (two clusters of normal behavior), so a single-Gaussian /
  per-feature z-score approach would misflag a chunk of legitimate traces.
- Isolation Forest makes no distributional assumptions, handles
  multi-modal data well, and in 5-fold cross-validation on held-out clean
  data it produced the flag-rate closest to the target contamination level
  (i.e. best-calibrated false-positive rate) among IsolationForest,
  One-Class SVM (RBF), LOF, and a 2-component GMM.

Usage:
    python train_detector.py --csv trace.csv --contamination 0.02
"""
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold


def calibrate_flag_rate(X, contamination, n_splits=5, seed=42):
    """K-fold sanity check: what fraction of held-out CLEAN data gets
    flagged at this contamination level? Should be close to `contamination`.
    Since we have no backdoor examples, this is our only calibration check."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    flagged_fracs = []
    for train_idx, val_idx in kf.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xval = scaler.transform(X[val_idx])
        model = IsolationForest(n_estimators=200, contamination='auto',
                                 random_state=seed).fit(Xtr)
        scores_tr = -model.decision_function(Xtr)
        scores_val = -model.decision_function(Xval)
        thresh = np.percentile(scores_tr, 100 * (1 - contamination))
        flagged_fracs.append((scores_val > thresh).mean())
    return float(np.mean(flagged_fracs)), float(np.std(flagged_fracs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='trace.csv', help='Path to benign HPC trace CSV')
    ap.add_argument('--contamination', type=float, default=0.02,
                     help='Target fraction of clean training data to treat as the '
                          'anomaly-score threshold boundary (lower = stricter, fewer '
                          'false positives but risk of missing subtle backdoors)')
    ap.add_argument('--out-dir', default='.', help='Where to save model artifacts')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    feature_names = df.columns.tolist()
    X = df.values
    print(f'Loaded {X.shape[0]} clean traces, {X.shape[1]} features: {feature_names}')

    # Calibration check on held-out clean folds
    mean_flag, std_flag = calibrate_flag_rate(X, args.contamination)
    print(f'K-fold calibration: held-out clean flag rate = '
          f'{mean_flag:.4f} +/- {std_flag:.4f} (target {args.contamination})')

    # Fit final scaler + model on ALL clean data
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = IsolationForest(n_estimators=200, contamination='auto',
                             random_state=42).fit(Xs)

    # Threshold = percentile of TRAINING anomaly scores, matching contamination
    train_scores = -model.decision_function(Xs)
    threshold = float(np.percentile(train_scores, 100 * (1 - args.contamination)))
    print(f'Final threshold (score > threshold => flag as backdoor): {threshold:.4f}')

    # Save artifacts
    joblib.dump(scaler, f'{args.out_dir}/scaler.joblib')
    joblib.dump(model, f'{args.out_dir}/isolation_forest.joblib')
    # Clean-score distribution, used by detect.py to calibrate file-level verdicts
    np.save(f'{args.out_dir}/clean_train_scores.npy', train_scores)
    meta = {
        'feature_names': feature_names,
        'contamination': args.contamination,
        'threshold': threshold,
        'train_flag_rate_mean': mean_flag,
        'train_flag_rate_std': std_flag,
        'n_train_samples': int(X.shape[0]),
    }
    with open(f'{args.out_dir}/model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved scaler.joblib, isolation_forest.joblib, model_meta.json to {args.out_dir}')


if __name__ == '__main__':
    main()
