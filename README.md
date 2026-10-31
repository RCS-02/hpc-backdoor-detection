# Problem 1: Backdoor Detection via Hardware Performance Counters

Unsupervised anomaly detector that flags backdoor-triggered model inferences
using HPC traces (`cache-references`, `cycles`, `LLC-loads`), trained only
on benign traces.

## Files

| File | Purpose |
|---|---|
| `train_detector.py` | Trains the detector on benign traces, saves model artifacts |
| `detect.py` | Loads the saved model and classifies new traces |
| `scaler.joblib` | Fitted `StandardScaler` |
| `isolation_forest.joblib` | Fitted `IsolationForest` detector |
| `model_meta.json` | Feature order, decision threshold, calibration stats |
| `clean_train_scores.npy` | Clean anomaly scores used to calibrate file-level verdicts |
| `trace.csv` | Provided benign training traces (800 rows) |
| `report.md` | Method write-up (convert to PDF for submission) |

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train_detector.py --csv trace.csv --contamination 0.02
```

`--contamination` sets the target false-positive rate on clean data (2% by
default). Lower it for a stricter detector (fewer false alarms, but risk of
missing subtle backdoors); raise it to catch more anomalies at the cost of
more false positives on clean inputs.

## Detect

```bash
python detect.py --csv new_traces.csv --out predictions.csv
```

Outputs an `anomaly_score` (higher = more anomalous) and a `prediction`
(`clean` / `backdoor`) column for each row.

The data format can be read two ways, so `detect.py` supports both:

| `--mode` | Meaning |
|---|---|
| `row` (default) | Each row is one independent inference; every row gets its own label |
| `trace` | The whole file is one inference; row scores are averaged into one file-level verdict |
| `auto` | 1 row -> row mode; several rows -> row labels plus a file-level verdict |

The file-level threshold is calibrated by resampling clean training scores
(`clean_train_scores.npy`) to the same number of rows as the input file.

## Method summary

See `report.md` for the full write-up. In short: exploratory analysis
showed the benign traces are **bimodal** in `cache-references` and
`LLC-loads` (two clusters of normal behavior), which rules out simple
per-feature Gaussian/z-score thresholds. We compared four unsupervised
detectors (Isolation Forest, One-Class SVM, LOF, 2-component GMM) via
5-fold cross-validation, checking how well each detector's flag rate on
**held-out clean folds** matched the target contamination level (our only
calibration signal, since no backdoor traces are available for training).
Isolation Forest was best calibrated and was selected as the final model.
