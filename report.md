# Detection of Backdoor Attacks using HPCs - Method Report


## 1. Problem and data interpretation

We are given hardware performance counter (HPC) traces from benign model inferences only: 800 rows with three columns (cache-references, cycles, LLC-loads). Following the problem statement, each row is one clean inference. The task is to train an anomaly detector that, at evaluation time, labels traces from backdoor inferences differently from clean ones, without ever seeing a backdoor example. This is a strict one-class (novelty detection) setting.


## 2. Exploratory analysis

- No missing values; 800 rows, 3 numeric features.
- cache-references and LLC-loads are strongly correlated (r = 0.92) and both are bimodal: two clusters of normal behaviour (Figure 1, 2).
- cycles is roughly unimodal with a short right tail of a few extreme points (about 4.85e9 to 5.0e9, against a typical 4.70e9 to 4.80e9). It correlates moderately with the other two (r about 0.53).
Consequence: a per-feature Gaussian or z-score threshold would treat one legitimate cluster as anomalous, and treating features independently would discard the strong cache-references / LLC-loads correlation. The detector should model the joint, multi-modal structure.


![Figure 1: Distribution of each HPC feature (clean traces).](hist.png)


![Figure 2: Pairwise scatter of the clean traces.](scatter.png)


## 3. Model selection

Candidates: Isolation Forest, One-Class SVM (RBF), Local Outlier Factor (novelty mode), and a 2-component Gaussian Mixture Model (matching the observed bimodality). Because no backdoor traces are available, true positive rate cannot be measured during development. Instead we used 5-fold cross-validation on clean data only: in each fold the scaler and model are fit on the training folds, the score threshold is set at the target contamination percentile (2%) of the training scores, and we measure the fraction of the held-out clean fold that gets flagged. A well-calibrated detector flags close to 2% of unseen clean data; much more means excess false positives, much less means a threshold that may be too loose.


| Model | Held-out clean flag rate (target 2%) |
|---|---|
| Isolation Forest (selected) | 1.75% |
| LOF (novelty) | 2.25% |
| GMM (k=2) | 2.75% |
| One-Class SVM (RBF) | 3.25% |

Isolation Forest was closest to the target and was selected. It also makes no distributional assumptions and needs no cluster-count or kernel tuning.


## 4. Final pipeline

- StandardScaler fit on all 800 clean traces.
- IsolationForest (200 trees, random_state=42) fit on the scaled data.
- Anomaly score = negative decision_function (higher = more anomalous).
- Decision threshold = 98th percentile of the clean training scores (0.0862). A trace scoring above it is labelled backdoor, otherwise clean.
This flags 2.00% of the training set by construction and 1.75% (+/- 0.73%) of held-out clean folds in cross-validation, so the false positive rate is expected to stay near 2% on unseen clean inputs. The code is in train_detector.py (training) and detect.py (inference); the model is saved as scaler.joblib, isolation_forest.joblib and model_meta.json.


## 5. Limitations

- Only 3 HPC channels and 800 samples were available; more channels or more benign samples would likely improve separability.
- The 2% contamination level is a knob trading FPR against TPR. Without backdoor examples we can calibrate only the FPR side. If labelled backdoor traces become available, the threshold should be re-tuned against the ROC curve (AUROC / F1).
- No backdoor results are reported because no backdoor data was provided; the private-set metrics are therefore not claimed here.
