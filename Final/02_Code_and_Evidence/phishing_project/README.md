# Phishing website detection – code and evidence (PRT565 Assessment 3)

Reproducible comparison of machine learning and deep learning models on the UCI Phishing Websites dataset. The report and presentation are in `../../01_Submission`; rehearsal notes are in `../../03_Study_Only_Do_Not_Submit`.

## Folder structure

```
phishing_project/
├── README.md                        this file
├── train.py                         main experiment (scikit-learn)
├── deep_learning_extension.py       LSTM + simple RNN (PyTorch), run after train.py
├── Phishing_Lab.ipynb               executed walkthrough notebook (dataset → conclusion)
├── run_project.py                   launcher: setup | check | notebook | train | deep
├── Start_Mac.command                one-click Mac setup, then opens the notebook
├── requirements.txt                 pinned core dependencies
├── requirements-deep-learning.txt   adds PyTorch for the optional LSTM/RNN step
├── data/
│   ├── uci_original.zip             original UCI download
│   ├── feature_documentation.txt    text of the feature definitions
│   └── source/                      Training Dataset.arff + Phishing Websites Features.docx
└── results/                         every number, table and figure used in the report/slides
```

## What the code does

`train.py` (≈ 2 min on a laptop CPU)
1. Loads the ARFF (11,055 rows, 30 categorical inputs), maps phishing (−1) to the positive class 1.
2. Audits missing values, repeated inputs (5,270) and conflicting labels (64).
3. EDA: class balance, mutual information, Spearman correlation heat map.
4. Group-safe split: identical input patterns never cross train / validation / test (StratifiedGroupKFold, seed 565).
5. Tunes 9 models on identical grouped folds by phishing F1: Decision Tree, Random Forest, Categorical NB, Multilayer ANN (required) + Logistic Regression, k-NN, SVM (RBF), Gradient Boosting, Deep ANN (3 layers).
6. Selects the winner by CV F1 **before** testing; evaluates all models once on 2,210 test rows.
7. Adds group-bootstrap 95% CIs for F1, exact McNemar tests with Holm correction, permutation importance, a top-10 feature-subset extension, ROC curves, confusion matrices and ANN loss curves.

`deep_learning_extension.py` (≈ 1.5 min) trains bidirectional LSTM and simple RNN networks on the same split and folds, saves their metrics, predictions, McNemar tests, loss curves and the zoomed ROC figure.

## Run it

Python 3.11–3.13 (3.12 recommended). On Mac, open Terminal, type `cd `, drag this folder into Terminal, press Return, then:

```bash
bash Start_Mac.command                                   # creates .venv, installs, opens notebook
.venv/bin/python run_project.py check                    # verify the setup
.venv/bin/python run_project.py train                    # re-run the main experiment
.venv/bin/python -m pip install -r requirements-deep-learning.txt
.venv/bin/python run_project.py deep                     # re-run LSTM / RNN
```

Windows / other: `python -m venv .venv`, activate it, `pip install -r requirements.txt`, then `python train.py` and `python deep_learning_extension.py`. In the notebook choose the **PRT565 Phishing (.venv)** kernel. `RUN_TRAINING = False` shows the saved results; set it to True to regenerate (this overwrites `results/`, so copy it first). If startup fails, read `startup_error.txt`.

## Key results (held-out test, phishing = positive)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| **Random Forest (CV-selected)** | **96.38%** | 96.10% | **95.71%** | **95.91%** | 0.993 |
| SVM (RBF) | 95.38% | 96.11% | 93.36% | 94.72% | 0.992 |
| Deep ANN (3 layers) | 95.07% | **97.18%** | 91.52% | 94.27% | 0.993 |
| Gradient Boosting | 94.89% | 94.46% | 93.97% | 94.21% | 0.993 |
| Multilayer ANN | 94.39% | 95.05% | 92.13% | 93.57% | 0.991 |
| LSTM (bidirectional) | 93.89% | 94.05% | 92.03% | 93.03% | 0.989 |
| Decision Tree | 93.03% | 93.38% | 90.70% | 92.02% | 0.958 |
| Categorical NB | 92.94% | 93.09% | 90.81% | 91.93% | 0.981 |
| Majority baseline | 55.70% | 0.00% | 0.00% | 0.00% | 0.500 |

Full table incl. k-NN, Logistic Regression, simple RNN and the top-10 subset: `results/metrics.csv`, `results/deep_learning_metrics.csv`.

## Results files

| File | Contents |
|---|---|
| `audit.json`, `split_manifest.csv`, `feature_codes.csv` | data audit, row-level split, observed codes |
| `eda_overview.png`, `eda_correlation.png`, `eda_mutual_information.csv` | exploratory analysis |
| `cv_summary.csv`, `*_cv.csv`, `deep_learning_cv.csv` | tuning results for every candidate |
| `metrics.csv`, `deep_learning_metrics.csv`, `classification_reports.json` | test metrics with F1 CIs |
| `predictions.csv`, `deep_learning_predictions.csv` | per-row test predictions and scores |
| `significance_mcnemar.csv`, `deep_learning_mcnemar.csv` | McNemar tests vs Random Forest |
| `permutation_importance.csv/.png`, `selection.json` | explainability and feature subset |
| `model_comparison.png`, `confusion_matrices.png`, `roc_curves.png`, `roc_curves_zoom.png`, `ann_loss_curves.png`, `dl_loss_curves.png` | figures |
| `environment.json`, `deep_learning_environment.json`, `warnings.json`, `demo_examples.json` | reproducibility record, demo cases |

Recorded environment: Python 3.12, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, PyTorch 2.14 (CPU, 1 thread), seed 565. Fitted models are not serialised; the scripts refit them.

**Dataset:** Mohammad, R., & McCluskey, L. (2012). *Phishing websites* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X (CC BY 4.0).
