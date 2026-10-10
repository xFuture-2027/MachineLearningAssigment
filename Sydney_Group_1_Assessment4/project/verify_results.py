"""Check bundled evidence without retraining or changing outputs.

Run from any directory: python /path/to/project/verify_results.py
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from train import load_data, SEED
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    X, y, groups, source = load_data()
    results = ROOT / 'results'
    audit = json.loads((results / 'audit.json').read_text())
    require(hashlib.sha256(source.read_bytes()).hexdigest() == audit['sha256_arff'],
            'Source checksum does not match the recorded audit.')
    manifest = pd.read_csv(results / 'split_manifest.csv')
    require(np.array_equal(manifest.row_id, np.arange(len(X))), 'Invalid manifest row IDs.')
    require(np.array_equal(manifest.group.to_numpy(), groups), 'Input groups do not match source.')
    train, test = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups))
    expected = np.full(len(X), 'train', dtype=object)
    expected[test] = 'test'
    require(np.array_equal(manifest.split, expected), 'Split differs from the declared procedure.')
    require(not set(groups[train]) & set(groups[test]), 'Train/test group overlap.')
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
    for a, b in inner.split(X.iloc[train], y.iloc[train], groups[train]):
        require(not set(groups[train][a]) & set(groups[train][b]), 'Inner-fold group overlap.')
    prepared = pd.read_csv(results / 'assessment4/prepared_dataset.csv')
    require(prepared.shape == (len(X), X.shape[1] + 3), 'Prepared CSV dimensions differ.')
    pd.testing.assert_frame_equal(prepared[X.columns], X, check_dtype=False)
    require(np.array_equal(prepared.row_id, manifest.row_id), 'Prepared row IDs differ.')
    require(np.array_equal(prepared.phishing, y), 'Prepared target differs.')
    require(np.array_equal(prepared.split, manifest.split), 'Prepared split differs.')
    metrics = pd.concat([pd.read_csv(results / 'metrics.csv'), pd.read_csv(results / 'deep_learning_metrics.csv')], ignore_index=True)
    predictions = pd.concat([pd.read_csv(results / 'predictions.csv', float_precision='round_trip'), pd.read_csv(results / 'deep_learning_predictions.csv', float_precision='round_trip')], ignore_index=True)
    require(metrics.model.is_unique, 'Duplicate metric rows.')
    require(set(metrics.model) == set(predictions.model), 'Model lists differ.')
    require(len(metrics) == 13, 'Expected eleven classifiers, one subset and one baseline.')
    for row in metrics.itertuples(index=False):
        pred = predictions[predictions.model.eq(row.model)].sort_values('row_id')
        require(np.array_equal(pred.row_id, np.sort(test)), f'{row.model}: missing or duplicate test IDs.')
        require(np.array_equal(pred.actual, y.iloc[pred.row_id].to_numpy()), f'{row.model}: labels differ from source.')
        require(set(pred.predicted).issubset({0, 1}), f'{row.model}: invalid predictions.')
        require(np.isfinite(pred.phishing_score).all() and pred.phishing_score.between(0, 1).all(),
                f'{row.model}: invalid scores.')
        for name, func in [('accuracy', accuracy_score), ('precision', precision_score),
                           ('recall', recall_score), ('f1', f1_score)]:
            kwargs = {} if name == 'accuracy' else {'zero_division': 0}
            value = func(pred.actual, pred.predicted, **kwargs)
            require(np.isclose(value, getattr(row, name), atol=1e-12, rtol=0), f'{row.model}: {name} mismatch.')
        require(np.isclose(roc_auc_score(pred.actual, pred.phishing_score), row.roc_auc,
                           atol=1e-12, rtol=0), f'{row.model}: AUC mismatch.')
        require(np.array_equal(confusion_matrix(pred.actual, pred.predicted, labels=[0, 1]).ravel(),
                               [row.TN, row.FP, row.FN, row.TP]), f'{row.model}: confusion matrix mismatch.')
    print('PASS: source checksum, original/prepared rows, targets, grouped splits, all thirteen models\' test IDs, metrics and confusion counts agree.')
    print('This verifies saved evidence consistency; it does not retrain models or validate original labels.')

if __name__ == '__main__':
    main()
