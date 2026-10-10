from pathlib import Path
import json, hashlib, platform, time, warnings, zipfile, urllib.request
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.io import arff
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve, classification_report)
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SEED = 565
URL = 'https://archive.ics.uci.edu/static/public/327/phishing+websites.zip'
REQUIRED = ['Decision Tree', 'Random Forest', 'Categorical NB', 'Multilayer ANN']
INK, TEAL, SAND, BLUE, RED, GREY = '#10213a', '#0f8480', '#dca84b', '#5b8fb0', '#c0463c', '#9aa5b1'


def category_codes(X):
    """Shift codes -1/0/1 to 0/1/2 (CategoricalNB and HistGB need non-negative codes)."""
    return np.asarray(X, dtype=int) + 1


def discrete_mi(X, y):
    return mutual_info_classif(X, y, discrete_features=True, random_state=SEED)


def load_data():
    p = ROOT / 'data/source/Training Dataset.arff'
    if not p.exists():
        z = ROOT / 'data/uci_original.zip'; z.parent.mkdir(parents=True, exist_ok=True)
        if not z.exists():
            with urllib.request.urlopen(URL, timeout=90) as r: z.write_bytes(r.read())
        with zipfile.ZipFile(z) as archive:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(archive.read('Training Dataset.arff'))
    records, _ = arff.loadarff(p)
    df = pd.DataFrame(records).astype(int)
    assert df.shape == (11055, 31), 'Unexpected dataset version: inspect before proceeding'
    X = df.drop(columns='Result')
    assert set(np.unique(X)).issubset({-1, 0, 1}) and set(df.Result) == {-1, 1}
    # Original -1 = phishing, +1 = legitimate. Our positive class is phishing.
    y = (df.Result == -1).astype(int)
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    assert len(np.unique(groups)) == len(X.drop_duplicates()), 'Hash collision'
    return X, y, groups, p


def make_split(X, y, groups):
    """Outer group-safe holdout (first of 5 folds) + three grouped inner folds."""
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(outer.split(X, y, groups))
    assert not set(groups[train]) & set(groups[test])
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
    folds = list(inner.split(X.iloc[train], y.iloc[train], groups[train]))
    for a, b in folds: assert not set(groups[train][a]) & set(groups[train][b])
    return train, test, folds


def model_specs():
    def onehot(): return OneHotEncoder(categories=[[-1, 0, 1]] * 30, handle_unknown='error', sparse_output=False)
    codes = lambda: FunctionTransformer(category_codes)
    return {
        # ---- four models required by the brief ----
        'Decision Tree': (Pipeline([('encode', onehot()), ('model', DecisionTreeClassifier(random_state=SEED))]),
                          {'model__max_depth': [5, None], 'model__min_samples_leaf': [1, 5]}),
        'Random Forest': (Pipeline([('encode', onehot()), ('model', RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=SEED))]),
                          {'model__max_depth': [10, None], 'model__min_samples_leaf': [1, 3]}),
        'Categorical NB': (Pipeline([('encode', codes()), ('model', CategoricalNB(min_categories=3))]),
                           {'model__alpha': [0.1, 1., 10.]}),
        'Multilayer ANN': (Pipeline([('encode', onehot()), ('model', MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=250, early_stopping=False, random_state=SEED))]),
                           {'model__alpha': [0.001, 0.01]}),
        # ---- additional algorithms ("plus any other algorithms as you see fit") ----
        'Logistic Regression': (Pipeline([('encode', onehot()), ('model', LogisticRegression(max_iter=2000))]),
                                {'model__C': [0.1, 1., 10.]}),
        'k-Nearest Neighbours': (Pipeline([('encode', onehot()), ('model', KNeighborsClassifier())]),
                                 {'model__n_neighbors': [3, 7, 15], 'model__weights': ['uniform', 'distance']}),
        'SVM (RBF)': (Pipeline([('encode', onehot()), ('model', SVC(kernel='rbf', probability=True, random_state=SEED))]),
                      {'model__C': [1., 10.]}),
        'Gradient Boosting': (Pipeline([('encode', codes()), ('model', HistGradientBoostingClassifier(categorical_features=[True] * 30, random_state=SEED))]),
                              {'model__learning_rate': [0.05, 0.1], 'model__max_leaf_nodes': [15, 31]}),
        # ---- deeper multilayer network (deep learning extension) ----
        'Deep ANN (3 layers)': (Pipeline([('encode', onehot()), ('model', MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, early_stopping=False, random_state=SEED))]),
                                {'model__alpha': [0.001, 0.01]}),
    }


def group_bootstrap_f1(y_true, pred, groups, n_boot=1000, seed=SEED):
    """95% CI for F1 by resampling whole input-pattern groups (respects repeated rows)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    tp = np.bincount(inv, weights=(y_true == 1) & (pred == 1), minlength=len(uniq))
    fp = np.bincount(inv, weights=(y_true == 0) & (pred == 1), minlength=len(uniq))
    fn = np.bincount(inv, weights=(y_true == 1) & (pred == 0), minlength=len(uniq))
    scores = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        TP, FP, FN = w @ tp, w @ fp, w @ fn
        scores[b] = 2 * TP / (2 * TP + FP + FN) if TP else 0.
    return np.percentile(scores, [2.5, 97.5])


def plot_model_comparison(results, output_path):
    """Show zero baseline scores explicitly; does not change model evaluation."""
    fig, ax = plt.subplots(figsize=(12, 5.2))
    results.set_index('model')[['precision', 'recall', 'f1']].plot.bar(
        ax=ax, color=[BLUE, SAND, TEAL], width=0.8)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score (phishing = positive)')
    ax.set_xlabel(''); ax.spines[['top', 'right']].set_visible(False)
    baseline = results.reset_index(drop=True).index[results.model.eq('Majority baseline')][0]
    ax.annotate(f'All three scores = 0\nAlways predicts legitimate\nAccuracy = {results.loc[results.model.eq("Majority baseline"), "accuracy"].iloc[0]*100:.2f}%',
                xy=(baseline, 0), xytext=(baseline-.2, .22), ha='right', fontsize=9,
                arrowprops={'arrowstyle': '->', 'color': INK})
    plt.xticks(rotation=25, ha='right'); fig.tight_layout()
    fig.savefig(output_path, dpi=170); plt.close(fig)


def main():
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    X, y, groups, p = load_data()
    train, test, folds = make_split(X, y, groups)
    # Keep repeats, including conflicting labels, together. Do not discard difficult cases.
    audit = {'rows': len(X), 'features': X.shape[1], 'missing': int(X.isna().sum().sum()),
             'duplicate_feature_rows': int(X.duplicated().sum()),
             'duplicate_full_rows': int(pd.concat([X, y.rename('target')], axis=1).duplicated().sum()),
             'unique_feature_patterns': len(np.unique(groups)),
             'conflicting_label_patterns': int(pd.DataFrame({'g': groups, 'y': y}).groupby('g').y.nunique().gt(1).sum()),
             'phishing': int(y.sum()), 'legitimate': int((1 - y).sum()), 'train_rows': len(train), 'test_rows': len(test),
             'train_phishing': int(y.iloc[train].sum()), 'test_phishing': int(y.iloc[test].sum()),
             'split_seed': SEED, 'target_mapping': {'original_-1': 'phishing=1', 'original_1': 'legitimate=0'},
             'sha256_arff': hashlib.sha256(p.read_bytes()).hexdigest()}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2))
    pd.DataFrame({'row_id': np.arange(len(X)), 'group': groups,
                  'split': np.where(np.isin(np.arange(len(X)), test), 'test', 'train')}).to_csv(out / 'split_manifest.csv', index=False)
    pd.DataFrame({'feature': X.columns, 'observed_codes': [str(sorted(X[c].unique().tolist())) for c in X]}).to_csv(out / 'feature_codes.csv', index=False)
    print(json.dumps(audit, indent=2), flush=True)

    # ---------- Exploratory data analysis (development rows only for MI) ----------
    mi = pd.Series(discrete_mi(X.iloc[train], y.iloc[train]), index=X.columns).sort_values(ascending=False)
    mi.rename('mutual_information').to_csv(out / 'eda_mutual_information.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 2.2]})
    counts = [audit['legitimate'], audit['phishing']]
    axes[0].bar(['Legitimate', 'Phishing'], counts, color=[BLUE, RED])
    for i, c in enumerate(counts): axes[0].text(i, c + 80, f'{c:,}\n({c / len(X):.1%})', ha='center', fontsize=11)
    axes[0].set_ylim(0, 7600); axes[0].set_title('Class balance (all 11,055 rows)'); axes[0].spines[['top', 'right']].set_visible(False)
    top = mi.head(12)[::-1]
    axes[1].barh(top.index, top.values, color=TEAL); axes[1].set_xlabel('Mutual information with label (development rows)')
    axes[1].set_title('Most informative indicators'); axes[1].spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'eda_overview.png', dpi=170); plt.close(fig)
    corr = X.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(9, 8)); im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(30)); ax.set_yticks(range(30)); ax.set_xticklabels(X.columns, rotation=90, fontsize=6); ax.set_yticklabels(X.columns, fontsize=6)
    fig.colorbar(im, fraction=0.046); ax.set_title('Spearman correlation between the 30 indicators'); fig.tight_layout()
    fig.savefig(out / 'eda_correlation.png', dpi=170); plt.close(fig)

    # ---------- Tune every model on identical grouped folds ----------
    specs = model_specs()
    fitted = {}; cvrows = []; warning_log = []
    with threadpool_limits(limits=1):
        for name, (pipe, grid) in specs.items():
            print('Training', name, flush=True); start = time.perf_counter()
            search = GridSearchCV(pipe, grid, scoring='f1', cv=folds, n_jobs=1, refit=True, return_train_score=True, error_score='raise')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); search.fit(X.iloc[train], y.iloc[train])
            warning_log.extend([{'model': name, 'category': w.category.__name__, 'message': str(w.message)[:300]} for w in caught
                                if not issubclass(w.category, (DeprecationWarning, FutureWarning))])
            fitted[name] = search.best_estimator_
            pd.DataFrame(search.cv_results_).to_csv(out / (name.replace(' ', '_').replace('(', '').replace(')', '') + '_cv.csv'), index=False)
            ix = search.best_index_
            cvrows.append({'model': name, 'required': name in REQUIRED, 'cv_f1': float(search.best_score_),
                           'cv_sd': float(search.cv_results_['std_test_score'][ix]),
                           'train_cv_f1': float(search.cv_results_['mean_train_score'][ix]),
                           'search_seconds': time.perf_counter() - start, 'parameters': str(search.best_params_)})
        # Choose the primary model before touching the test outcomes.
        winner = max(cvrows, key=lambda r: r['cv_f1'])['model']
        winner_required = max([r for r in cvrows if r['required']], key=lambda r: r['cv_f1'])['model']
        # Prespecified extension: top 10 original features, selected inside every CV fold.
        base = clone(fitted[winner_required]); encoder = base.named_steps['encode']
        if isinstance(encoder, OneHotEncoder): encoder.set_params(categories=[[-1, 0, 1]] * 10)
        subset = Pipeline([('select', SelectKBest(discrete_mi, k=10)), *base.steps])
        subset_cv = GridSearchCV(subset, {}, scoring='f1', cv=folds, n_jobs=1, error_score='raise')
        subset_cv.fit(X.iloc[train], y.iloc[train]); sub_name = 'Top 10 ' + winner_required; fitted[sub_name] = subset_cv.best_estimator_
        selected = X.columns[subset_cv.best_estimator_.named_steps['select'].get_support()].tolist()
        cvrows.append({'model': sub_name, 'required': False, 'cv_f1': float(subset_cv.best_score_),
                       'cv_sd': float(subset_cv.cv_results_['std_test_score'][0]),
                       'parameters': 'Fixed best full-model settings; exploratory extension'})
        fitted['Majority baseline'] = DummyClassifier(strategy='most_frequent').fit(X.iloc[train], y.iloc[train])

        # ---------- Single final evaluation on the held-out test set ----------
        yt = y.iloc[test].to_numpy(); gt = groups[test]
        rows = []; predictions = []; reports = {}; preds = {}; probs = {}
        for name, model in fitted.items():
            pred = model.predict(X.iloc[test]); prob = model.predict_proba(X.iloc[test])[:, 1]
            preds[name], probs[name] = pred, prob
            tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
            lo, hi = group_bootstrap_f1(yt, pred, gt)
            rows.append({'model': name, 'accuracy': accuracy_score(yt, pred), 'precision': precision_score(yt, pred, zero_division=0),
                         'recall': recall_score(yt, pred, zero_division=0), 'f1': f1_score(yt, pred, zero_division=0),
                         'f1_ci_low': lo, 'f1_ci_high': hi, 'roc_auc': roc_auc_score(yt, prob),
                         'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)})
            predictions.append(pd.DataFrame({'row_id': test, 'model': name, 'actual': yt, 'predicted': pred, 'phishing_score': prob}))
            reports[name] = classification_report(yt, pred, target_names=['legitimate', 'phishing'], output_dict=True, zero_division=0)
        results = pd.DataFrame(rows); results.to_csv(out / 'metrics.csv', index=False)
        pd.DataFrame(cvrows).to_csv(out / 'cv_summary.csv', index=False)
        all_predictions = pd.concat(predictions, ignore_index=True)
        temporary = out / 'predictions.tmp'; temporary.write_text(all_predictions.to_csv(index=False)); temporary.replace(out / 'predictions.csv')
        assert len(pd.read_csv(out / 'predictions.csv')) == len(test) * len(fitted)
        (out / 'classification_reports.json').write_text(json.dumps(reports, indent=2))

        # McNemar exact test: is the CV-selected model's error pattern different from each other model?
        sig = []
        for name in fitted:
            if name in (winner, 'Majority baseline'): continue
            a = preds[winner] == yt; b = preds[name] == yt
            n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
            pval = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            sig.append({'reference': winner, 'model': name, 'reference_right_other_wrong': n01,
                        'reference_wrong_other_right': n10, 'mcnemar_exact_p': pval})
        sig = pd.DataFrame(sig).sort_values('mcnemar_exact_p')
        # Holm step-down correction for testing several comparisons at once.
        m = len(sig); sig['holm_adjusted_p'] = np.minimum(1, np.maximum.accumulate(sig.mcnemar_exact_p.to_numpy() * (m - np.arange(m))))
        sig['significant_at_0.05'] = sig.holm_adjusted_p < 0.05
        sig.to_csv(out / 'significance_mcnemar.csv', index=False)

        # Permutation importance of the selected model on held-out rows (explanation only, no selection).
        pi = permutation_importance(fitted[winner], X.iloc[test], yt, scoring='f1', n_repeats=10, random_state=SEED, n_jobs=1)
        imp = pd.DataFrame({'feature': X.columns, 'importance_mean': pi.importances_mean, 'importance_sd': pi.importances_std}).sort_values('importance_mean', ascending=False)
        imp.to_csv(out / 'permutation_importance.csv', index=False)

        (out / 'selection.json').write_text(json.dumps({
            'primary_model_selected_by_cv': winner, 'best_required_model_by_cv': winner_required,
            'subset_base_model': winner_required, 'subset_features': selected, 'subset_cv_f1': subset_cv.best_score_,
            'note': 'Selection uses development CV F1 only. No changes to models or selection after test evaluation.'}, indent=2))
        (out / 'warnings.json').write_text(json.dumps(warning_log, indent=2))
        (out / 'environment.json').write_text(json.dumps({'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                                                          'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__, 'seed': SEED}, indent=2))

    # ---------- Figures ----------
    main_models = [m for m in specs]
    r = results.set_index('model').loc[main_models + [sub_name, 'Majority baseline']]
    plot_model_comparison(r.reset_index(), out / 'model_comparison.png')

    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for ax, name in zip(axes.ravel(), main_models):
        rr = results.set_index('model').loc[name]; m = np.array([[rr.TN, rr.FP], [rr.FN, rr.TP]], dtype=int); ax.imshow(m, cmap='Blues')
        for i in range(2):
            for j in range(2): ax.text(j, i, str(m[i, j]), ha='center', va='center', color='white' if m[i, j] > m.max() / 2 else 'black', fontsize=15)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'], xlabel='Predicted', ylabel='Actual', title=name)
    fig.tight_layout(); fig.savefig(out / 'confusion_matrices.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    palette = dict(zip(main_models, ['#8c6bb1', TEAL, SAND, RED, GREY, '#6b8e23', BLUE, INK, '#e377c2']))
    for name in main_models:
        fpr, tpr, _ = roc_curve(yt, probs[name]); ax.plot(fpr, tpr, lw=2.2 if name == winner else 1.3, color=palette[name],
                                                          label=f"{name} (AUC {results.set_index('model').loc[name, 'roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls='--', color='#999', lw=1, label='Chance (AUC 0.500)')
    ax.set(xlabel='False positive rate', ylabel='True positive rate (recall)', title='ROC curves on the held-out test set')
    ax.legend(fontsize=8, loc='lower right'); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'roc_curves.png', dpi=170); plt.close(fig)

    t = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5)); ax.barh(t.feature, t.importance_mean, xerr=t.importance_sd, color=TEAL)
    ax.set_xlabel('Drop in test F1 when the feature is shuffled'); ax.set_title(f'Permutation importance: {winner}'); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'permutation_importance.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, col in [('Multilayer ANN', RED), ('Deep ANN (3 layers)', INK)]:
        ax.plot(fitted[name].named_steps['model'].loss_curve_, color=col, label=name)
    ax.set(xlabel='Epoch', ylabel='Training log-loss', title='ANN training loss (final refit on development rows)'); ax.legend(); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'ann_loss_curves.png', dpi=170); plt.close(fig)

    # ---------- Saved example predictions for the demo ----------
    model = fitted[winner]; pred = preds[winner]; examples = []
    for tag, mask in [('correct_phishing', (yt == 1) & (pred == 1)), ('missed_phishing', (yt == 1) & (pred == 0)), ('false_alarm', (yt == 0) & (pred == 1))]:
        ids = np.flatnonzero(mask)
        if len(ids):
            k = ids[0]; examples.append({'case': tag, 'row_id': int(test[k]), 'actual': int(yt[k]), 'prediction': int(pred[k]), 'features': X.iloc[test[k]].to_dict()})
    (out / 'demo_examples.json').write_text(json.dumps(examples, indent=2))
    print(results.round(4).to_string(index=False), flush=True)
    print('CV-selected model:', winner, '| best required model:', winner_required, flush=True)
    return results


if __name__ == '__main__': main()
