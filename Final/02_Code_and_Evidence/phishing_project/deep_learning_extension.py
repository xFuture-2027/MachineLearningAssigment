"""PRT565 Assessment 3: optional deep-learning extension (LSTM and simple RNN, PyTorch).

Run AFTER train.py:   python deep_learning_extension.py
Needs:                pip install -r requirements-deep-learning.txt   (adds PyTorch)

Why: the brief suggests LSTM/RNN to make the study more comprehensive. Each website
is turned into a sequence of 30 tokens (one per indicator, token = feature x code).
The recurrent network reads the tokens in the fixed ARFF column order. There is no
natural time order in these features, so this tests whether sequence models add
anything over the feed-forward ANN on tabular data.

Protocol is identical to train.py: same grouped development/test split, same three
grouped folds, phishing F1 as the selection metric, test touched once.
 - For each fold: train with early stopping on that fold's validation part.
 - CV F1 = mean validation F1. Final epoch count = median best epoch.
 - Refit on all development rows for that many epochs, evaluate on the test set.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import load_data, make_split, group_bootstrap_f1, SEED

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
MAX_EPOCHS, PATIENCE, BATCH, LR = 60, 8, 128, 2e-3
torch.set_num_threads(1)


def tokens(X):
    """Feature i with code c (-1/0/1) -> token 3*i + (c+1): 90 possible tokens."""
    codes = np.asarray(X, dtype=int) + 1
    return torch.tensor(codes + 3 * np.arange(codes.shape[1]), dtype=torch.long)


class SeqNet(nn.Module):
    def __init__(self, cell='lstm', emb=16, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(90, emb)
        rnn = nn.LSTM if cell == 'lstm' else nn.RNN
        self.rnn = rnn(emb, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * hidden, 1))

    def forward(self, t):
        h, _ = self.rnn(self.embed(t))
        return self.head(h.mean(dim=1)).squeeze(1)   # mean-pool over the 30 steps


def predict(net, T):
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(T)).numpy()


def fit(cell, Ttr, ytr, Tva=None, yva=None, epochs=MAX_EPOCHS):
    torch.manual_seed(SEED); np.random.seed(SEED)
    net = SeqNet(cell); opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(); ytr_t = torch.tensor(ytr, dtype=torch.float32)
    g = torch.Generator().manual_seed(SEED)
    best, best_ep, best_state, wait, curve = -1, 0, None, 0, []
    for ep in range(1, epochs + 1):
        net.train(); perm = torch.randperm(len(Ttr), generator=g); total = 0.
        for i in range(0, len(Ttr), BATCH):
            b = perm[i:i + BATCH]; opt.zero_grad()
            loss = loss_fn(net(Ttr[b]), ytr_t[b]); loss.backward(); opt.step(); total += loss.item() * len(b)
        curve.append(total / len(Ttr))
        if Tva is not None:
            f1 = f1_score(yva, predict(net, Tva) >= 0.5)
            if f1 > best: best, best_ep, best_state, wait = f1, ep, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= PATIENCE: break
    if best_state is not None: net.load_state_dict(best_state)
    return net, best, best_ep, curve


def main():
    X, y, groups, _ = load_data()
    train, test, folds = make_split(X, y, groups)
    Tdev, ydev = tokens(X.iloc[train]), y.iloc[train].to_numpy()
    Tte, yte = tokens(X.iloc[test]), y.iloc[test].to_numpy()
    names = {'lstm': 'LSTM (bidirectional)', 'rnn': 'Simple RNN (bidirectional)'}
    cvrows, rows, preds, curves = [], [], [], {}
    for cell, name in names.items():
        start = time.perf_counter(); scores, eps = [], []
        for k, (a, b) in enumerate(folds):
            _, f1, ep, _ = fit(cell, Tdev[a], ydev[a], Tdev[b], ydev[b])
            scores.append(f1); eps.append(ep); print(f'{name} fold {k + 1}: val F1 {f1:.4f} at epoch {ep}', flush=True)
        n_ep = int(np.median(eps))
        net, _, _, curve = fit(cell, Tdev, ydev, epochs=n_ep); curves[name] = curve
        prob = predict(net, Tte); pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        lo, hi = group_bootstrap_f1(yte, pred, groups[test])
        cvrows.append({'model': name, 'cv_f1': float(np.mean(scores)), 'cv_sd': float(np.std(scores)), 'epochs_per_fold': str(eps),
                       'final_epochs': n_ep, 'seconds': time.perf_counter() - start,
                       'parameters': 'embed 16, hidden 32 x2 directions, dropout 0.2, Adam lr 0.002, batch 128'})
        rows.append({'model': name, 'accuracy': accuracy_score(yte, pred), 'precision': precision_score(yte, pred, zero_division=0),
                     'recall': recall_score(yte, pred), 'f1': f1_score(yte, pred), 'f1_ci_low': lo, 'f1_ci_high': hi,
                     'roc_auc': roc_auc_score(yte, prob), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)})
        preds.append(pd.DataFrame({'row_id': test, 'model': name, 'actual': yte, 'predicted': pred, 'phishing_score': prob}))
    res = pd.DataFrame(rows); res.to_csv(OUT / 'deep_learning_metrics.csv', index=False)
    pd.DataFrame(cvrows).to_csv(OUT / 'deep_learning_cv.csv', index=False)
    pd.concat(preds).to_csv(OUT / 'deep_learning_predictions.csv', index=False)

    # McNemar vs the model selected in train.py (saved predictions, same test rows).
    sel = json.loads((OUT / 'selection.json').read_text())['primary_model_selected_by_cv']
    base = pd.read_csv(OUT / 'predictions.csv').query('model == @sel').set_index('row_id').loc[test]
    ok_ref = base.predicted.to_numpy() == yte; sig = []
    for d in preds:
        ok = d.predicted.to_numpy() == yte; n01, n10 = int((ok_ref & ~ok).sum()), int((~ok_ref & ok).sum())
        sig.append({'reference': sel, 'model': d.model.iloc[0], 'reference_right_other_wrong': n01, 'reference_wrong_other_right': n10,
                    'mcnemar_exact_p': binomtest(min(n01, n10), n01 + n10, 0.5).pvalue})
    pd.DataFrame(sig).to_csv(OUT / 'deep_learning_mcnemar.csv', index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (name, c), col in zip(curves.items(), ['#0f8480', '#c0463c']): ax.plot(range(1, len(c) + 1), c, color=col, label=name)
    ax.set(xlabel='Epoch', ylabel='Training BCE loss', title='Recurrent networks: final refit on development rows'); ax.legend()
    ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig(OUT / 'dl_loss_curves.png', dpi=170); plt.close(fig)
    # Zoomed ROC comparing key models from both scripts (used in the slides and report).
    from sklearn.metrics import roc_curve
    allp = pd.concat([pd.read_csv(OUT / 'predictions.csv'), pd.concat(preds)])
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for nm, col, lw in [('Random Forest', '#0f8480', 3), ('Gradient Boosting', '#10213a', 1.6), ('SVM (RBF)', '#5b8fb0', 1.6), ('Deep ANN (3 layers)', '#c0463c', 1.6),
                        ('LSTM (bidirectional)', '#dca84b', 1.6), ('Categorical NB', '#8c6bb1', 1.4), ('Decision Tree', '#9aa5b1', 1.4)]:
        d = allp[allp.model == nm]; f, t, _ = roc_curve(d.actual, d.phishing_score)
        ax.plot(f, t, color=col, lw=lw, label=f'{nm}  AUC {roc_auc_score(d.actual, d.phishing_score):.3f}')
    ax.set_xlim(0, 0.3); ax.set_ylim(0.6, 1.005); ax.set_xlabel('False positive rate (legitimate sites flagged)', fontsize=11)
    ax.set_ylabel('True positive rate (phishing detected)', fontsize=11); ax.legend(loc='lower right', fontsize=9.5, frameon=False)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.25); ax.set_title('Zoomed ROC: FPR 0-0.3', fontsize=11, loc='left', color='#555')
    fig.tight_layout(); fig.savefig(OUT / 'roc_curves_zoom.png', dpi=200); plt.close(fig)
    (OUT / 'deep_learning_environment.json').write_text(json.dumps({'torch': torch.__version__, 'seed': SEED, 'threads': 1}, indent=2))
    print(res.round(4).to_string(index=False)); print(pd.DataFrame(sig).to_string(index=False))


if __name__ == '__main__': main()
