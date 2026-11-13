"""
Shared evaluation for Tier 3: one threshold rule and one metric set for all models
===============================================================================

Every model hands over a continuous anomaly score s(x) (higher = more likely an
attack). The threshold tau is chosen on val only, by best F1, and then frozen
for test. Scores are saved so other threshold rules can be applied later
without refitting.
"""

import json
import time

import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)

from preprocess import BASE_DIR, SEED

RESULT_DIR = BASE_DIR / "results"
SEEDS = (SEED, SEED + 1, SEED + 2)
SINGLE_FLOW_TRIALS = 1_000


# --------------------------- Threshold and metrics --------------------------
def best_f1_threshold(y, scores):
    """Tau that maximises F1 on the given (val) labels; attack if s >= tau."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=precision + recall > 0)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y, scores, labels, tau):
    """Threshold-free and at-tau metrics, plus recall for every attack type."""
    pred = scores >= tau
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    attacks = sorted(set(labels[y == 1]))
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "tau": float(tau),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if tp else 0.0,
        "fpr": fp / int((y == 0).sum()),
        "recall_by_type": {a: float(pred[labels == a].mean()) for a in attacks},
        "n_by_type": {a: int((labels == a).sum()) for a in attacks},
    }


# --------------------------- Timing -----------------------------------------
def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def single_flow_latency_ms(score_fn, X, seed):
    """Median wall time to score one flow at a time, as a live detector would."""
    rows = np.random.default_rng(seed).choice(len(X), SINGLE_FLOW_TRIALS,
                                              replace=False)
    times = []
    for i in rows:
        _, seconds = timed(score_fn, X[i:i + 1])
        times.append(seconds * 1e3)
    return float(np.median(times))


# --------------------------- Saving -----------------------------------------
def summarize(runs):
    """Mean and sample std over seeds for every numeric entry."""
    def walk(values):
        if isinstance(values[0], dict):
            return {k: walk([v[k] for v in values]) for k in values[0]}
        if isinstance(values[0], (int, float)):
            arr = np.asarray(values, dtype=float)
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            return {"mean": float(arr.mean()), "std": std}
        return values[0]
    return walk(runs)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def load_json(path):
    return json.loads(path.read_text())


def print_metrics(name, m):
    by_type = ", ".join(f"{a} {r:.3f}" for a, r in m["recall_by_type"].items())
    print(f"{name}: PR-AUC {m['pr_auc']:.4f}  ROC-AUC {m['roc_auc']:.4f}  "
          f"F1 {m['f1']:.4f}  FPR {m['fpr']:.4f}  tau {m['tau']:.4g}")
    print(f"    recall by type: {by_type}")
