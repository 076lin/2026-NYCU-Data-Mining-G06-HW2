"""
Tier 3 (unsupervised): ECOD fitted on normal train flows only
===============================================================================

How to run
----------
    uv run python prepare_data.py       # once, builds artifacts/
    uv run python ecod_tier3.py         # fit, score val, set tau
    uv run python ecod_tier3.py --test  # final run on Friday, once

ECOD (Li et al., IEEE TKDE 2022) scores a flow by how far into the tails of
each feature's empirical CDF it falls, summed over features. It has no
hyperparameters and no randomness.

Important design decisions:
- Normal-only: fitted on train_benign, the same rows as Tier 2.
- Own implementation instead of pyod.models.ECOD. PyOD's decision_function
  appends the scored rows to the training rows and recomputes the ECDF and
  skewness, so val/test rows would shape the model. Here the sorted train
  columns and skewness are frozen at fit time; a new flow is ranked against
  train only, with +1 smoothing so no tail probability is zero.
- Aggregation follows the paper: max of the left, right and skewness-chosen
  tail sums. PyOD's variant (per-feature max, then sum) is scored as well for
  reference; tau and the headline numbers use the paper's version.
- Raw view, NaN filled with the train-benign median. ECOD only uses ranks, so
  the log/IQR scaling is irrelevant, and the scaled view's clip at +-10 would
  only merge the most extreme values into ties.
"""

import argparse

import numpy as np
from scipy.stats import skew

from common import (
    RESULT_DIR, SEEDS, best_f1_threshold, evaluate, load_json, print_metrics,
    save_json, single_flow_latency_ms, timed,
)
from preprocess import load_split

OUT_DIR = RESULT_DIR / "ecod"
CHUNK = 100_000


class ECOD:
    """ECOD with tail probabilities frozen on the training rows."""

    def fit(self, X):
        self.sorted_ = np.sort(X, axis=0)
        self.left_tail_ = skew(X, axis=0) < 0
        return self

    def tails(self, X):
        """-log P(train <= x) and -log P(train >= x) for every feature."""
        n = len(self.sorted_)
        left = np.empty(X.shape)
        right = np.empty(X.shape)
        for j in range(X.shape[1]):
            column = self.sorted_[:, j]
            at_most = np.searchsorted(column, X[:, j], side="right")
            at_least = n - np.searchsorted(column, X[:, j], side="left")
            left[:, j] = -np.log((at_most + 1) / (n + 1))
            right[:, j] = -np.log((at_least + 1) / (n + 1))
        return left, right

    def score(self, X, aggregation="paper"):
        out = np.empty(len(X))
        for start in range(0, len(X), CHUNK):
            left, right = self.tails(X[start:start + CHUNK])
            if aggregation == "paper":
                chosen = np.where(self.left_tail_, left, right)
                sums = (left.sum(axis=1), right.sum(axis=1), chosen.sum(axis=1))
                out[start:start + CHUNK] = np.max(sums, axis=0)
            else:  # pyod: per-feature max, then sum
                out[start:start + CHUNK] = np.maximum(left, right).sum(axis=1)
        return out


def load_filled(name, median=None, **kwargs):
    X, y, meta = load_split(name, view="raw", **kwargs)
    X = X.astype("float64")
    if median is None:
        median = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), median, X)
    return X, y, meta["Label"].to_numpy(), median


def fit_and_validate():
    X, y, _, median = load_filled("train_benign")
    assert (y == 0).all()
    counts = {"normal": len(X), "anomaly": 0}
    print(f"Train rows: {counts}")

    model, train_seconds = timed(ECOD().fit, X)
    X_val, y_val, labels_val, _ = load_filled("val", median)

    variants = {}
    for aggregation in ("paper", "pyod"):
        val_scores, val_seconds = timed(model.score, X_val, aggregation)
        tau = best_f1_threshold(y_val, val_scores)
        metrics = evaluate(y_val, val_scores, labels_val, tau)
        metrics |= {"batch_us_per_flow": val_seconds / len(X_val) * 1e6}
        print_metrics(f"val ({aggregation})", metrics)
        np.save(OUT_DIR / f"val_scores_{aggregation}.npy", val_scores)
        variants[aggregation] = metrics

    variants["paper"] |= {
        "train_seconds": train_seconds,
        "single_flow_ms": single_flow_latency_ms(model.score, X_val, SEEDS[0]),
    }
    # X_raw.npy is float32, so storing the sorted columns as float32 is exact.
    np.savez(OUT_DIR / "model.npz", sorted=model.sorted_.astype("float32"),
             left_tail=model.left_tail_, median=median)
    save_json({"train_rows": counts, "deterministic": True,
               "val": variants}, OUT_DIR / "summary_val.json")


def test_once():
    saved = np.load(OUT_DIR / "model.npz")
    model = ECOD()
    model.sorted_ = saved["sorted"].astype("float64")
    model.left_tail_ = saved["left_tail"]
    X_test, y_test, labels_test, _ = load_filled(
        "test", saved["median"], allow_test=True)

    results = {}
    for aggregation, metrics in load_json(OUT_DIR / "summary_val.json")["val"].items():
        test_scores = model.score(X_test, aggregation)
        results[aggregation] = evaluate(y_test, test_scores, labels_test,
                                        metrics["tau"])
        print_metrics(f"test ({aggregation})", results[aggregation])
        np.save(OUT_DIR / f"test_scores_{aggregation}.npy", test_scores)
    save_json(results, OUT_DIR / "summary_test.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="evaluate the saved model on Friday, once")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.test:
        test_once()
    else:
        fit_and_validate()


if __name__ == "__main__":
    main()
