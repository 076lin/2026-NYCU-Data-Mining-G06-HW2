"""
Tier 3 (supervised): XGBoost on labelled train flows
===============================================================================

How to run
----------
    uv run python prepare_data.py      # once, builds artifacts/
    uv run python xgb_tier3.py         # tune on train holdout, 3 seeds, val
    uv run python xgb_tier3.py --test  # final run on Friday, once

Important design decisions:
- Train on train_dedup with labels (TA: supervised models may keep the train
  anomalies). Raw view: XGBoost routes NaN itself, so no imputation.
- 20% of train_dedup, stratified by Label, is held out for early stopping and
  the grid search; the model is fitted on the other 80%. This follows Xu & Liu
  (2025), whose supervised models stop early on a validation split drawn from
  training data. Val (unseen attacks) is used only to set tau, as for Tier 1/2.
  results/xgb_val_early_stop/ keeps the earlier run that stopped early on val.
- Hulk is about 90% of train attacks. Sample weights give benign and attacks
  half the total weight each, and every attack type an equal share of the
  attack half, so the model is not a Hulk detector. The holdout is weighted
  the same way.
- Early stopping and the grid use weighted log loss: PR-AUC saturates near 1
  on attacks the model has seen, so it cannot rank settings.
- s(x) = predicted attack probability.
- --test reloads the saved models and val tau; nothing is refit on test.
"""

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from common import (
    RESULT_DIR, SEEDS, best_f1_threshold, evaluate, load_json, print_metrics,
    save_json, single_flow_latency_ms, summarize, timed,
)
from preprocess import SEED, feature_names, load_split

OUT_DIR = RESULT_DIR / "xgb"
HOLDOUT = 0.2
GRID = [
    {"max_depth": depth, "learning_rate": rate}
    for depth in (4, 6, 8) for rate in (0.05, 0.1)
]
FIXED = {
    "n_estimators": 2_000,
    "early_stopping_rounds": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "n_jobs": -1,
}


def balanced_weights(labels):
    """Benign and attacks share the weight equally; attack types equally too."""
    attack_types = sorted(set(labels) - {"BENIGN"})
    share = {"BENIGN": 0.5} | {a: 0.5 / len(attack_types) for a in attack_types}
    counts = labels.value_counts()
    weights = labels.map(lambda label: share[label] / counts[label]).to_numpy()
    return weights * len(weights)  # mean weight 1


def fit(params, seed, fit_part, holdout):
    (X, y, w), (X_hold, y_hold, w_hold) = fit_part, holdout
    model = xgb.XGBClassifier(**FIXED, **params, random_state=seed)
    _, seconds = timed(model.fit, X, y, sample_weight=w,
                       eval_set=[(X_hold, y_hold)],
                       sample_weight_eval_set=[w_hold], verbose=False)
    return model, seconds


def score(model, X):
    return model.predict_proba(X)[:, 1]


def holdout_metrics(model, holdout):
    X_hold, y_hold, w_hold = holdout
    return {
        "holdout_logloss": float(model.best_score),
        "holdout_pr_auc": float(average_precision_score(
            y_hold, score(model, X_hold), sample_weight=w_hold)),
    }


def split_train():
    """80/20 split of train_dedup, stratified by Label, fixed for every seed."""
    X, y, meta = load_split("train_dedup", view="raw")
    labels = meta["Label"].reset_index(drop=True)
    fit_idx, hold_idx = train_test_split(
        np.arange(len(y)), test_size=HOLDOUT, stratify=labels,
        random_state=SEED)
    parts = []
    for idx in (fit_idx, hold_idx):
        part_labels = labels.iloc[idx].reset_index(drop=True)
        parts.append((X[idx], y[idx], balanced_weights(part_labels)))
    counts = {
        name: {"normal": int((y[idx] == 0).sum()),
               "anomaly": int((y[idx] == 1).sum())}
        for name, idx in (("fit", fit_idx), ("holdout", hold_idx))
    }
    return parts[0], parts[1], counts


def train_and_validate():
    fit_part, holdout, counts = split_train()
    X_val, y_val, meta_val = load_split("val", view="raw")
    labels_val = meta_val["Label"].to_numpy()
    print(f"Train rows: {counts}")

    # Grid search on the first seed, ranked by weighted holdout log loss.
    grid = []
    for params in GRID:
        model, seconds = fit(params, SEEDS[0], fit_part, holdout)
        row = holdout_metrics(model, holdout)
        grid.append({"params": params, "n_trees": model.best_iteration + 1,
                     "train_seconds": seconds} | row)
        print(f"  {params} -> holdout log loss {row['holdout_logloss']:.5f}, "
              f"PR-AUC {row['holdout_pr_auc']:.5f}, "
              f"{model.best_iteration + 1} trees, {seconds:.0f}s")
    grid.sort(key=lambda row: row["holdout_logloss"])
    best = grid[0]["params"]
    print(f"Selected {best}")

    runs = []
    for seed in SEEDS:
        model, train_seconds = fit(best, seed, fit_part, holdout)
        val_scores, val_seconds = timed(score, model, X_val)
        tau = best_f1_threshold(y_val, val_scores)
        metrics = evaluate(y_val, val_scores, labels_val, tau)
        metrics |= holdout_metrics(model, holdout) | {
            "train_seconds": train_seconds,
            "batch_us_per_flow": val_seconds / len(X_val) * 1e6,
            "single_flow_ms": single_flow_latency_ms(
                lambda rows: score(model, rows), X_val, seed),
            "n_trees": model.best_iteration + 1,
        }
        print_metrics(f"seed {seed} val", metrics)

        model.save_model(OUT_DIR / f"model_seed{seed}.json")
        np.save(OUT_DIR / f"val_scores_seed{seed}.npy", val_scores)
        save_json(metrics, OUT_DIR / f"val_seed{seed}.json")
        runs.append(metrics)

        gain = model.get_booster().get_score(importance_type="gain")
        importance = pd.Series(
            {name: gain.get(f"f{i}", 0.0) for i, name in enumerate(feature_names())}
        )
        importance.sort_values(ascending=False).to_csv(
            OUT_DIR / f"importance_gain_seed{seed}.csv", header=["gain"])

    save_json({"train_rows": counts, "params": FIXED | best,
               "weighting": "benign 0.5, each attack type 0.5 / 5",
               "grid": grid, "val": summarize(runs)},
              OUT_DIR / "summary_val.json")


def test_once():
    X_test, y_test, meta_test = load_split("test", view="raw", allow_test=True)
    labels_test = meta_test["Label"].to_numpy()
    runs = []
    for seed in SEEDS:
        model = xgb.XGBClassifier()
        model.load_model(OUT_DIR / f"model_seed{seed}.json")
        tau = load_json(OUT_DIR / f"val_seed{seed}.json")["tau"]
        test_scores = score(model, X_test)
        metrics = evaluate(y_test, test_scores, labels_test, tau)
        print_metrics(f"seed {seed} test", metrics)
        np.save(OUT_DIR / f"test_scores_seed{seed}.npy", test_scores)
        save_json(metrics, OUT_DIR / f"test_seed{seed}.json")
        runs.append(metrics)
    save_json(summarize(runs), OUT_DIR / "summary_test.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="evaluate the saved models on Friday, once")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.test:
        test_once()
    else:
        train_and_validate()


if __name__ == "__main__":
    main()
