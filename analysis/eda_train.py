"""
Training-only EDA numbers  --  Section 4
===============================================================================

    python analysis/eda_train.py

Uses the deduplicated TRAINING split only. Prints and saves
  - the Kolmogorov-Smirnov distance between benign flows and each of the five
    training attack types, for every feature (results/eda_ks.csv)
  - the Spearman correlation among the six features with the highest mean KS
The feature with the highest mean KS is the one Tier 1 thresholds.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from _common import RESULTS, load

SEED = 20261101
BENIGN_SAMPLE = 200_000


def main():
    meta, _, raw, features, ids = load(meta_columns=["Label"])
    index = ids["train_dedup"]
    labels = meta["Label"].to_numpy()[index]
    X = np.asarray(raw[index], dtype="float64")
    types = sorted(set(labels) - {"BENIGN"})
    print("Training rows after deduplication:",
          {name: int((labels == name).sum()) for name in ["BENIGN"] + types})
    benign = np.random.default_rng(SEED).choice(np.flatnonzero(labels == "BENIGN"),
                                                BENIGN_SAMPLE, replace=False)
    rows = []
    for j, name in enumerate(features):
        reference = X[benign, j]
        reference = reference[~np.isnan(reference)]
        distances = []
        for attack in types:
            values = X[labels == attack, j]
            distances.append(ks_2samp(reference, values[~np.isnan(values)]).statistic)
        rows.append([name, np.mean(distances), np.min(distances)] + distances)
    table = pd.DataFrame(rows, columns=["feature", "mean_ks", "min_ks"] + types)
    table = table.sort_values("mean_ks", ascending=False)
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "eda_ks.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nHighest mean KS (benign vs. each training attack type):")
    print(table.head(8).round(3).to_string(index=False))
    top = table["feature"].head(6).tolist()
    logged = pd.DataFrame(np.log1p(X[benign][:, [features.index(f) for f in top]]), columns=top)
    print("\nSpearman correlation among the top six (benign training flows):")
    print(logged.corr(method="spearman").round(2).to_string())
    print(f"\nSaved {RESULTS / 'eda_ks.csv'}")


if __name__ == "__main__":
    main()
