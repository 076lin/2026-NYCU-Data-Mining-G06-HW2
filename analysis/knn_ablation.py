"""
Tier 2 ablations on the VALIDATION split  --  Table 4 (Section 7.2)
===============================================================================

    python analysis/knn_ablation.py             # about 20 min on 2 CPU cores

One factor of the kNN detector is changed at a time, for the three Tier 2 seeds:
  - k in {1, 5, 10, 20}                 (score = distance to the k-th neighbour)
  - score = mean of the 5 nearest distances instead of the 5th distance
  - reference size in {4,096, 16,384, 32,768}
The test split is never read. Writes results/knn_ablation_results.jsonl (one
line per seed and variant) and prints the mean, std and delta table.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from _common import (ARTIFACTS, KNN_SEEDS, REFERENCE_SIZE, RESULTS, best_f1_threshold,
                     confusion, knn_distances, load, reference_rows)

FULL = "k=5"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default=ARTIFACTS)
    parser.add_argument("--out", type=Path, default=RESULTS / "knn_ablation_results.jsonl")
    args = parser.parse_args()
    meta, scaled, _, _, ids = load(args.artifacts, meta_columns=["is_attack"])
    val, y = np.asarray(scaled[ids["val"]]), meta["is_attack"].to_numpy()[ids["val"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(variant, seed, size, scores):
        at_tau = confusion(y, scores, best_f1_threshold(y, scores))
        rows.append({"variant": variant, "seed": seed, "n_ref": size,
                     "ap": float(average_precision_score(y, scores)),
                     "auroc": float(roc_auc_score(y, scores)),
                     "f1": at_tau["f1"], "tau": at_tau["threshold"], "recall": at_tau["recall"]})
        print(rows[-1], flush=True)

    for seed in KNN_SEEDS:
        for size in (REFERENCE_SIZE, 4_096, 16_384):
            distances = knn_distances(val, reference_rows(scaled, ids["train_benign"], seed, size),
                                      k_max=20)
            if size == REFERENCE_SIZE:
                for k in (1, 5, 10, 20):
                    record(f"k={k}", seed, size, distances[:, k - 1])
                record("mean of 5 nearest", seed, size, distances[:, :5].mean(1))
            else:
                record(f"ref={size}", seed, size, distances[:, 4])
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))

    table = pd.DataFrame(rows)
    for metric in ("ap", "f1"):
        wide = table.pivot(index="seed", columns="variant", values=metric)
        summary = pd.DataFrame({"mean": wide.mean(), "std": wide.std(ddof=1),
                                "delta_vs_full": wide.mean() - wide[FULL].mean()})
        print(f"\nValidation {metric.upper()} (3 seeds)\n{summary.round(3).to_string()}")


if __name__ == "__main__":
    main()
