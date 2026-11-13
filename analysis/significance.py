"""
Significance tests and derived numbers  --  Sections 7.1 and 7.2
===============================================================================

    python analysis/significance.py

Uses only reported per-seed results: Tier 2 from tier2_knn/results_final/summary.json,
Tier 1 (deterministic) and Tier 3 (mean and std over 3 seeds) as constants below,
and the ablation runs in results/knn_ablation_results.jsonl.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from _common import REPO, RESULTS

TIER1_TEST_F1 = 0.5097538429663158           # tier1_single_feature/results/results_test.xlsx
TIER3 = {"f1": (0.431, 0.010), "ap": (0.782, 0.008), "auroc": (0.858, 0.006)}  # mean, std, n = 3
PREVALENCE = 288_923 / 703_245


def main():
    runs = json.loads((REPO / "tier2_knn/results_final/summary.json").read_text())["runs"]
    tier2 = {m: np.array([run["test"][m] for run in runs]) for m in ("f1", "ap", "roc_auc")}
    tier2["auroc"] = tier2.pop("roc_auc")
    for metric, values in tier2.items():
        print(f"Tier 2 test {metric}: {values.mean():.4f} +- {values.std(ddof=1):.4f}")
    print(f"\nFlag-everything F1 at {PREVALENCE:.2%} attacks: {2 * PREVALENCE / (1 + PREVALENCE):.3f}")

    t = stats.ttest_1samp(tier2["f1"], TIER1_TEST_F1)
    print(f"Tier 2 vs Tier 1, test F1 (one-sample t): t(2) = {t.statistic:.2f}, p = {t.pvalue:.4f}")
    mean, std = TIER3["f1"]
    t3 = (mean - TIER1_TEST_F1) / (std / np.sqrt(3))
    print(f"Tier 3 vs Tier 1, test F1 (one-sample t): t(2) = {t3:.2f}, p = {2 * stats.t.sf(abs(t3), 2):.4f}")
    for metric in ("ap", "auroc"):
        mean, std = TIER3[metric]
        w = stats.ttest_ind_from_stats(tier2[metric].mean(), tier2[metric].std(ddof=1), 3,
                                       mean, std, 3, equal_var=False)
        print(f"Tier 2 vs Tier 3, test {metric} (Welch t): t = {w.statistic:.2f}, p = {w.pvalue:.4f}")

    path = RESULTS / "knn_ablation_results.jsonl"
    if path.exists():
        table = pd.DataFrame([json.loads(line) for line in path.read_text().splitlines()])
        for metric in ("ap", "f1"):
            wide = table.pivot(index="seed", columns="variant", values=metric)
            print(f"\nAblation, validation {metric.upper()}: full = {wide['k=5'].mean():.3f} "
                  f"+- {wide['k=5'].std(ddof=1):.3f}")
            for variant in wide.columns.drop("k=5"):
                paired = stats.ttest_rel(wide[variant], wide["k=5"])
                print(f"  {variant:18s} {wide[variant].mean():.3f} +- {wide[variant].std(ddof=1):.3f}  "
                      f"delta {wide[variant].mean() - wide['k=5'].mean():+.3f}  paired t p = {paired.pvalue:.4f}")


if __name__ == "__main__":
    main()
