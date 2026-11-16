"""
Check a Tier 3 re-run against the numbers quoted in the paper
===============================================================================

How to run (after `xgb_tier3.py` and `xgb_tier3.py --test` have finished)
----------
    python check_tier3.py

Reads results/xgb/ (summaries, per-seed tau and saved test scores); nothing is
refitted. Prints each reported number next to the re-run value, plus the Tier 3
numbers that the paper leaves open or takes from the analysis notes:
AUROC per attack family (including DDoS), recall at Tier 1's false-positive
rate, the suspect BENIGN flows between victim and attacker, and the
false-positive rate on port 22. These use Friday labels and are diagnostic only.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import RESULT_DIR, SEEDS, load_json
from preprocess import load_split

OUT_DIR = RESULT_DIR / "xgb"
VICTIM, ATTACKER = "192.168.10.50", "172.16.0.1"
TIER1_FPR = 0.20538614893729998
REPORTED = {   # (mean, std) as quoted in the paper / analysis notes
    "val": {"pr_auc": (0.119, 0.005), "roc_auc": (0.934, 0.004), "precision": (0.164, 0.004),
            "recall": (0.763, 0.013), "f1": (0.270, 0.005), "fpr": (0.0648, 0.0028),
            "tau": (0.0066, 0.0024)},
    "test": {"pr_auc": (0.782, 0.008), "roc_auc": (0.858, 0.006), "precision": (0.788, 0.008),
             "recall": (0.297, 0.010), "f1": (0.431, 0.010), "fpr": (0.0556, 0.0039)},
}
REPORTED_EXTRA = {"AUROC PortScan": 0.790, "AUROC Bot": 0.716, "AUROC DDoS": None,
                  "recall PortScan": 0.01, "recall Bot": 0.00, "recall DDoS": 0.661,
                  "recall at Tier 1's FPR": 0.712, "share of suspect flows flagged": 0.035,
                  "FPR on other benign flows": 0.057, "AP, suspect flows excluded": 0.799,
                  "FPR on destination port 22": 0.217}


def line(name, reported, values):
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    if reported is None:
        print(f"  {name:34s} reported   --            re-run {values.mean():.4f} +- {std:.4f}")
        return
    mean_reported = reported[0] if isinstance(reported, tuple) else reported
    spread = max(std, reported[1] if isinstance(reported, tuple) else 0.0, 0.005)
    verdict = "ok" if abs(values.mean() - mean_reported) <= 2 * spread else "DIFFERS"
    print(f"  {name:34s} reported {mean_reported:.4f}        re-run {values.mean():.4f} +- {std:.4f}   {verdict}")


def main():
    print("== Summary files (mean over seeds)")
    val = load_json(OUT_DIR / "summary_val.json")["val"]
    test = load_json(OUT_DIR / "summary_test.json")
    for split, summary in (("val", val), ("test", test)):
        for metric, reported in REPORTED[split].items():
            mean, std = summary[metric]["mean"], summary[metric]["std"]
            verdict = "ok" if abs(mean - reported[0]) <= 2 * max(std, reported[1], 0.005) else "DIFFERS"
            print(f"  {split:4s} {metric:10s} reported {reported[0]:.4f} +- {reported[1]:.4f}   "
                  f"re-run {mean:.4f} +- {std:.4f}   {verdict}")
    print("  trees:", val.get("n_trees"), " train seconds:", val.get("train_seconds"),
          " batch us/flow:", val.get("batch_us_per_flow"))

    print("\n== Diagnostics from the saved Friday scores (per seed, then mean +- std)")
    _, y, meta = load_split("test", view="raw", allow_test=True)
    labels = meta["Label"].to_numpy()
    benign = labels == "BENIGN"
    src, dst = meta["Source IP"].to_numpy(), meta["Destination IP"].to_numpy()
    suspect = benign & (((src == VICTIM) & (dst == ATTACKER)) | ((src == ATTACKER) & (dst == VICTIM)))
    port22 = benign & (pd.to_numeric(meta["Destination Port"], errors="coerce").to_numpy() == 22)
    print(f"  Friday flows {len(y):,}; suspect BENIGN flows {suspect.sum():,}; "
          f"benign flows to port 22: {port22.sum():,}")
    rows = {name: [] for name in REPORTED_EXTRA}
    for seed in SEEDS:
        scores = np.load(OUT_DIR / f"test_scores_seed{seed}.npy")
        flagged = scores >= load_json(OUT_DIR / f"val_seed{seed}.json")["tau"]
        for family in ("PortScan", "Bot", "DDoS"):
            keep = benign | (labels == family)
            rows[f"AUROC {family}"].append(roc_auc_score(y[keep], scores[keep]))
            rows[f"recall {family}"].append(flagged[labels == family].mean())
        threshold = np.quantile(scores[benign], 1 - TIER1_FPR)
        rows["recall at Tier 1's FPR"].append((scores[~benign] > threshold).mean())
        rows["share of suspect flows flagged"].append(flagged[suspect].mean())
        rows["FPR on other benign flows"].append(flagged[benign & ~suspect].mean())
        rows["AP, suspect flows excluded"].append(
            average_precision_score(y[~suspect], scores[~suspect]))
        rows["FPR on destination port 22"].append(flagged[port22].mean())
    for name, values in rows.items():
        line(name, REPORTED_EXTRA[name], values)
    print("\n'DIFFERS' marks a gap larger than twice the seed std (at least 0.01).")


if __name__ == "__main__":
    main()
