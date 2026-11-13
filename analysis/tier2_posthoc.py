"""
Post hoc diagnostics for Tier 2 (kNN) on Friday  --  Sections 7.1, 7.2 and 8.1
===============================================================================

    python analysis/tier2_posthoc.py            # about 30 min on 2 CPU cores

Re-scores validation and Friday with the Tier 2 definition (k = 5, 32,768
normal reference flows per seed, tau = best validation F1) on CPU, then reports
  - metrics at the frozen tau, overall and per attack family
  - recall at the false-positive rates of Tier 1 and Tier 3 (matched FPR)
  - the best F1 any threshold could reach in hindsight
  - the effect of excluding the suspect BENIGN flows between the victim web
    server and the attacker's address
  - score and exact per-feature shares of s(x)^2 for the Table 6 cases
Everything here uses Friday labels AFTER the frozen run: it is diagnostic and
was never used for model selection. Writes results/tier2_posthoc.json.

The official Tier 2 numbers come from tier2_knn/ on GPU. This CPU re-run uses
the same seeds and sampling rule; numbers can differ slightly when the pandas
version changes the deduplicated train_benign index (see README).
"""
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from _common import (ARTIFACTS, KNN_SEEDS, RESULTS, best_f1_threshold, confusion,
                     knn_distances, load, reference_rows, save_json, suspect_flows)

MATCHED_FPR = {"tier1_fpr": 0.20538614893729998, "tier3_fpr": 0.0556}
CASE_ROWS = [2454412, 2286836, 2687946, 2723650, 2132365, 2285373]
CASE_SEED = 20261104


def recall_at_fpr(labels, scores, fpr):
    threshold = np.quantile(scores[labels == 0], 1 - fpr)
    return float((scores[labels == 1] > threshold).mean())


def best_f1(labels, scores):
    return confusion(labels, scores, best_f1_threshold(labels, scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default=ARTIFACTS)
    parser.add_argument("--out", default=RESULTS / "tier2_posthoc.json")
    args = parser.parse_args()
    meta, scaled, _, features, ids = load(args.artifacts)
    labels_all, types_all = meta["is_attack"].to_numpy(), meta["Label"].to_numpy()
    suspect_all = suspect_flows(meta)
    val, test = np.asarray(scaled[ids["val"]]), np.asarray(scaled[ids["test"]])
    y_val, y_test = labels_all[ids["val"]], labels_all[ids["test"]]
    types, suspect = types_all[ids["test"]], suspect_all[ids["test"]]
    print(f"train_benign rows: {len(ids['train_benign']):,}; suspect Friday flows: {suspect.sum():,}")

    runs = []
    for seed in KNN_SEEDS:
        reference = reference_rows(scaled, ids["train_benign"], seed)
        tau = best_f1_threshold(y_val, knn_distances(val, reference)[:, -1])
        scores = knn_distances(test, reference)[:, -1]
        at_tau = confusion(y_test, scores, tau)
        flagged = scores >= tau
        benign = y_test == 0
        keep = ~suspect
        run = {
            "seed": seed, "at_tau": at_tau,
            "ap": float(average_precision_score(y_test, scores)),
            "auroc": float(roc_auc_score(y_test, scores)),
            "by_family": {
                name: {"recall": float(flagged[types == name].mean()),
                       "auroc_vs_benign": float(roc_auc_score(
                           y_test[benign | (types == name)], scores[benign | (types == name)])),
                       "median_score": float(np.median(scores[types == name]))}
                for name in ("PortScan", "DDoS", "Bot")},
            "portscan_median_percentile_of_benign": float(
                (scores[benign] < np.median(scores[types == "PortScan"])).mean()),
            "recall_at_matched_fpr": {k: recall_at_fpr(y_test, scores, v)
                                      for k, v in MATCHED_FPR.items()},
            "hindsight_best_f1": best_f1(y_test, scores),
            "suspect": {
                "flows": int(suspect.sum()),
                "share_flagged": float(flagged[suspect].mean()),
                "share_of_false_positives": float((flagged & suspect).sum() / at_tau["fp"]),
                "excluded_at_tau": confusion(y_test[keep], scores[keep], tau),
                "excluded_ap": float(average_precision_score(y_test[keep], scores[keep])),
            },
        }
        print(f"seed {seed}: tau {tau:.4f}  F1 {at_tau['f1']:.4f}  recall {at_tau['recall']:.4f}  "
              f"FPR {at_tau['fpr']:.4f}  AP {run['ap']:.4f}  AUROC {run['auroc']:.4f}", flush=True)
        if seed == CASE_SEED:   # exact per-feature decomposition of s(x)^2
            run["cases"] = []
            for row in CASE_ROWS:
                x = np.asarray(scaled[row], dtype="float64")
                d2 = ((reference - x) ** 2).sum(1)
                kth = reference[np.argsort(d2)[4]]
                terms = (x - kth) ** 2
                top = np.argsort(-terms)[:3]
                run["cases"].append({
                    "row_id": row, "label": str(types_all[row]), "score": float(np.sqrt(terms.sum())),
                    "minus_tau": float(np.sqrt(terms.sum()) - tau),
                    "top_features": [{"feature": features[j], "share": float(terms[j] / terms.sum())}
                                     for j in top]})
        runs.append(run)
        save_json({"runs": runs}, args.out)

    def mean_std(path):
        values = []
        for run in runs:
            value = run
            for key in path:
                value = value[key]
            values.append(value)
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}

    summary = {name: mean_std(path) for name, path in {
        "f1_at_tau": ("at_tau", "f1"), "recall_at_tau": ("at_tau", "recall"),
        "fpr_at_tau": ("at_tau", "fpr"), "precision_at_tau": ("at_tau", "precision"),
        "ap": ("ap",), "auroc": ("auroc",),
        "recall_at_tier1_fpr": ("recall_at_matched_fpr", "tier1_fpr"),
        "recall_at_tier3_fpr": ("recall_at_matched_fpr", "tier3_fpr"),
        "hindsight_best_f1": ("hindsight_best_f1", "f1"),
        "suspect_share_of_false_positives": ("suspect", "share_of_false_positives"),
        "suspect_share_flagged": ("suspect", "share_flagged"),
        "fpr_suspect_excluded": ("suspect", "excluded_at_tau", "fpr"),
        "precision_suspect_excluded": ("suspect", "excluded_at_tau", "precision"),
        "f1_suspect_excluded": ("suspect", "excluded_at_tau", "f1"),
        "ap_suspect_excluded": ("suspect", "excluded_ap"),
        "auroc_portscan": ("by_family", "PortScan", "auroc_vs_benign"),
        "auroc_ddos": ("by_family", "DDoS", "auroc_vs_benign"),
        "auroc_bot": ("by_family", "Bot", "auroc_vs_benign"),
    }.items()}
    for name, value in summary.items():
        print(f"{name:36s} {value['mean']:.4f} +- {value['std']:.4f}")
    save_json({"summary": summary, "runs": runs}, args.out)


if __name__ == "__main__":
    main()
