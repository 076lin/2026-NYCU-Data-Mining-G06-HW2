"""
Counts and example rows quoted in the paper, recomputed from the raw data
===============================================================================

    python analysis/verify_numbers.py

No model is fitted. Prints
  - split sizes and attack families                        (Table 2)
  - the two example flows of Section 1
  - Tier 1 on Friday: confusion counts, the 0-byte false positives, the 6-byte
    PortScan false negatives, the decision rule in bytes, alert volumes,
    recall at Tier 3's false-positive rate                 (Sections 7.1, 8.1, 8.2)
  - raw feature values of every case row                   (Table 6)
  - the suspect BENIGN flows between victim and attacker   (Section 7.1)
  - descriptive statistics used for Tier 3                 (Section 8.1)
Row IDs are positions in artifacts/meta.parquet.
"""
import json

import numpy as np
import pandas as pd

from _common import ARTIFACTS, VICTIM, load, suspect_flows

FEATURE = "Bwd Packet Length Max"
TIER1_TAU = 1.7172181606292725           # frozen on validation (tier1 baseline.py)
TIER3_FPR = 0.0556
CASES = {"Section 1 normal example": [4437],
         "Tier 1 (Table 6)": [2127500, 2357130, 2183874, 2398923],
         "Tier 2 (Table 6)": [2454412, 2286836, 2687946, 2723650, 2132365],
         "Tier 3 (docs)": [2466594, 2191085, 2355969]}
SHOW = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Bwd Packets", "Bwd Packet Length Max", "Bwd Packet Length Min"]


def main():
    meta, _, raw, features, ids = load()
    params = json.loads((ARTIFACTS / "preprocess_params.json").read_text())
    column = lambda name: np.asarray(raw[:, features.index(name)], dtype="float64")
    label, split = meta["Label"].to_numpy(), meta["split"].to_numpy()
    test, benign = split == "test", label == "BENIGN"

    print("== Split sizes (Table 2)")
    print(meta.groupby("split")["is_attack"].agg(flows="size", attacks="sum", rate="mean")
          .loc[["train", "val", "test"]].to_string())
    print(meta[~benign].groupby(["split", "Label"]).size().to_string())
    print("deduplicated normal training flows:", len(ids["train_benign"]),
          "(depends on the pandas version, see README)")

    print("\n== Tier 1 on Friday")
    k = params["columns"].index(FEATURE)
    median, iqr = params["median"][k], params["iqr"][k]
    x = column(FEATURE)
    filled = np.where(np.isnan(x), np.expm1(median), x)
    score = np.abs(np.clip((np.log1p(filled) - median) / iqr, -10, 10)).astype("float32").astype("float64")
    flagged = score > TIER1_TAU
    low, high = np.expm1(median - TIER1_TAU * iqr), np.expm1(median + TIER1_TAU * iqr)
    print(f"rule in bytes: flag if x < {low:.2f} or x > {high:.2f}; benign median {np.expm1(median):.0f} B")
    tp, fp = (test & ~benign & flagged).sum(), (test & benign & flagged).sum()
    fn, tn = (test & ~benign & ~flagged).sum(), (test & benign & ~flagged).sum()
    print(f"TP {tp:,}  FP {fp:,}  FN {fn:,}  TN {tn:,}  alerts {tp + fp:,} "
          f"({(tp + fp) / test.sum():.1%} of Friday flows)")
    zero_fp = (test & benign & flagged & (x == 0)).sum()
    print(f"false positives with value 0: {zero_fp:,} ({zero_fp / fp:.2%})")
    scan = test & (label == "PortScan")
    print(f"PortScan with value 6: {(scan & (x == 6)).sum():,} of {scan.sum():,} "
          f"({(scan & (x == 6)).sum() / scan.sum():.2%}); benign Friday flows with value 6: "
          f"{(test & benign & (x == 6)).sum():,}")
    print(f"PortScan share of false negatives: {(scan & ~flagged).sum() / fn:.2%}")
    for name in ("DDoS", "PortScan", "Bot"):
        family = test & (label == name)
        print(f"  {name}: recall {flagged[family].mean():.4f}, median {np.nanmedian(x[family]):.0f} B")
    print(f"  benign Friday median: {np.nanmedian(x[test & benign]):.0f} B")
    threshold = np.quantile(score[test & benign], 1 - TIER3_FPR)
    print(f"recall at Tier 3's FPR ({TIER3_FPR:.2%}): {(score[test & ~benign] > threshold).mean():.4f}")
    val = split == "val"
    vtp, vfp = (val & ~benign & flagged).sum(), (val & benign & flagged).sum()
    print(f"validation: {vtp + vfp:,} alerts, {vfp / vtp:.2f} false alarms per detected attack")

    print("\n== Case rows (raw values)")
    shown = {name: column(name) for name in SHOW}
    for group, rows in CASES.items():
        print(group)
        for r in rows:
            values = ", ".join(f"{name}={shown[name][r]:.0f}" for name in SHOW)
            print(f"  #{r} {label[r]} [{split[r]}] {values}; Tier 1 score {score[r]:.6f} "
                  f"({score[r] - TIER1_TAU:+.6f} vs tau)")

    print("\n== Suspect BENIGN flows between victim and attacker (Section 7.1)")
    suspect = suspect_flows(meta)
    from_victim_80 = suspect & (meta["Source IP"].to_numpy() == VICTIM) & (
        pd.to_numeric(meta["Source Port"], errors="coerce").to_numpy() == 80)
    minute = meta["minute_of_day"].to_numpy()
    for name in ("train", "val", "test"):
        part = split == name
        print(f"  {name}: both directions {(suspect & part).sum():,}; "
              f"victim port 80 -> attacker {(from_victim_80 & part).sum():,}")
    ddos = test & (label == "DDoS")
    inside = suspect & test & (minute >= minute[ddos].min()) & (minute <= minute[ddos].max())
    print(f"  Friday suspect flows inside the DDoS window: {inside.sum():,} "
          f"({inside.sum() / (suspect & test).sum():.1%})")

    print("\n== Descriptive statistics used for Tier 3 (Section 8.1)")
    bmin = column("Bwd Packet Length Min")
    train_dedup = np.zeros(len(meta), dtype=bool); train_dedup[ids["train_dedup"]] = True
    print(f"  median Bwd Packet Length Min, deduplicated training: benign "
          f"{np.nanmedian(bmin[train_dedup & benign]):.0f} B; "
          + ", ".join(f"{a} {np.nanmedian(bmin[train_dedup & (label == a)]):.0f} B"
                      for a in sorted(set(label[train_dedup]) - {"BENIGN"})))
    for name in ("PortScan", "Bot"):
        print(f"  median Bwd Packet Length Min, Friday {name}: {np.nanmedian(bmin[test & (label == name)]):.0f} B")
    tiny = (column("Total Fwd Packets") <= 1) & (column("Total Backward Packets") <= 1)
    print(f"  at most one packet per direction: PortScan {tiny[scan].mean():.1%}, "
          f"benign Friday {tiny[test & benign].mean():.1%}")


if __name__ == "__main__":
    main()
