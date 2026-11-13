"""Shared helpers for the analysis scripts (paths, loading, CPU kNN)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
KNN_SEEDS = (20261104, 20261105, 20261106)   # Tier 2 seeds
REFERENCE_SIZE, NEIGHBORS = 32_768, 5
VICTIM, ATTACKER = "192.168.10.50", "172.16.0.1"


def load(artifacts=ARTIFACTS, meta_columns=None):
    """Return (meta, X_scaled memmap, X_raw memmap, feature names, split indices)."""
    artifacts = Path(artifacts)
    meta = pd.read_parquet(artifacts / "meta.parquet", columns=meta_columns)
    scaled = np.load(artifacts / "X_scaled.npy", mmap_mode="r")
    raw = np.load(artifacts / "X_raw.npy", mmap_mode="r")
    features = json.loads((artifacts / "features.json").read_text())
    with np.load(artifacts / "splits.npz") as splits:
        ids = {name: splits[name] for name in splits.files}
    return meta, scaled, raw, features, ids


def reference_rows(scaled, train_benign_ids, seed, size=REFERENCE_SIZE):
    """Same sampling rule as tier2_knn/tier2_alternatives.py."""
    chosen = np.random.default_rng(seed).choice(len(train_benign_ids), size, replace=False)
    return np.asarray(scaled[train_benign_ids[chosen]], dtype="float64")


def knn_distances(queries, reference, k_max=NEIGHBORS, chunk=2048):
    """Sorted Euclidean distances to the k_max nearest reference rows (CPU, exact).

    Identical query rows are scored once. Returns an array (len(queries), k_max).
    """
    unique, inverse = np.unique(np.asarray(queries), axis=0, return_inverse=True)
    unique = unique.astype("float64")
    ref_sq = (reference * reference).sum(1)
    out = np.empty((len(unique), k_max))
    for start in range(0, len(unique), chunk):
        q = unique[start:start + chunk]
        d2 = (q * q).sum(1)[:, None] + ref_sq[None, :] - 2.0 * (q @ reference.T)
        nearest = np.partition(d2, k_max - 1, axis=1)[:, :k_max]
        nearest.sort(axis=1)
        out[start:start + chunk] = np.sqrt(np.maximum(nearest, 0.0))
    return out[inverse.ravel()]


def best_f1_threshold(labels, scores):
    """Same rule as tier2_knn/tier2_ae.py: largest threshold among the best-F1 ties."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def confusion(labels, scores, threshold):
    predicted = scores >= threshold
    attack = labels == 1
    tp, fp = int((predicted & attack).sum()), int((predicted & ~attack).sum())
    fn, tn = int((~predicted & attack).sum()), int((~predicted & ~attack).sum())
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn), "fpr": fp / (fp + tn)}


def suspect_flows(meta):
    """BENIGN flows between the victim web server and the attacker's NAT address."""
    src, dst = meta["Source IP"].to_numpy(), meta["Destination IP"].to_numpy()
    between = ((src == VICTIM) & (dst == ATTACKER)) | ((src == ATTACKER) & (dst == VICTIM))
    return between & (meta["Label"].to_numpy() == "BENIGN")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
    print(f"Saved {path}")
