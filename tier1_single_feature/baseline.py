# Minimal handoff: scientific configuration is embedded as FROZEN_CONFIG.
# Supplied Excel files contain the completed validation and final test results.
# Dataset dependencies: HW2/artifacts and HW2/TrafficLabelling; numpy/pandas/pyarrow/sklearn.
# Validation reproduction: python baseline.py --no-excel
# Fixed-setting test reproduction: python baseline.py --final-test --no-excel
# The original final Test has already been evaluated. Do not change settings using Test results.
"""Shared, auditable data loading and evaluation for the three Tier-1 candidates.

Default runs evaluate validation only. Friday features/labels are not loaded.
Raw CSV values are joined to the supplied metadata by verified source order,
timestamp and label. row_id is the global positional index in meta.parquet.
The teammate's preprocessing parameters and raw files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

METHOD_DIR = Path(__file__).resolve().parent
HW2 = next((p for p in METHOD_DIR.parents
            if (p / "artifacts/meta.parquet").exists() and (p / "TrafficLabelling").is_dir()), None)
if HW2 is None:
    raise RuntimeError("Place this method folder inside HW2, alongside artifacts and TrafficLabelling.")
FROZEN_CONFIG = {'method': 'single_feature', 'params': {'feature': 'Bwd Packet Length Max', 'median_log1p': 4.770684624465665, 'iqr_log1p': 1.9781710112735347, 'clip': 10.0, 'source_params_sha256': 'fd79dada80bbf8c18ebf60d57b946a2af782c9e8a499f127844be2fe7f8e531b', 'reference_population': 'deduplicated train BENIGN, supplied teammate scaler'}, 'seeds': [20261101, 20261102, 20261103], 'threshold_rule': 'score > threshold', 'feature_selection': 'train-only EDA macro KS; fixed before validation', 'threshold': 1.7172181606292725, 'threshold_selection': 'maximum validation F1; tie: fewer FP, then larger tau', 'tuning_time_s': 0.059625099995173514, 'validation_index_sha256': 'ce245df62dcd0e35fbc718edbceee5f90b3312cda50bf7fe7713ced69e0919c1'}
ROOT = HW2 / "tier1_baselines"
ROOT.mkdir(exist_ok=True)
FEATURE = "Bwd Packet Length Max"
SEEDS = [20261101, 20261102, 20261103]
EXPECTED_VAL = (764_639, 12_525)
CACHE = ROOT / "_shared"
ARTIFACTS = HW2 / "artifacts"
RUN_METRICS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2",
    "false_positive_rate", "specificity", "average_precision", "roc_auc",
    "alert_rate", "tp", "fp", "tn", "fn", "fit_time_s", "tuning_time_s",
    "inference_batch_ms", "inference_ms_per_flow",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(a):
    return hashlib.sha256(np.asarray(a, dtype="<i8").tobytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def parameter_bundle():
    path = ARTIFACTS / "preprocess_params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    j = params["columns"].index(FEATURE)
    return {"feature": FEATURE, "median_log1p": params["median"][j],
            "iqr_log1p": params["iqr"][j], "clip": params["clip"],
            "source_params_sha256": sha256_file(path),
            "reference_population": "deduplicated train BENIGN, supplied teammate scaler"}


def read_source_validation_or_test(split):
    """Only read Wednesday/Thursday for validation, or Friday for explicit test."""
    if split == "val":
        files = ["Wednesday-workingHours", "Thursday-WorkingHours-Morning-WebAttacks",
                 "Thursday-WorkingHours-Afternoon-Infilteration"]
    elif split == "test":
        files = ["Friday-WorkingHours-Morning", "Friday-WorkingHours-Afternoon-PortScan",
                 "Friday-WorkingHours-Afternoon-DDos"]
    else:
        raise ValueError(split)
    frames, source_hashes = [], {}
    for stem in files:
        path = HW2 / "TrafficLabelling" / f"{stem}.pcap_ISCX.csv"
        source_hashes[path.name] = sha256_file(path)
        for part in pd.read_csv(path, encoding="latin-1", skipinitialspace=True,
                                usecols=lambda c: c.strip() in [FEATURE, "Label", "Timestamp"],
                                chunksize=150_000, low_memory=False):
            part.columns = part.columns.str.strip()
            part = part.loc[part["Label"].notna()].copy()
            if part.empty:
                continue
            part["Label"] = part["Label"].str.replace("\x96", "-", regex=False).str.strip()
            if stem.startswith("Wednesday"):
                parts = part["Timestamp"].str.extract(r"\d+/\d+/\d+ (\d+):(\d+)").astype(int)
                hour = parts[0].where(parts[0] >= 8, parts[0] + 12)
                part = part.loc[hour * 60 + parts[1] >= 11 * 60 + 8]
            frames.append(part)
    return pd.concat(frames, ignore_index=True), source_hashes


def load_data(split="val"):
    """Build a small cache, preserving the frozen global metadata indices."""
    if split not in ("val", "test"):
        raise ValueError(split)
    CACHE.mkdir(exist_ok=True)
    meta_path = ARTIFACTS / "meta.parquet"
    signature = {"meta_sha256": sha256_file(meta_path),
                 "params_sha256": sha256_file(ARTIFACTS / "preprocess_params.json")}
    cache_path = CACHE / f"{split}.parquet"
    manifest_path = CACHE / f"{split}_manifest.json"
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_valid = all(sha256_file(HW2 / "TrafficLabelling" / name) == digest
                           for name, digest in manifest["source_csv_sha256"].items())
        if manifest["signature"] == signature and source_valid:
            data = pd.read_parquet(cache_path)
            assert array_hash(data["row_id"]) == manifest["row_id_sha256"]
            return data, manifest
        raise RuntimeError("Source/version changed: preserve old outputs and rebuild the cache explicitly.")

    # Only split membership is read for the full metadata. Test labels/features
    # are not inspected during a validation run.
    membership = pd.read_parquet(meta_path, columns=["split"])["split"]
    index = np.flatnonzero(membership.to_numpy() == split)
    meta = pd.read_parquet(meta_path, filters=[("split", "=", split)],
                           columns=["Timestamp", "Label", "is_attack", "day"])
    meta = meta.reset_index(drop=True)
    source, source_hashes = read_source_validation_or_test(split)
    if len(source) != len(index) or len(meta) != len(index):
        raise AssertionError("Source rows and frozen metadata split have different lengths.")
    for col in ("Timestamp", "Label"):
        pd.testing.assert_series_equal(source[col], meta[col], check_names=False)
    raw = pd.to_numeric(source[FEATURE], errors="coerce").to_numpy(dtype=np.float64)
    raw[(~np.isfinite(raw)) | (raw < 0)] = np.nan
    data = meta.rename(columns={"is_attack": "y_true"})
    data.insert(0, "row_id", index)
    data["raw_feature"] = raw
    assert np.array_equal((data["Label"] != "BENIGN").astype(int), data["y_true"])
    if split == "val":
        assert (len(data), int(data["y_true"].sum())) == EXPECTED_VAL
        assert not (data["day"] == "Friday").any()
    manifest = {
        "split": split, "signature": signature, "source_csv_sha256": source_hashes,
        "row_id_definition": "zero-based positional index in supplied artifacts/meta.parquet",
        "row_id_sha256": array_hash(index), "n": len(data),
        "attacks": int(data["y_true"].sum()), "benign": int((data["y_true"] == 0).sum()),
        "class_counts": {str(k): int(v) for k, v in data["Label"].value_counts().items()},
        "alignment": "Every source Timestamp and Label matched frozen metadata in order.",
        "test_evaluated": split == "test",
    }
    data.to_parquet(cache_path, index=False)
    np.save(CACHE / f"{split}_row_ids.npy", index)
    write_json(manifest_path, manifest)
    return data, manifest


def single_scores(raw, params):
    logged = np.log1p(raw)
    logged = np.where(np.isnan(logged), params["median_log1p"], logged)
    scaled = np.clip((logged - params["median_log1p"]) / params["iqr_log1p"],
                     -params["clip"], params["clip"]).astype(np.float32)
    return np.abs(scaled).astype(np.float64)


def best_f1_threshold(y, score):
    """Exact sweep over tied score groups, using strict score > threshold.

    Include all-positive/all-negative predictions. Ties in F1 are resolved by
    fewer false positives, then the largest threshold. Never split tied scores.
    """
    order = np.argsort(score, kind="stable")
    s, labels = score[order], y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    cumulative = np.cumsum(labels, dtype=np.int64)
    p, n = int(y.sum()), len(y) - int(y.sum())
    thresholds = np.r_[np.nextafter(s[0], -np.inf), s[ends]]
    tp = np.r_[p, p - cumulative[ends]]
    fp = np.r_[n, n - ((ends + 1) - cumulative[ends])]
    fn = p - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn,
                   out=np.zeros(len(tp), dtype=float), where=(2 * tp + fp + fn) != 0)
    best = np.lexsort((-thresholds, fp, -f1))[0]
    table = pd.DataFrame({"threshold": thresholds, "tp": tp, "fp": fp,
                          "fn": fn, "f1": f1, "selected": np.arange(len(tp)) == best})
    return float(thresholds[best]), table


def safe_ratio(a, b):
    return float(a / b) if b else None


def evaluate(y, score, prediction):
    y, prediction = np.asarray(y, dtype=np.int8), np.asarray(prediction, dtype=np.int8)
    tp = int(((y == 1) & (prediction == 1)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    tn = int(((y == 0) & (prediction == 0)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    recall, specificity = safe_ratio(tp, tp + fn), safe_ratio(tn, tn + fp)
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": safe_ratio(tp, tp + fp), "recall": recall,
        "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "f2": safe_ratio(5 * tp, 5 * tp + 4 * fn + fp),
        "false_positive_rate": safe_ratio(fp, fp + tn), "specificity": specificity,
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "alert_rate": (tp + fp) / len(y), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def per_class(data, prediction, variant):
    rows = []
    for label in sorted(data["Label"].unique()):
        mask = data["Label"].to_numpy() == label
        flagged = int(prediction[mask].sum())
        rows.append({"variant": variant, "label": label, "support": int(mask.sum()),
                     "flagged": flagged, "not_flagged": int(mask.sum()) - flagged,
                     "flag_rate": flagged / int(mask.sum()),
                     "interpretation": "FPR" if label == "BENIGN" else "Recall"})
    return rows


def time_predict(predict_fn, repeats=7):
    predict_fn()  # warm-up, no fitting and no tuning
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        score, pred = predict_fn()
        times.append((time.perf_counter() - start) * 1000)
    return score, pred, float(np.median(times))


def summarize(runs):
    result = {}
    for metric in RUN_METRICS:
        values = [r[metric] for r in runs if r[metric] is not None]
        result[metric] = {"mean": float(np.mean(values)) if values else None,
                          "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)}
    return result


def export_excel(report_path):
    preferred = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
    node = os.environ.get("NODE_EXECUTABLE") or (str(preferred) if preferred.exists() else shutil.which("node"))
    if not node:
        raise RuntimeError("Results JSON/CSV saved. Excel export needs Node and @oai/artifact-tool.")
    subprocess.run([node, str(ROOT / "_support/build_workbook.mjs"), str(report_path)], check=True)


def run_method(method, folder):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-test", action="store_true",
                        help="Evaluate locked configuration on Friday ONCE, after the team freezes all tiers.")
    parser.add_argument("--no-excel", action="store_true", help="Save JSON/CSV; export Excel separately.")
    args = parser.parse_args()
    out = Path(folder)
    config_path = out / "locked_config.json"
    split = "test" if args.final_test else "val"
    if split == "test":
        # Distribution copy includes the final frozen configuration in this file.
        # A local file, if present, must match the frozen scientific settings.
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if any(existing.get(key) != FROZEN_CONFIG.get(key)
                   for key in ("method", "params", "threshold", "threshold_rule")):
                raise RuntimeError("Local configuration differs from the embedded frozen configuration.")
        # Exclusive marker prevents accidental repeated test access, even on a failed run.
        with open(out / "FINAL_TEST_STARTED.json", "x", encoding="utf-8") as f:
            json.dump({"started_at": pd.Timestamp.now(tz="UTC").isoformat()}, f)
        config = json.loads(json.dumps(FROZEN_CONFIG))
        if config["params"]["source_params_sha256"] != parameter_bundle()["source_params_sha256"]:
            raise RuntimeError("Preprocessing parameters changed after validation.")
    else:
        if (out / "FINAL_TEST_STARTED.json").exists():
            raise RuntimeError("Test has been opened: do not retune this method.")
        config = {"method": method, "params": parameter_bundle(), "seeds": SEEDS,
                  "threshold_rule": "score > threshold",
                  "feature_selection": ("train-only EDA macro KS; fixed before validation"
                                        if method == "single_feature" else "none")}

    data, manifest = load_data(split)
    y = data["y_true"].to_numpy(dtype=np.int8)
    raw = data["raw_feature"].to_numpy(dtype=np.float64)
    tuning_time = 0.0
    if method == "single_feature" and split == "val":
        start = time.perf_counter()
        tau, sweep = best_f1_threshold(y, single_scores(raw, config["params"]))
        tuning_time = time.perf_counter() - start
        config.update(threshold=tau, threshold_selection="maximum validation F1; tie: fewer FP, then larger tau",
                      tuning_time_s=tuning_time, validation_index_sha256=manifest["row_id_sha256"])
        sweep.to_csv(out / "validation_threshold_sweep.csv", index=False)
    elif method == "single_feature":
        tuning_time = config["tuning_time_s"]
    if method == "random_guess":
        config.update(threshold=0.5, score="independent Uniform[0,1), numpy default_rng PCG64",
                      attack_probability=0.5, seed_selection="three predeclared seeds; never select the best seed")
    if method == "constant_prediction":
        config.update(variants=["always_normal", "always_attack"], threshold=0.5)

    runs, by_class, prediction_files = [], [], []
    if method == "single_feature":
        variants = [("single_feature", None)]
    elif method == "constant_prediction":
        variants = [("always_normal", None), ("always_attack", None)]
    else:
        variants = [(f"seed_{seed}", seed) for seed in config["seeds"]]
    for variant, seed in variants:
        if method == "single_feature":
            def predictor():
                s = single_scores(raw, config["params"])
                return s, (s > config["threshold"]).astype(np.int8)
        elif method == "constant_prediction":
            value = int(variant == "always_attack")
            def predictor():
                return np.full(len(y), value, dtype=np.float64), np.full(len(y), value, dtype=np.int8)
        else:
            def predictor():
                s = np.random.default_rng(seed).random(len(y))
                return s, (s > config["threshold"]).astype(np.int8)
        score, pred, batch_ms = time_predict(predictor)
        metrics = evaluate(y, score, pred)
        metrics.update(variant=variant, seed=seed, fit_time_s=0.0,
                       tuning_time_s=tuning_time, inference_batch_ms=batch_ms,
                       inference_ms_per_flow=batch_ms / len(y))
        runs.append(metrics)
        by_class.extend(per_class(data, pred, variant))
        saved = data[["row_id", "Label", "y_true"]].copy()
        if method == "single_feature":
            saved["raw_feature"] = raw
        saved["score"], saved["prediction"] = score, pred
        prediction_path = out / f"predictions_{split}_{variant}.parquet"
        saved.to_parquet(prediction_path, index=False)
        prediction_files.append(prediction_path.name)

    report = {
        "method": method, "split": split, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "data": manifest, "config": config, "runs": runs,
        "summary": ({r["variant"]: summarize([r]) for r in runs}
                    if method == "constant_prediction" else summarize(runs)),
        "by_class": by_class, "prediction_files": prediction_files,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__,
                        "os": platform.platform(), "cpu_count": os.cpu_count()},
        "notes": [
            "Validation selection results are development results, not final unknown-attack test performance.",
            "Positive class: attack=1. BENIGN=0. No resampling or deduplication of validation/test.",
            "Average precision (AP) uses sklearn.average_precision_score; not trapezoidal PR-AUC.",
            "Undefined precision for no predicted positives is null/n.a.; F1 remains 0 when attacks are missed.",
            "Constant scores have ROC-AUC=0.5 and AP equal to attack prevalence.",
            "No model weights are fitted. Single feature reuses supplied normal-train scaler; validation threshold tuning is timed separately.",
            "Latency is median of 7 warm batched runs, excluding disk I/O and metrics; not single-request latency.",
            "Deterministic methods have one distinct prediction run; seed std is zero, not three independent fits.",
            "Validation contains 5 DoS slowloris rows also represented in Train. Frozen split is preserved.",
        ],
    }
    report_path = out / f"results_{split}.json"
    write_json(report_path, report)
    pd.DataFrame(runs).to_csv(out / f"metrics_{split}.csv", index=False)
    pd.DataFrame(by_class).to_csv(out / f"by_attack_{split}.csv", index=False)
    if split == "val":
        write_json(config_path, config)
    print(json.dumps({"method": method, "split": split, "n": len(y),
                      "threshold": config.get("threshold"), "runs": runs}, indent=2))
    if not args.no_excel and (ROOT / "_support/build_workbook.mjs").exists():
        export_excel(report_path)
    elif not args.no_excel:
        print("JSON/CSV/Parquet results saved. Prebuilt Excel files are included in the handoff.")
    return report


if __name__ == "__main__":
    run_method('single_feature', METHOD_DIR)
