"""Evaluate the frozen kNN configuration on Friday using the saved validation thresholds."""

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import tier2_ae as base
from tier2_alternatives import NEIGHBORS, REFERENCE_SIZE, knn_scores

EXPECTED_TEST = "8c0e5fef35edb97c7ce5c6d5316ddaa99caba8f95ba0b96a5d3c172c9b5237e0"


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.validation.read_text())
    assert prior["knn_reference_size"] == REFERENCE_SIZE and prior["knn_neighbors"] == NEIGHBORS
    assert tuple(row["seed"] for row in prior["seeds"]) == base.SEEDS
    path = base.ARTIFACTS / "splits.npz"
    with path.open("rb") as stream:
        split_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    assert split_hash == prior["split_sha256"]
    with np.load(path) as splits:
        train_ids, test_ids = splits["train_benign"], splits["test"]
    test_hash = hashlib.sha256(test_ids.astype("<i8").tobytes()).hexdigest()
    assert test_hash == EXPECTED_TEST and (len(train_ids), len(test_ids)) == (848406, 703245)

    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    if not gpu.isdigit() or base.gpu_free_mib(gpu) < 6144:
        raise SystemExit("Select one GPU with at least 6 GiB free")
    before = base.cpu_counters()
    time.sleep(1)
    after = base.cpu_counters()
    load = base.cpu_load(before, after)
    if load >= 60:
        raise SystemExit(f"Whole-host CPU too busy at startup: {load:.1f}%")
    stop, overloaded = threading.Event(), threading.Event()
    samples = [load]
    monitor = threading.Thread(target=base.monitor_cpu, args=(after, stop, overloaded, samples), daemon=True)
    monitor.start()
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.cuda.set_per_process_memory_fraction(0.10)
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "locked_config.json").open("x") as lock:
            config = {
                "model": "kNN anomaly detector",
                "features": 57, "reference_population": "train_benign",
                "reference_size": REFERENCE_SIZE, "k": NEIGHBORS,
                "distance": "Euclidean distance to the kth nearest reference row",
                "batch_size": base.BATCH_SIZE, "prediction_rule": "score >= threshold",
                "threshold_selection": "per-seed maximum validation F1",
                "thresholds": {str(row["seed"]): row["knn"]["threshold"] for row in prior["seeds"]},
                "split_sha256": split_hash, "test_row_id_sha256": test_hash,
                "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            }
            json.dump(config, lock, indent=2)
        # The exclusive configuration file is the final-test marker.
        matrix = np.load(base.ARTIFACTS / "X_scaled.npy", mmap_mode="r")
        meta = pd.read_parquet(base.ARTIFACTS / "meta.parquet", columns=["split", "Label", "is_attack"])
        assert meta.iloc[train_ids]["is_attack"].eq(0).all()
        test = meta.iloc[test_ids]
        assert test["split"].eq("test").all()
        labels, types = test["is_attack"].to_numpy(), test["Label"].to_numpy()
        assert labels.sum() == 288923 and np.array_equal(types != "BENIGN", labels == 1)
        assert set(types) == {"BENIGN", "Bot", "PortScan", "DDoS"}
        rows = torch.from_numpy(matrix[test_ids])
        assert rows.shape == (703245, 57) and torch.isfinite(rows).all()
        results = []
        for previous in prior["seeds"]:
            seed = previous["seed"]
            ids = np.random.default_rng(seed).choice(len(train_ids), REFERENCE_SIZE, replace=False)
            reference = torch.from_numpy(matrix[train_ids[ids]])
            threshold = config["thresholds"][str(seed)]
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            scores = knn_scores(reference, rows, overloaded)
            elapsed = time.perf_counter() - start
            assert np.isfinite(scores).all()
            test_metrics = base.threshold_metrics(labels, scores, threshold)
            test_metrics.update(
                ap=float(average_precision_score(labels, scores)),
                roc_auc=float(roc_auc_score(labels, scores)),
                inference_seconds=elapsed,
                inference_ms_per_flow=1000 * elapsed / len(labels),
                attack_recall={
                    name: {"count": int((types == name).sum()),
                           "flagged": int(((types == name) & (scores >= threshold)).sum())}
                    for name in ("Bot", "PortScan", "DDoS")
                },
            )
            result = {
                "seed": seed, "validation": previous["knn"], "test": test_metrics,
                "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            }
            save_json(args.output / f"results_{seed}.json", result)
            results.append(result)
            print(f"Seed {seed}: Test AP={test_metrics['ap']:.4f} F1={test_metrics['f1']:.4f}", flush=True)
        save_json(args.output / "summary.json", {
            "config": config, "validation_flows": 764639, "validation_attacks": 12525,
            "test_flows": len(labels), "test_attacks": int(labels.sum()),
            "max_sampled_host_cpu_percent": max(samples), "runs": results,
        })
    finally:
        stop.set()
        monitor.join()


if __name__ == "__main__":
    main()
