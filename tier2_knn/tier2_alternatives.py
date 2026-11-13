"""Compare normal-only Isolation Forest and kNN on the frozen validation split."""

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest

import tier2_ae as base
from tier2_ae_refine import summarize

REFERENCE_SIZE = 32_768
NEIGHBORS = 5
N_TREES = 128
TREE_SAMPLE = 4096


@torch.inference_mode()
def knn_scores(reference, val, overloaded):
    reference = reference.cuda()
    scores = []
    for rows in val.split(base.BATCH_SIZE):
        if overloaded.is_set():
            raise SystemExit(75)
        distance = torch.cdist(rows.cuda(), reference)
        nearest = distance.topk(NEIGHBORS, dim=1, largest=False).values
        scores.append(nearest.max(dim=1).values.cpu())
    return torch.cat(scores).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    gpu_index = os.environ["CUDA_VISIBLE_DEVICES"]
    if not gpu_index.isdigit() or base.gpu_free_mib(gpu_index) < 6144:
        raise SystemExit("Select one physical GPU with at least 6 GiB free")
    before = base.cpu_counters()
    time.sleep(1)
    after = base.cpu_counters()
    initial_cpu = base.cpu_load(before, after)
    if initial_cpu >= 60:
        raise SystemExit(f"Whole-host CPU too busy at startup: {initial_cpu:.1f}%")
    stop, overloaded = threading.Event(), threading.Event()
    cpu_samples = [initial_cpu]
    monitor = threading.Thread(
        target=base.monitor_cpu, args=(after, stop, overloaded, cpu_samples), daemon=True,
    )
    monitor.start()
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise SystemExit("Expected exactly one visible CUDA device")
        torch.cuda.set_per_process_memory_fraction(0.10)
        print(f"Physical GPU {gpu_index}: {torch.cuda.get_device_name(0)}", flush=True)
        train, val, labels = base.load_data()
        split_path = base.ARTIFACTS / "splits.npz"
        with np.load(split_path) as splits:
            val_ids = splits["val"]
        with split_path.open("rb") as stream:
            split_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        attack_types = pd.read_parquet(
            base.ARTIFACTS / "meta.parquet", columns=["Label"],
        )["Label"].iloc[val_ids].to_numpy()
        assert np.array_equal(attack_types != "BENIGN", labels == 1)
        args.output.mkdir(parents=True, exist_ok=False)

        results = []
        for seed in base.SEEDS:
            reference_ids = np.random.default_rng(seed).choice(len(train), REFERENCE_SIZE, replace=False)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            nearest = knn_scores(train[reference_ids], val, overloaded)
            result = {
                "seed": seed,
                "knn": summarize(labels, attack_types, nearest),
                "knn_seconds": time.monotonic() - start,
                "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            }
            print(f"seed={seed} kNN val AP={result['knn']['ap']:.4f}", flush=True)
            forest = IsolationForest(
                n_estimators=N_TREES, max_samples=TREE_SAMPLE, random_state=seed, n_jobs=4,
            )
            if overloaded.is_set():
                raise SystemExit(75)
            start = time.monotonic()
            forest.fit(train.numpy())
            result["isolation_forest_train_seconds"] = time.monotonic() - start
            chunks = []
            with joblib.parallel_backend("threading", n_jobs=4):
                for rows in val.split(32_768):
                    if overloaded.is_set():
                        raise SystemExit(75)
                    chunks.append(-forest.score_samples(rows.numpy()))
            result["isolation_forest"] = summarize(labels, attack_types, np.concatenate(chunks))
            print(f"seed={seed} Isolation Forest val AP={result['isolation_forest']['ap']:.4f}", flush=True)
            joblib.dump(forest, args.output / f"isolation_forest_{seed}.joblib", compress=3)
            (args.output / f"validation_{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
        (args.output / "summary.json").write_text(json.dumps({
            "split_sha256": split_hash,
            "knn_reference_size": REFERENCE_SIZE, "knn_neighbors": NEIGHBORS,
            "forest_trees": N_TREES, "forest_sample_per_tree": TREE_SAMPLE,
            "max_sampled_host_cpu_percent": max(cpu_samples), "seeds": results,
        }, indent=2) + "\n")
    finally:
        stop.set()
        monitor.join()


if __name__ == "__main__":
    main()
