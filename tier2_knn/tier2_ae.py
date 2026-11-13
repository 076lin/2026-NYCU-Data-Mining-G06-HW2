"""Train the CIC-IDS2017 Tier 2 autoencoder on the frozen train/val split."""

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SEEDS = (20261104, 20261105, 20261106)
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 1e-3
TOP_K = 5
NORMAL_PERCENTILE = 0.99


def cpu_counters():
    ticks = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
    return sum(ticks), ticks[3] + ticks[4]


def cpu_load(before, after):
    return 100 * (1 - (after[1] - before[1]) / (after[0] - before[0]))


def monitor_cpu(previous, stop, overloaded, samples):
    while not stop.wait(1):
        current = cpu_counters()
        load = cpu_load(previous, current)
        samples.append(load)
        if load >= 68:
            overloaded.set()
            print(f"Stopping training at {load:.1f}% whole-host CPU", flush=True)
            return
        previous = current


def gpu_free_mib(gpu_index):
    return int(subprocess.check_output(
        ["nvidia-smi", "-i", gpu_index, "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], text=True,
    ).strip())


def autoencoder():
    return nn.Sequential(
        nn.Linear(57, 64), nn.ReLU(), nn.Linear(64, 16),
        nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 57),
    )


def load_data():
    split_path = ARTIFACTS / "splits.npz"
    with np.load(split_path) as split:
        train_ids, val_ids = split["train_benign"], split["val"]
    meta = pd.read_parquet(ARTIFACTS / "meta.parquet", columns=["is_attack", "split"])
    train_meta, val_meta = meta.iloc[train_ids], meta.iloc[val_ids]
    assert train_meta["split"].eq("train").all() and train_meta["is_attack"].eq(0).all()
    assert val_meta["split"].eq("val").all()
    matrix = np.load(ARTIFACTS / "X_scaled.npy", mmap_mode="r")
    train, val = matrix[train_ids], matrix[val_ids]
    assert train.shape == (848406, 57) and val.shape == (764639, 57)
    assert np.isfinite(train).all() and np.isfinite(val).all()
    return torch.from_numpy(train), torch.from_numpy(val), val_meta["is_attack"].to_numpy()


def best_f1_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def threshold_metrics(labels, scores, threshold):
    predicted = scores >= threshold
    attack = labels == 1
    tp = int((predicted & attack).sum())
    fp = int((predicted & ~attack).sum())
    fn = int((~predicted & attack).sum())
    tn = int((~predicted & ~attack).sum())
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "false_positive_rate": fp / (fp + tn),
    }


def run_seed(seed, train, val, labels, output, overloaded):
    torch.manual_seed(seed)
    order_rng = torch.Generator().manual_seed(seed)
    model = autoencoder().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    start = time.monotonic()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        squared_error = torch.zeros((), device="cuda")
        for ids in torch.randperm(len(train), generator=order_rng).split(BATCH_SIZE):
            if overloaded.is_set():
                raise SystemExit(75)
            batch = train[ids].cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = (model(batch) - batch).square().mean()
            loss.backward()
            optimizer.step()
            squared_error += loss.detach() * len(ids)
        epoch_mse = squared_error.item() / len(train)
        if not np.isfinite(epoch_mse):
            raise RuntimeError(f"Non-finite training loss in seed {seed}, epoch {epoch}")
        losses.append(epoch_mse)
        print(f"seed={seed} epoch={epoch:02d}/{EPOCHS} train_mse={epoch_mse:.6f}", flush=True)
    train_seconds = time.monotonic() - start

    model.eval()
    mean_scores, topk_scores = [], []
    with torch.inference_mode():
        for rows in val.split(BATCH_SIZE):
            if overloaded.is_set():
                raise SystemExit(75)
            rows = rows.cuda()
            errors = (model(rows) - rows).square()
            mean_scores.append(errors.mean(dim=1).cpu())
            topk_scores.append(errors.topk(TOP_K, dim=1).values.mean(dim=1).cpu())
    mean_scores = torch.cat(mean_scores).numpy()
    topk_scores = torch.cat(topk_scores).numpy()
    assert np.isfinite(mean_scores).all() and np.isfinite(topk_scores).all()
    best_f1 = best_f1_threshold(labels, mean_scores)
    normal_p99 = np.quantile(mean_scores[labels == 0], NORMAL_PERCENTILE)
    result = {
        "seed": seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_seconds": train_seconds,
        "train_mse": losses,
        "val_ap_mean_mse": float(average_precision_score(labels, mean_scores)),
        "val_ap_top5_mse": float(average_precision_score(labels, topk_scores)),
        "val_best_f1": threshold_metrics(labels, mean_scores, best_f1),
        "val_normal_p99": threshold_metrics(labels, mean_scores, normal_p99),
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    torch.save(model.cpu().state_dict(), output / f"ae_{seed}.pt")
    (output / f"validation_{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpu_index = os.environ["CUDA_VISIBLE_DEVICES"]
    if not gpu_index.isdigit() or gpu_free_mib(gpu_index) < 6144:
        raise SystemExit("Select one physical GPU with at least 6 GiB free")
    before = cpu_counters()
    time.sleep(1)
    after = cpu_counters()
    load = cpu_load(before, after)
    if load >= 60:
        raise SystemExit(f"Whole-host CPU too busy at startup: {load:.1f}%")

    stop, overloaded = threading.Event(), threading.Event()
    cpu_samples = [load]
    monitor = threading.Thread(
        target=monitor_cpu, args=(after, stop, overloaded, cpu_samples), daemon=True,
    )
    monitor.start()
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise SystemExit("Expected exactly one visible CUDA device")
        torch.cuda.set_per_process_memory_fraction(0.10)
        args.output.mkdir(parents=True, exist_ok=False)
        print(f"Physical GPU {gpu_index}: {torch.cuda.get_device_name(0)}", flush=True)
        train, val, labels = load_data()
        print(f"Train {tuple(train.shape)}; validation {tuple(val.shape)}; val attacks {labels.sum()}", flush=True)
        results = [run_seed(seed, train, val, labels, args.output, overloaded) for seed in SEEDS]
        summary = {
            "split_sha256": hashlib.file_digest((ARTIFACTS / "splits.npz").open("rb"), "sha256").hexdigest(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "seeds": results,
            "max_sampled_host_cpu_percent": max(cpu_samples),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Completed all seeds; maximum sampled host CPU {max(cpu_samples):.1f}%", flush=True)
    finally:
        stop.set()
        monitor.join()


if __name__ == "__main__":
    main()
