"""Compare AE scoring and denoising on the frozen validation split."""

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
from sklearn.metrics import average_precision_score

import tier2_ae as base

MASK_RATE = 0.15
CALIBRATION_ROWS = 16_384


def train_denoising(seed, train, overloaded):
    torch.manual_seed(seed)
    order_rng = torch.Generator().manual_seed(seed)
    mask_rng = torch.Generator(device="cuda").manual_seed(seed)
    model = base.autoencoder().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE)
    start = time.monotonic()
    for epoch in range(1, base.EPOCHS + 1):
        epoch_loss = torch.zeros((), device="cuda")
        for ids in torch.randperm(len(train), generator=order_rng).split(base.BATCH_SIZE):
            if overloaded.is_set():
                raise SystemExit(75)
            clean = train[ids].cuda()
            mask = torch.rand(clean.shape, device=clean.device, generator=mask_rng) < MASK_RATE
            optimizer.zero_grad(set_to_none=True)
            loss = (model(clean.masked_fill(mask, 0)) - clean).square().mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() * len(ids)
        loss_value = epoch_loss.item() / len(train)
        if not np.isfinite(loss_value):
            raise RuntimeError(f"Non-finite denoising loss in seed {seed}, epoch {epoch}")
        print(f"denoising seed={seed} epoch={epoch:02d}/{base.EPOCHS} mse={loss_value:.6f}", flush=True)
    return model, time.monotonic() - start


@torch.inference_mode()
def score(model, reference, val, overloaded):
    model.eval()
    errors = []
    for rows in reference.split(base.BATCH_SIZE):
        rows = rows.cuda()
        errors.append((model(rows) - rows).square())
    sorted_errors = torch.cat(errors).T.contiguous().sort(dim=1).values
    count = len(reference)
    mean_scores, calibrated_scores = [], []
    for rows in val.split(base.BATCH_SIZE):
        if overloaded.is_set():
            raise SystemExit(75)
        rows = rows.cuda()
        residual = (model(rows) - rows).square()
        rank = torch.searchsorted(sorted_errors, residual.T.contiguous(), right=False)
        tail = (count + 1 - rank) / (count + 1)
        mean_scores.append(residual.mean(dim=1).cpu())
        calibrated_scores.append(-tail.log().mean(dim=0).cpu())
    return {
        "mean_mse": torch.cat(mean_scores).numpy(),
        "calibrated_tail": torch.cat(calibrated_scores).numpy(),
    }


def summarize(labels, attack_types, scores, original_threshold=None):
    assert np.isfinite(scores).all()
    threshold = base.best_f1_threshold(labels, scores) if original_threshold is None else original_threshold
    result = base.threshold_metrics(labels, scores, threshold)
    result["ap"] = float(average_precision_score(labels, scores))
    result["attack_recall"] = {
        name: {"count": int(group.sum()), "flagged": int((scores[group] >= threshold).sum())}
        for name in np.unique(attack_types[labels == 1])
        for group in [attack_types == name]
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
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

        previous = json.loads((args.baseline / "summary.json").read_text())
        with (base.ARTIFACTS / "splits.npz").open("rb") as frozen_split:
            split_hash = hashlib.file_digest(frozen_split, "sha256").hexdigest()
        assert split_hash == previous["split_sha256"]
        train, val, labels = base.load_data()
        with np.load(base.ARTIFACTS / "splits.npz") as splits:
            val_ids = splits["val"]
        attack_types = pd.read_parquet(
            base.ARTIFACTS / "meta.parquet", columns=["Label"],
        )["Label"].iloc[val_ids].to_numpy()
        assert np.array_equal(attack_types != "BENIGN", labels == 1)
        reference_ids = np.random.default_rng(20261104).choice(len(train), CALIBRATION_ROWS, replace=False)
        reference = train[reference_ids]
        args.output.mkdir(parents=True, exist_ok=False)
        results = []
        for row in previous["seeds"]:
            seed = row["seed"]
            model = base.autoencoder()
            model.load_state_dict(torch.load(
                args.baseline / f"ae_{seed}.pt", map_location="cpu", weights_only=True,
            ))
            baseline_scores = score(model.cuda(), reference, val, overloaded)
            assert abs(average_precision_score(labels, baseline_scores["mean_mse"]) - row["val_ap_mean_mse"]) < 1e-8
            torch.cuda.reset_peak_memory_stats()
            denoising, train_seconds = train_denoising(seed, train, overloaded)
            denoising_scores = score(denoising, reference, val, overloaded)
            result = {"seed": seed, "denoising_train_seconds": train_seconds}
            for name, values in (("ae", baseline_scores), ("denoising", denoising_scores)):
                for score_name, values_by_row in values.items():
                    threshold = row["val_best_f1"]["threshold"] if name == "ae" and score_name == "mean_mse" else None
                    result[f"{name}_{score_name}"] = summarize(labels, attack_types, values_by_row, threshold)
            result["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
            torch.save(denoising.cpu().state_dict(), args.output / f"denoising_{seed}.pt")
            (args.output / f"validation_{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            results.append(result)
        (args.output / "summary.json").write_text(json.dumps({
            "split_sha256": split_hash, "calibration_rows": CALIBRATION_ROWS,
            "calibration_source": "train_benign", "mask_rate": MASK_RATE,
            "max_sampled_host_cpu_percent": max(cpu_samples), "seeds": results,
        }, indent=2) + "\n")
    finally:
        stop.set()
        monitor.join()


if __name__ == "__main__":
    main()
