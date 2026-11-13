"""
Low-memory variant of preprocess.py (about 5 GB RAM instead of about 16 GB)
===============================================================================

    python preprocess_lowmem.py

Applies the same rules, in the same order, with the same checks as
preprocess.py: it imports that file and reuses its constants and functions, but
reads one CSV at a time and keeps numeric columns as arrays. It writes the same
files to artifacts/. Use it only when preprocess.py runs out of memory.
The count of deduplicated normal training flows is printed, not asserted,
because it depends on the pandas version (see README).
"""
import sys, json, importlib.util, gc
import numpy as np, pandas as pd
from pathlib import Path
spec = importlib.util.spec_from_file_location("preprocess", Path(__file__).resolve().parent / "preprocess.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
P.ARTIFACT_DIR.mkdir(exist_ok=True)

metas, blocks, cols = [], [], None
drop = set(P.IDENTIFIERS + P.LEAKY_COLUMNS + ["Label"])
for name in P.FILES:
    f = pd.read_csv(P.DATA_DIR / f"{name}.pcap_ISCX.csv", encoding="latin-1",
                    low_memory=False, skipinitialspace=True)
    f.columns = [c.strip() for c in f.columns]
    blank = int(f["Label"].isna().sum())
    f = f[f["Label"].notna()]
    num_cols = [c for c in f.columns if c not in drop]
    if cols is None: cols = num_cols
    assert num_cols == cols
    num = np.array(f[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64"))
    num[~np.isfinite(num)] = np.nan
    num[num < 0] = np.nan
    blocks.append(num)
    m = f[P.IDENTIFIERS + ["Label"]].copy()
    for c in ("Source Port", "Destination Port", "Protocol"):
        m[c] = pd.to_numeric(m[c], errors="coerce").astype("Int64").astype("string")
    m["day"] = name.split("-")[0]
    metas.append(m)
    print(f"{name}: {len(m):,} flows, {blank:,} blank rows dropped", flush=True)
    del f, num; gc.collect()

df = pd.concat(metas, ignore_index=True); del metas
df["Label"] = df["Label"].str.replace("\x96", "-").str.strip()
df = P.add_time_and_split(df)
P.check_split(df)
X = np.vstack(blocks); del blocks; gc.collect()
numeric_train = pd.DataFrame(X[(df["split"] == "train").to_numpy()], columns=cols)
constant, duplicate = P.find_redundant(numeric_train)
flow_cols = [c for c in cols if c not in constant and c not in duplicate]
print(f"Dropped {len(constant)} constant: {constant}")
print(f"Dropped {len(duplicate)} exact duplicates: {duplicate}")
print(f"Kept {len(flow_cols)} flow features")
keep = [cols.index(c) for c in flow_cols]
is_train = (df["split"] == "train").to_numpy(); benign = (df["is_attack"] == 0).to_numpy()
key = numeric_train[flow_cols].copy(); key["is_attack"] = df["is_attack"].to_numpy()[is_train]
train_dedup = np.zeros(len(df), dtype=bool)
train_dedup[np.flatnonzero(is_train)] = ~key.duplicated().to_numpy()
del key, numeric_train; gc.collect()
train_benign = train_dedup & benign
print("train_dedup", int(train_dedup.sum()), "train_benign", int(train_benign.sum()),
      "expected", P.EXPECTED_TRAIN_BENIGN, "pandas", pd.__version__)
X_raw = np.ascontiguousarray(X[:, keep]); del X; gc.collect()
median, iqr = P.fit_scaler(X_raw[train_benign])
np.save(P.ARTIFACT_DIR / "X_raw.npy", X_raw.astype("float32"))
X_scaled = np.empty(X_raw.shape, dtype="float32")
for s in range(0, len(X_raw), 500_000):
    X_scaled[s:s + 500_000] = P.apply_scaler(X_raw[s:s + 500_000], median, iqr)
assert np.isfinite(X_scaled).all()
np.save(P.ARTIFACT_DIR / "X_scaled.npy", X_scaled)
np.savez(P.ARTIFACT_DIR / "splits.npz", train=np.flatnonzero(is_train),
         train_dedup=np.flatnonzero(train_dedup), train_benign=np.flatnonzero(train_benign),
         val=np.flatnonzero(df["split"] == "val"), test=np.flatnonzero(df["split"] == "test"))
json.dump(flow_cols, open(P.ARTIFACT_DIR / "features.json", "w"), indent=1)
json.dump({"columns": flow_cols, "median": median.tolist(), "iqr": iqr.tolist(),
           "dropped_leaky": P.LEAKY_COLUMNS, "dropped_constant": constant,
           "dropped_duplicate": duplicate, "clip": P.CLIP},
          open(P.ARTIFACT_DIR / "preprocess_params.json", "w"), indent=1)
df.to_parquet(P.ARTIFACT_DIR / "meta.parquet", index=False)
print(f"Saved {X_scaled.shape[0]:,} rows x {X_scaled.shape[1]} columns")
j = flow_cols.index("Bwd Packet Length Max")
print("Tier1 feature median/iqr", median[j], iqr[j])
