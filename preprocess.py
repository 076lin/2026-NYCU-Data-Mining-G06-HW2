"""
CIC-IDS2017: shared cleaning, frozen split, and feature matrices for all tiers
===============================================================================

How to run
----------
1. Keep the "TrafficLabelling" folder next to this script (its name really
   does end with a space).
2. Install dependencies:
   pip install pandas numpy pyarrow
3. Run:
   python preprocess.py

The script creates an artifacts/ directory (about 1.3 GB). Every tier then loads
the same rows through load_split():

    from preprocess import load_split
    X, y, meta = load_split("train_benign", view="scaled")   # Tier 2
    X, y, meta = load_split("train_dedup", view="raw")       # Tier 3
    X, y, meta = load_split("val", view="scaled")            # tuning and tau
    X, y, meta = load_split("test", allow_test=True)         # final run, once

Important design decisions:
- Split is chronological: train = Mon to Wed 11:08, val = Wed 11:08 to Thu end,
  test = Fri. All Friday attack types are absent from train and val.
- Identifiers (Flow ID, IPs, ports, Protocol, Timestamp) are never model
  inputs. They stay in meta.parquet for the error analysis only.
- Columns that describe a machine's TCP setup rather than attack behaviour are
  removed (LEAKY_COLUMNS). eda_tables/leak_audit.txt holds the evidence.
- Per-source context features are not used: they raised flags on benign flows
  from the attacker's NAT address to 15.8%, against 1.0% for other benign flows.
- Inf and negative values become NaN.
- Constant and exact-duplicate columns are found on train only.
- Scaler and medians are fitted on deduplicated train benign flows only.
- Only train is deduplicated, on the model's inputs plus the label. Val and
  test keep repeats, as real traffic does.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

# --------------------------- Configuration ----------------------------------
SEED = 20261101
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "TrafficLabelling"
ARTIFACT_DIR = BASE_DIR / "artifacts"

FILES = [
    "Monday-WorkingHours",
    "Tuesday-WorkingHours",
    "Wednesday-workingHours",
    "Thursday-WorkingHours-Morning-WebAttacks",
    "Thursday-WorkingHours-Afternoon-Infilteration",
    "Friday-WorkingHours-Morning",
    "Friday-WorkingHours-Afternoon-PortScan",
    "Friday-WorkingHours-Afternoon-DDos",
]

# DoS Hulk ends at 11:07 and DoS GoldenEye starts at 11:10.
VAL_START_MINUTE_ON_WEDNESDAY = 11 * 60 + 8
FIRST_MORNING_HOUR = 8  # the 12-hour clock has no AM/PM; 1-7 means afternoon
CLIP = 10.0

IDENTIFIERS = [
    "Flow ID", "Source IP", "Source Port", "Destination IP",
    "Destination Port", "Protocol", "Timestamp",
]
# Set by the operating system or TCP options, so they identify the machine.
LEAKY_COLUMNS = [
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "min_seg_size_forward",
    "Fwd Header Length", "Fwd Header Length.1", "Bwd Header Length",
]

EXPECTED = {
    "train": (1_362_859, 256_198),
    "val": (764_639, 12_525),
    "test": (703_245, 288_923),
}
EXPECTED_TRAIN_BENIGN = 848_406


# --------------------------- Load and label ---------------------------------
def load_flows():
    """Read all days in order, dropping blank rows and fixing column names."""
    frames = []
    for name in FILES:
        frame = pd.read_csv(
            DATA_DIR / f"{name}.pcap_ISCX.csv",
            encoding="latin-1", low_memory=False, skipinitialspace=True,
        )
        frame.columns = [c.strip() for c in frame.columns]
        blank = frame["Label"].isna().sum()
        frame = frame[frame["Label"].notna()].copy()
        frame["day"] = name.split("-")[0]
        print(f"{name}: {len(frame):,} flows, {blank:,} blank rows dropped")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    # The en dash in "Web Attack - ..." is a broken byte in these files.
    df["Label"] = df["Label"].str.replace("\x96", "-").str.strip()
    return df


def add_time_and_split(df):
    """Add minute columns and the frozen train/val/test assignment."""
    parts = df["Timestamp"].str.extract(r"(\d+)/\d+/\d+ (\d+):(\d+)").astype(int)
    date, hour, minute = parts[0], parts[1], parts[2]
    hour = hour.where(hour >= FIRST_MORNING_HOUR, hour + 12)

    df["minute_of_day"] = (hour * 60 + minute).astype("int32")
    df["minute_abs"] = (date * 1440 + df["minute_of_day"]).astype("int32")

    late_wednesday = (df["day"] == "Wednesday") & (
        df["minute_of_day"] >= VAL_START_MINUTE_ON_WEDNESDAY
    )
    df["split"] = "train"
    df.loc[late_wednesday | (df["day"] == "Thursday"), "split"] = "val"
    df.loc[df["day"] == "Friday", "split"] = "test"
    df["is_attack"] = (df["Label"] != "BENIGN").astype("int8")
    return df


# --------------------------- Features ---------------------------------------
def clean_numeric(df):
    """Return non-leaky flow statistics with inf and negatives set to NaN."""
    meta_cols = IDENTIFIERS + LEAKY_COLUMNS + [
        "Label", "day", "minute_of_day", "minute_abs", "split", "is_attack",
    ]
    numeric = df.drop(columns=meta_cols).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.mask(numeric < 0)


def find_redundant(train_numeric):
    """Find constant columns and exact duplicates using train rows only."""
    constant = [c for c in train_numeric if train_numeric[c].nunique() <= 1]
    seen, duplicate = {}, {}
    for col in train_numeric.columns.difference(constant, sort=False):
        key = train_numeric[col].fillna(-1.5e18).to_numpy().tobytes()
        if key in seen:
            duplicate[col] = seen[key]
        else:
            seen[key] = col
    return constant, duplicate


def signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


def fit_scaler(values):
    """Median, and IQR falling back to 1 for near-constant columns."""
    logged = signed_log1p(values)
    median = np.nanmedian(logged, axis=0)
    q25, q75 = np.nanpercentile(logged, [25, 75], axis=0)
    iqr = q75 - q25
    return median, np.where(iqr > 0, iqr, 1.0)


def apply_scaler(values, median, iqr):
    logged = signed_log1p(values)
    logged = np.where(np.isnan(logged), median, logged)
    return np.clip((logged - median) / iqr, -CLIP, CLIP)


# --------------------------- Checks -----------------------------------------
def check_split(df):
    """Stop if the split drifts from the numbers the plan was built on."""
    for name, (flows, attacks) in EXPECTED.items():
        part = df[df["split"] == name]
        found = (len(part), int(part["is_attack"].sum()))
        assert found == (flows, attacks), f"{name}: {found} != {(flows, attacks)}"

    train, val, test = (df[df["split"] == s] for s in ("train", "val", "test"))
    assert train["minute_abs"].max() < val["minute_abs"].min()
    assert val["minute_abs"].max() < test["minute_abs"].min()

    test_attacks = set(test["Label"]) - {"BENIGN"}
    assert not test_attacks & (set(train["Label"]) | set(val["Label"]))


# --------------------------- Loading for all tiers --------------------------
def load_split(name, view="scaled", allow_test=False):
    """
    Return X, y and meta for one frozen split.

    name:   train | train_dedup | train_benign | val | test
    view:   scaled (Tier 1 and 2) or raw (Tier 3, NaN kept)
    """
    if name == "test" and not allow_test:
        raise ValueError("Test is evaluated once: pass allow_test=True.")

    index = np.load(ARTIFACT_DIR / "splits.npz")[name]
    matrix = np.load(ARTIFACT_DIR / f"X_{view}.npy", mmap_mode="r")
    X = np.asarray(matrix[index])
    meta = pd.read_parquet(ARTIFACT_DIR / "meta.parquet").iloc[index]
    return X, meta["is_attack"].to_numpy(), meta


def feature_names():
    with open(ARTIFACT_DIR / "features.json") as f:
        return json.load(f)


# --------------------------- Main -------------------------------------------
def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)

    df = add_time_and_split(load_flows())
    check_split(df)

    numeric = clean_numeric(df)
    is_train = (df["split"] == "train").to_numpy()
    benign = (df["is_attack"] == 0).to_numpy()

    constant, duplicate = find_redundant(numeric[is_train])
    flow_cols = [
        c for c in numeric.columns if c not in constant and c not in duplicate
    ]
    print(f"Dropped {len(constant)} constant: {constant}")
    print(f"Dropped {len(duplicate)} exact duplicates: {duplicate}")
    print(f"Kept {len(flow_cols)} flow features")

    # Repeats are judged on what the model sees, and within train only:
    # Wednesday's rows are not in time order in the file.
    flow_key = numeric[flow_cols].join(df["is_attack"])
    train_dedup = np.zeros(len(df), dtype=bool)
    train_dedup[np.flatnonzero(is_train)] = ~flow_key[is_train].duplicated().to_numpy()
    train_benign = train_dedup & benign
    assert train_benign.sum() == EXPECTED_TRAIN_BENIGN

    X_raw = numeric[flow_cols].to_numpy(dtype="float64")
    median, iqr = fit_scaler(X_raw[train_benign])  # train benign only
    X_scaled = apply_scaler(X_raw, median, iqr)
    assert np.isfinite(X_scaled).all()

    np.save(ARTIFACT_DIR / "X_raw.npy", X_raw.astype("float32"))
    np.save(ARTIFACT_DIR / "X_scaled.npy", X_scaled.astype("float32"))
    np.savez(
        ARTIFACT_DIR / "splits.npz",
        train=np.flatnonzero(is_train),
        train_dedup=np.flatnonzero(train_dedup),
        train_benign=np.flatnonzero(train_benign),
        val=np.flatnonzero(df["split"] == "val"),
        test=np.flatnonzero(df["split"] == "test"),
    )
    with open(ARTIFACT_DIR / "features.json", "w") as f:
        json.dump(flow_cols, f, indent=1)
    with open(ARTIFACT_DIR / "preprocess_params.json", "w") as f:
        json.dump(
            {
                "columns": flow_cols,
                "median": median.tolist(),
                "iqr": iqr.tolist(),
                "dropped_leaky": LEAKY_COLUMNS,
                "dropped_constant": constant,
                "dropped_duplicate": duplicate,
                "clip": CLIP,
            },
            f, indent=1,
        )
    df[IDENTIFIERS + ["Label", "day", "split", "is_attack", "minute_of_day",
                      "minute_abs"]].astype({"Source Port": "string",
                                             "Destination Port": "string",
                                             "Protocol": "string"}).to_parquet(
        ARTIFACT_DIR / "meta.parquet", index=False,
    )
    print(f"Saved {X_scaled.shape[0]:,} rows x {X_scaled.shape[1]} columns")


if __name__ == "__main__":
    main()
