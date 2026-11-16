"""
Figures for the error analysis: PCA by split, Tier 1 feature, traffic timeline
===============================================================================

How to run (from the repository root, after preprocess.py has built artifacts/)
----------
    python analysis/make_figures.py
    python analysis/make_figures.py --artifacts path/to/artifacts --out figures/

Needs: numpy, pandas, pyarrow, scipy, scikit-learn, matplotlib. CPU only.

Output
  fig_pca.pdf               (a) training days, (b) validation, (c) Friday test, all
                            on the first two principal components of NORMAL
                            TRAINING traffic. PCA is fitted on train_benign only.
  fig_eda_feature.pdf       Bwd Packet Length Max on the TRAINING split only, by
                            attack family (for the EDA section).
  fig_tier1_feature.pdf     Bwd Packet Length Max on Friday with the two regions
                            that the frozen Tier 1 rule flags.
  fig_traffic_timeline.pdf  One chart, Monday to Friday: share of flows per 10
                            minutes that belong to each attack family, with the
                            train / validation / test boundaries marked.
  fig_pca_region_shares.csv, fig_traffic_by_day.csv   numbers quoted in the text

How the "95% of normal training flows" outline is drawn
  1. Project train_benign onto PC1/PC2 and count flows on an 80 x 80 grid.
  2. Smooth the counts lightly (Gaussian, 1 bin) so the outline is not ragged.
  3. Sort the bins from densest to sparsest and keep bins until they hold 95%
     of the flows. The outline is the edge of the kept bins.
  It is a highest-density region in this 2-D projection only. It is NOT the
  kNN decision boundary, which lives in all 57 dimensions.

PCA uses training data only: the two components are fitted on train_benign
(the kNN reference population). Validation and Friday flows are only projected
with that fixed transform; the plot range is the only thing that looks at them.

Many flows share exactly the same feature values, so they land on the same
spot and a scatter of single points hides how many flows are there. Each attack
family is therefore drawn as bubbles: the plane is cut into a grid, and each
cell holding at least 0.5% of the family gets one marker whose AREA is the
share of that family's flows in the cell. All flows are used (no subsampling).
Infiltration and Heartbleed (47 flows) are not drawn.
Validation and Friday labels are used for plotting only (post hoc analysis).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

SEED = 20261101
TIER1_FEATURE = "Bwd Packet Length Max"
TIER1_TAU = 1.7172181606292725          # frozen on validation (Tier 1 locked config)
VAL_START_MINUTE_ON_WEDNESDAY = 11 * 60 + 8
GRID = 80                               # bins per axis for the 95% normal region
BUBBLE_GRID = 45                        # bins per axis for the attack bubbles
BUBBLE_MIN, BUBBLE_SCALE = 6, 300       # marker area = MIN + SCALE * share
BIN_MINUTES = 10                        # timeline resolution

# One fixed colour and marker per family, the same in every figure.
# Families that share a panel are checked for colour-blind separation; markers
# repeat the identity for print and for the shared legend.
INK, MUTED, GRID_INK = "#1a1a19", "#6b6a63", "#d9d8d2"
STYLE = {
    "Brute force (FTP/SSH)": ("#6250d6", "P"),
    "Slow DoS":              ("#e34948", "X"),
    "DoS Hulk":              ("#008300", "h"),
    "DoS GoldenEye":         ("#2a78d6", "o"),
    "Web attacks":           ("#eb6834", "s"),
    "DDoS":                  ("#1baf7a", "^"),
    "PortScan":              ("#eda100", "D"),
    "Bot":                   ("#e87ba4", "v"),
}
GROUPS = {
    "FTP-Patator": "Brute force (FTP/SSH)", "SSH-Patator": "Brute force (FTP/SSH)",
    "DoS slowloris": "Slow DoS", "DoS Slowhttptest": "Slow DoS",
}
PANELS = [
    ("train", "(a) Training (Mon to Wed 11:07)", ["DoS Hulk", "Slow DoS", "Brute force (FTP/SSH)"]),
    ("val",   "(b) Validation (Wed 11:08 to Thu)", ["DoS GoldenEye", "Web attacks"]),
    ("test",  "(c) Friday test", ["DDoS", "PortScan", "Bot"]),
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
LIGHT_GREYS = LinearSegmentedColormap.from_list("light_greys", ["#ececea", "#8f8e88"])


def family_of(labels):
    """Group labels into the families of STYLE; other labels stay as they are."""
    labels = pd.Series(labels.astype(str))
    grouped = labels.replace(GROUPS)
    grouped[labels.str.startswith("Web Attack")] = "Web attacks"
    return grouped.to_numpy()


def normal_region(points, lim):
    """Boolean GRID x GRID mask of the densest bins holding 95% of the points."""
    counts, xe, ye = np.histogram2d(points[:, 0], points[:, 1], bins=GRID, range=lim)
    counts = gaussian_filter(counts, sigma=1.0)   # smooth the outline only
    order = np.argsort(counts, axis=None)[::-1]
    keep = np.cumsum(counts.ravel()[order]) <= 0.95 * counts.sum()
    mask = np.zeros(counts.size, dtype=bool)
    mask[order[keep]] = True
    return mask.reshape(counts.shape), xe, ye


def share_inside(points, mask, xe, ye):
    i = np.searchsorted(xe, points[:, 0], side="right") - 1
    j = np.searchsorted(ye, points[:, 1], side="right") - 1
    ok = (i >= 0) & (i < GRID) & (j >= 0) & (j < GRID)
    inside = np.zeros(len(points), dtype=bool)
    inside[ok] = mask[i[ok], j[ok]]
    return float(inside.mean())


def style():
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })


def save(fig, out, name):
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"{name}.{suffix}", dpi=400)
    plt.close(fig)
    print(f"Saved {out / name}.pdf and .png")


# ------------------------------- Figure 1: PCA -------------------------------
def figure_pca(scaled, ids, labels, features, out):
    # Fitted on normal TRAINING flows only; other splits are projected, never fitted.
    pca = PCA(n_components=2, random_state=SEED).fit(np.asarray(scaled[ids["train_benign"]]))
    var = pca.explained_variance_ratio_
    print(f"Explained variance: PC1 {var[0]:.1%}, PC2 {var[1]:.1%}")
    for k in range(2):
        top = np.argsort(-np.abs(pca.components_[k]))[:5]
        print(f"  PC{k + 1} top loadings: "
              + ", ".join(f"{features[i]} ({pca.components_[k][i]:+.2f})" for i in top))

    proj, fam = {}, {}
    for split in ("train", "val", "test"):
        proj[split] = pca.transform(np.asarray(scaled[ids[split]]))
        fam[split] = family_of(labels[ids[split]])
    train_benign = pca.transform(np.asarray(scaled[ids["train_benign"]]))

    everything = np.vstack(list(proj.values()))
    lo, hi = np.percentile(everything, [0.2, 99.8], axis=0)
    pad = 0.05 * (hi - lo)
    lim = [[lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]]
    mask, xe, ye = normal_region(train_benign, lim)
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2

    print("\nShare inside the region holding 95% of normal training flows (PC1/PC2):")
    rows = [{"split": "train_benign", "family": "BENIGN (deduplicated)",
             "flows": len(train_benign),
             "inside_normal_95": share_inside(train_benign, mask, xe, ye)}]
    for split, _, families in PANELS:
        for name in ["BENIGN"] + families:
            points = proj[split][fam[split] == name]
            rows.append({"split": split, "family": name, "flows": len(points),
                         "inside_normal_95": share_inside(points, mask, xe, ye)})
    for row in rows:
        print(f"  {row['split']:12s} {row['family']:22s} n={row['flows']:>9,}  "
              f"inside {row['inside_normal_95']:.1%}")
    pd.DataFrame(rows).to_csv(out / "fig_pca_region_shares.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.15), sharex=True, sharey=True,
                             constrained_layout=True)
    halo = [pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()]
    for ax, (split, title, families) in zip(axes, PANELS):
        benign = train_benign if split == "train" else proj[split][fam[split] == "BENIGN"]
        ax.hexbin(benign[:, 0], benign[:, 1], gridsize=60, bins="log", cmap=LIGHT_GREYS,
                  extent=(*lim[0], *lim[1]), linewidths=0, mincnt=1, rasterized=True)
        outline = ax.contour(xc, yc, mask.T.astype(float), levels=[0.5], colors=INK,
                             linewidths=1.3, linestyles="solid", zorder=4)
        outline.set_path_effects(halo)
        handles = [Patch(facecolor="#b5b4ae", edgecolor="none", label="Benign (density)")]
        # Flows pile up on a few exact positions, so single points hide the mass.
        # Each family is binned on a grid; one bubble per cell, area = share of
        # the family's flows in that cell (cells under 0.5% are left out).
        for name in families:
            points = proj[split][fam[split] == name]
            counts, bx, by = np.histogram2d(points[:, 0], points[:, 1], bins=BUBBLE_GRID,
                                            range=lim)
            sx, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=BUBBLE_GRID,
                                      range=lim, weights=points[:, 0])
            sy, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=BUBBLE_GRID,
                                      range=lim, weights=points[:, 1])
            cell_share = counts / len(points)
            big = cell_share >= 0.005
            colour, marker = STYLE[name]
            ax.scatter(sx[big] / counts[big], sy[big] / counts[big],
                       s=BUBBLE_MIN + BUBBLE_SCALE * cell_share[big], c=colour,
                       marker=marker, alpha=0.75, edgecolors="white", linewidths=0.4,
                       zorder=5)
            handles.append(Line2D([], [], color=colour, marker=marker, ls="none",
                                  markersize=4.5, label=f"{name} (n={len(points):,})"))
            print(f"  {name} ({split}): {int(big.sum())} bubbles cover "
                  f"{cell_share[big].sum():.1%} of its flows")
        ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"PC1 ({var[0]:.0%} of normal variance)")
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.27),
                  ncol=1, frameon=False, handletextpad=0.4)
    axes[0].set_ylabel(f"PC2 ({var[1]:.0%})")
    fig.legend(handles=[
        Line2D([], [], color=INK, lw=1.3, label="Outline: densest region holding 95% of normal training flows"),
        Line2D([], [], color=MUTED, marker="o", markersize=6, ls="none",
               label="Marker area: share of the family's flows at that position"),
    ], loc="outside lower center", ncol=2, frameon=False)
    save(fig, out, "fig_pca")


# -------------------------- Figure 2: Tier 1 feature -------------------------
def figure_tier1(raw, ids, labels, features, params, out):
    j = features.index(TIER1_FEATURE)
    k = params["columns"].index(TIER1_FEATURE)
    median, iqr = params["median"][k], params["iqr"][k]
    low_cut = np.expm1(median - TIER1_TAU * iqr)     # flagged below this (bytes)
    high_cut = np.expm1(median + TIER1_TAU * iqr)    # flagged above this (bytes)
    values = np.asarray(raw[ids["test"], j], dtype="float64")
    values = np.where(np.isnan(values), np.expm1(median), values)  # as in preprocessing
    family = family_of(labels[ids["test"]])
    logged = np.log1p(values)
    edges = np.linspace(0, max(logged.max(), np.log1p(high_cut)) * 1.02, 60)

    fig, ax = plt.subplots(figsize=(3.03, 2.3), constrained_layout=True)
    ax.axvspan(edges[0], np.log1p(low_cut), color="#e34948", alpha=0.10, lw=0)
    ax.axvspan(np.log1p(high_cut), edges[-1], color="#e34948", alpha=0.10, lw=0)
    print(f"\nTier 1 rule in bytes: flag if x < {low_cut:.2f} or x > {high_cut:.1f}")
    for name, colour in [("BENIGN", MUTED), ("DDoS", STYLE["DDoS"][0]),
                         ("PortScan", STYLE["PortScan"][0]), ("Bot", STYLE["Bot"][0])]:
        group = values[family == name]
        share, _ = np.histogram(np.log1p(group), bins=edges)
        ax.step(edges[:-1], share / share.sum(), where="post", color=colour, lw=1.2,
                label="Benign" if name == "BENIGN" else name)
        print(f"  Tier 1 flags {np.mean((group < low_cut) | (group > high_cut)):.1%} "
              f"of Friday {name}")
    ticks = [0, 6, 100, 1_000, 10_000]
    ax.set_xticks(np.log1p(ticks)); ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("Bwd Packet Length Max (bytes, log scale)")
    ax.set_ylabel("Share of the family's flows")
    ax.grid(axis="y", color=GRID_INK, lw=0.4); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#e34948", alpha=0.10, label="Flagged by Tier 1"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3,
              frameon=False, handlelength=1.2, columnspacing=0.8, handletextpad=0.4)
    save(fig, out, "fig_tier1_feature")


# ---------------------- Figure for Section 4: training-only EDA ---------------
def figure_eda(raw, ids, labels, features, out):
    """Tier 1's feature on the TRAINING split only (deduplicated), by family."""
    j = features.index(TIER1_FEATURE)
    values = np.asarray(raw[ids["train_dedup"], j], dtype="float64")
    family = family_of(labels[ids["train_dedup"]])
    keep = ~np.isnan(values)
    logged = np.log1p(values)
    edges = np.linspace(0, np.nanmax(logged) * 1.02, 45)
    fig, ax = plt.subplots(figsize=(3.03, 2.25), constrained_layout=True)
    for name, colour in [("BENIGN", MUTED), ("DoS Hulk", STYLE["DoS Hulk"][0]),
                         ("Slow DoS", STYLE["Slow DoS"][0]),
                         ("Brute force (FTP/SSH)", STYLE["Brute force (FTP/SSH)"][0])]:
        share, _ = np.histogram(logged[keep & (family == name)], bins=edges)
        ax.step(edges[:-1], share / share.sum(), where="post", color=colour,
                lw=1.6 if name == "BENIGN" else 1.2,
                label="Benign" if name == "BENIGN" else name)
    ticks = [0, 10, 100, 1_000, 10_000]
    ax.set_xticks(np.log1p(ticks)); ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("Bwd Packet Length Max (bytes, log scale)")
    ax.set_ylabel("Share of the family's flows")
    ax.grid(axis="y", color=GRID_INK, lw=0.4); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False,
              handlelength=1.2, columnspacing=0.8, handletextpad=0.4)
    save(fig, out, "fig_eda_feature")


# ------------------------- Figure 3: traffic timeline ------------------------
def figure_timeline(meta, out):
    frame = pd.DataFrame({
        "day": meta["day"].to_numpy(),
        "bin": (meta["minute_of_day"].to_numpy() // BIN_MINUTES) * BIN_MINUTES,
        "family": family_of(meta["Label"].to_numpy()),
    })
    counts = frame.groupby(["day", "bin", "family"]).size().unstack(fill_value=0)
    share = 100 * counts.div(counts.sum(axis=1), axis=0)

    by_day = frame.groupby(["day", "family"]).size().unstack(fill_value=0).reindex(DAYS)
    summary = (100 * by_day.div(by_day.sum(axis=1), axis=0)).round(3)
    summary.insert(0, "flows", by_day.sum(axis=1))
    summary.to_csv(out / "fig_traffic_by_day.csv")
    print("\nShare of flows by day (%):")
    print(summary.to_string())

    # One continuous axis: the working hours of the five days placed side by side.
    start, end = int(frame["bin"].min()), int(frame["bin"].max()) + BIN_MINUTES
    span = end - start
    offset = {day: i * span for i, day in enumerate(DAYS)}
    fig, ax = plt.subplots(figsize=(6.9, 2.5), constrained_layout=True)
    used = []
    for day in DAYS:
        if day not in share.index.get_level_values(0):
            continue
        full = share.loc[day].reindex(np.arange(start, end, BIN_MINUTES), fill_value=0)
        x = offset[day] + (full.index - start) + BIN_MINUTES / 2
        for name, (colour, marker) in STYLE.items():
            if name in full and full[name].max() > 0:
                ax.plot(x, full[name], color=colour, lw=1.3, marker=marker,
                        markersize=2.4, markevery=2)
                if name not in used:
                    used.append(name)
    for i in range(1, len(DAYS)):                       # day separators
        ax.axvline(i * span, color=GRID_INK, lw=0.6, zorder=0)
    # Split boundaries: Wednesday 11:08 (train | validation), Friday start (validation | test).
    cuts = [0, offset["Wednesday"] + VAL_START_MINUTE_ON_WEDNESDAY - start,
            offset["Friday"], len(DAYS) * span]
    for cut in cuts[1:3]:
        ax.axvline(cut, color=INK, lw=1.0, ls=(0, (4, 2)), zorder=5)
    for name, left, right in zip(("Train", "Validation", "Test"), cuts[:-1], cuts[1:]):
        ax.text((left + right) / 2, 1.03, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=INK)
        ax.annotate("", xy=(left, 1.015), xytext=(right, 1.015),
                    xycoords=ax.get_xaxis_transform(), textcoords=ax.get_xaxis_transform(),
                    arrowprops=dict(arrowstyle="|-|", color=MUTED, lw=0.6,
                                    mutation_scale=2.5, shrinkA=1.5, shrinkB=1.5))
    ax.text(cuts[1], 1.035, "Wed 11:08", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=6, color=MUTED)
    hours = [h for h in (9, 12, 15) if start <= h * 60 < end]
    ax.set_xticks([offset[d] + h * 60 - start for d in DAYS for h in hours])
    ax.set_xticklabels([f"{h}:00" for _ in DAYS for h in hours])
    for day in DAYS:
        ax.text(offset[day] + span / 2, -0.17, day, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color=INK)
    ax.set_xlim(0, len(DAYS) * span); ax.set_ylim(0, 100)
    ax.set_ylabel(f"Share of flows per {BIN_MINUTES} min (%)")
    ax.set_xlabel("Day and time of day", labelpad=12)
    ax.grid(axis="y", color=GRID_INK, lw=0.4); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Line2D([], [], color=STYLE[n][0], marker=STYLE[n][1], lw=1.3,
                               markersize=3.5, label=n) for n in STYLE if n in used],
               loc="outside lower center", ncol=8, frameon=False,
               handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
    save(fig, out, "fig_traffic_timeline")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()

    with np.load(args.artifacts / "splits.npz") as splits:
        ids = {name: splits[name] for name in ("train", "train_dedup", "train_benign", "val", "test")}
    meta = pd.read_parquet(args.artifacts / "meta.parquet",
                           columns=["Label", "day", "split", "minute_of_day"])
    labels = meta["Label"].to_numpy()
    assert (labels[ids["train_benign"]] == "BENIGN").all()
    assert meta["split"].iloc[ids["test"]].eq("test").all()
    scaled = np.load(args.artifacts / "X_scaled.npy", mmap_mode="r")
    raw = np.load(args.artifacts / "X_raw.npy", mmap_mode="r")
    features = json.loads((args.artifacts / "features.json").read_text())
    params = json.loads((args.artifacts / "preprocess_params.json").read_text())

    figure_pca(scaled, ids, labels, features, args.out)
    figure_eda(raw, ids, labels, features, args.out)
    figure_tier1(raw, ids, labels, features, params, args.out)
    figure_timeline(meta, args.out)


if __name__ == "__main__":
    main()
