#!/usr/bin/env python3
"""
Aggregate ProMEP results:
  1. Violin plot of per-protein Spearman correlations.
  2. Heatmap of stability-score distributions across all 66 DMS datasets.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from pathlib import Path

RESULTS_ROOT = Path("results/promep")
RUN_TAG = "promep_lrP3.5e-4_lrH2e-4"
OUT_DIR = RESULTS_ROOT / "total"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Root directory that holds one sub-folder per DMS dataset.
DATA_ROOT = Path("/data/sinfante/protstab/data/proMEP")

# Viral proteins excluded by the ProMEP paper
VIRAL_PROTEINS = {"REV_HV1H2_Fernandes_2016"}

# CSV score column name used by every dataset.
SCORE_COL = "target"


def collect_spearman(results_root: Path, run_tag: str) -> list[tuple[str, float]]:
    records = []
    for protein_dir in sorted(results_root.iterdir()):
        if not protein_dir.is_dir() or protein_dir.name == "total":
            continue
        if protein_dir.name in VIRAL_PROTEINS:
            continue
        test_json = protein_dir / run_tag / "logs" / f"test__{run_tag}.json"
        if not test_json.exists():
            continue
        with open(test_json) as f:
            data = json.load(f)
        records.append((protein_dir.name, float(data["test_spearman"])))
    return records


def plot_violin(values: np.ndarray, out_path: Path, macro_avg: float) -> None:
    fig, ax = plt.subplots(figsize=(2.2, 5.0))

    parts = ax.violinplot(values, positions=[0], showmedians=False,
                          showextrema=False, widths=0.7)

    for pc in parts["bodies"]:
        pc.set_facecolor("#C0604A")
        pc.set_edgecolor("#C0604A")
        pc.set_alpha(0.85)

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    whisker_lo = max(values.min(), q1 - 1.5 * (q3 - q1))
    whisker_hi = min(values.max(), q3 + 1.5 * (q3 - q1))

    ax.vlines(0, whisker_lo, whisker_hi, color="black", linewidth=1.2, zorder=2)
    ax.vlines(0, q1, q3, color="black", linewidth=4.5, zorder=3)
    ax.plot(0, median, "o", color="white", markersize=5, zorder=4,
            markeredgecolor="black", markeredgewidth=0.5)

    ax.set_ylim(-0.25, 1.05)
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Spearman's rank correlation", fontsize=10)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.axhline(macro_avg, color="#C0604A", linewidth=1.0,
               linestyle="--", alpha=0.7, zorder=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUT] Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# DMS score-distribution heatmap
# ---------------------------------------------------------------------------

def load_dms_scores_all(
    data_root: Path,
    score_col: str,
    splits: tuple = ("train", "valid", "test"),
) -> dict:
    """
    Load scores from all splits combined for every dataset under *data_root*,
    skipping folders listed in VIRAL_PROTEINS.

    Returns
    -------
    dict mapping dataset name -> 1-D float array of all scores across splits
    """
    datasets: dict = {}

    if not data_root.exists():
        print(f"  [warn] DATA_ROOT not found: {data_root}")
        return datasets

    for ds_dir in sorted(data_root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name in VIRAL_PROTEINS:
            continue

        all_scores = []
        for split in splits:
            csv_path = ds_dir / f"{split}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if score_col not in df.columns:
                continue
            scores = df[score_col].dropna().to_numpy(dtype=float)
            all_scores.append(scores)

        if all_scores:
            combined = np.concatenate(all_scores)
            if len(combined) > 0:
                datasets[ds_dir.name] = combined

    return datasets


def load_dms_scores_split(
    data_root: Path,
    split_file: str,
    score_col: str,
) -> dict:
    """
    Load stability scores for a single CSV split from every dataset directory
    under *data_root*, skipping folders listed in VIRAL_PROTEINS.

    Parameters
    ----------
    split_file : e.g. "train.csv", "valid.csv", "test.csv"

    Returns
    -------
    dict mapping dataset name -> 1-D float array of scores
    """
    datasets = {}

    if not data_root.exists():
        print(f"  [warn] DATA_ROOT not found: {data_root}")
        return datasets

    for ds_dir in sorted(data_root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name in VIRAL_PROTEINS:
            continue

        csv_path = ds_dir / split_file
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if score_col not in df.columns:
            print(f"  [warn] '{score_col}' not found in {csv_path} – skipping.")
            continue

        scores = df[score_col].dropna().to_numpy(dtype=float)
        if len(scores) > 0:
            datasets[ds_dir.name] = scores

    return datasets


def compute_quantile_bin_edges(all_scores: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Compute quantile-spaced bin edges from a pooled score array.

    Using percentile-based edges ensures that each bin column covers roughly
    the same fraction of the pooled data density.  The dense central region
    therefore gets many narrow bins (high resolution) while the sparse tails
    get wide bins (compressed), preventing a handful of extreme-value datasets
    from visually collapsing all other distributions into a tiny x-axis slice.

    Duplicate percentile values (which occur in discrete score distributions)
    are nudged apart by a tiny epsilon to keep edges strictly increasing.
    """
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(all_scores, quantiles)

    # Guarantee strict monotonicity.
    eps = max(np.finfo(float).eps * np.abs(edges).max() * 100, 1e-10)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps

    return edges


def build_histogram_matrix(
    datasets: dict,
    bin_edges: np.ndarray,
) -> tuple:
    """
    Build a (n_datasets × n_bins) row-normalised frequency matrix using
    pre-computed *bin_edges* (shared across all splits for comparability).

    Returns
    -------
    matrix : shape (n_datasets, n_bins), each row sums to 1
    names  : list of dataset names in row order
    """
    n_bins = len(bin_edges) - 1
    names  = list(datasets.keys())
    matrix = np.zeros((len(names), n_bins), dtype=float)

    for i, name in enumerate(names):
        counts, _ = np.histogram(datasets[name], bins=bin_edges)
        total = counts.sum()
        if total > 0:
            matrix[i] = counts / total  # row-normalise → each row sums to 1

    return matrix, names


def plot_dms_heatmap(
    matrix: np.ndarray,
    bin_edges: np.ndarray,
    names: list,
    out_path: Path,
    split_label: str = "",
    log_color: bool = False,
) -> None:
    """
    Render the (n_datasets × n_bins) normalised-frequency matrix as a heatmap.

    Layout
    ------
    y-axis  : dataset names (one compact row per DMS assay)
    x-axis  : quantile-spaced stability-score bins; tick labels show actual
              score values so the reader can still interpret the scale
    colour  : normalised frequency (row-independent → rows are comparable)

    Notes on quantile bins
    ----------------------
    Because bin edges are quantile-spaced the columns are *not* linearly
    spaced in score value.  The dense central region is shown at higher
    resolution while extreme tails are compressed into wide edge bins.
    Tick labels display the actual score values at each shown bin edge.
    """
    n_rows, n_cols = matrix.shape
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Compact row height so 66 rows stay readable without a very tall figure.
    row_height_in = 0.15
    fig_height = max(7, n_rows * row_height_in + 2.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    # Colour normalisation.
    norm = (
        LogNorm(vmin=matrix[matrix > 0].min(), vmax=matrix.max())
        if log_color
        else None
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdPu",         
        norm=norm,
        vmin=None if log_color else 0.0,
        vmax=None if log_color else matrix.max(),
        interpolation="nearest",
    )

    # y-axis: one labelled row per dataset.
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(names, fontsize=6.5, fontfamily="monospace")
    ax.tick_params(axis="y", length=0, pad=2)

    # x-axis: show ~10 tick labels with actual score values.
    desired_ticks = 10
    step = max(1, n_cols // desired_ticks)
    xtick_pos = np.arange(0, n_cols, step)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(
        [f"{bin_centres[k]:.2f}" for k in xtick_pos],
        fontsize=9, rotation=45, ha="right",
    )

    ax.set_xlabel("DMS score (quantile-spaced bins)", fontsize=11, labelpad=6)
    ax.set_ylabel("DMS dataset / assay", fontsize=11, labelpad=8)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.018, shrink=0.8)
    cbar.set_label(
        "Normalised frequency" + (" (log)" if log_color else ""),
        fontsize=9,
    )
    cbar.ax.tick_params(labelsize=7)

    title = f"DMS score distributions — {n_rows} datasets"
    if split_label:
        title += f"  [{split_label} split]"
    ax.set_title(title, fontsize=12, pad=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUT] Heatmap saved to {out_path}")


# ---------------------------------------------------------------------------
# DMS dataset summary statistics
# ---------------------------------------------------------------------------

def compute_dms_stats(
    data_root: Path,
    score_col: str = "target",
    splits: tuple = ("train", "valid", "test"),
) -> tuple:
    """
    Compute per-assay and aggregate split statistics for all DMS datasets.

    Label normalization note
    ------------------------
    Labels are the RAW DMS_score values written by promep_to_lomus_csv.py.
    features_CSV.py standardises only the X feature matrix (StandardScaler);
    the y_*_aligned.npy arrays are saved verbatim from the CSV 'target' column.
    featuresdataset.py and train/test scripts use those raw labels unchanged.

    Returns
    -------
    per_assay : pd.DataFrame  — one row per dataset, columns:
                dataset, {split}_n, {split}_mean, {split}_sd for each split
    agg       : pd.DataFrame  — one row per split with aggregate stats
    """

    def _iqr(x: np.ndarray) -> float:
        return float(np.percentile(x, 75) - np.percentile(x, 25))

    rows = []
    skipped = []

    for ds_dir in sorted(data_root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name in VIRAL_PROTEINS:
            skipped.append(ds_dir.name)
            continue

        record: dict = {"dataset": ds_dir.name}
        ok = True
        for split in splits:
            csv_path = ds_dir / f"{split}.csv"
            if not csv_path.exists():
                print(f"  [warn] {ds_dir.name}: missing {split}.csv – skipped.")
                ok = False
                break
            df = pd.read_csv(csv_path)
            if score_col not in df.columns:
                print(f"  [warn] {ds_dir.name}: no '{score_col}' in {split}.csv – skipped.")
                ok = False
                break
            vals = df[score_col].dropna().values
            record[f"{split}_n"]    = int(len(vals))
            record[f"{split}_mean"] = float(np.mean(vals))
            record[f"{split}_sd"]   = float(np.std(vals, ddof=1))   # sample SD

        if ok:
            rows.append(record)

    per_assay = pd.DataFrame(rows)

    print(f"[stats] Datasets included : {len(per_assay)}  |  skipped : {len(skipped)}"
          f"  ({', '.join(skipped) or 'none'})")

    # Aggregate: statistics computed per-assay first, then summarised across assays.
    agg_rows = []
    for split in splits:
        ns    = per_assay[f"{split}_n"].values
        means = per_assay[f"{split}_mean"].values
        sds   = per_assay[f"{split}_sd"].values
        agg_rows.append({
            "split"              : split,
            "n_assays"           : len(per_assay),
            "total_samples"      : int(ns.sum()),
            "median_n"           : float(np.median(ns)),
            "iqr_n"              : _iqr(ns),
            "min_n"              : int(ns.min()),
            "max_n"              : int(ns.max()),
            "median_mean_label"  : float(np.median(means)),
            "iqr_mean_label"     : _iqr(means),
            "median_sd_label"    : float(np.median(sds)),
            "iqr_sd_label"       : _iqr(sds),
        })

    return per_assay, pd.DataFrame(agg_rows)


def main():
    records = collect_spearman(RESULTS_ROOT, RUN_TAG)
    if not records:
        raise RuntimeError(f"No test results found under {RESULTS_ROOT} for run tag '{RUN_TAG}'")

    proteins = [r[0] for r in records]
    values = np.array([r[1] for r in records])
    macro_avg = float(values.mean())

    print(f"Proteins evaluated : {len(proteins)}")
    print(f"Macro-average Spearman : {macro_avg:.4f}")
    print(f"Median                 : {np.median(values):.4f}")
    print(f"Std                    : {values.std():.4f}")
    print(f"Min / Max              : {values.min():.4f} / {values.max():.4f}")

    # Per-protein table
    print("\nPer-protein results (sorted by Spearman):")
    for prot, sp in sorted(records, key=lambda x: x[1], reverse=True):
        print(f"  {sp:+.4f}  {prot}")

    # Save summary JSON
    summary = {
        "run_tag": RUN_TAG,
        "n_proteins": len(proteins),
        "macro_avg_spearman": macro_avg,
        "median_spearman": float(np.median(values)),
        "std_spearman": float(values.std()),
        "min_spearman": float(values.min()),
        "max_spearman": float(values.max()),
        "per_protein": {p: float(s) for p, s in records},
    }
    summary_path = OUT_DIR / f"summary__{RUN_TAG}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OUT] Summary saved to {summary_path}")

    plot_violin(values, OUT_DIR / f"violin__{RUN_TAG}.png", macro_avg)

    # ------------------------------------------------------------------ #
    # DMS dataset summary statistics                                       #
    # ------------------------------------------------------------------ #
    print("\nComputing DMS dataset summary statistics …")
    per_assay, agg_stats = compute_dms_stats(DATA_ROOT)

    per_assay_path = OUT_DIR / "dms_per_assay_stats.csv"
    agg_stats_path = OUT_DIR / "dms_aggregate_stats.csv"
    per_assay.to_csv(per_assay_path, index=False, float_format="%.6f")
    agg_stats.to_csv(agg_stats_path, index=False, float_format="%.6f")
    print(f"[OUT] Per-assay stats  → {per_assay_path}")
    print(f"[OUT] Aggregate stats  → {agg_stats_path}")
    print("\nAggregate DMS split statistics:")
    print(agg_stats.to_string(index=False))

    # ------------------------------------------------------------------ #
    # DMS score-distribution heatmap (all splits combined)                 #
    # ------------------------------------------------------------------ #
    print("\nBuilding DMS score-distribution heatmap (all splits combined) …")

    all_datasets = load_dms_scores_all(DATA_ROOT, SCORE_COL)

    if not all_datasets:
        print(f"  [warn] No DMS data found under {DATA_ROOT} – heatmap skipped.")
    else:
        pooled = np.concatenate(list(all_datasets.values()))
        bin_edges = compute_quantile_bin_edges(pooled, n_bins=60)
        matrix, names = build_histogram_matrix(all_datasets, bin_edges)
        plot_dms_heatmap(
            matrix,
            bin_edges,
            names,
            out_path=OUT_DIR / "dms_score_heatmap_all.png",
        )


if __name__ == "__main__":
    main()
