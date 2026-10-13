"""Builds a slide-ready PNG comparison table across experiments, covering:
  - MAE (val_mae_mces_bucket_avg)
  - Overlap: self-bin vs the first 3 non-self GT-MCES bins (skip0/1/2)
  - Decoy hit@1/5/20 (val_hit_at_*, in-batch validation decoy pool)
  - MSG/ICEBERG hit@1/5/20 (iceberg_hit_at_*, real formula-matched candidates)
  - Analog discovery, Search A (NIST20+MassSpecGym) only: median best-candidate
    MCES (top-10), TPR@FPR=0.05, ROC AUC, hit@10/100/1000 -- all from the
    model's own simba_raw score (see analog_discovery_analyze.py)

Reads training-time metrics from each experiment's own Lightning CSVLogger
output (<experiment>/metrics.csv, last non-null value per column = final
validation check) and analog-discovery metrics from
data/analog_discovery/results/<experiment>/summary.json. Renders one table
per call, colored per-column (green=best, red=worst) relative to the rows in
that call -- meant to be re-run with a growing row list as each new "trick"
slide is built, so colors are always relative to what's actually being shown.

Usage:
    uv run python tools/build_metrics_table.py \\
        --rows mces_only_baseline_full_run:Baseline cosine_only_full_run:"+Cosine head" \\
        --output /path/to/slide_table.png \\
        --title "Baseline vs. +cosine-similarity head"
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


EXPERIMENTS_DIR = Path(
    "/sofia/projects/2026_053/simba_project/simba-triplet-experiments"
)
RESULTS_DIR = Path(
    "/sofia/projects/2026_053/simba_project/data/analog_discovery/results"
)

# Sentinel row name for the plain, non-learned spectral-cosine baseline (no
# model at all). Its validation-style metrics (overlap/hit@k) were computed
# separately and logged into full_run_diagnostics/tb_logs/cosine purely so
# they'd sit on the same dashboard as real checkpoints -- NOT a real/trained
# Lightning run, so it has no MAE logged (no regression head to compute MAE
# from) and must be read from that TB event file directly, not metrics.csv
# (which holds full_run_diagnostics's own real training run instead). Its
# analog-discovery numbers are the "cosine" method already present in every
# experiment's own summary.json (identical everywhere, since that score
# doesn't depend on any checkpoint) -- reused from full_run_diagnostics here
# only as a convenient, arbitrary source.
COSINE_BASELINE = "__cosine_baseline__"
_COSINE_TB_LOGDIR = EXPERIMENTS_DIR / "full_run_diagnostics" / "tb_logs" / "cosine"
_COSINE_ANALOG_SOURCE_EXP = "full_run_diagnostics"

_COSINE_TB_TAGS = {
    "overlap0": "val_overlap/self (MCES=0)_vs_(0,5]_skip0",
    "overlap1": "val_overlap/self (MCES=0)_vs_(5,10]_skip1",
    "overlap2": "val_overlap/self (MCES=0)_vs_(10,15]_skip2",
    "dhit1": "val_hit_at_1",
    "dhit5": "val_hit_at_5",
    "dhit20": "val_hit_at_20",
    "mhit1": "iceberg_hit_at_1",
    "mhit5": "iceberg_hit_at_5",
    "mhit20": "iceberg_hit_at_20",
}

_tb_event_accumulator_cache = {}


def _last_tb_scalar(logdir: Path, tag: str) -> float:
    from tensorboard.backend.event_processing import event_accumulator

    key = str(logdir)
    if key not in _tb_event_accumulator_cache:
        ea = event_accumulator.EventAccumulator(
            key, size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        _tb_event_accumulator_cache[key] = ea
    ea = _tb_event_accumulator_cache[key]
    events = ea.Scalars(tag)
    return events[-1].value


CSV_METRIC_COLUMNS = {
    "mae": "val_mae_mces_bucket_avg",
    "overlap0": "val_overlap/self (MCES=0)_vs_(0,5]_skip0",
    "overlap1": "val_overlap/self (MCES=0)_vs_(5,10]_skip1",
    "overlap2": "val_overlap/self (MCES=0)_vs_(10,15]_skip2",
    "dhit1": "val_hit_at_1",
    "dhit5": "val_hit_at_5",
    "dhit20": "val_hit_at_20",
    "mhit1": "iceberg_hit_at_1",
    "mhit5": "iceberg_hit_at_5",
    "mhit20": "iceberg_hit_at_20",
}

# "<experiment>#corrected" as a row's exp name uses these CORN-corrected
# columns/method instead -- only meaningful for a checkpoint with an
# mces_bucket head (see corn_schema). MAE has no corrected variant for the
# "self" bin alone, but val_mae_mces_corrected_bucket_avg (the average
# across buckets) exists and is what the MAE column actually shows.
CORRECTED_SUFFIX = "#corrected"
CSV_METRIC_COLUMNS_CORRECTED = {
    "mae": "val_mae_mces_corrected_bucket_avg",
    "overlap0": "val_overlap_corrected/self (MCES=0)_vs_(0,5]_skip0",
    "overlap1": "val_overlap_corrected/self (MCES=0)_vs_(5,10]_skip1",
    "overlap2": "val_overlap_corrected/self (MCES=0)_vs_(10,15]_skip2",
    "dhit1": "val_hit_at_corrected_1",
    "dhit5": "val_hit_at_corrected_5",
    "dhit20": "val_hit_at_corrected_20",
    "mhit1": "iceberg_hit_at_corrected_1",
    "mhit5": "iceberg_hit_at_corrected_5",
    "mhit20": "iceberg_hit_at_corrected_20",
}

GROUPS = [
    ("MAE", [("mae", "avg across\nbuckets", "lower")]),
    (
        "Overlap (self vs bin)",
        [
            ("overlap0", "self vs\n(0,5]", "lower"),
            ("overlap1", "self vs\n(5,10]", "lower"),
            ("overlap2", "self vs\n(10,15]", "lower"),
        ],
    ),
    (
        "Decoy hit@k",
        [
            ("dhit1", "hit@1", "higher"),
            ("dhit5", "hit@5", "higher"),
            ("dhit20", "hit@20", "higher"),
        ],
    ),
    (
        "MSG hit@k (ICEBERG)",
        [
            ("mhit1", "hit@1", "higher"),
            ("mhit5", "hit@5", "higher"),
            ("mhit20", "hit@20", "higher"),
        ],
    ),
    (
        "Analog discovery (Search A: NIST20+MSG)",
        [
            ("amedian", "med. MCES\n(top-10)", "lower"),
            ("atpr", "TPR@FPR\n=.05", "higher"),
            ("aauc", "AUC", "higher"),
            ("ahit10", "top-10", "higher"),
            ("ahit100", "top-100", "higher"),
            ("ahit1000", "top-1000", "higher"),
        ],
    ),
]


def _last_csv_value(exp: str, column: str) -> float:
    path = EXPERIMENTS_DIR / exp / "metrics.csv"
    val = None
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get(column, "")
            if v not in (None, ""):
                val = float(v)
    if val is None:
        raise ValueError(f"{exp}: column {column!r} never logged in metrics.csv")
    return val


def _analog_value(
    exp: str, field: str, k: int | None = None, method: str = "simba_raw"
) -> float:
    d = json.loads((RESULTS_DIR / exp / "summary.json").read_text())["search_a"]
    if field == "median":
        return d["boxplot"][method]["median"]
    if field == "tpr":
        return d["roc"][method]["tpr_at_fpr_0.05"]
    if field == "auc":
        return d["roc"][method]["auc"]
    if field == "hitk":
        return d["ranking_performance"][method]["cum_frac_at_k"][str(k)]
    raise ValueError(field)


def fetch(exp: str, key: str) -> float:
    if exp == COSINE_BASELINE:
        return _fetch_cosine_baseline(key)
    if exp.endswith(CORRECTED_SUFFIX):
        base_exp = exp[: -len(CORRECTED_SUFFIX)]
        if key in CSV_METRIC_COLUMNS_CORRECTED:
            return _last_csv_value(base_exp, CSV_METRIC_COLUMNS_CORRECTED[key])
        method = "simba_corn"
        exp = base_exp
    else:
        method = "simba_raw"
    if key in CSV_METRIC_COLUMNS:
        return _last_csv_value(exp, CSV_METRIC_COLUMNS[key])
    if key == "amedian":
        return _analog_value(exp, "median", method=method)
    if key == "atpr":
        return _analog_value(exp, "tpr", method=method)
    if key == "aauc":
        return _analog_value(exp, "auc", method=method)
    if key == "ahit10":
        return _analog_value(exp, "hitk", 10, method=method)
    if key == "ahit100":
        return _analog_value(exp, "hitk", 100, method=method)
    if key == "ahit1000":
        return _analog_value(exp, "hitk", 1000, method=method)
    raise KeyError(key)


def _fetch_cosine_baseline(key: str) -> float:
    if key == "mae":
        return float("nan")  # no regression head -> no MAE for this baseline
    if key in _COSINE_TB_TAGS:
        return _last_tb_scalar(_COSINE_TB_LOGDIR, _COSINE_TB_TAGS[key])
    field_map = {
        "amedian": ("median", None),
        "atpr": ("tpr", None),
        "aauc": ("auc", None),
        "ahit10": ("hitk", 10),
        "ahit100": ("hitk", 100),
        "ahit1000": ("hitk", 1000),
    }
    if key in field_map:
        field, k = field_map[key]
        return _analog_value(_COSINE_ANALOG_SOURCE_EXP, field, k, method="cosine")
    raise KeyError(key)


def _split_into_bands(groups, n_bands: int):
    """Greedily splits GROUPS (in order) into n_bands lists of groups, each
    covering roughly total_cols/n_bands columns -- keeps a slide-friendly
    aspect ratio instead of one very wide row of 16+ columns."""
    total_cols = sum(len(cols) for _, cols in groups)
    target = total_cols / n_bands
    bands, current, current_cols = [], [], 0
    for g in groups:
        current.append(g)
        current_cols += len(g[1])
        if current_cols >= target and len(bands) < n_bands - 1:
            bands.append(current)
            current, current_cols = [], 0
    if current:
        bands.append(current)
    return bands


def build_table(
    rows: list[tuple[str, str]],
    output_path: str,
    title: str | None = None,
    n_bands: int = 2,
):
    """rows: list of (experiment_dir_name, display_label), in the order they
    should appear top-to-bottom. The metric groups are wrapped into n_bands
    vertically-stacked blocks (same experiment rows repeated per block) so
    the figure's aspect ratio stays slide-friendly instead of one very wide
    16-column row.
    """
    flat_cols = [
        (k, label, direction) for _, cols in GROUPS for (k, label, direction) in cols
    ]
    data = np.zeros((len(rows), len(flat_cols)))
    for i, (exp, _) in enumerate(rows):
        for j, (key, _, _) in enumerate(flat_cols):
            data[i, j] = fetch(exp, key)

    # Per-column color score (0.5=white, 1=full green, 0=full red).
    #
    # Direction (red vs. green) for every row, including Cosine, comes from
    # ONE consistent reference: the column's own mean across all rows shown
    # -- so a row is never on the "wrong side" of white relative to where its
    # actual value sits.
    #
    # Intensity: rows are grouped by sign (same side of the mean) and
    # stretched into [K_MIN, 1.0] using the GROUP's own min/max -- this is
    # what keeps a small-but-real baseline-vs-trick improvement visibly
    # lighter/darker even when some other row's gap is 5-10x larger (if that
    # larger gap set the scale too, the small one would get crushed into a
    # barely-distinguishable pale shade right next to white). The K_MIN floor
    # means even the mildest row in a group still reads as clearly red/green,
    # never fading all the way to white.
    #
    # Cosine ("no NN") is excluded from SETTING a group's min/max whenever
    # that group already has >=2 other rows to form a meaningful spread from
    # -- it's a fixed, non-learned reference and very often an extreme
    # outlier (e.g. far better on hit@k), and letting it set the scale would
    # crush the baseline-vs-trick resolution that's actually being compared.
    # But it's still PLACED on that same scale (clipped), rather than always
    # forced to full saturation regardless of its real position -- otherwise
    # a trick that genuinely surpasses Cosine on some metric would render
    # paler than Cosine, backwards from the truth. Only when Cosine has no
    # such group to join (it's the sole row on its side) does it get full
    # saturation, as a standalone visual anchor. NaN-safe throughout: a row
    # with no value for a metric is rendered as a neutral "n/a" cell below.
    K_MIN = 0.35
    color = np.full_like(data, 0.5)
    is_cosine_row = np.array([exp == COSINE_BASELINE for exp, _ in rows])
    for j, (_, _, direction) in enumerate(flat_cols):
        col = data[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        overall_mean = col[finite].mean()
        diff = col - overall_mean
        if direction == "lower":
            diff = -diff  # positive diff always means "improvement" from here on
        sign = np.sign(diff)
        mag = np.abs(diff)

        row_color = np.full(col.shape, 0.5)
        for s in (1.0, -1.0):
            group = np.where(finite & (sign == s))[0]
            if group.size == 0:
                continue
            non_cosine = group[~is_cosine_row[group]]
            scale_idx = non_cosine if non_cosine.size >= 2 else group
            if scale_idx.size == 1 and is_cosine_row[scale_idx[0]]:
                row_color[group] = 0.5 + 0.5 * s * 1.0  # Cosine alone on this side
                continue
            gmin, gmax = mag[scale_idx].min(), mag[scale_idx].max()
            if gmax - gmin < 1e-12:
                k = np.full(group.shape, (K_MIN + 1.0) / 2)
            else:
                frac = np.clip((mag[group] - gmin) / (gmax - gmin), 0.0, 1.0)
                k = K_MIN + (1 - K_MIN) * frac
            row_color[group] = 0.5 + 0.5 * s * k
        row_color[finite & (mag <= 1e-12)] = 0.5  # exactly at the mean

        color[:, j] = np.where(finite, row_color, 0.5)

    cmap = LinearSegmentedColormap.from_list("rg", ["#f4a6a6", "#ffffff", "#a8e6a1"])

    bands = _split_into_bands(GROUPS, n_bands)
    n_rows = len(rows)

    # Everything below is in inches directly (the axes limits are set to
    # match figsize 1:1). Every band uses the same cell width (set by the
    # widest band, normally the first/top one) -- a narrower band just
    # leaves dead space on the right instead of stretching its cells wider.
    LABEL_W = 1.7
    DATA_W = 8.0
    ROW_H = 0.48
    GROUP_HEADER_H = 0.32
    COL_HEADER_H = 0.68
    BAND_GAP = 0.12
    TOP_MARGIN = 0.5 if title else 0.15
    BOTTOM_MARGIN = 0.15

    FS_GROUP_HEADER = 11
    FS_COL_HEADER = 9.5
    FS_CELL = 13
    FS_ROW_LABEL = 11.5
    FS_TITLE = 12

    # Every band uses the SAME cell width, set by the widest band (normally
    # the first/top one) -- a narrower band (e.g. analog discovery's 6
    # columns vs. the top band's 10) just leaves dead space on the right
    # instead of stretching its cells wider than the top band's.
    max_band_cols = max(sum(len(cols) for _, cols in band) for band in bands)
    cell_w = DATA_W / max_band_cols

    fig_w = LABEL_W + DATA_W
    band_h = GROUP_HEADER_H + COL_HEADER_H + n_rows * ROW_H
    fig_h = TOP_MARGIN + n_bands * band_h + (n_bands - 1) * BAND_GAP + BOTTOM_MARGIN

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes((0, 0, 1, 1))  # axes fill the whole figure: no default
    # matplotlib margins fighting the manual inch-based layout below.
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis("off")

    if title:
        ax.text(
            fig_w / 2,
            TOP_MARGIN * 0.35,
            title,
            ha="center",
            va="center",
            fontsize=FS_TITLE,
        )

    y_cursor = TOP_MARGIN
    col_offset_global = 0  # tracks flat_cols index into `data`
    for band in bands:
        band_cols = sum(len(cols) for _, cols in band)

        j = 0
        for group_name, cols in band:
            span = len(cols)
            x0 = LABEL_W + j * cell_w
            ax.text(
                x0 + span * cell_w / 2,
                y_cursor,
                group_name,
                ha="center",
                va="top",
                fontsize=FS_GROUP_HEADER,
                fontweight="bold",
            )
            ax.plot(
                [x0 + 0.03, x0 + span * cell_w - 0.03],
                [y_cursor + GROUP_HEADER_H, y_cursor + GROUP_HEADER_H],
                color="black",
                linewidth=1,
            )
            j += span
        y_cursor += GROUP_HEADER_H

        j = 0
        for _, cols in band:
            for _key, label, direction in cols:
                arrow = "↓" if direction == "lower" else "↑"
                x0 = LABEL_W + j * cell_w
                ax.text(
                    x0 + cell_w / 2,
                    y_cursor + COL_HEADER_H,
                    f"{label} {arrow}",
                    ha="center",
                    va="bottom",
                    fontsize=FS_COL_HEADER,
                    linespacing=1.3,
                )
                j += 1
        y_cursor += COL_HEADER_H

        for i in range(n_rows):
            y = y_cursor + i * ROW_H
            for j in range(band_cols):
                val = data[i, col_offset_global + j]
                is_na = not np.isfinite(val)
                col_color = color[i, col_offset_global + j]
                facecolor = "#e0e0e0" if is_na else cmap(col_color)
                x0 = LABEL_W + j * cell_w
                ax.add_patch(
                    Rectangle(
                        (x0, y), cell_w, ROW_H, facecolor=facecolor, edgecolor="white"
                    )
                )
                text = "n/a" if is_na else f"{val:.3f}"
                ax.text(
                    x0 + cell_w / 2,
                    y + ROW_H / 2,
                    text,
                    ha="center",
                    va="center",
                    fontsize=FS_CELL,
                )
            ax.text(
                LABEL_W - 0.1,
                y + ROW_H / 2,
                rows[i][1],
                ha="right",
                va="center",
                fontsize=FS_ROW_LABEL,
                fontweight="bold",
            )

        col_offset_global += band_cols
        y_cursor += n_rows * ROW_H + BAND_GAP

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--rows",
        nargs="+",
        required=True,
        help="experiment_dir:Display Label pairs, in row order",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--title", default=None)
    p.add_argument(
        "--n_bands",
        type=int,
        default=2,
        help="How many vertically-stacked column-bands to wrap the 5 metric "
        "groups into (keeps the figure's aspect ratio slide-friendly)",
    )
    args = p.parse_args()
    rows = [tuple(r.split(":", 1)) for r in args.rows]
    build_table(rows, args.output, args.title, args.n_bands)


if __name__ == "__main__":
    main()
