"""Analog discovery, stage 4: combine stage 2's per-method score matrices
with stage 3's exhaustive exact GT-MCES matrix into the paper's three
Figure-2-style panels, for TWO searches at once (search A: NIST20 +
MassSpecGym; search B: GNPS) -- one combined figure, 2 rows (one per
search) x 3 columns:

  (a) boxplot of best TRUE MCES among each query's top-10 inferred candidates
      (the paper's "retrieved analog" procedure, following Huber et al.)
  (b) ROC curve discriminating true analogs from unrelated molecules, pooled
      over every (query, library) pair with a resolved GT MCES -- NOT just
      top-10, which is the whole reason stage 3 computes the GT matrix
      exhaustively rather than only for each method's own top-10 (different
      methods have different top-10s; an exhaustive common GT source is what
      makes the curves comparable and lets every pair vote, not just the
      ones a given method already ranked high)
  (c) ranking-performance: cumulative fraction of queries whose single
      TRUE best analog (lowest GT MCES in the whole library) is retrieved
      within the model's own top-K, for K = 1..max_rank_k (default 1000)

Every panel overlays every available method on the same axes: SIMBA
raw-MCES-regression, SIMBA CORN-corrected (only when the checkpoint has an
mces_bucket head -- see analog_discovery_embed_rank.py's score_simba), and
plain binned-peak cosine (no SIMBA involved). The boxplot additionally shows
an oracle (best GT MCES anywhere in the whole library, ignoring ranking).

Two paper simplifications carried over from NOTES_014_2_ANALOG_DISCOVERY.md
(agreed with the user before building this pipeline): SIMBA vs plain cosine
only (no modified-cosine/Spec2Vec/MS2DeepScore/DreaMS), and RAW MCES
throughout rather than the paper's molecular-size-normalized MCES -- so the
paper's normalized-MCES<0.3 "true analog" cutoff doesn't directly transfer.
--roc_threshold (default 4.0) is a raw-MCES stand-in, chosen to match this
project's own already-trained CORN bucket boundary (bucket edges [2,4,6,8])
rather than an arbitrary pick -- pass a different value to see the ROC curve
at another cutoff; it does not require recomputing anything upstream since
the underlying GT matrix is continuous.

GT MCES values above the exact-MCES computation's threshold=20 are
LOWER BOUNDS, not exact distances (myopic_mces's own documented early-
rejection behavior -- see analog_discovery_exact_mces.py's docstring). This
only matters for very close cutoffs; a lower bound > roc_threshold still
conclusively means "not a true analog" at any threshold <= 20, which covers
every roc_threshold this script would sensibly be run with.

Usage:
    uv run python tools/analog_discovery_analyze.py \\
        --search_a_scores_dir data/analog_discovery/scores/<experiment>/search_A \\
        --search_a_exact_mces_dir data/analog_discovery/search_A_exact_mces \\
        --search_a_label "Search A (NIST20 + MassSpecGym)" \\
        --search_b_scores_dir data/analog_discovery/scores/<experiment>/search_B \\
        --search_b_exact_mces_dir data/analog_discovery/search_B_exact_mces \\
        --search_b_label "Search B (GNPS)" \\
        --output_dir data/analog_discovery/results/<experiment>
"""

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


ALL_METHODS = ["simba_raw", "simba_corn", "cosine"]
METHOD_LABELS = {
    "simba_raw": "SIMBA (raw MCES)",
    "simba_corn": "SIMBA (CORN-corrected)",
    "cosine": "Cosine (plain)",
}
METHOD_COLORS = {
    "simba_raw": "tab:blue",
    "simba_corn": "tab:orange",
    "cosine": "tab:green",
}


def load_scores(scores_dir: Path) -> tuple[list, list, dict, list]:
    """methods is whichever of ALL_METHODS actually has a score_*.npy file --
    e.g. simba_corn is absent for checkpoints with no mces_bucket head (see
    analog_discovery_embed_rank.py's score_simba), which is most of this
    project's non-FILIP, non-"full" ablation experiments."""
    q_smiles = (scores_dir / "smiles_query.txt").read_text().splitlines()
    lib_smiles = (scores_dir / "smiles_library.txt").read_text().splitlines()
    methods = [m for m in ALL_METHODS if (scores_dir / f"score_{m}.npy").exists()]
    scores = {m: np.load(scores_dir / f"score_{m}.npy") for m in methods}
    for m, arr in scores.items():
        assert arr.shape == (len(q_smiles), len(lib_smiles)), (
            f"score_{m}.npy shape {arr.shape} != ({len(q_smiles)}, {len(lib_smiles)})"
        )
    return q_smiles, lib_smiles, scores, methods


def build_gt_matrix(
    q_smiles: list, lib_smiles: list, exact_mces_dir: Path
) -> np.ndarray:
    """(n_query, n_library) GT-MCES matrix, NaN where unresolved (solver
    failure or, in principle, a pair analog_discovery_exact_mces.py's
    `prepare` never generated -- shouldn't happen here since it builds the
    full query x library cross product by construction)."""
    exact_smiles = (exact_mces_dir / "smiles.txt").read_text().splitlines()
    arr = np.load(exact_mces_dir / "mces_exact.npy")

    q_pos = {s: i for i, s in enumerate(q_smiles)}
    lib_pos = {s: i for i, s in enumerate(lib_smiles)}
    to_q_idx = np.full(len(exact_smiles), -1, dtype=np.int64)
    to_lib_idx = np.full(len(exact_smiles), -1, dtype=np.int64)
    for i, s in enumerate(exact_smiles):
        if s in q_pos:
            to_q_idx[i] = q_pos[s]
        if s in lib_pos:
            to_lib_idx[i] = lib_pos[s]

    idx_a = arr[:, 0].astype(np.int64)
    idx_b = arr[:, 1].astype(np.int64)
    mces = arr[:, 2]

    # By analog_discovery_exact_mces.py's own `prepare` construction, column
    # 0 is always the query-side molecule and column 1 the library-side --
    # not swapped, so no need to also try the reverse mapping.
    row = to_q_idx[idx_a]
    col = to_lib_idx[idx_b]

    resolved = (mces != -1.0) & ~np.isnan(mces)
    mapped = (row >= 0) & (col >= 0)
    n_unmapped = int((resolved & ~mapped).sum())
    if n_unmapped:
        print(
            f"  WARNING: {n_unmapped} resolved GT pairs could not be mapped to the "
            "score matrix's query/library SMILES -- canonical-SMILES mismatch "
            "between analog_discovery_exact_mces.py and analog_discovery_embed_rank.py "
            "runs? Excluded from analysis."
        )

    keep = resolved & mapped
    gt = np.full((len(q_smiles), len(lib_smiles)), np.nan, dtype=np.float32)
    gt[row[keep], col[keep]] = mces[keep]

    n_total = gt.size
    n_resolved = int(keep.sum())
    print(
        f"  GT matrix: {n_resolved:,} / {n_total:,} query-library pairs resolved ({100 * n_resolved / n_total:.1f}%)"
    )
    return gt


def panel_boxplot(ax, scores: dict, gt: np.ndarray, top_k: int, methods: list) -> dict:
    n_q = gt.shape[0]
    per_method = {}
    summary = {}
    for m in methods:
        best_mces = []
        n_no_gt = 0
        for i in range(n_q):
            top = np.argsort(scores[m][i])[:top_k]
            vals = gt[i, top]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                n_no_gt += 1
                continue
            best_mces.append(float(vals.min()))
        per_method[m] = best_mces
        summary[m] = {
            "n": len(best_mces),
            "n_excluded_no_gt": n_no_gt,
            "mean": float(np.mean(best_mces)) if best_mces else float("nan"),
            "median": float(np.median(best_mces)) if best_mces else float("nan"),
        }
        print(
            f"  [{m}] best-in-top{top_k} MCES: n={len(best_mces)} "
            f"mean={summary[m]['mean']:.2f} median={summary[m]['median']:.2f} "
            f"({n_no_gt} queries excluded, no resolved GT in top{top_k})"
        )

    # Oracle: best GT MCES anywhere in the WHOLE library per query, ignoring
    # every method's own ranking entirely -- the ceiling no ranking method
    # can beat, since it's not constrained to any top-k. Uses the exhaustive
    # GT matrix directly, which is why stage 3 computed it exhaustively
    # rather than per-method top-k-only.
    oracle_mces = []
    n_no_gt_oracle = 0
    for i in range(n_q):
        row = gt[i]
        if np.all(np.isnan(row)):
            n_no_gt_oracle += 1
            continue
        oracle_mces.append(float(np.nanmin(row)))
    summary["oracle"] = {
        "n": len(oracle_mces),
        "n_excluded_no_gt": n_no_gt_oracle,
        "mean": float(np.mean(oracle_mces)) if oracle_mces else float("nan"),
        "median": float(np.median(oracle_mces)) if oracle_mces else float("nan"),
    }
    print(
        f"  [oracle] best-in-library MCES: n={len(oracle_mces)} "
        f"mean={summary['oracle']['mean']:.2f} median={summary['oracle']['median']:.2f} "
        f"({n_no_gt_oracle} queries excluded, no resolved GT anywhere)"
    )

    plot_methods = [*methods, "oracle"]
    plot_labels = {**METHOD_LABELS, "oracle": "Oracle"}
    per_method["oracle"] = oracle_mces

    data = [per_method[m] for m in plot_methods]
    ax.boxplot(data, tick_labels=[plot_labels[m] for m in plot_methods], showmeans=True)
    ax.tick_params(axis="x", labelsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(15)
        label.set_ha("right")
    ax.set_ylabel(f"best MCES (top-{top_k})")
    return summary


def panel_roc(
    ax,
    scores: dict,
    gt: np.ndarray,
    roc_threshold: float,
    methods: list,
    target_fpr: float = 0.05,
) -> dict:
    """target_fpr mirrors the SIMBA paper's own Figure-2 ROC panel, which
    marks TPR at a fixed FPR=0.05 (dashed vertical line) alongside the full
    curve/AUC -- interpolated from the curve's own (fpr, tpr) points since
    roc_curve rarely lands exactly on FPR=0.05."""
    valid = ~np.isnan(gt)
    labels = (gt[valid] < roc_threshold).astype(np.int32)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    pct_pos = 100 * n_pos / (n_pos + n_neg)
    print(
        f"  ROC pool: {valid.sum():,} pairs, {n_pos:,} true analogs (<{roc_threshold}), {n_neg:,} unrelated"
    )

    summary = {}
    for m in methods:
        pred = -scores[m][
            valid
        ]  # higher = closer/more-likely-analog, matching roc_curve's convention
        fpr, tpr, _ = roc_curve(labels, pred)
        roc_auc = auc(fpr, tpr)
        tpr_at_target_fpr = float(np.interp(target_fpr, fpr, tpr))
        summary[m] = {
            "auc": float(roc_auc),
            f"tpr_at_fpr_{target_fpr:g}": tpr_at_target_fpr,
        }
        print(
            f"  [{m}] AUC={roc_auc:.4f}  TPR@FPR={target_fpr:g}: {tpr_at_target_fpr:.4f}"
        )
        ax.plot(
            fpr,
            tpr,
            label=f"{METHOD_LABELS[m]} (AUC={roc_auc:.3f})",
            color=METHOD_COLORS[m],
        )
    ax.axvline(
        target_fpr, color="k", linestyle=":", linewidth=1, label=f"FPR={target_fpr:g}"
    )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"threshold={roc_threshold:g}  ({pct_pos:.2f}% label=1)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    return summary


def panel_ranking_performance(
    ax, scores: dict, gt: np.ndarray, max_k: int, methods: list
) -> dict:
    n_q, n_lib = gt.shape
    max_k = min(max_k, n_lib)
    ks = np.arange(1, max_k + 1)

    ranks_by_method = {m: [] for m in methods}
    n_no_best_analog = 0
    for i in range(n_q):
        row = gt[i]
        if np.all(np.isnan(row)):
            n_no_best_analog += 1
            continue
        best_j = int(np.nanargmin(row))
        for m in methods:
            order = np.argsort(scores[m][i])
            rank = int(np.nonzero(order == best_j)[0][0]) + 1  # 1-indexed
            ranks_by_method[m].append(rank)

    print(
        f"  {n_no_best_analog} / {n_q} queries excluded (no resolved GT anywhere in library)"
    )

    milestone_ks = [k for k in (1, 5, 10, 20, 50, 100, 200, 500, 1000) if k <= max_k]

    summary = {}
    for m in methods:
        ranks = np.asarray(ranks_by_method[m])
        cum_frac = [(ranks <= k).mean() for k in ks]
        summary[m] = {
            "cum_frac": [float(v) for v in cum_frac],  # full curve, K=1..max_k
            "cum_frac_at_k": {str(k): float(cum_frac[k - 1]) for k in milestone_ks},
        }
        ax.plot(ks, cum_frac, label=METHOD_LABELS[m], color=METHOD_COLORS[m])
        print(
            f"  [{m}] hit@1={cum_frac[0]:.3f}  "
            + "  ".join(
                f"hit@{k}={summary[m]['cum_frac_at_k'][str(k)]:.3f}"
                for k in milestone_ks
                if k != 1
            )
        )

    ax.set_xlabel("K")
    ax.set_ylabel("fraction with true best in top-K")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    return summary


def _run_one_search(
    axes_row,
    scores_dir: Path,
    exact_mces_dir: Path,
    search_label: str,
    top_k: int,
    roc_threshold: float,
    max_rank_k: int,
) -> dict:
    print(f"\n=== {search_label} ===")
    print(f"Loading scores from {scores_dir} ...")
    q_smiles, lib_smiles, scores, methods = load_scores(scores_dir)
    print(f"  {len(q_smiles)} queries, {len(lib_smiles)} library molecules")
    print(f"  available methods: {methods}")

    print(f"Loading exact GT MCES from {exact_mces_dir} ...")
    gt = build_gt_matrix(q_smiles, lib_smiles, exact_mces_dir)

    ax_box, ax_roc, ax_rank = axes_row
    ax_box.set_title(f"{search_label}: best MCES (top-{top_k})", fontsize=10)

    print(f"\n(a) Boxplot (top-{top_k}) ...")
    boxplot_summary = panel_boxplot(ax_box, scores, gt, top_k, methods)

    print(f"\n(b) ROC (threshold={roc_threshold}) ...")
    roc_summary = panel_roc(ax_roc, scores, gt, roc_threshold, methods)

    print(f"\n(c) Ranking performance (K=1..{max_rank_k}) ...")
    ranking_summary = panel_ranking_performance(
        ax_rank, scores, gt, max_rank_k, methods
    )

    return {
        "search_label": search_label,
        "methods": methods,
        "n_query": len(q_smiles),
        "n_library": len(lib_smiles),
        "top_k": top_k,
        "roc_threshold": roc_threshold,
        "boxplot": boxplot_summary,
        "roc": roc_summary,
        "ranking_performance": ranking_summary,
    }


def run(
    search_a_scores_dir: str,
    search_a_exact_mces_dir: str,
    search_b_scores_dir: str,
    search_b_exact_mces_dir: str,
    output_dir: str,
    search_a_label: str = "Search A (NIST20 + MassSpecGym)",
    search_b_label: str = "Search B (GNPS)",
    top_k: int = 10,
    roc_threshold: float = 4.0,
    max_rank_k: int = 1000,
):
    """One combined figure, 2 rows (search A, search B) x 3 columns
    (boxplot, ROC, ranking performance), every available method overlaid on
    the same axes per panel (SIMBA raw + SIMBA CORN-corrected when present +
    plain cosine)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    summary_a = _run_one_search(
        axes[0],
        Path(search_a_scores_dir),
        Path(search_a_exact_mces_dir),
        search_a_label,
        top_k,
        roc_threshold,
        max_rank_k,
    )
    summary_b = _run_one_search(
        axes[1],
        Path(search_b_scores_dir),
        Path(search_b_exact_mces_dir),
        search_b_label,
        top_k,
        roc_threshold,
        max_rank_k,
    )

    fig.tight_layout()
    fig_path = output_dir / "analog_discovery_combined.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    summary = {"search_a": summary_a, "search_b": summary_b}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {fig_path} + summary.json to {output_dir}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--search_a_scores_dir",
        required=True,
        help="Output of analog_discovery_embed_rank.py for search A",
    )
    p.add_argument(
        "--search_a_exact_mces_dir",
        required=True,
        help="Output of analog_discovery_exact_mces.py for search A",
    )
    p.add_argument("--search_a_label", default="Search A (NIST20 + MassSpecGym)")
    p.add_argument(
        "--search_b_scores_dir",
        required=True,
        help="Output of analog_discovery_embed_rank.py for search B",
    )
    p.add_argument(
        "--search_b_exact_mces_dir",
        required=True,
        help="Output of analog_discovery_exact_mces.py for search B",
    )
    p.add_argument("--search_b_label", default="Search B (GNPS)")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K for the boxplot panel (paper uses 10)",
    )
    p.add_argument(
        "--roc_threshold",
        type=float,
        default=4.0,
        help="Raw-MCES 'true analog' cutoff for the ROC panel",
    )
    p.add_argument(
        "--max_rank_k",
        type=int,
        default=1000,
        help="Max K for the ranking-performance panel",
    )
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
