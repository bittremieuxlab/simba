"""Analog discovery (paper Figure 2 reproduction): build the final plots.

Combines each search's SIMBA score matrices (analog_discovery_score_simba.py,
both the `dedup` and `no_mol_agg` subcommands) and baseline-model score
matrices (Spec2Vec/MS2DeepScore/DreaMS, see tools/ANALOG_DISCOVERY_README.md
for how those are produced) with the exact + normalized ground-truth MCES
(analog_discovery_prepare.py) into two combined figures -- one per
normalized-MCES metric (paper's Sebastian tier-similarity vs. the exact
Johnson-similarity formula) -- each 2 rows (Search A / Search B) x 3 panels:

  (a) violin+box of best normalized MCES among each query's top-10 ranked
      candidate SPECTRA (duplicate molecules allowed, matching Sebastian's
      own AnalogDiscovery.get_k_candidates -- no molecule-level dedup
      anywhere), selected by the normalized metric itself. Plus an Oracle
      column: best GT normalized MCES anywhere in the whole library,
      ignoring ranking -- the ceiling no ranking method can beat.
  (b) ROC curve discriminating true analogs (normalized MCES < 0.3) from
      unrelated molecules, pooled over every (query, spectrum) pair with a
      resolved GT MCES.
  (c) ranking performance: cumulative fraction of queries whose true best
      analog MOLECULE (lowest raw GT MCES anywhere in the library) has at
      least one of its own spectra ranked within the top-K raw spectrum
      slots (duplicates allowed, no molecule-level re-ranking -- this is
      Sebastian's own literal algorithm, verified from his simba_v1 source,
      not an approximation).

A baseline's own similarity.npy + smiles_library.txt are used as-is (not
assumed to share indexing with SIMBA's or each other's -- e.g. Spec2Vec's
matchms filters can drop a handful of spectra, so its library is a few rows
shorter than MS2DeepScore's). Baselines have no edit-distance tiebreak
(only SIMBA predicts that), so their ranking is plain descending similarity.

Caching: the expensive per-search work (building the raw ground-truth
matrix, and sorting every method's -- especially a baseline's, which can be
hundreds of thousands of columns -- full similarity matrix into a rank
matrix) does NOT depend on which of the two metrics is being plotted, so
it's computed once per search and cached to `--cache_dir` across runs, not
once per metric per run. Pass --refresh_cache after any upstream score
file changes (new/updated baseline, rerun SIMBA, etc.) to invalidate it.

Run analog_discovery_prepare.py's normalize_combine subcommand first for
each search + metric so both normalized_mces_cache*.npy already exist, and
analog_discovery_score_simba.py's `dedup` + `no_mol_agg` subcommands for
both searches -- this script only reads cached results, no fallback.

Usage:
    python tools/analog_discovery_plot.py \\
        --no_mol_agg_dir /mnt/data2/nkubrakov/analog_discovery/results_final_no_mol_agg \\
        --baseline_scores_dir /mnt/data2/nkubrakov/analog_discovery/baseline_scores \\
        --output_dir tools/analog_discovery_figures
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

TOP_K = 10
K_VALUES = [5, 10, 20, 40, 80, 100]  # matches Sebastian's own precision_candidates.ipynb exactly
ROC_THRESHOLD = 0.3

# name -> (color, linestyle, baseline_scores subdir or None for SIMBA/oracle)
METHODS = [
    ("oracle", "#4C72B0", "-", None),
    ("Spec2Vec (ours)", "#55A868", "-", "spec2vec_ours"),
    ("Spec2Vec (paper)", "#55A868", "--", "spec2vec_paper"),
    ("MS2DeepScore (ours)", "#8172B2", "-", "ms2deepscore_ours"),
    ("MS2DeepScore (paper)", "#8172B2", "--", "ms2deepscore_paper"),
    ("DreaMS (ours)", "#937860", "-", "dreams_ours"),
    ("DreaMS (paper embedding)", "#937860", "--", "dreams_paper_embedding"),
    ("SIMBA", "#C44E52", "-", None),
]


def load_scores(scores_dir: Path) -> tuple[list, list]:
    """Reads the `dedup` subcommand's output: the canonical per-molecule
    library ordering everything else (ground-truth caches, baseline-model
    alignment) is keyed against."""
    q_smiles = (scores_dir / "smiles_query.txt").read_text().splitlines()
    lib_smiles = (scores_dir / "smiles_library.txt").read_text().splitlines()
    return q_smiles, lib_smiles


def build_raw_gt_matrix(q_smiles: list, lib_smiles: list, exact_mces_dir: Path) -> np.ndarray:
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

    row = to_q_idx[idx_a]
    col = to_lib_idx[idx_b]
    resolved = (mces != -1.0) & ~np.isnan(mces)
    mapped = (row >= 0) & (col >= 0)
    keep = resolved & mapped

    gt_raw = np.full((len(q_smiles), len(lib_smiles)), np.nan, dtype=np.float32)
    gt_raw[row[keep], col[keep]] = mces[keep]

    n_resolved = int(keep.sum())
    print(f"  Raw GT matrix: {n_resolved:,} / {gt_raw.size:,} pairs resolved ({100 * n_resolved / gt_raw.size:.1f}%)")
    return gt_raw


def load_baseline(subdir: Path, lib_smiles_dedup: list):
    sim = np.load(subdir / "similarity.npy")
    lib_smiles_this = Path(subdir / "smiles_library.txt").read_text().splitlines()
    lib_pos = {s: i for i, s in enumerate(lib_smiles_dedup)}
    lib_mol_idx = np.array([lib_pos.get(s, -1) for s in lib_smiles_this], dtype=np.int64)
    valid_cols = lib_mol_idx >= 0
    sim = sim[:, valid_cols]
    lib_mol_idx = lib_mol_idx[valid_cols].astype(np.int32)
    # int32 not int64 (argsort's default): rank/index values never exceed
    # library size, and this array is cached to disk at up to ~500K columns
    # x 132 rows x ~7 methods x 2 searches -- int64 made that multiple GB.
    rank = np.argsort(np.argsort(-sim, axis=1), axis=1).astype(np.int32)  # descending similarity -> ascending rank, 0=best
    return rank, lib_mol_idx


def flat_rank_of_true_best(rank, lib_mol_idx, gt_raw):
    """Matches Sebastian's actual panel-c algorithm, verified directly from
    his own simba_v1 code (PerformanceAnalysis.obtain_precision_k_candidates):
    K counts raw ranked SPECTRUM slots, duplicates allowed -- there is NO
    molecule-level re-ranking step anywhere in his code. A query counts as
    "found within top-K" iff ANY of its true best molecule's own spectra has
    per-spectrum rank < K, i.e. simply the minimum per-spectrum rank among
    that molecule's spectra (+1 since rank is 0-indexed). An earlier version
    of this function instead collapsed to a per-molecule dense rank before
    comparing to K -- that was never actually his method, just something
    that happened to move our numbers closer to the paper's; keeping this
    comment so it isn't reintroduced by assuming "closer to paper" means
    "more correct"."""
    n_q = rank.shape[0]
    found = []
    for i in range(n_q):
        row = gt_raw[i]
        if np.all(np.isnan(row)):
            continue
        best_mol = int(np.nanargmin(row))
        spec_mask = lib_mol_idx == best_mol
        if not spec_mask.any():
            continue  # true best molecule has no spectra in this method's own (possibly filtered) library
        found.append(int(rank[i, spec_mask].min()) + 1)
    return np.asarray(found)


def _safe_key(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_")


def prepare_search_data(key: str, scores_dir: Path, exact_dir: Path, no_mol_agg_dir: Path,
                         baseline_dir: Path, label: str, cache_dir: Path, refresh: bool) -> dict:
    """Everything about a search that does NOT depend on which of the two
    normalized-MCES metrics is being plotted: the raw GT matrix, and every
    available method's (rank, lib_mol_idx). Cached to cache_dir/{key}.npz
    since the expensive part -- sorting a baseline's full similarity
    matrix, which can be hundreds of thousands of columns -- is otherwise
    redone twice per run (once per metric) and again on every re-run."""
    cache_path = cache_dir / f"{key}.npz"
    q_smiles, lib_smiles_dedup = load_scores(scores_dir)

    if cache_path.exists() and not refresh:
        print(f"  [{key}] loading cached rank data from {cache_path}")
        data = np.load(cache_path)
        methods = {}
        for name, _, _, _ in METHODS:
            if name == "oracle":
                continue
            sk = _safe_key(name)
            if f"rank__{sk}" in data:
                methods[name] = (data[f"rank__{sk}"], data[f"libidx__{sk}"])
        return {"q_smiles": q_smiles, "lib_smiles_dedup": lib_smiles_dedup,
                "gt_raw": data["gt_raw"], "methods": methods}

    print(f"  [{key}] building rank data (cache miss or --refresh_cache) ...")
    gt_raw = build_raw_gt_matrix(q_smiles, lib_smiles_dedup, exact_dir)

    methods = {"SIMBA": (np.load(no_mol_agg_dir / f"{label}_rank.npy").astype(np.int32),
                         np.load(no_mol_agg_dir / f"{label}_lib_mol_idx.npy").astype(np.int32))}
    for name, _, _, subdir in METHODS:
        if name in ("oracle", "SIMBA"):
            continue
        sim_path = baseline_dir / subdir / key.replace("Search ", "search_") / "similarity.npy"
        if sim_path.exists():
            methods[name] = load_baseline(sim_path.parent, lib_smiles_dedup)

    cache_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"gt_raw": gt_raw}
    for name, (rank, lib_mol_idx) in methods.items():
        sk = _safe_key(name)
        save_kwargs[f"rank__{sk}"] = rank
        save_kwargs[f"libidx__{sk}"] = lib_mol_idx
    np.savez_compressed(cache_path, **save_kwargs)
    print(f"  [{key}] cached rank data to {cache_path}")

    return {"q_smiles": q_smiles, "lib_smiles_dedup": lib_smiles_dedup, "gt_raw": gt_raw, "methods": methods}


def build_figure(searches, prepared: dict, exact_dirs: dict, metric_name, cache_filename, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    summaries = {}

    for (key, label, _, _), ax_row in zip(searches, axes):
        p = prepared[key]
        gt_raw = p["gt_raw"]
        gt_norm = np.load(exact_dirs[key] / cache_filename)
        n_q = gt_raw.shape[0]

        ax_box, ax_roc, ax_rank = ax_row
        box_data, box_labels, box_colors = [], [], []
        summaries.setdefault(key, {})

        for name, color, ls, _ in METHODS:
            if name == "oracle":
                oracle_norm = []
                for i in range(n_q):
                    row = gt_norm[i]
                    if not np.all(np.isnan(row)):
                        oracle_norm.append(float(np.nanmin(row)))
                box_data.append(np.array(oracle_norm)); box_labels.append("Best candidate\nMCES"); box_colors.append(color)
                summaries[key]["oracle"] = {"n": len(oracle_norm), "median": float(np.median(oracle_norm))}
                continue
            if name not in p["methods"]:
                continue  # not ready yet / not computed
            rank, lib_mol_idx = p["methods"][name]

            # (a) best-in-top-10 by normalized metric
            best_norm = []
            for i in range(n_q):
                top = np.argsort(rank[i])[:TOP_K]
                cand_mol = lib_mol_idx[top]
                norm_vals = gt_norm[i, cand_mol]
                ok = ~np.isnan(norm_vals)
                if ok.any():
                    best_norm.append(float(np.min(norm_vals[ok])))
            box_data.append(np.array(best_norm)); box_labels.append(name); box_colors.append(color)

            # (b) ROC
            gt_norm_per_spec = gt_norm[np.arange(n_q)[:, None], lib_mol_idx[None, :]]
            valid = ~np.isnan(gt_norm_per_spec)
            labels_roc = (gt_norm_per_spec[valid] < ROC_THRESHOLD).astype(np.int32)
            pred = -rank[valid]
            fpr, tpr, _ = roc_curve(labels_roc, pred)
            roc_auc_val = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.6, label=f"{name} (AUC={roc_auc_val:.2f})")

            # (c) ranking performance, at Sebastian's own K_VALUES only
            found_ranks = flat_rank_of_true_best(rank, lib_mol_idx, gt_raw)
            cum_frac = [(found_ranks <= k).mean() for k in K_VALUES]
            ax_rank.plot(K_VALUES, cum_frac, color=color, linestyle=ls, linewidth=1.4, marker="*", markersize=7, label=name)

            summaries[key][name] = {"n": len(best_norm), "median": float(np.median(best_norm)) if best_norm else None,
                                     "auc": float(roc_auc_val), "recall_at_100": float(cum_frac[-1])}
            print(f"[{metric_name}] {label} {name}: n={len(best_norm)} median={summaries[key][name]['median']:.3f} "
                  f"AUC={roc_auc_val:.4f} recall@100={cum_frac[-1]:.3f}")

        for i, (data, color) in enumerate(zip(box_data, box_colors)):
            parts = ax_box.violinplot([data], positions=[i + 1], showmeans=False, showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.85); pc.set_edgecolor("black"); pc.set_linewidth(0.6)
        ax_box.boxplot(box_data, widths=0.1, showfliers=True, patch_artist=False,
                       medianprops=dict(color="black", linewidth=1.5), whiskerprops=dict(color="black"),
                       capprops=dict(color="black"), boxprops=dict(color="black"),
                       flierprops=dict(marker="o", markersize=2, markerfacecolor="none"))
        ax_box.set_xticks(range(1, len(box_labels) + 1))
        ax_box.set_xticklabels(box_labels, fontsize=7, rotation=20, ha="right")
        ax_box.set_ylabel("Normalized MCES distance")
        ax_box.set_title(f"{label}: {metric_name}", fontsize=10)
        ax_box.grid(True, alpha=0.6, color="gray", linewidth=0.6)
        ax_box.set_axisbelow(True)

        ax_roc.axvline(0.05, color="black", linestyle=":", linewidth=1)
        ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
        ax_roc.legend(loc="lower right", fontsize=6.5, frameon=True)
        ax_roc.grid(True, alpha=0.6, color="gray", linewidth=0.6)
        ax_roc.set_axisbelow(True)

        ax_rank.set_xlabel("K candidates"); ax_rank.set_ylabel("Precision")
        ax_rank.set_xlim(0, 100)
        ax_rank.set_ylim(0, 0.4)
        ax_rank.grid(True, alpha=0.6, color="gray", linewidth=0.6)
        ax_rank.set_axisbelow(True)
        ax_rank.legend(loc="lower right", fontsize=6.5, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    (out_path.parent / f"summary_{out_path.stem}.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote {out_path}\n")


def run(
    search_a_scores_dir: str,
    search_a_exact_mces_dir: str,
    search_b_scores_dir: str,
    search_b_exact_mces_dir: str,
    no_mol_agg_dir: str,
    baseline_scores_dir: str,
    output_dir: str,
    cache_dir: str | None = None,
    refresh_cache: bool = False,
    search_a_label: str = "Search A (NIST20 + MassSpecGym)",
    search_b_label: str = "Search B (GNPS)",
):
    searches = [
        ("Search A", search_a_label, Path(search_a_scores_dir), Path(search_a_exact_mces_dir)),
        ("Search B", search_b_label, Path(search_b_scores_dir), Path(search_b_exact_mces_dir)),
    ]
    exact_dirs = {key: exact_dir for key, _, _, exact_dir in searches}
    no_mol_agg_dir = Path(no_mol_agg_dir)
    baseline_dir = Path(baseline_scores_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else no_mol_agg_dir / "plot_cache"

    print("Preparing per-search rank data (cached across metrics and runs) ...")
    prepared = {
        key: prepare_search_data(key, scores_dir, exact_dir, no_mol_agg_dir, baseline_dir, label, cache_dir, refresh_cache)
        for key, label, scores_dir, exact_dir in searches
    }

    build_figure(searches, prepared, exact_dirs, "paper metric", "normalized_mces_cache_sebastian.npy",
                 out_dir / "combined_all_methods_paper_metric.png")
    build_figure(searches, prepared, exact_dirs, "exact metric", "normalized_mces_cache.npy",
                 out_dir / "combined_all_methods_exact_metric.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search_a_scores_dir", default="/mnt/data2/nkubrakov/analog_discovery/scores/search_A")
    p.add_argument("--search_a_exact_mces_dir", default="/mnt/data2/nkubrakov/analog_discovery/search_A_exact_mces_final")
    p.add_argument("--search_a_label", default="Search A (NIST20 + MassSpecGym)")
    p.add_argument("--search_b_scores_dir", default="/mnt/data2/nkubrakov/analog_discovery/scores/search_B")
    p.add_argument("--search_b_exact_mces_dir", default="/mnt/data2/nkubrakov/analog_discovery/search_B_exact_mces_final")
    p.add_argument("--search_b_label", default="Search B (GNPS)")
    p.add_argument("--no_mol_agg_dir", default="/mnt/data2/nkubrakov/analog_discovery/results_final_no_mol_agg")
    p.add_argument("--baseline_scores_dir", default="/mnt/data2/nkubrakov/analog_discovery/baseline_scores")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None, help="default: <no_mol_agg_dir>/plot_cache")
    p.add_argument("--refresh_cache", action="store_true", help="recompute rank data even if a cache file exists (after any upstream score file changes)")
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
