"""Write the step-2 ground truth in the format of Sebastian's notebooks, and
optionally compare it with his earlier ground-truth files.

His notebooks (precision_candidates.ipynb) load
  mces_similarity_casmi_NEURIPS[_gnps]_results_analog_discovery_unknown_compounds.pkl
with dill and use dict["similarities"]: (queries x library SPECTRA), his
row/column order, normalized MCES similarity, 0 = not computed. This writes
the same files from gt_results (expanded from molecules to spectra), with:
  similarities      float64, Johnson similarity of the actual MCES; 0 where missing
  computed          bool, False where step 2 has no value (status in gt_status.npy)
  smiles_janssen    query SMILES (row order), as in his result files
  smiles_reference  library SMILES per spectrum (column order)

--compare_dir: directory holding his earlier files with the same names; adds
a hexbin of old vs new values and summary numbers (gt_comparison.png/.json).

Usage:
  .venv/bin/python tools/analog_discovery_benchmark/export_gt_notebook_format.py \
      --gt_dir $WORK/gt_results --sets_dir $WORK/gt_inputs --out_dir $WORK/for_sebastian [--compare_dir <dir>]
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

FILE_NAMES = {"A": "mces_similarity_casmi_NEURIPS_results_analog_discovery_unknown_compounds.pkl",
              "B": "mces_similarity_casmi_NEURIPS_gnps_results_analog_discovery_unknown_compounds.pkl"}
SEARCH_LABELS = {"A": "Search A (NIST20 + MassSpecGym)", "B": "Search B (GNPS)"}


def load_old(path):
    """His files only contain numpy arrays: refuse anything else while unpickling."""
    import dill._dill

    allowed = {("dill._dill", "_create_array"), ("numpy.core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype")}

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in allowed:
                raise pickle.UnpicklingError(f"refusing {module}.{name}")
            return getattr(dill._dill, name) if module == "dill._dill" else super().find_class(module, name)

    with open(path, "rb") as fh:
        return Restricted(fh).load()["similarities"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--sets_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--compare_dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5)) if args.compare_dir else (None, None)

    for col, search in enumerate(("A", "B")):
        sets = Path(args.sets_dir) / f"search_{search}"
        gt = np.load(Path(args.gt_dir) / f"search_{search}" / "gt_sim.npy")
        mol_idx = np.load(sets / "spectrum_mol_idx.npy")
        new = gt[:, mol_idx].astype(np.float64)
        computed = ~np.isnan(new)
        result = {"similarities": np.where(computed, new, 0.0), "computed": computed,
                  "smiles_janssen": sets.joinpath("queries.txt").read_text().split(),
                  "smiles_reference": sets.joinpath("reference_spectra_smiles.txt").read_text().split()}
        with open(out / FILE_NAMES[search], "wb") as fh:
            pickle.dump(result, fh, protocol=4)
        print(f"Search {search}: wrote {out / FILE_NAMES[search]} {new.shape}, computed {computed.mean():.4f}")

        if not args.compare_dir:
            continue
        old = load_old(Path(args.compare_dir) / FILE_NAMES[search])
        if old.shape != new.shape:
            raise SystemExit(f"Search {search}: old {old.shape} vs new {new.shape}")
        both = computed & (old > 0)
        x, y = old[both], new[both]
        old_best, new_best = np.argmax(old, axis=1), np.argmax(np.where(computed, new, -1), axis=1)
        same_best = mol_idx[old_best] == mol_idx[new_best]
        comparison[search] = {
            "pairs": int(new.size),
            "old_nonzero_fraction": float((old > 0).mean()),
            "new_computed_fraction": float(computed.mean()),
            "pairs_in_both": int(both.sum()),
            "spearman_on_both": float(spearmanr(x[::50], y[::50]).correlation),
            "median_old_minus_new": float(np.median(x - y)),
            "fraction_new_below_old": float((y < x - 1e-9).mean()),
            "queries_same_best_molecule": f"{int(same_best.sum())}/{len(same_best)}",
        }
        ax = axes[col]
        hb = ax.hexbin(x, y, gridsize=120, bins="log", mincnt=1, cmap="viridis", extent=(0, 1, 0, 1))
        ax.plot([0, 1], [0, 1], color="red", lw=0.8, ls="--")
        ax.set(xlabel="Old: tier-based estimate (calculate_mces_sim)", ylabel="New: exact (calculate_mces_sim_exact)",
               title=f"{SEARCH_LABELS[search]}\n{both.sum():,} pairs computed in both, "
                     f"Spearman {comparison[search]['spearman_on_both']:.3f}",
               xlim=(0, 1), ylim=(0, 1), aspect="equal")
        fig.colorbar(hb, ax=ax, label="pairs (log)")
        print(f"Search {search}: {json.dumps(comparison[search])}")

    if args.compare_dir:
        fig.suptitle("Normalized MCES similarity per (query, library spectrum)")
        fig.tight_layout()
        fig.savefig(out / "gt_comparison.png", dpi=150)
        (out / "gt_comparison.json").write_text(json.dumps(comparison, indent=1) + "\n")


if __name__ == "__main__":
    main()
