"""Step 6: analog-discovery metrics + figure with the normalized-MCES ground
truth (Johnson similarity of the actual MCES, output of
02_mces_ground_truth.py combine; NaN = not computed).

Ground truth is per (query, library molecule); every method's score matrix is
per (query, library spectrum) in 01_build_sets.py order. Higher score = better
(SIMBA: his ranking matrix; baselines: similarity; NaN score = ranked last).

Metrics per method and search (s = Johnson similarity, distance = normalized MCES distance = 1 - s):
  best-of-10 distance (his analog plot): per query, min distance over the
    top-10 retrieved spectra (ignoring NaN GT); median and q05/q25/q75/q95
    over queries. "Oracle" = the query's true best analog.
  precision@K (his obtain_precision_k_candidates, min threshold 0): the true
    best = argmax of the per-spectrum GT row (first spectrum on ties); queries
    with best similarity <= 0 are skipped; hit if the best molecule is among
    the molecules of the top-K retrieved spectra. K = 5, 10, 20, 40, 80, 100.
  ROC AUC: pooled over all (query, spectrum) pairs with GT; positive =
    distance < 0.3; score = -rank of the spectrum in the query's ranking.

--methods takes Label=dir pairs; each dir holds search_{A,B}/ranking.npy (SIMBA,
step 5) or search_{A,B}/similarity.npy (baselines, step 4).
Output: <out_dir>/summary.json and <out_dir>/analog_discovery.png
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

K_VALUES = [5, 10, 20, 40, 80, 100]
TOP_N_ANALOG = 10
POSITIVE_DISTANCE = 0.3
QUANTILES = {"q05": 5, "q25": 25, "median": 50, "q75": 75, "q95": 95}
SEARCH_LABELS = {"A": "Search A (NIST20 + MassSpecGym)", "B": "Search B (GNPS)"}


def load_scores(method_dir: Path, search: str, n_q: int, n_spec: int) -> np.ndarray:
    d = method_dir / f"search_{search}"
    path = d / "ranking.npy" if (d / "ranking.npy").exists() else d / "similarity.npy"
    scores = np.load(path).astype(np.float32)
    if scores.shape != (n_q, n_spec):
        raise SystemExit(f"{path}: shape {scores.shape}, expected {(n_q, n_spec)}")
    return np.where(np.isnan(scores), -np.inf, scores)


def evaluate_method(scores, gt, mol_idx, best_mol, include):
    n_q, n_spec = scores.shape
    order = np.argsort(-scores, axis=1, kind="stable")
    top_mols = mol_idx[order[:, : max(K_VALUES + [TOP_N_ANALOG])]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN top-10 rows -> NaN, counted below
        best10 = np.nanmax(np.take_along_axis(gt, top_mols[:, :TOP_N_ANALOG], axis=1), axis=1)
    dist10 = 1 - best10
    valid = ~np.isnan(dist10)

    precision = {}
    for k in K_VALUES:
        hits = (top_mols[include, :k] == best_mol[include, None]).any(axis=1)
        precision[k] = float(hits.mean())

    ranks = np.empty((n_q, n_spec), dtype=np.float32)
    rows = np.arange(n_q)[:, None]
    ranks[rows, order] = np.arange(n_spec, dtype=np.float32)[None, :]
    gt_spec = gt[:, mol_idx]
    has_gt = ~np.isnan(gt_spec)
    labels = (1 - gt_spec[has_gt]) < POSITIVE_DISTANCE
    score = -ranks[has_gt]
    auc = float(roc_auc_score(labels, score))
    fpr, tpr, _ = roc_curve(labels, score)
    step = max(1, len(fpr) // 2000)
    return {
        "best_of_10_distance": {name: float(np.percentile(dist10[valid], q)) for name, q in QUANTILES.items()},
        "n_queries_without_gt_in_top10": int((~valid).sum()),
        "roc_auc": auc,
        "tpr_at_fpr_0.05": float(np.interp(0.05, fpr, tpr)),
        "precision_at_k": precision,
        "_dist10": dist10[valid],
        "_roc": (fpr[::step], tpr[::step]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", required=True, help="Label=dir pairs")
    ap.add_argument("--gt_dir", required=True, help="output of 02_mces_ground_truth.py")
    ap.add_argument("--sets_dir", required=True, help="output of 01_build_sets.py")
    ap.add_argument("--searches", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    gt_dir, sets_dir, out = Path(args.gt_dir), Path(args.sets_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = [m.split("=", 1) for m in args.methods]

    summary = {}
    fig, axes = plt.subplots(len(args.searches), 3, figsize=(18, 5.5 * len(args.searches)), squeeze=False)
    for row, search in enumerate(args.searches):
        gt = np.load(gt_dir / f"search_{search}" / "gt_sim.npy")
        mol_idx = np.load(sets_dir / f"search_{search}" / "spectrum_mol_idx.npy")
        n_q, n_spec = gt.shape[0], len(mol_idx)
        gt_spec_best = np.where(np.isnan(gt), -np.inf, gt)[:, mol_idx]
        best_spec = np.argmax(gt_spec_best, axis=1)  # his rule: first spectrum with the max value
        best_val = gt_spec_best[np.arange(n_q), best_spec]
        best_mol = mol_idx[best_spec]
        include = best_val > 0
        del gt_spec_best
        oracle = 1 - np.nanmax(gt, axis=1)

        res = {}
        for label, rel in methods:
            print(f"search {search}: {label}", flush=True)
            res[label] = evaluate_method(load_scores(Path(rel), search, n_q, n_spec), gt, mol_idx, best_mol, include)
        summary[search] = {
            "n_queries": n_q,
            "n_queries_in_precision": int(include.sum()),
            "oracle_best_distance": {name: float(np.nanpercentile(oracle, q)) for name, q in QUANTILES.items()},
            "methods": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()},
        }

        ax_box, ax_roc, ax_prec = axes[row]
        data = [oracle[~np.isnan(oracle)]] + [r["_dist10"] for r in res.values()]
        names = ["Best candidate\n(oracle)"] + [k.replace(" ", "\n", 1) for k in res]
        ax_box.boxplot(data, labels=names, showfliers=False, whis=(5, 95))
        ax_box.set_ylabel("Normalized MCES distance, best of top-10 (lower = better)")
        ax_box.set_title(f"{SEARCH_LABELS[search]}: retrieved analogs")
        ax_box.tick_params(axis="x", labelsize=8)
        for label, r in res.items():
            fpr, tpr = r["_roc"]
            ax_roc.plot(fpr, tpr, label=f"{label} (AUC {r['roc_auc']:.3f})")
            ax_prec.plot(K_VALUES, [r["precision_at_k"][k] for k in K_VALUES], marker="*", label=label)
        ax_roc.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
        ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
                   title=f"ROC, positive = distance < {POSITIVE_DISTANCE}")
        ax_roc.legend(fontsize=8, loc="lower right")
        ax_prec.set(xlabel="K candidates", ylabel="Precision", xlim=(0, 100), ylim=(0, None),
                    title=f"Precision@K ({int(include.sum())} queries)")
        ax_prec.grid(True)
        ax_prec.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out / "analog_discovery.png", dpi=150)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for search, s in summary.items():
        print(f"\n== Search {search} ({s['n_queries']} queries) | oracle median {s['oracle_best_distance']['median']:.4f}")
        for label, r in s["methods"].items():
            d, p = r["best_of_10_distance"], r["precision_at_k"]
            print(f"  {label:28s} median {d['median']:.4f} q25 {d['q25']:.4f} q75 {d['q75']:.4f} | AUC {r['roc_auc']:.4f} "
                  f"| P@10 {p[10]:.4f} P@100 {p[100]:.4f}")
    print(f"\nwrote {out}/analog_discovery.png and summary.json")


if __name__ == "__main__":
    main()
