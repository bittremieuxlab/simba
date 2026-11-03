"""Step 6: analog-discovery metrics + figure with the normalized-MCES ground
truth (Johnson similarity of the actual MCES, output of
02_mces_ground_truth.py combine; NaN = not computed).

Metric definitions: see benchmark_metrics.py.

--methods takes Label=dir pairs; each dir holds search_{A,B}/ranking.npy (SIMBA,
step 5) or search_{A,B}/similarity.npy (baselines, step 4). A label starting
with a tool name (SIMBA, Spec2Vec, MS2DeepScore, DreaMS, Modified cosine) gets
that tool's color; "(paper ...)" labels are drawn dashed.
Output: <out_dir>/summary.json and <out_dir>/analog_discovery.{png,svg}
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark_metrics import QUANTILES, GroundTruth, draw_search  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", required=True, help="Label=dir pairs")
    ap.add_argument("--gt_dir", required=True, help="output of 02_mces_ground_truth.py")
    ap.add_argument("--sets_dir", required=True, help="output of 01_build_sets.py")
    ap.add_argument("--searches", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = [m.split("=", 1) for m in args.methods]

    summary = {}
    fig, axes = plt.subplots(len(args.searches), 3, figsize=(18, 5.5 * len(args.searches)), squeeze=False)
    for row, search in enumerate(args.searches):
        gt = GroundTruth(args.gt_dir, args.sets_dir, search)
        results = {}
        for label, method_dir in methods:
            print(f"search {search}: {label}", flush=True)
            results[label] = gt.evaluate(gt.load_scores(method_dir))
        summary[search] = {
            "n_queries": gt.n_queries,
            "n_queries_in_precision": int(gt.include.sum()),
            "oracle_best_distance": {name: float(np.nanpercentile(gt.oracle, q)) for name, q in QUANTILES.items()},
            "methods": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()},
        }
        draw_search(axes[row], gt, results)

    fig.tight_layout()
    fig.savefig(out / "analog_discovery.png", dpi=150)
    fig.savefig(out / "analog_discovery.svg")
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
