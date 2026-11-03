"""Write method scores (steps 4-5) in the result-file format of Sebastian's
notebooks, so notebooks/precision_candidates.ipynb (simba_v1) can load them.

For each Name=dir pair and each search, writes
  <out_dir>/<Name>_<query_set>_NEURIPS[_gnps]_results_analog_discovery_unknown_compounds.pkl
a pickled dict like the ones his analog notebook saves:
  similarities      float32 (queries x library spectra), higher = better
                    (SIMBA: his ranking matrix; baselines: similarity; NaN -> -inf)
  smiles_janssen    query SMILES (row order)
  smiles_reference  library SMILES per spectrum (column order)

--gt_dir: also write mces_retrieved_<query_set>_NEURIPS.pkl, the 'mces_retrieved' entry of
his analog notebook for every model and search (per query: MCES similarity of
the best molecule among the top-10 retrieved spectra, None if not computed):
  {"<Name>_casmi_NEURIPS[_gnps]": {"mces_retrieved": [...], "smiles_janssen": [...]}}
--skip_scores: write only that file.

Usage:
  .venv/bin/python tools/analog_discovery_benchmark/export_results_notebook_format.py --sets_dir $WORK/gt_inputs \
      --out_dir $WORK/notebook_format --methods simba_paper_checkpoint=$WORK/simba_scores/paper_checkpoint \
      spec2vec_ours=$WORK/baseline_scores/spec2vec_ours ...
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from benchmark_metrics import TOP_N_ANALOG, GroundTruth

def suffixes(query_set="casmi"):
    return {"A": f"_{query_set}_NEURIPS", "B": f"_{query_set}_NEURIPS_gnps"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="Name=dir pairs (dir holds search_{A,B}/)")
    ap.add_argument("--gt_dir", default=None, help="step-2 ground truth: also write mces_retrieved_casmi_NEURIPS.pkl")
    ap.add_argument("--skip_scores", action="store_true", help="do not write the (large) score files")
    ap.add_argument("--query_set", default="casmi", help="name used in the file names (casmi, msnlib)")
    args = ap.parse_args()
    SUFFIX = suffixes(args.query_set)
    retrieved_name = f"mces_retrieved_{args.query_set}_NEURIPS.pkl"
    retrieved = {}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for search in ("A", "B"):
        sets = Path(args.sets_dir) / f"search_{search}"
        smiles_q = sets.joinpath("queries.txt").read_text().split()
        smiles_ref = sets.joinpath("reference_spectra_smiles.txt").read_text().split()
        for name, method_dir in (m.split("=", 1) for m in args.methods):
            d = Path(method_dir) / f"search_{search}"
            path = d / "ranking.npy" if (d / "ranking.npy").exists() else d / "similarity.npy"
            scores = np.load(path).astype(np.float32)
            if scores.shape != (len(smiles_q), len(smiles_ref)):
                raise SystemExit(f"{path}: shape {scores.shape} does not match the sets")
            scores = np.where(np.isnan(scores), -np.inf, scores).astype(np.float32)
            if args.gt_dir:
                gt = GroundTruth(args.gt_dir, args.sets_dir, search)
                top = gt.mol_idx[np.argsort(-scores, axis=1, kind="stable")[:, :TOP_N_ANALOG]]
                best = np.take_along_axis(gt.gt, top, axis=1)
                retrieved[f"{name}{SUFFIX[search]}"] = {
                    "mces_retrieved": [None if np.isnan(row).all() else float(np.nanmax(row)) for row in best],
                    "smiles_janssen": smiles_q}
            if args.skip_scores:
                continue
            result = {"similarities": scores, "smiles_janssen": smiles_q, "smiles_reference": smiles_ref}
            target = out / f"{name}{SUFFIX[search]}_results_analog_discovery_unknown_compounds.pkl"
            with open(target, "wb") as fh:
                pickle.dump(result, fh, protocol=4)
            print(f"Search {search}: {target.name} {scores.shape}")
    if args.gt_dir:
        with open(out / retrieved_name, "wb") as fh:
            pickle.dump(retrieved, fh, protocol=4)
        print(f"wrote {out / retrieved_name}: {sorted(retrieved)}")


if __name__ == "__main__":
    main()
