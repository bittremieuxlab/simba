"""Step 1: write the query/library sets of Search A and B (see casmi_sets.py).

Output, per search, in <out_dir>/search_{A,B}/:
  queries.txt                   canonical SMILES per query (row order of every matrix)
  reference_spectra_smiles.txt  canonical SMILES per library spectrum (column order of score matrices)
  molecules.txt                 unique library molecules, first-appearance order (ground-truth columns)
  spectrum_mol_idx.npy          int64 per library spectrum: its row in molecules.txt
  meta.json                     counts and input files
"""

import argparse
import json
from pathlib import Path

import numpy as np

from casmi_sets import SEARCHES, SetLoader, canonical


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_pkl", required=True, help="SIMBA training dataset (edit_distance_neurips_nist_exhaustive.pkl)")
    ap.add_argument("--casmi_pkl", required=True, help="CASMI 2022 spec_df pickle")
    ap.add_argument("--gnps_mgf", required=True, help="GNPS library MGF (ALL_GNPS_NO_PROPOGATED snapshot)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    loader = SetLoader(args.dataset_pkl, args.casmi_pkl, args.gnps_mgf)

    for search in SEARCHES:
        queries, library = loader.load(search)
        q_smiles, lib_smiles = canonical(queries), canonical(library)
        molecules = list(dict.fromkeys(lib_smiles))
        position = {s: i for i, s in enumerate(molecules)}
        d = Path(args.out_dir) / f"search_{search}"
        d.mkdir(parents=True, exist_ok=True)
        d.joinpath("queries.txt").write_text("\n".join(q_smiles) + "\n")
        d.joinpath("reference_spectra_smiles.txt").write_text("\n".join(lib_smiles) + "\n")
        d.joinpath("molecules.txt").write_text("\n".join(molecules) + "\n")
        np.save(d / "spectrum_mol_idx.npy", np.array([position[s] for s in lib_smiles], dtype=np.int64))
        meta = {"n_queries": len(q_smiles), "n_reference_spectra": len(lib_smiles), "n_molecules": len(molecules),
                "n_gt_pairs": len(q_smiles) * len(molecules),
                "inputs": {"dataset_pkl": args.dataset_pkl, "casmi_pkl": args.casmi_pkl, "gnps_mgf": args.gnps_mgf}}
        d.joinpath("meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"Search {search}: {meta['n_queries']} queries, {meta['n_reference_spectra']} library spectra, "
              f"{meta['n_molecules']} molecules -> {d}")


if __name__ == "__main__":
    main()
