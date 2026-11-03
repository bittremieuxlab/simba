"""Step 1: write the query/library sets of Search A and B (see benchmark_sets.py).

Output, per search, in <out_dir>/search_{A,B}/:
  queries.txt                   canonical SMILES per query (row order of every matrix)
  reference_spectra_smiles.txt  canonical SMILES per library spectrum (column order of score matrices)
  molecules.txt                 unique library molecules, first-appearance order (ground-truth columns)
  spectrum_mol_idx.npy          int64 per library spectrum: its row in molecules.txt
  meta.json                     counts and input files
--queries casmi (default) or msnlib (needs --msnlib_mgf). With --gt_job_dir, the
ground truth is computed for every query molecule of the input file; step 6
then picks the rows of the evaluated queries by SMILES.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_sets import QUERY_SETS, SEARCHES, SetLoader, canonical


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_pkl", required=True, help="SIMBA training dataset (edit_distance_neurips_nist_exhaustive.pkl)")
    ap.add_argument("--casmi_pkl", required=True, help="CASMI 2022 spec_df pickle")
    ap.add_argument("--gnps_mgf", required=True, help="GNPS library MGF (ALL_GNPS_NO_PROPOGATED snapshot)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--queries", default="casmi", choices=QUERY_SETS, help="query set (default casmi)")
    ap.add_argument("--msnlib_mgf", default=None, help="MSnLib ENAMDISC MGF, for --queries msnlib")
    ap.add_argument("--gt_job_dir", default=None,
                    help="also write ground-truth job inputs (02) for ALL query molecules of the input file, "
                         "before filtering and exclusion: <dir>/search_X/{queries,molecules}.txt")
    args = ap.parse_args()
    loader = SetLoader(args.dataset_pkl, args.casmi_pkl, args.gnps_mgf, args.queries, args.msnlib_mgf)

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
        meta = {"query_set": args.queries, "n_queries": len(q_smiles), "n_reference_spectra": len(lib_smiles), "n_molecules": len(molecules),
                "n_gt_pairs": len(q_smiles) * len(molecules),
                "inputs": {"dataset_pkl": args.dataset_pkl, "casmi_pkl": args.casmi_pkl, "gnps_mgf": args.gnps_mgf,
                           "msnlib_mgf": args.msnlib_mgf}}
        d.joinpath("meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"Search {search}: {meta['n_queries']} queries, {meta['n_reference_spectra']} library spectra, "
              f"{meta['n_molecules']} molecules -> {d}")
        if args.gt_job_dir:
            job = Path(args.gt_job_dir) / f"search_{search}"
            job.mkdir(parents=True, exist_ok=True)
            all_queries = loader.all_query_smiles()
            job.joinpath("queries.txt").write_text("\n".join(all_queries) + "\n")
            job.joinpath("molecules.txt").write_text("\n".join(molecules) + "\n")
            print(f"Search {search}: ground-truth job inputs {len(all_queries)} queries x {len(molecules)} molecules -> {job}")


if __name__ == "__main__":
    main()
