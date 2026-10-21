"""Step 3: export the query and library spectra of Search A and B for the
baseline tools (Spec2Vec, MS2DeepScore, DreaMS).

Spectra are the originals (only intensities scaled to max = 1); each tool
applies its own documented preprocessing. Row order = 01_build_sets.py output
(asserted). Schema = what the simba-paper-other-tools batch-scoring scripts read:
  IDENTIFIER, SMILES, PRECURSOR M/Z, CHARGE, PARSED PEAKS (2 x n array: mz, intensity)

Output: <out_dir>/search_{A,B}_query.pkl and search_{A,B}_library.pkl (pandas pickles).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from casmi_sets import SEARCHES, SetLoader, check_order


def scaled_intensity(spectrum):
    intensity = np.asarray(spectrum.intensity, dtype=np.float64)
    m = intensity.max() if intensity.size else 0.0
    return intensity / m if m > 0 else intensity


def to_df(spectra):
    return pd.DataFrame.from_records([
        {
            "IDENTIFIER": str(i),
            "SMILES": Chem.CanonSmiles(s.smiles),
            "PRECURSOR M/Z": float(s.precursor_mz),
            "CHARGE": int(s.precursor_charge) if s.precursor_charge else 1,
            "PARSED PEAKS": np.stack([np.asarray(s.mz, dtype=np.float64), scaled_intensity(s)]),
        }
        for i, s in enumerate(spectra)
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_pkl", required=True)
    ap.add_argument("--casmi_pkl", required=True)
    ap.add_argument("--gnps_mgf", required=True)
    ap.add_argument("--sets_dir", required=True, help="output of 01_build_sets.py")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    loader = SetLoader(args.dataset_pkl, args.casmi_pkl, args.gnps_mgf)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for search in SEARCHES:
        queries, library = loader.load(search)
        check_order(args.sets_dir, search, queries, library)
        to_df(queries).to_pickle(out / f"search_{search}_query.pkl")
        to_df(library).to_pickle(out / f"search_{search}_library.pkl")
        print(f"Search {search}: {len(queries)} queries, {len(library)} library spectra -> {out}")


if __name__ == "__main__":
    main()
