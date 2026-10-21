"""Step 4 (Spec2Vec only): batch scoring for the analog-discovery comparison, ALIGNED to the
input order (spec2vec 0.8.0 / matchms 0.26.4).

Same method as simba-paper-other-tools/spec2vec/experiments/simba_training/
03_score_analog_discovery.py (matchms DEFAULT_FILTERS, Spec2Vec with
intensity_weighting_power=0.5, allowed_missing_percentage=100), except that a
spectrum rejected by DEFAULT_FILTERS is NOT dropped: its row/column is NaN, so
similarity.npy is always (n_query, n_library) in the exact input order.

Output: similarity.npy (float32, higher = more similar, NaN = filtered out),
smiles_query.txt / smiles_library.txt (all input rows), kept_query.npy /
kept_library.npy (bool masks).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from matchms import Spectrum, SpectrumProcessor
from matchms.filtering.default_pipelines import DEFAULT_FILTERS
from spec2vec import Spec2Vec

_processor = SpectrumProcessor(DEFAULT_FILTERS)


def df_to_spectra(df):
    spectra, kept = [], np.zeros(len(df), dtype=bool)
    for pos, (_, row) in enumerate(df.iterrows()):
        mz, intensity = row["PARSED PEAKS"]
        order = np.argsort(mz)
        s = Spectrum(
            mz=mz[order].astype(np.float64),
            intensities=intensity[order].astype(np.float64),
            metadata={"precursor_mz": float(row["PRECURSOR M/Z"]), "charge": int(row["CHARGE"]), "ionmode": "positive"},
        )
        s = _processor.process_spectrum(s)
        if s is not None:
            spectra.append(s)
            kept[pos] = True
    return spectra, kept


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query-pkl", required=True, type=Path)
    p.add_argument("--library-pkl", required=True, type=Path)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-library-limit", type=int, default=None, help="for smoke testing on a subset")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    q_df, lib_df = pd.read_pickle(args.query_pkl), pd.read_pickle(args.library_pkl)
    if args.n_library_limit:
        lib_df = lib_df.iloc[: args.n_library_limit]
    t0 = time.time()
    q_spectra, q_kept = df_to_spectra(q_df)
    lib_spectra, lib_kept = df_to_spectra(lib_df)
    print(f"{q_kept.sum()}/{len(q_df)} queries and {lib_kept.sum()}/{len(lib_df)} library spectra pass "
          f"DEFAULT_FILTERS ({time.time() - t0:.0f}s)", flush=True)

    model = Spec2Vec(Word2Vec.load(str(args.model_path)), intensity_weighting_power=0.5,
                     allowed_missing_percentage=100, progress_bar=False)
    t0 = time.time()
    scored = model.matrix(lib_spectra, q_spectra, array_type="numpy", is_symmetric=False).T  # (n_q_kept, n_lib_kept)
    print(f"scored {scored.shape} in {time.time() - t0:.0f}s", flush=True)

    sim = np.full((len(q_df), len(lib_df)), np.nan, dtype=np.float32)
    sim[np.ix_(q_kept, lib_kept)] = scored
    np.save(args.output_dir / "similarity.npy", sim)
    np.save(args.output_dir / "kept_query.npy", q_kept)
    np.save(args.output_dir / "kept_library.npy", lib_kept)
    (args.output_dir / "smiles_query.txt").write_text("\n".join(q_df["SMILES"]) + "\n")
    (args.output_dir / "smiles_library.txt").write_text("\n".join(lib_df["SMILES"]) + "\n")
    print(f"wrote {args.output_dir}/similarity.npy {sim.shape}")


if __name__ == "__main__":
    main()
