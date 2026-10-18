"""Export CASMI queries + full (non-deduplicated) library spectra for both
searches into a tool-agnostic DataFrame schema, so the three baseline tools
(DreaMS, MS2DeepScore, Spec2Vec) can each convert it into their own spectrum
object format without depending on the `simba` package or its pickle shims.

Schema matches DreaMS's own existing simba_finetune/01_extract_simba_spectra.py
convention exactly (so DreaMS's MSData.load() can read these files directly):
    SMILES, PRECURSOR M/Z, CHARGE, PARSED PEAKS (2, n_peaks np.array: mz, intensity)
Plus one extra column, IDENTIFIER, a stable per-row index string, since unlike
SIMBA's own training data these rows are NOT deduplicated by molecule --
multiple spectra can share the same SMILES, and downstream analysis needs to
map each row back to its molecule (via SMILES) and recover ranking position.

Usage:
    python tools/export_spectra_for_baselines.py --output_dir /mnt/data2/nkubrakov/analog_discovery/baseline_spectra
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.analog_discovery_score_simba import load_spectra as load_mgf_spectra
from tools.analog_discovery_prepare import canon

MIN_N_PEAKS = 6


def spectra_to_df(smiles_list, spectra) -> pd.DataFrame:
    records = []
    for idx, (smi, s) in enumerate(zip(smiles_list, spectra)):
        mz = np.asarray(s.mz, dtype=np.float64)
        intensity = np.asarray(s.intensity, dtype=np.float64)
        if mz.size == 0:
            continue
        records.append({
            "IDENTIFIER": str(idx),
            "SMILES": smi,
            "PRECURSOR M/Z": float(s.precursor_mz),
            "CHARGE": int(s.precursor_charge) if s.precursor_charge else 1,
            "PARSED PEAKS": np.stack([mz, intensity]),
        })
    return pd.DataFrame.from_records(records)


def load_library_a_all(training_pkl):
    import simba
    sys.modules["src"] = simba
    import dill
    with open(training_pkl, "rb") as f:
        dataset = dill.load(f)
    exact_members = set()
    for split in ("molecule_pairs_train", "molecule_pairs_val", "molecule_pairs_test"):
        for s in dataset[split].spectrums:
            c = canon(s.smiles)
            if c:
                exact_members.add(c)
    lib_smiles_all, lib_spectra_all = [], []
    for split in ("molecule_pairs_train", "molecule_pairs_val", "molecule_pairs_test"):
        for s in dataset[split].spectrums_original:
            if len(s.mz) < MIN_N_PEAKS:
                continue
            c = canon(s.smiles)
            if c is None or c not in exact_members:
                continue
            lib_smiles_all.append(c)
            lib_spectra_all.append(s)
    return lib_smiles_all, lib_spectra_all


def load_library_b_all(gnps_mgf):
    from simba.spectrum_ext import SpectrumExt
    lib_smiles_all, lib_spectra_all = [], []
    cur, n_seen = None, 0
    with open(gnps_mgf, errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                cur = {"_mz": [], "_intensity": []}
                continue
            if line == "END IONS":
                if cur is not None:
                    n_seen += 1
                    smi = canon(cur.get("SMILES"))
                    ionmode = (cur.get("IONMODE") or "").strip().lower()
                    try:
                        precursor_mz = float(cur.get("PEPMASS", 0) or 0)
                    except ValueError:
                        precursor_mz = 0.0
                    if (smi and len(cur["_mz"]) >= MIN_N_PEAKS
                            and ionmode in ("", "positive") and precursor_mz > 0):
                        m = max(cur["_intensity"]) if cur["_intensity"] else 0.0
                        inten = [i / m for i in cur["_intensity"]] if m > 0 else cur["_intensity"]
                        lib_smiles_all.append(smi)
                        lib_spectra_all.append(SpectrumExt(
                            identifier=str(n_seen), precursor_mz=precursor_mz, precursor_charge=1,
                            mz=np.asarray(cur["_mz"], dtype=np.float64), intensity=np.asarray(inten, dtype=np.float64),
                            retention_time=0.0, params={"smiles": smi}, library=None, inchi=None, smiles=smi,
                            ionmode="positive", bms=None, superclass=None, classe=None, subclass=None,
                        ))
                cur = None
                continue
            if cur is None:
                continue
            if "=" in line and not line[0].isdigit():
                key, _, val = line.partition("=")
                cur[key.strip().upper()] = val.strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz, inten = float(parts[0]), float(parts[1])
                    except ValueError:
                        continue
                    cur["_mz"].append(mz)
                    cur["_intensity"].append(inten)
    return lib_smiles_all, lib_spectra_all


def run(output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Search A query ...")
    q_smiles_a, q_spectra_a = load_mgf_spectra("/mnt/data2/nkubrakov/analog_discovery/search_A_nist_msg.mgf", "query")
    spectra_to_df(q_smiles_a, q_spectra_a).to_pickle(out_dir / "search_A_query.pkl")
    print(f"  {len(q_spectra_a)} query spectra")

    print("Search A library (no dedup) ...")
    lib_smiles_a, lib_spectra_a = load_library_a_all(
        "/mnt/data2/nkubrakov/preprocessing_ed_mces_20250123/edit_distance_neurips_nist_exhaustive.pkl")
    spectra_to_df(lib_smiles_a, lib_spectra_a).to_pickle(out_dir / "search_A_library.pkl")
    print(f"  {len(lib_spectra_a)} library spectra")

    print("Search B query ...")
    q_smiles_b, q_spectra_b = load_mgf_spectra("/mnt/data2/nkubrakov/analog_discovery/search_B_gnps.mgf", "query")
    spectra_to_df(q_smiles_b, q_spectra_b).to_pickle(out_dir / "search_B_query.pkl")
    print(f"  {len(q_spectra_b)} query spectra")

    print("Search B library (no dedup, GNPS) ...")
    lib_smiles_b, lib_spectra_b = load_library_b_all("/home/nkubrakov/simba-paper/data/gnps/ALL_GNPS_NO_PROPOGATED.mgf")
    spectra_to_df(lib_smiles_b, lib_spectra_b).to_pickle(out_dir / "search_B_library.pkl")
    print(f"  {len(lib_spectra_b)} library spectra")

    print(f"\nWrote all 4 files to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    run(args.output_dir)
