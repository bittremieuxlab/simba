"""Theoretical precursor m/z and MIST-CF/BUDDY-style precursor m/z noise, as in
the simba-triplet branch (simba/core/chemistry/chem_utils.py there):

- theoretical: precursor m/z recomputed from the molecule (RDKit ExactMolWt of
  the SMILES) and the adduct, instead of the value stored with the spectrum;
- noise: Gaussian with std = instrument-specific ppm tolerance / 5.
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt

RDLogger.DisableLog("rdApp.*")

# singly charged monomer adducts (same values as simba-triplet's ADDUCT_TO_MASS)
ADDUCT_OFFSET = {
    "[M+H]+": 1.007276,
    "[M+NH4]+": 18.033823,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M-H]-": -1.007276,
}
# NIST spectra carry the adduct only as the last line of their name, e.g. "Asn-Ser\n M+H"
NIST_NAME_ADDUCT = {"M+H": "[M+H]+", "M+NH4": "[M+NH4]+", "M+Na": "[M+Na]+", "M+K": "[M+K]+", "M-H": "[M-H]-"}

# per-instrument-type ppm tolerance (BUDDY's table)
INSTRUMENT_PPM_TOLERANCE = {"orbitrap": 5, "qtof": 10, "iontrap": 15, "fticr": 5, "unknown": 15}
_RAW_INSTRUMENT_TO_TYPE = {
    "orbitrap": "orbitrap", "qtof": "qtof", "q-tof": "qtof", "q tof": "qtof",
    "iontrap": "iontrap", "ion trap": "iontrap", "fticr": "fticr", "ft-icr": "fticr",
}


def spectrum_adduct(spectrum):
    params = spectrum.params or {}
    if params.get("adduct"):
        return params["adduct"]
    name = params.get("name")
    if name:
        return NIST_NAME_ADDUCT.get(str(name).split("\n")[-1].strip())
    return None


def instrument_ppm_tolerance(spectrum):
    raw = (spectrum.params or {}).get("instrument_type")
    kind = _RAW_INSTRUMENT_TO_TYPE.get(str(raw).strip().lower(), "unknown") if raw else "unknown"
    return INSTRUMENT_PPM_TOLERANCE[kind]


def theoretical_precursor_mz(spectrum, cache):
    """Theoretical m/z, or None when the SMILES or the adduct is not usable."""
    offset = ADDUCT_OFFSET.get(spectrum_adduct(spectrum))
    smiles = getattr(spectrum, "smiles", None) or (spectrum.params or {}).get("smiles")
    if offset is None or not smiles:
        return None
    if smiles not in cache:
        mol = Chem.MolFromSmiles(smiles)
        cache[smiles] = ExactMolWt(mol) if mol is not None else None
    return None if cache[smiles] is None else cache[smiles] + offset


def resample_precursor_mz(precursor_mz, tolerance_ppm):
    return precursor_mz + np.random.normal(0, precursor_mz * tolerance_ppm / 5 / 1e6)
