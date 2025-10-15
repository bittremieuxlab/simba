import matplotlib.pyplot as plt
import numpy as np
import pubchempy as pcp
import spectrum_utils.plot as sup
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

import simba
from simba.config import Config
from simba.simba.analog_discovery import AnalogDiscovery
from simba.simba.ground_truth import GroundTruth
from simba.simba.plotting import Plotting
from simba.simba.preprocessing_simba import PreprocessingSimba
from simba.simba.simba import Simba


def get_compound_name(smiles):
    compound = pcp.get_compounds(smiles, "smiles")  # aspirin
    try:
        return compound[0].synonyms[0]
    except:
        return compound[0].iupac_name


def formula_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    try:
        if mol is None:
            raise ValueError("Invalid SMILES")
            return "None"
        return rdMD.CalcMolFormula(mol)  # e.g. "C6H6"
    except:
        return "None"


def get_formula_subformulas(
    spec,
    use_best_subformula=True,
    USE_PERFECT_FORMULA=False,
    USE_3_FORMULAS=True,
):
    if USE_PERFECT_FORMULA:
        formula_results = formula_from_smiles(spec.params["smiles"])
        top = formula_results
    else:
        formula_results, top, results_formula = predict_formula_msms(
            mz_list=spec.mz,
            intensity_list=spec.intensity,
            precursor_mass=spec.precursor_mz,
            charge=1,
        )

    # calculate the subformulas for 3 formulas different
    keys = [k for k in results_formula if k.startswith("formula_rank_")]
    keys = keys[0:3]

    if use_best_subformula:
        subformula_results = predict_subformulas_msms(
            mz_list=spec.mz,
            intensity_list=spec.intensity,
            precursor_formula=top,
        )
    else:
        subformula_results = {}
        for k in keys:
            print(results_formula[k])
            if results_formula[k] is not None:
                subformula_results[k] = predict_subformulas_msms(
                    mz_list=spec.mz,
                    intensity_list=spec.intensity,
                    precursor_formula=results_formula[k],
                )
            else:
                if "formula_rank_1" in subformula_results:
                    subformula_results[k] = [
                        {"best_subformula": None}
                        for s in subformula_results["formula_rank_1"]
                    ]
                else:
                    subformula_results[k] = [{"best_subformula": None}]
        subformula_results = [
            [subformula_results[k][n]["best_subformula"] for k in keys]
            for n in range(0, len(subformula_results["formula_rank_1"]))
        ]

    if use_best_subformula:
        formulas_peaks = [s["best_subformula"] for s in subformula_results]
    else:
        formulas_peaks = [s for s in subformula_results]
    if USE_3_FORMULAS:
        return formula_results, formulas_peaks
    else:
        return top, formulas_peaks


from rdkit import Chem


def extract_functional_groups(smiles: str) -> dict:
    """
    Returns a dictionary of functional groups present in a SMILES string.
    Keys are group names, values are lists of atom indices where the group occurs.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    # Dictionary of functional group SMARTS patterns
    functional_groups = {
        "Alcohol": "[OX2H]",  # -OH
        "Phenol": "c[OH]",  # aromatic -OH
        "Carboxylic Acid": "[CX3](=O)[OX2H1]",  # -COOH
        "Ester": "[CX3](=O)[OX2H0][#6]",  # -COOR
        "Amide": "[NX3][CX3](=O)[#6]",  # -CONR2
        "Amine (Primary/Secondary)": "[NX3;H2,H1;!$(NC=O)]",  # NH2 or NHR
        "Ketone": "[#6][CX3](=O)[#6]",  # RCOR
        "Aldehyde": "[CX3H1](=O)[#6]",  # RCHO
        "Ether": "[OD2]([#6])[#6]",  # R-O-R
        "Thiol": "[#16X2H]",  # -SH
        "Disulfide": "[#16X2][#16X2]",  # R-S-S-R
        "Nitrile": "[CX2]#N",  # -C≡N
        "Halide": "[F,Cl,Br,I]",  # Halogens
        "Sulfonic Acid": "S(=O)(=O)[OH]",  # -SO3H
        "Nitro": "[NX3](=O)[O-]",  # -NO2
        "Aromatic Ring": "a1aaaaa1",  # benzene-like
    }

    matches = {}

    for name, smarts in functional_groups.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        hits = mol.GetSubstructMatches(patt)
        if hits:
            matches[name] = hits  # list of tuples of atom indices

    results = ""
    for name, atoms in matches.items():
        results = results + (f"{name}: {len(atoms)} match(es), ")
    return results


"""
predict_formula_msms.py
Example use of msbuddy to predict a molecular formula from one MS/MS spectrum
"""

from typing import Any, Dict, Sequence, Tuple

import numpy as np
from msbuddy import Msbuddy, MsbuddyConfig  # engine + settings
from msbuddy.base import MetaFeature, Spectrum  # in-memory objects


def predict_formula_msms(
    mz_list: Sequence[float],
    intensity_list: Sequence[float],
    precursor_mass: float,
    charge: int,
    *,
    instrument: str = "orbitrap",  # or "qtof", "fticr"
    ms1_tol_ppm: float = 5,
    ms2_tol_ppm: float = 10,
) -> Tuple[str, Dict[str, Any]]:
    """
    Run msbuddy on a single spectrum and return (best_formula, result_dict).

    ── Parameters ───────────────────────────────────────────────────────────
    mz_list, intensity_list : centroided fragment spectrum
    precursor_mass          : monoisotopic neutral mass (Da)
    charge                  : precursor charge (+1, –1, …)
    instrument              : lets msbuddy auto-tune tolerances
    ms1_tol_ppm / ms2_tol_ppm : override tolerances if instrument=None
    """
    # 1 . Pick tolerances / chemistry rules
    cfg = MsbuddyConfig(
        ms_instr=instrument,  # auto-sets ppm/Da windows
        ppm=True,  # treat ms1_tol and ms2_tol as ppm
        ms1_tol=ms1_tol_ppm,
        ms2_tol=ms2_tol_ppm,
        halogen=True,  # consider F, Cl, Br, I
    )

    engine = Msbuddy(cfg)

    # 2 . Wrap the MS/MS into msbuddy’s objects
    precursor_mz = precursor_mass / abs(charge)
    ms2_spec = Spectrum(
        np.asarray(mz_list, dtype=float),
        np.asarray(intensity_list, dtype=float),
    )

    feature = MetaFeature(
        identifier="feat-1", mz=precursor_mz, charge=charge, ms2=ms2_spec
    )

    # 3 . Annotate
    engine.add_data([feature])
    engine.annotate_formula()
    result = engine.get_summary()[0]  # only one feature → take first dict

    # concatenate results
    top = result["formula_rank_1"]
    full = result

    string_total = f"Top formula: {top}"
    # string_total = ""
    for k, v in full.items():
        string_total = string_total + f" {k:15s} {v}"

    return string_total, top, result


from typing import Any, Dict, List, Sequence

import numpy as np

# from msbuddy.utils import assign_subformula, form_arr_to_str  # core helpers
# from msbuddy.utils import assign_subformula, form_arr_to_str  # core helpers
from msbuddy import assign_subformula
from msbuddy.utils import form_arr_to_str


def predict_subformulas_msms(
    mz_list: Sequence[float],
    intensity_list: Sequence[float],
    precursor_formula: str,
    *,
    adduct: str = "[M+H]+",
    ms2_tol_ppm: float = 10,
    dbe_cutoff: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Annotate each MS² fragment with chemically valid sub-formulas.

    ── Parameters ───────────────────────────────────────────────────────────
    mz_list, intensity_list : centroided MS² spectrum (m/z and intensities)
    precursor_formula       : uncharged formula string, e.g. "C15H16O5"
    adduct                  : precursor adduct, e.g. "[M+H]+", "[M-H]-"
    ms2_tol_ppm             : fragment m/z tolerance (ppm)
    dbe_cutoff              : filter out sub-formulas with DBE < cutoff

    ── Returns ──────────────────────────────────────────────────────────────
    List[dict] with keys
        idx               : peak index in the original list
        mz_obs            : observed m/z
        intensity         : peak intensity (same order as input)
        best_subformula   : top-ranked Hill string or None
        subformula_list   : all accepted sub-formulas (Hill strings)
    """
    # ensure NumPy arrays (not strictly required, but speeds things up)
    mz_array = np.asarray(mz_list, dtype=float)
    int_array = np.asarray(intensity_list, dtype=float)

    # 1   Compute sub-formula candidates for ALL peaks in one call
    sub_results = assign_subformula(
        mz_array,
        precursor_formula=precursor_formula,
        adduct=adduct,
        ms2_tol=ms2_tol_ppm,
        ppm=True,
        dbe_cutoff=dbe_cutoff,
    )  # → list of SubformulaResult objects :contentReference[oaicite:0]{index=0}

    # 2   Re-format for downstream scripting / display
    if sub_results is not None:
        summary = []
        for res in sub_results:
            i = res.idx  # index of this peak
            sub_list = [
                (
                    sf
                    if isinstance(sf, str)  # already a string?
                    else (
                        sf.formula
                        if hasattr(sf, "formula")
                        else form_arr_to_str(sf)
                    )
                )
                for sf in res.subform_list
            ]

            # compute the neutral loss
            summary.append(
                {
                    "idx": int(i),
                    "mz_obs": float(mz_array[i]),
                    "intensity": float(int_array[i]),
                    "best_subformula": sub_list[0] if sub_list else None,
                    "subformula_list": sub_list,
                }
            )
    else:
        summary = []

        for i in range(0, len(mz_array)):
            summary.append(
                {
                    "idx": int(i),
                    "mz_obs": float(mz_array[i]),
                    "intensity": float(int_array[i]),
                    "best_subformula": None,
                    "subformula_list": None,
                }
            )

    return summary


from rdkit import Chem
from rdkit.Chem import Crippen, Lipinski
from rdkit.Chem import rdMolDescriptors as rdmd

HALOGENS = {"F", "Cl", "Br", "I"}


def core_struct_features_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "N_count": None,
            "O_count": None,
            "S_count": None,
            "P_count": None,
            "Halogen_count": None,
            "NumRings": None,
            "NumAromaticRings": None,
            "FractionCSP3": None,
            "TPSA": None,
            "RotatableBonds": None,
            "ChiralCenters": None,
            "FormalCharge": None,
        }
    else:
        # ensure explicit Hs are not needed
        mol = Chem.AddHs(
            mol, addCoords=False
        )  # keeps formula/charge consistent

        # element counts
        elem_counts = {"N": 0, "O": 0, "S": 0, "P": 0, "Hal": 0}
        for a in mol.GetAtoms():
            sym = a.GetSymbol()
            if sym in elem_counts:
                elem_counts[sym] += 1
            if sym in HALOGENS:
                elem_counts["Hal"] += 1

        # rings & aromaticity
        num_rings = rdmd.CalcNumRings(mol)
        num_arom_rings = rdmd.CalcNumAromaticRings(mol)

        # topology / polarity / flexibility
        frac_csp3 = rdmd.CalcFractionCSP3(mol)
        tpsa = rdmd.CalcTPSA(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)

        # stereochemistry
        chiral_centers = len(
            Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        )

        # charge
        formal_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())

        return {
            "N_count": elem_counts["N"],
            "O_count": elem_counts["O"],
            "S_count": elem_counts["S"],
            "P_count": elem_counts["P"],
            "Halogen_count": elem_counts["Hal"],
            "NumRings": int(num_rings),
            "NumAromaticRings": int(num_arom_rings),
            "FractionCSP3": float(frac_csp3),
            "TPSA": float(tpsa),
            "RotatableBonds": int(rot_bonds),
            "ChiralCenters": int(chiral_centers),
            "FormalCharge": int(formal_charge),
        }
