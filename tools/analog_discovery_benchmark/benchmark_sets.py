"""Query and library sets of the SIMBA paper's analog-discovery benchmark.

Port of the loading/filtering cell of Sebastian's
notebooks/precision_candidates.ipynb (branch simba_v1). The order of the
returned lists is the row/column order of his ground-truth matrices
(verified: Search A 132 x 328,555 spectra, Search B 148 x 53,618 spectra).

  queries   one query set, minus the queries whose molecule is in that
            search's library (exclusion is per search):
              casmi   CASMI 2022 spectra (LoadData.get_all_spectrums_casmi),
                      >= 6 peaks after preprocessing (his precision notebook)
              msnlib  MSnLib ENAMDISC spectra (PreprocessingSimba.load_spectra,
                      as in his analog notebook), >= 6 peaks
  library A train + val + test spectrums_original of the SIMBA training
            dataset pickle (NIST20 + MassSpecGym), >= 6 peaks.
  library B GNPS spectra loaded with PreprocessingSimba.load_spectra(
            use_gnps_format=True) (his GNPS validity filter), parsable SMILES,
            >= 6 peaks.
The ORIGINAL spectra are returned; preprocessing is only used to filter.
"""

import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dill  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

import simba  # noqa: E402

sys.modules["src"] = simba  # the training dataset pickle refers to the old package name
from simba.config import Config  # noqa: E402
from simba.load_data import LoadData  # noqa: E402
from simba.preprocessor import Preprocessor  # noqa: E402
from simba.simba.preprocessing_simba import PreprocessingSimba  # noqa: E402

RDLogger.DisableLog("rdApp.*")
SEARCHES = ("A", "B")
QUERY_SETS = ("casmi", "msnlib")
# (queries, library spectra) per search; casmi = the shapes of his ground-truth matrices
EXPECTED_SIZES = {"casmi": {"A": (132, 328555), "B": (148, 53618)},
                  "msnlib": {"A": (490, 328555), "B": (491, 53618)}}


def filter_spectra(spectra, pp):
    """His filter_spectra: preprocess a copy, keep the ORIGINAL if >= 6 peaks remain."""
    kept = []
    for original in spectra:
        processed = pp.preprocess_spectrum(
            copy.deepcopy(original),
            fragment_tol_mass=10,
            fragment_tol_mode="ppm",
            min_intensity=0.01,
            max_num_peaks=100,
            scale_intensity=None,
        )
        if len(processed.mz) >= 6:
            kept.append(original)
    return kept


class SetLoader:
    def __init__(self, dataset_pkl, casmi_pkl, gnps_mgf, queries="casmi", msnlib_mgf=None):
        if queries not in QUERY_SETS or (queries == "msnlib" and not msnlib_mgf):
            raise SystemExit(f"queries must be one of {QUERY_SETS}; msnlib needs msnlib_mgf")
        self.dataset_pkl, self.casmi_pkl, self.gnps_mgf = str(dataset_pkl), str(casmi_pkl), str(gnps_mgf)
        self.query_set, self.msnlib_mgf = queries, msnlib_mgf and str(msnlib_mgf)
        self.config, self.pp = Config(), Preprocessor()
        self._queries = None

    def candidate_queries(self):
        if self._queries is None:
            if self.query_set == "casmi":
                spectra = LoadData.get_all_spectrums_casmi(self.casmi_pkl, config=self.config)
            else:
                spectra = PreprocessingSimba.load_spectra(self.msnlib_mgf, self.config)
            self._queries = filter_spectra(spectra, self.pp)
        return self._queries

    def all_query_smiles(self):
        """Canonical SMILES of every query molecule in the input file, before any
        filtering (to compute ground truth for spectra the loader drops too)."""
        if self.query_set == "casmi":
            return list(dict.fromkeys(canonical(LoadData.get_all_spectrums_casmi(self.casmi_pkl, config=self.config))))
        smiles = [line.strip().split("=", 1)[1] for line in open(self.msnlib_mgf) if line.startswith("SMILES=")]
        return list(dict.fromkeys(Chem.CanonSmiles(s) for s in smiles))

    def library(self, search):
        if search == "A":
            with open(self.dataset_pkl, "rb") as fh:
                dataset = dill.load(fh)
            spectra = (
                dataset["molecule_pairs_train"].spectrums_original
                + dataset["molecule_pairs_val"].spectrums_original
                + dataset["molecule_pairs_test"].spectrums_original
            )
        else:
            spectra = PreprocessingSimba.load_spectra(self.gnps_mgf, self.config, use_gnps_format=True)
            spectra = [s for s in spectra if Chem.MolFromSmiles(s.smiles) is not None]
        return filter_spectra(spectra, self.pp)

    def load(self, search):
        """-> (queries, library) spectrum lists of one search, in his order."""
        library = self.library(search)
        library_molecules = {Chem.CanonSmiles(s.smiles) for s in library}
        queries = [s for s in self.candidate_queries() if Chem.CanonSmiles(s.smiles) not in library_molecules]
        expected = EXPECTED_SIZES[self.query_set][search]
        if (len(queries), len(library)) != expected:
            raise SystemExit(f"Search {search} ({self.query_set}): {len(queries)} queries x {len(library)} spectra, "
                             f"expected {expected} -- different input files?")
        return queries, library


def canonical(spectra):
    return [Chem.CanonSmiles(s.smiles) for s in spectra]


def check_order(sets_dir, search, queries, library):
    """Assert that the lists are in the order written by 01_build_sets.py."""
    d = Path(sets_dir) / f"search_{search}"
    if canonical(queries) != d.joinpath("queries.txt").read_text().split():
        raise SystemExit(f"Search {search}: query order differs from {d}/queries.txt")
    if canonical(library) != d.joinpath("reference_spectra_smiles.txt").read_text().split():
        raise SystemExit(f"Search {search}: library order differs from {d}/reference_spectra_smiles.txt")
