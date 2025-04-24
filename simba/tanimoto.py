from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from rdkit.Chem.inchi import MolFromInchi
import functools

# disable logging info
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")
from numba import njit

from functools import lru_cache
import numpy as np
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

# get the right size
_example = Chem.RDKFingerprint(Chem.MolFromSmiles("CC"))
FP_SIZE = _example.GetNumBits()

class Tanimoto:

    @functools.lru_cache
    def compute_tanimoto(fp1, fp2, nbits=2048, use_inchi=False):
        if (fp1 is not None) and (fp2 is not None):
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            return similarity
        else:
            return None

 
    @lru_cache(maxsize=None)
    def compute_fingerprint(smiles):
        # build or fallback to a zero‐vector bitvect
        try:
            if smiles and smiles not in ("", "N/A"):
                m = Chem.MolFromSmiles(Chem.CanonSmiles(smiles))
                if m is not None:
                    bv = Chem.RDKFingerprint(m)
                else:
                    bv = ExplicitBitVect(FP_SIZE)
            else:
                bv = ExplicitBitVect(FP_SIZE)
        except Exception:
            bv = ExplicitBitVect(FP_SIZE)

        # convert to a plain numpy array of ints
        arr = np.zeros((FP_SIZE,), dtype=int)
        ConvertToNumpyArray(bv, arr)
        return arr

    '''
    @functools.lru_cache
    def compute_fingerprint(smiles):
        if smiles != "" and smiles != "N/A":
            try:
                smiles_canon = Chem.CanonSmiles(smiles)
                mol = Chem.MolFromSmiles(smiles_canon)
                fp = Chem.RDKFingerprint(mol) if mol is not None else None
            except:  # sometimes the smiles is not correctly understood by the rdkit
                fp = None
        else:
            fp = None

        return fp
    '''
    @staticmethod
    # @functools.lru_cache
    def compute_tanimoto_from_smiles(smiles0, smiles1):
        fp0 = Tanimoto.compute_fingerprint(smiles0)
        fp1 = Tanimoto.compute_fingerprint(smiles1)
        return Tanimoto.compute_tanimoto(fp0, fp1)
