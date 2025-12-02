from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric

from simba.logger_setup import logger


class MurckoScaffold:
    """
    code for computing murcko scaffold for dividing train, val and test sets
    """

    def get_bm_scaffold(smiles):
        try:
            scaffold = Chem.MolToSmiles(
                MakeScaffoldGeneric(mol=Chem.MolFromSmiles(smiles))
            )
        except Exception:
            logger.warning(f"No scaffold for given SMILES ({smiles})")
            scaffold = ""
        return scaffold
