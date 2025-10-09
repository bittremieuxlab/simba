import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from myopic_mces.myopic_mces import MCES as MCES2
from rdkit import Chem, DataStructs, Geometry
from rdkit.Chem import AllChem, Draw, PandasTools, rdFMCS
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from tqdm import tqdm

import simba.edit_distance.mol_utils as mu
from simba.config import Config
from simba.logger_setup import logger


class EditDistance:

    @staticmethod
    def create_input_df(smiles, indexes_0, indexes_1):
        df = pd.DataFrame()
        logger.info(f"Number of spectra: {len(smiles)}")

        df["smiles_0"] = [smiles[int(r)] for r in indexes_0]
        df["smiles_1"] = [smiles[int(r)] for r in indexes_1]

        return df

    def compute_ed_or_mces(
        smiles: List[str],
        sampled_index: np.int64,
        batch_size: int,
        identifier: int,
        random_sampling: bool,
        config: Config,
        fps: List[ExplicitBitVect],
        mols: List[Mol],
        use_edit_distance: bool,
    ) -> np.ndarray:
        """
        Compute the edit distance or MCES for a batch of molecule pairs.

        Parameters
        ----------
        smiles : List[str]
            List of SMILES strings.
        sampled_index : np.int64
            Index to sample from the smiles list.
        batch_size : int
            The size of the batch to process.
        identifier : int
            An identifier for the batch (used for random seed).
        random_sampling : bool
            Whether to use random sampling of pairs.
        config : Config
            Configuration object containing parameters.
        fps : List[ExplicitBitVect]
            List of fingerprints corresponding to the smiles.
        mols : List[Mol]
            List of RDKit Mol objects corresponding to the smiles.
        use_edit_distance : bool
            Whether to compute edit distance (True) or MCES (False).

        Returns
        -------
        np.ndarray
            A 2D numpy array with each row containing (index1, index2, distance).
        """
        # 2D array to store the indexes and distances
        pair_distances = np.zeros(
            (int(batch_size), 3),
        )
        # initialize randomness
        if random_sampling:
            np.random.seed(identifier)
            pair_distances[:, 0] = np.random.randint(
                0, len(smiles), int(batch_size)
            )
            pair_distances[:, 1] = np.random.randint(
                0, len(smiles), int(batch_size)
            )
        else:
            pair_distances[:, 0] = sampled_index
            pair_distances[:, 1] = np.arange(0, batch_size)

        distances = []

        for index in range(0, pair_distances.shape[0]):
            pair = pair_distances[index]

            s0 = smiles[int(pair[0])]
            s1 = smiles[int(pair[1])]
            fp0 = fps[int(pair[0])]
            fp1 = fps[int(pair[1])]
            mol0 = mols[int(pair[0])]
            mol1 = mols[int(pair[1])]
            if use_edit_distance:
                dist, _ = EditDistance.simba_solve_pair_edit_distance(
                    s0, s1, fp0, fp1, mol0, mol1
                )
            else:
                dist, _ = EditDistance.simba_solve_pair_mces(
                    s0, s1, fp0, fp1, mol0, mol1, config.THRESHOLD_MCES
                )
            distances.append(dist)

        pair_distances[:, 2] = distances
        return pair_distances

    def get_number_of_modification_edges(mol, substructure):
        if not mol.HasSubstructMatch(substructure):
            #    raise ValueError("The substructure is not a substructure of the molecule.")
            return None
        matches = mol.GetSubstructMatch(substructure)
        intersect = set(matches)
        modification_edges = []
        for bond in mol.GetBonds():
            if (
                bond.GetBeginAtomIdx() in intersect
                and bond.GetEndAtomIdx() in intersect
            ):
                continue
            if (
                bond.GetBeginAtomIdx() in intersect
                or bond.GetEndAtomIdx() in intersect
            ):
                modification_edges.append(bond.GetIdx())

        return modification_edges

    def get_edit_distance_from_smiles(smiles1, smiles2, return_nans=True):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        return EditDistance.simba_get_edit_distance(
            mol1, mol2, return_nans=return_nans
        )

    def simba_get_edit_distance(mol1, mol2, return_nans=True):
        """
        Calculates the edit distance between mol1 and mol2.
        Input:
            mol1: first molecule
            mol2: second molecule
        Output:
            edit_distance: edit distance between mol1 and mol2
        """
        # if mol1.GetNumAtoms() > 60 or mol2.GetNumAtoms() > 60:
        #    print("The molecules are too large.")
        #    return np.nan
        mcs1 = rdFMCS.FindMCS([mol1, mol2])
        mcs_mol = Chem.MolFromSmarts(mcs1.smartsString)
        if return_nans:
            if (
                mcs_mol.GetNumAtoms() < mol1.GetNumAtoms() // 2
                and mcs_mol.GetNumAtoms() < mol2.GetNumAtoms() // 2
            ):
                # print("The molecules are too small.")
                return np.nan
            if mcs_mol.GetNumAtoms() < 2:

                # print("The molecules are too small.")
                return np.nan

        # dist1 = EditDistance.get_number_of_modification_edges(mol1, mcs_mol)
        # dist2 =  EditDistance.get_number_of_modification_edges(mol2, mcs_mol)

        # if (dist1 is not None) and (dist2 is not None):
        #    return len(dist1) + len(dist2)
        # else:
        #    return np.nan
        # print("going to calculate edit distance")
        dist1, dist2 = mu.get_edit_distance_detailed(mol1, mol2, mcs_mol)

        distance = dist1 + dist2

        return distance

    from functools import lru_cache

    @lru_cache(
        maxsize=1000
    )  # Set maxsize to None for an unbounded cache or a specific integer for a bounded cache
    def return_mol(smiles):
        # Simulate some processing
        return Chem.MolFromSmiles(smiles)

    def simba_solve_pair_edit_distance(s0, s1, fp0, fp1, mol0, mol1):

        tanimoto = DataStructs.TanimotoSimilarity(fp0, fp1)

        if tanimoto < 0.2:
            # print("The Tanimoto similarity is too low.")
            # distance = np.nan
            distance = (
                666  # very high number to remark that they are very different
            )
        else:
            # mol0 = EditDistance.return_mol(s0)
            # mol1 = EditDistance.return_mol(s1)
            if (mol0.GetNumAtoms() > 60) or (mol1.GetNumAtoms() > 60):
                # raise ValueError("The molecules are too large.")
                return np.nan, tanimoto
            else:
                distance = EditDistance.simba_get_edit_distance(mol0, mol1)

        return distance, tanimoto

    def simba_solve_pair_mces(
        s0,
        s1,
        fp0,
        fp1,
        mol0,
        mol1,
        threshold,
        TIME_LIMIT=2,  # 2 seconds
    ):

        tanimoto = DataStructs.TanimotoSimilarity(fp0, fp1)

        if tanimoto < 0.2:
            # print("The Tanimoto similarity is too low.")
            # distance = np.nan
            distance = (
                666  # very high number to remark that they are very different
            )
        else:
            # mol0 = EditDistance.return_mol(s0)
            # mol1 = EditDistance.return_mol(s1)
            # distance = MCES(s0, s1, threshold=threshold)
            if (mol0.GetNumAtoms() > 60) or (mol1.GetNumAtoms() > 60):
                # raise ValuseError("The molecules are too large.")
                return np.nan, tanimoto
            else:
                result = MCES2(
                    s0,
                    s1,
                    threshold=threshold,
                    i=0,
                    # solver='CPLEX_CMD',       # or another fast solver you have installed
                    solver="PULP_CBC_CMD",
                    solver_options={
                        "threads": 1,
                        "msg": False,
                        "timeLimit": TIME_LIMIT,  # Stop CBC after 1 seconds
                    },
                    # solver_options={'threads': 1, 'msg': False},  # use single thread + no console messages
                    no_ilp_threshold=False,  # allow the ILP to stop early once the threshold is exceeded
                    always_stronger_bound=False,  # use dynamic bounding for speed
                    catch_errors=False,  # typically raise exceptions if something goes wrong
                )
                distance = result[1]
                time_taken = result[2]
                exact_answer = result[3]

                if time_taken >= (0.9 * TIME_LIMIT) and (exact_answer != 1):
                    distance = np.nan

        return distance, tanimoto

    def get_data(data, index, batch_count):
        batch_size = len(data) // batch_count
        if index < len(data) % batch_count:
            batch_size += 1
            start = index * batch_size
            end = start + batch_size
        else:
            start = index * batch_size + len(data) % batch_count
            end = start + batch_size

        if end > len(data):
            end = len(data)

        res = data.iloc[start:end]
        # reset index
        res = res.reset_index(drop=True)
        return res
