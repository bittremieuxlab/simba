"""Unit tests for Edit Distance computations."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols

from simba.edit_distance.edit_distance import (
    VERY_HIGH_DISTANCE,
    get_edit_distance_from_smiles,
    simba_get_edit_distance,
    simba_solve_pair_edit_distance,
    simba_solve_pair_mces,
)


pytestmark = pytest.mark.unit


class TestEditDistanceBasic:
    """Test basic edit distance computations."""

    def test_get_edit_distance_from_smiles_identical(self):
        """Test edit distance for identical molecules returns 0."""
        smiles = "CCO"  # Ethanol
        distance = get_edit_distance_from_smiles(smiles, smiles, return_nans=False)

        assert distance == 0

    def test_get_edit_distance_from_smiles_similar(self):
        """Test edit distance for similar molecules."""
        smiles1 = "CCO"  # Ethanol
        smiles2 = "CCCO"  # Propanol
        distance = get_edit_distance_from_smiles(smiles1, smiles2, return_nans=False)

        assert distance == 1

    def test_get_edit_distance_from_smiles_different(self):
        """Test edit distance for very different molecules returns NaN."""
        smiles1 = "C"  # Methane (small)
        smiles2 = "c1ccc2c(c1)ccc1ccccc12"  # Anthracene (large aromatic)
        distance = get_edit_distance_from_smiles(smiles1, smiles2, return_nans=True)

        # Very different molecules with return_nans=True should return NaN
        assert np.isnan(distance)

    def test_get_edit_distance_from_smiles_different_no_nans(self):
        """Test edit distance for very different molecules with return_nans=False."""
        smiles1 = "C"  # Methane (small)
        smiles2 = "c1ccc2c(c1)ccc1ccccc12"  # Anthracene (large aromatic)
        distance = get_edit_distance_from_smiles(smiles1, smiles2, return_nans=False)

        # With return_nans=False, should return concrete distance value of 2
        assert distance == 2


class TestSimbaGetEditDistance:
    """Test simba_get_edit_distance function."""

    def test_identical_molecules(self):
        """Test edit distance for identical molecules."""
        mol = Chem.MolFromSmiles("CCO")
        distance = simba_get_edit_distance(mol, mol, return_nans=False)

        assert distance == 0

    def test_similar_molecules(self):
        """Test edit distance for similar molecules."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCCO")
        distance = simba_get_edit_distance(mol1, mol2, return_nans=False)

        assert distance == 1

    def test_return_nans_for_dissimilar(self):
        """Test that function returns NaN for very dissimilar molecules."""
        mol1 = Chem.MolFromSmiles("C")  # Methane
        mol2 = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # Naphthalene (large)
        distance = simba_get_edit_distance(mol1, mol2, return_nans=True)

        assert np.isnan(distance)


class TestSimbaSolvePairEditDistance:
    """Test simba_solve_pair_edit_distance function."""

    def test_identical_molecules_pair(self):
        """Test pair solving for identical molecules."""
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        fp = FingerprintMols.FingerprintMol(mol)

        distance, tanimoto = simba_solve_pair_edit_distance(
            smiles, smiles, fp, fp, mol, mol
        )

        assert distance == 0
        assert tanimoto == 1.0

    def test_similar_molecules_pair(self):
        """Test pair solving for similar molecules."""
        smiles1 = "CCO"
        smiles2 = "CCCO"
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)

        distance, tanimoto = simba_solve_pair_edit_distance(
            smiles1, smiles2, fp1, fp2, mol1, mol2
        )

        assert distance == 1
        assert tanimoto == pytest.approx(0.6, abs=0.01)

    def test_very_dissimilar_molecules_pair(self):
        """Test pair solving returns VERY_HIGH_DISTANCE for dissimilar molecules."""
        smiles1 = "C"  # Methane
        smiles2 = "CCCCCCCCCCCCCCCCCCCC"  # C20 chain (Tanimoto = 0.0)
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)

        distance, tanimoto = simba_solve_pair_edit_distance(
            smiles1, smiles2, fp1, fp2, mol1, mol2
        )

        assert distance == VERY_HIGH_DISTANCE  # 666
        assert tanimoto == pytest.approx(0.0, abs=0.01)

    def test_large_molecules_return_nan(self):
        """Test that large molecules (>60 atoms) return NaN when Tanimoto >= 0.2."""
        # Create large similar molecules (>60 atoms, high Tanimoto = 1.0)
        smiles1 = "C" * 65  # Linear chain with 65 carbons
        smiles2 = "C" * 70  # Linear chain with 70 carbons
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)

        distance, tanimoto = simba_solve_pair_edit_distance(
            smiles1, smiles2, fp1, fp2, mol1, mol2
        )

        # Large molecules (>60 atoms) with high Tanimoto should return NaN
        assert np.isnan(distance)
        assert tanimoto == pytest.approx(1.0, abs=0.01)


class TestSimbaSolvePairMCES:
    """Test simba_solve_pair_mces function."""

    def test_identical_molecules_mces(self):
        """Test MCES for identical molecules returns 0."""
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        fp = FingerprintMols.FingerprintMol(mol)

        distance, tanimoto = simba_solve_pair_mces(
            smiles, smiles, fp, fp, mol, mol, threshold=20, TIME_LIMIT=2
        )

        assert distance == 0
        assert tanimoto == 1.0

    def test_similar_molecules_mces(self):
        """Test MCES for similar molecules."""
        smiles1 = "CCO"
        smiles2 = "CCCO"
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)

        distance, tanimoto = simba_solve_pair_mces(
            smiles1, smiles2, fp1, fp2, mol1, mol2, threshold=20, TIME_LIMIT=2
        )

        assert distance == 1.0
        assert tanimoto == pytest.approx(0.6, abs=0.01)

    def test_very_dissimilar_molecules_mces(self):
        """Test MCES returns VERY_HIGH_DISTANCE for dissimilar molecules."""
        smiles1 = "C"
        smiles2 = "CCCCCCCCCCCCCCCCCCCC"
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)

        distance, tanimoto = simba_solve_pair_mces(
            smiles1, smiles2, fp1, fp2, mol1, mol2, threshold=20, TIME_LIMIT=2
        )

        assert distance == VERY_HIGH_DISTANCE  # 666
        assert tanimoto == pytest.approx(0.0, abs=0.01)

    def test_mces_threshold_parameter(self):
        """Test MCES with different threshold values."""
        smiles1 = "CCO"
        smiles2 = "CCCO"
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)

        distance1, _ = simba_solve_pair_mces(
            smiles1, smiles2, fp1, fp2, mol1, mol2, threshold=10, TIME_LIMIT=2
        )
        distance2, _ = simba_solve_pair_mces(
            smiles1, smiles2, fp1, fp2, mol1, mol2, threshold=30, TIME_LIMIT=2
        )

        # Both should return valid distances
        assert isinstance(distance1, (int, float, np.number))
        assert isinstance(distance2, (int, float, np.number))
