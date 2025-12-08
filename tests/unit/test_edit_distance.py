"""Unit tests for Edit Distance computations."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols

from simba.edit_distance.edit_distance import (
    VERY_HIGH_DISTANCE,
    create_input_df,
    compute_ed_or_mces,
    get_number_of_modification_edges,
    get_data,
    return_mol,
    get_edit_distance_from_smiles,
    simba_get_edit_distance,
    simba_solve_pair_edit_distance,
    simba_solve_pair_mces,
)
from simba.config import Config


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


class TestCreateInputDF:
    """Test create_input_df function."""

    def test_create_input_df_basic(self):
        """Test creating input DataFrame with basic inputs."""
        smiles = ["CCO", "CCCO", "C", "CC"]
        indexes_0 = [0, 1, 2]
        indexes_1 = [1, 2, 3]

        df = create_input_df(smiles, indexes_0, indexes_1)

        assert len(df) == 3
        assert "smiles_0" in df.columns
        assert "smiles_1" in df.columns
        assert df["smiles_0"].tolist() == ["CCO", "CCCO", "C"]
        assert df["smiles_1"].tolist() == ["CCCO", "C", "CC"]


class TestGetNumberOfModificationEdges:
    """Test get_number_of_modification_edges function."""

    def test_get_modification_edges_exact_match(self):
        """Test with exact match (no modification edges)."""
        mol = Chem.MolFromSmiles("CCO")
        substructure = Chem.MolFromSmiles("CCO")

        result = get_number_of_modification_edges(mol, substructure)

        assert result == []

    def test_get_modification_edges_with_additions(self):
        """Test with substructure that has additional bonds."""
        mol = Chem.MolFromSmiles("CCCO")  # Propanol
        substructure = Chem.MolFromSmiles("CCO")  # Ethanol (substructure)

        result = get_number_of_modification_edges(mol, substructure)

        # Should have modification edges
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_modification_edges_no_match(self):
        """Test with non-matching substructure."""
        mol = Chem.MolFromSmiles("CCO")
        substructure = Chem.MolFromSmiles("c1ccccc1")  # Benzene

        result = get_number_of_modification_edges(mol, substructure)

        assert result is None


class TestReturnMol:
    """Test return_mol caching function."""

    def test_return_mol_valid_smiles(self):
        """Test return_mol with valid SMILES."""
        smiles = "CCO"
        mol = return_mol(smiles)

        assert mol is not None
        assert mol.GetNumAtoms() == 3

    def test_return_mol_caching(self):
        """Test that return_mol uses caching."""
        smiles = "CCCO"

        mol1 = return_mol(smiles)
        mol2 = return_mol(smiles)

        # Should return same cached object
        assert mol1 is mol2


class TestGetData:
    """Test get_data batch splitting function."""

    def test_get_data_even_split(self):
        """Test splitting data evenly into batches."""
        import pandas as pd

        data = pd.DataFrame({"smiles": ["CCO", "CCCO", "C", "CC", "CCC", "CCCC"]})
        batch_count = 3

        batch0 = get_data(data, 0, batch_count)
        batch1 = get_data(data, 1, batch_count)
        batch2 = get_data(data, 2, batch_count)

        assert len(batch0) == 2
        assert len(batch1) == 2
        assert len(batch2) == 2
        assert batch0.index.tolist() == [0, 1]
        assert batch1.index.tolist() == [0, 1]
        assert batch2.index.tolist() == [0, 1]

    def test_get_data_uneven_split(self):
        """Test splitting data unevenly into batches."""
        import pandas as pd

        data = pd.DataFrame({"smiles": ["CCO", "CCCO", "C", "CC", "CCC"]})
        batch_count = 3

        batch0 = get_data(data, 0, batch_count)
        batch1 = get_data(data, 1, batch_count)
        batch2 = get_data(data, 2, batch_count)

        # First two batches get 2 items, last gets 1
        assert len(batch0) == 2
        assert len(batch1) == 2
        assert len(batch2) == 1


class TestComputeEdOrMces:
    """Test compute_ed_or_mces batch computation function."""

    def test_compute_ed_non_random_sampling(self):
        """Test compute_ed_or_mces with non-random sampling."""
        smiles = ["CCO", "CCCO", "C", "CC"]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        fps = [FingerprintMols.FingerprintMol(m) for m in mols]
        config = Config()

        result = compute_ed_or_mces(
            smiles=smiles,
            sampled_index=0,
            batch_size=3,
            identifier=0,
            random_sampling=False,
            config=config,
            fps=fps,
            mols=mols,
            use_edit_distance=True,
        )

        assert result.shape == (3, 3)
        assert result[0, 0] == 0  # First index
        assert result[0, 1] == 0  # Second index (starts from 0)
        assert result[0, 2] == 0  # Distance (CCO vs CCO = 0)

    def test_compute_ed_random_sampling(self):
        """Test compute_ed_or_mces with random sampling."""
        smiles = ["CCO", "CCCO", "C", "CC"]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        fps = [FingerprintMols.FingerprintMol(m) for m in mols]
        config = Config()

        result = compute_ed_or_mces(
            smiles=smiles,
            sampled_index=0,
            batch_size=2,
            identifier=42,
            random_sampling=True,
            config=config,
            fps=fps,
            mols=mols,
            use_edit_distance=True,
        )

        assert result.shape == (2, 3)
        # With random sampling, indices should be random
        assert 0 <= result[0, 0] < len(smiles)
        assert 0 <= result[0, 1] < len(smiles)

    def test_compute_mces_non_random(self):
        """Test compute_ed_or_mces with MCES computation."""
        smiles = ["CCO", "CCCO"]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        fps = [FingerprintMols.FingerprintMol(m) for m in mols]
        config = Config()
        config.THRESHOLD_MCES = 20

        result = compute_ed_or_mces(
            smiles=smiles,
            sampled_index=0,
            batch_size=2,
            identifier=0,
            random_sampling=False,
            config=config,
            fps=fps,
            mols=mols,
            use_edit_distance=False,
        )

        assert result.shape == (2, 3)
        assert result[0, 2] == 0  # CCO vs CCO = 0
        assert result[1, 2] == 1.0  # CCO vs CCCO = 1

    def test_compute_ed_batch_size_exceeds_molecules(self):
        """Test that ValueError is raised when batch_size exceeds molecule count."""
        smiles = ["CCO", "CCCO"]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        fps = [FingerprintMols.FingerprintMol(m) for m in mols]
        config = Config()

        with pytest.raises(ValueError, match="batch_size .* cannot exceed"):
            compute_ed_or_mces(
                smiles=smiles,
                sampled_index=0,
                batch_size=5,  # Exceeds len(smiles)
                identifier=0,
                random_sampling=False,
                config=config,
                fps=fps,
                mols=mols,
                use_edit_distance=True,
            )
