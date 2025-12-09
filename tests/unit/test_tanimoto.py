"""Unit tests for Tanimoto similarity computations."""

import pytest
from rdkit import Chem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from simba.tanimoto import Tanimoto


pytestmark = pytest.mark.unit


class TestTanimotoComputeTanimoto:
    """Test compute_tanimoto static method."""

    def test_compute_tanimoto_identical_fingerprints(self):
        """Test Tanimoto for identical fingerprints returns 1.0."""
        mol = Chem.MolFromSmiles("CCO")
        fp = Chem.RDKFingerprint(mol)

        similarity = Tanimoto.compute_tanimoto(fp, fp)

        assert similarity == pytest.approx(1.0)

    def test_compute_tanimoto_different_fingerprints(self):
        """Test Tanimoto for different fingerprints."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCCO")
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)

        similarity = Tanimoto.compute_tanimoto(fp1, fp2)

        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.0  # Similar molecules should have some similarity

    def test_compute_tanimoto_none_fingerprints(self):
        """Test Tanimoto with None fingerprints returns None."""
        mol = Chem.MolFromSmiles("CCO")
        fp = Chem.RDKFingerprint(mol)

        result1 = Tanimoto.compute_tanimoto(None, fp)
        result2 = Tanimoto.compute_tanimoto(fp, None)
        result3 = Tanimoto.compute_tanimoto(None, None)

        assert result1 is None
        assert result2 is None
        assert result3 is None


class TestTanimotoComputeFingerprint:
    """Test compute_fingerprint static method."""

    def test_compute_fingerprint_valid_smiles(self):
        """Test fingerprint computation for valid SMILES."""
        smiles = "CCO"

        fp = Tanimoto.compute_fingerprint(smiles)

        assert isinstance(fp, ExplicitBitVect)
        assert fp.GetNumBits() > 0

    def test_compute_fingerprint_empty_smiles(self):
        """Test fingerprint for empty SMILES returns zero vector."""
        fp = Tanimoto.compute_fingerprint("")

        assert isinstance(fp, ExplicitBitVect)
        assert fp.GetNumOnBits() == 0  # Zero vector fingerprint

    def test_compute_fingerprint_na_smiles(self):
        """Test fingerprint for 'N/A' SMILES returns zero vector."""
        fp = Tanimoto.compute_fingerprint("N/A")

        assert isinstance(fp, ExplicitBitVect)
        assert fp.GetNumOnBits() == 0  # Zero vector fingerprint

    def test_compute_fingerprint_invalid_smiles(self):
        """Test fingerprint for invalid SMILES returns zero vector."""
        fp = Tanimoto.compute_fingerprint("INVALID_SMILES_123")

        assert isinstance(fp, ExplicitBitVect)
        assert fp.GetNumOnBits() == 0  # Zero vector fingerprint

    def test_compute_fingerprint_caching(self):
        """Test that compute_fingerprint uses caching."""
        smiles = "CCO"

        # Call twice with same SMILES
        fp1 = Tanimoto.compute_fingerprint(smiles)
        fp2 = Tanimoto.compute_fingerprint(smiles)

        # Should return the exact same object due to lru_cache
        assert fp1 is fp2


class TestTanimotoComputeTanimotoFromSmiles:
    """Test compute_tanimoto_from_smiles static method."""

    def test_compute_tanimoto_from_smiles_identical(self):
        """Test Tanimoto for identical SMILES returns 1.0."""
        smiles = "CCO"

        similarity = Tanimoto.compute_tanimoto_from_smiles(smiles, smiles)

        assert similarity == pytest.approx(1.0)

    def test_compute_tanimoto_from_smiles_similar(self):
        """Test Tanimoto for similar SMILES."""
        smiles1 = "CCO"  # Ethanol
        smiles2 = "CCCO"  # Propanol

        similarity = Tanimoto.compute_tanimoto_from_smiles(smiles1, smiles2)

        assert 0.0 < similarity < 1.0
        assert similarity > 0.5  # Similar molecules

    def test_compute_tanimoto_from_smiles_different(self):
        """Test Tanimoto for very different SMILES."""
        smiles1 = "C"  # Methane
        smiles2 = "c1ccccc1"  # Benzene

        similarity = Tanimoto.compute_tanimoto_from_smiles(smiles1, smiles2)

        assert 0.0 <= similarity <= 1.0
        assert similarity < 0.3  # Very different molecules

    def test_compute_tanimoto_from_smiles_empty(self):
        """Test Tanimoto for empty SMILES returns 1.0 (both zero vectors)."""
        similarity = Tanimoto.compute_tanimoto_from_smiles("", "")

        # RDKit returns 1.0 when both are zero vectors
        assert similarity == pytest.approx(1.0)

    def test_compute_tanimoto_from_smiles_one_empty(self):
        """Test Tanimoto when one SMILES is empty."""
        smiles = "CCO"

        similarity = Tanimoto.compute_tanimoto_from_smiles(smiles, "")

        # Non-zero vs zero vector should give zero similarity
        assert similarity == pytest.approx(0.0)

    def test_compute_tanimoto_from_smiles_caching(self):
        """Test that compute_tanimoto_from_smiles uses caching."""
        smiles1 = "CCO"
        smiles2 = "CCCO"

        # Call twice with same SMILES pair
        similarity1 = Tanimoto.compute_tanimoto_from_smiles(smiles1, smiles2)
        similarity2 = Tanimoto.compute_tanimoto_from_smiles(smiles1, smiles2)

        # Should return the same value
        assert similarity1 == similarity2

    def test_compute_tanimoto_from_smiles_canonical(self):
        """Test that different representations of same molecule give same result."""
        # Different SMILES for ethanol
        smiles1 = "CCO"
        smiles2 = "OCC"  # Same molecule, different order

        similarity = Tanimoto.compute_tanimoto_from_smiles(smiles1, smiles2)

        # Should be 1.0 or very close since they're the same molecule
        assert similarity == pytest.approx(1.0, abs=0.01)
