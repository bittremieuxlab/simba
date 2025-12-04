"""Unit tests for MCES computation utilities."""

import numpy as np
import pytest

from simba.mces.mces_computation import MCES


pytestmark = pytest.mark.unit


class TestMCESCreateCombinations:
    """Test create_combinations utility function."""

    def test_create_combinations_small_list(self):
        """Test combinations creation with small list."""
        spectra = [1, 2, 3]  # Dummy spectra
        combinations = MCES.create_combinations(spectra)

        assert len(combinations) == 3  # C(3,2) = 3
        assert (0, 1) in combinations
        assert (0, 2) in combinations
        assert (1, 2) in combinations

    def test_create_combinations_two_elements(self):
        """Test combinations with exactly two elements."""
        spectra = ["a", "b"]
        combinations = MCES.create_combinations(spectra)

        assert len(combinations) == 1
        assert (0, 1) in combinations

    def test_create_combinations_four_elements(self):
        """Test combinations with four elements."""
        spectra = [1, 2, 3, 4]
        combinations = MCES.create_combinations(spectra)

        assert len(combinations) == 6  # C(4,2) = 6
        # Check all pairs are present
        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for pair in expected:
            assert pair in combinations


class TestMCESCreateInputDF:
    """Test create_input_df utility function."""

    def test_create_input_df_simple(self):
        """Test DataFrame creation with simple inputs."""
        smiles = ["CCO", "CCCO", "C"]
        indexes_0 = [0, 1]
        indexes_1 = [1, 2]

        df = MCES.create_input_df(smiles, indexes_0, indexes_1)

        assert len(df) == 2
        assert list(df.columns) == ["smiles_0", "smiles_1"]
        assert df.iloc[0]["smiles_0"] == "CCO"
        assert df.iloc[0]["smiles_1"] == "CCCO"
        assert df.iloc[1]["smiles_0"] == "CCCO"
        assert df.iloc[1]["smiles_1"] == "C"

    def test_create_input_df_same_molecules(self):
        """Test DataFrame creation with same molecule indices."""
        smiles = ["CCO", "CCCO", "C"]
        indexes_0 = [0, 0]
        indexes_1 = [0, 1]

        df = MCES.create_input_df(smiles, indexes_0, indexes_1)

        assert len(df) == 2
        assert df.iloc[0]["smiles_0"] == "CCO"
        assert df.iloc[0]["smiles_1"] == "CCO"
        assert df.iloc[1]["smiles_0"] == "CCO"
        assert df.iloc[1]["smiles_1"] == "CCCO"


class TestMCESNormalize:
    """Test normalize_mces function."""

    def test_normalize_mces_below_threshold(self):
        """Test normalization for values below max_mces."""
        mces = np.array([0.0, 1.0, 2.0, 3.0])
        max_mces = 5

        normalized = MCES.normalize_mces(mces, max_mces=max_mces)

        # Formula: 1 - (mces / max_mces)
        expected = np.array([1.0, 0.8, 0.6, 0.4])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_mces_at_threshold(self):
        """Test normalization when value equals max_mces."""
        mces = np.array([5.0])
        max_mces = 5

        normalized = MCES.normalize_mces(mces, max_mces=max_mces)

        # At max: 1 - (5/5) = 0
        assert normalized[0] == pytest.approx(0.0)

    def test_normalize_mces_above_threshold(self):
        """Test normalization for values above max_mces (clamped)."""
        mces = np.array([0.0, 3.0, 6.0, 10.0, 100.0])
        max_mces = 5

        normalized = MCES.normalize_mces(mces, max_mces=max_mces)

        # Values >= 5 should be clamped to 5, then normalized: 1 - (5/5) = 0
        expected = np.array([1.0, 0.4, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_mces_all_zeros(self):
        """Test normalization with all zero values."""
        mces = np.array([0.0, 0.0, 0.0])
        max_mces = 5

        normalized = MCES.normalize_mces(mces, max_mces=max_mces)

        # All zeros: 1 - (0/5) = 1.0
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(normalized, expected)


class TestMCESExpNormalize:
    """Test exponential normalization functions."""

    def test_exp_normalize_mces20_zero(self):
        """Test exponential normalization with zero distance."""
        distances = np.array([0.0])
        result = MCES.exp_normalize_mces20(distances, scale=20, low_threshold=0.20)

        # At distance=0, similarity should be 1.0
        # Formula: 1 / (1 + (0 / 20)) = 1.0
        assert result[0] == pytest.approx(1.0, abs=0.01)

    def test_exp_normalize_mces20_high_distance(self):
        """Test exponential normalization with high distance."""
        distances = np.array([20.0])
        result = MCES.exp_normalize_mces20(distances, scale=20, low_threshold=0.20)

        # At distance=20, formula: 1 / (1 + (20 / 20)) = 1 / 2 = 0.5
        # But values <= low_threshold (0.20) are set to 0
        # 0.5 > 0.20, so should be 0.5
        assert result[0] == pytest.approx(0.5, abs=0.01)

    def test_exp_normalize_mces20_medium_distance(self):
        """Test exponential normalization with medium distance."""
        distances = np.array([10.0])
        result = MCES.exp_normalize_mces20(distances, scale=20, low_threshold=0.20)

        # At distance=10, formula: 1 / (1 + (10 / 20)) = 1 / 1.5 = 0.666...
        assert result[0] == pytest.approx(0.666, abs=0.01)

    def test_exp_normalize_mces20_array(self):
        """Test exponential normalization with array input."""
        distances = np.array([0.0, 5.0, 10.0, 20.0])
        results = MCES.exp_normalize_mces20(distances, scale=20, low_threshold=0.20)

        # Check results are valid
        assert len(results) == 4
        # Results should be monotonically decreasing (or equal after threshold)
        assert results[0] >= results[1] >= results[2] >= results[3]


class TestMCESInverseExpNormalize:
    """Test inverse exponential normalization function."""

    def test_inverse_exp_normalize_perfect_similarity(self):
        """Test inverse normalization with perfect similarity (1.0)."""
        similarity = np.array([1.0])
        result = MCES.inverse_exp_normalize_mces20(similarity, scale=20)

        # similarity=1.0 should give distance close to 0
        # Formula: scale * ((1 / similarity) - 1) = 20 * (1 - 1) = 0
        assert result[0] == pytest.approx(0.0, abs=0.01)

    def test_inverse_exp_normalize_half_similarity(self):
        """Test inverse normalization at half similarity."""
        similarity = np.array([0.5])
        result = MCES.inverse_exp_normalize_mces20(similarity, scale=20)

        # similarity=0.5 should give: 20 * ((1 / 0.5) - 1) = 20 * 1 = 20
        assert result[0] == pytest.approx(20.0, abs=0.01)

    def test_inverse_exp_normalize_medium_similarity(self):
        """Test inverse normalization with medium similarity."""
        similarity = np.array([0.666])
        result = MCES.inverse_exp_normalize_mces20(similarity, scale=20)

        # similarity=0.666 should give distance between 0 and 20
        # Formula: 20 * ((1 / 0.666) - 1) ≈ 20 * 0.5 = 10
        assert 0.0 < result[0] < 20.0
        assert result[0] == pytest.approx(10.0, abs=0.5)

    def test_inverse_exp_normalize_roundtrip(self):
        """Test that forward and inverse functions are inverses."""
        original_distance = np.array([10.0])

        # Forward: distance -> similarity
        similarity = MCES.exp_normalize_mces20(
            original_distance, scale=20, low_threshold=0.0
        )  # Use 0.0 threshold to avoid clamping

        # Inverse: similarity -> distance
        recovered_distance = MCES.inverse_exp_normalize_mces20(similarity, scale=20)

        assert recovered_distance[0] == pytest.approx(original_distance[0], abs=0.01)


class TestMCESComputeMCESListSmiles:
    """Test compute_mces_list_smiles function."""

    @pytest.mark.skip(
        reason="Requires external 'myopic_mces' tool which may not be available"
    )
    def test_compute_mces_list_smiles_identical(self):
        """Test MCES computation for identical molecules."""
        smiles_0 = ["CCO"]
        smiles_1 = ["CCO"]

        result = MCES.compute_mces_list_smiles(smiles_0, smiles_1, threshold_mces=20)

        assert len(result) == 1
        assert (
            result["mces"].iloc[0] == 0.0
        )  # Identical molecules should have distance 0

    @pytest.mark.skip(
        reason="Requires external 'myopic_mces' tool which may not be available"
    )
    def test_compute_mces_list_smiles_similar(self):
        """Test MCES computation for similar molecules."""
        smiles_0 = ["CCO"]  # Ethanol
        smiles_1 = ["CCCO"]  # Propanol

        result = MCES.compute_mces_list_smiles(smiles_0, smiles_1, threshold_mces=20)

        assert len(result) == 1
        assert result["mces"].iloc[0] >= 0.0  # Should return valid distance
        assert (
            result["mces"].iloc[0] < 20.0
        )  # Should be below threshold for similar molecules

    @pytest.mark.skip(
        reason="Requires external 'myopic_mces' tool which may not be available"
    )
    def test_compute_mces_list_smiles_multiple_pairs(self):
        """Test MCES computation for multiple pairs."""
        smiles_0 = ["CCO", "CCO"]
        smiles_1 = ["CCO", "CCCO"]

        result = MCES.compute_mces_list_smiles(smiles_0, smiles_1, threshold_mces=20)

        assert len(result) == 2
        assert result["mces"].iloc[0] == 0.0  # First pair identical
        assert result["mces"].iloc[1] > 0.0  # Second pair different
