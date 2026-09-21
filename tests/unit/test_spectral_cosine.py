"""Tests for simba/core/data/spectral_cosine.py::pairwise_spectral_cosine"""

import numpy as np

from simba.core.data.spectral_cosine import pairwise_spectral_cosine


class TestPairwiseSpectralCosine:
    def test_identical_spectra(self):
        mz = np.array([100.0, 200.0, 300.0])
        intensity = np.array([1.0, 0.5, 0.25])

        cos = pairwise_spectral_cosine(mz, intensity, mz, intensity)

        assert np.isclose(cos, 1.0, atol=1e-6)

    def test_disjoint_spectra(self):
        mz_a = np.array([100.0, 200.0])
        intensity_a = np.array([1.0, 0.5])
        mz_b = np.array([400.0, 500.0])
        intensity_b = np.array([1.0, 1.0])

        cos = pairwise_spectral_cosine(mz_a, intensity_a, mz_b, intensity_b)

        assert np.isclose(cos, 0.0, atol=1e-6)

    def test_partial_overlap_matches_hand_computation(self):
        # side a: one peak at 100 (weight 1), one at 300 (weight 1)
        # side b: one peak at 100 (weight 1), one at 500 (weight 1)
        # sqrt-compressed intensities are both 1.0 for each peak here, so
        # each side's L2-normalized vector puts 1/sqrt(2) on its two bins.
        # Only the 100 bin overlaps -> cosine = (1/sqrt(2)) * (1/sqrt(2)) = 0.5
        mz_a = np.array([100.0, 300.0])
        intensity_a = np.array([1.0, 1.0])
        mz_b = np.array([100.0, 500.0])
        intensity_b = np.array([1.0, 1.0])

        cos = pairwise_spectral_cosine(mz_a, intensity_a, mz_b, intensity_b)

        assert np.isclose(cos, 0.5, atol=1e-6)

    def test_all_peaks_masked_out_returns_zero(self):
        mz = np.array([100.0, 200.0, 300.0])
        intensity = np.array([1.0, 0.5, 0.25])
        zeros = np.zeros(3)

        cos = pairwise_spectral_cosine(mz, intensity, zeros, zeros)

        assert cos == 0.0

    def test_padding_zeros_ignored(self):
        # padded arrays (fixed max_num_peaks), trailing zeros should not
        # contribute to the similarity.
        mz = np.array([100.0, 0.0, 0.0])
        intensity = np.array([1.0, 0.0, 0.0])

        cos = pairwise_spectral_cosine(mz, intensity, mz, intensity)

        assert np.isclose(cos, 1.0, atol=1e-6)

    def test_peaks_beyond_max_mz_dropped(self):
        mz = np.array([100.0, 2000.0])
        intensity = np.array([1.0, 1.0])

        cos = pairwise_spectral_cosine(mz, intensity, mz, intensity, max_mz=1100.0)

        # both sides identical after dropping the >max_mz peak -> still 1.0
        assert np.isclose(cos, 1.0, atol=1e-6)
