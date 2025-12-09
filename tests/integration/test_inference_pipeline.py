"""Integration tests for SIMBA inference pipeline.

Based on notebooks/final_tutorials/run_inference.ipynb
"""

import numpy as np
import pytest

from simba.config import Config
from simba.ordinal_classification.embedder_multitask import EmbedderMultitask
from simba.simba.preprocessing_simba import PreprocessingSimba
from simba.simba.simba import Simba


pytestmark = pytest.mark.integration


class TestInferencePipeline:
    """Test inference workflow from README use case 1."""

    def test_load_spectra_from_mgf_standard_format(self, sample_mgf):
        """Test loading spectra from standard MGF format."""
        config = Config()
        spectra = PreprocessingSimba.load_spectra(
            sample_mgf,
            config,
            min_peaks=5,
            n_samples=100,
            use_gnps_format=False,
        )

        assert len(spectra) > 0
        assert len(spectra) <= 3

        for spec in spectra:
            assert hasattr(spec, "precursor_mz")
            assert hasattr(spec, "mz")
            assert hasattr(spec, "intensity")
            assert len(spec.mz) > 0

    def test_load_spectra_from_mgf_casmi_format(self, sample_mgf_casmi):
        """Test loading spectra from CASMI2022 format with SMILES."""
        config = Config()
        spectra = PreprocessingSimba.load_spectra(
            sample_mgf_casmi,
            config,
            min_peaks=5,
            n_samples=100,
            use_gnps_format=False,
        )

        assert len(spectra) > 0

        for spec in spectra:
            assert hasattr(spec, "precursor_mz")
            assert len(spec.mz) > 0

    def test_inference_end_to_end(self, sample_mgf, mocker):
        """Test complete inference pipeline with model."""
        config = Config()

        spectra = PreprocessingSimba.load_spectra(
            sample_mgf,
            config,
            min_peaks=5,
            n_samples=100,
            use_gnps_format=False,
        )

        assert len(spectra) >= 2
        n_spectra = len(spectra)

        model = EmbedderMultitask(
            d_model=int(config.D_MODEL),
            n_layers=int(config.N_LAYERS),
            n_classes=config.EDIT_DISTANCE_N_CLASSES,
            use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
            use_element_wise=True,
            use_cosine_distance=config.use_cosine_distance,
            use_edit_distance_regresion=config.USE_EDIT_DISTANCE_REGRESSION,
            use_fingerprints=config.USE_FINGERPRINT,
            USE_LEARNABLE_MULTITASK=config.USE_LEARNABLE_MULTITASK,
        )
        model.eval()

        mocker.patch(
            "simba.ordinal_classification.embedder_multitask.EmbedderMultitask.load_from_checkpoint",
            return_value=model,
        )

        simba = Simba(
            "fake_model.ckpt", config=config, device="cpu", cache_embeddings=True
        )
        assert simba is not None
        assert simba.model is not None

        sim_ed, sim_mces = simba.predict(spectra, spectra)

        assert sim_ed.shape == (n_spectra, n_spectra)
        assert sim_mces.shape == (n_spectra, n_spectra)
        assert isinstance(sim_ed, np.ndarray)
        assert isinstance(sim_mces, np.ndarray)

    def test_embedding_caching(self, sample_mgf, mocker):
        """Test that embeddings caching works correctly."""
        config = Config()

        model = EmbedderMultitask(
            d_model=int(config.D_MODEL),
            n_layers=int(config.N_LAYERS),
            n_classes=config.EDIT_DISTANCE_N_CLASSES,
            use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
            use_element_wise=True,
            use_cosine_distance=config.use_cosine_distance,
            use_edit_distance_regresion=config.USE_EDIT_DISTANCE_REGRESSION,
            use_fingerprints=config.USE_FINGERPRINT,
            USE_LEARNABLE_MULTITASK=config.USE_LEARNABLE_MULTITASK,
        )
        model.eval()

        mocker.patch(
            "simba.ordinal_classification.embedder_multitask.EmbedderMultitask.load_from_checkpoint",
            return_value=model,
        )

        simba = Simba(
            "fake_model.ckpt", config=config, device="cpu", cache_embeddings=True
        )

        assert simba.cache_embeddings is True
        assert hasattr(simba, "_embedding_cache")
        assert isinstance(simba._embedding_cache, dict)

        simba_no_cache = Simba(
            "fake_model.ckpt", config=config, device="cpu", cache_embeddings=False
        )
        assert simba_no_cache.cache_embeddings is False
