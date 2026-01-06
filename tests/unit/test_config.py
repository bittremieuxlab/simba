"""Tests for simba/config.py"""

import os

import pytest

from simba.config import Config


class TestConfig:
    """Tests for the Config class."""

    def test_derived_variables(self):
        """Test that derived variables are computed correctly."""
        config = Config()

        assert isinstance(config.MODEL_CODE, str)
        assert str(config.D_MODEL) in config.MODEL_CODE
        assert str(config.N_LAYERS) in config.MODEL_CODE
        assert str(config.epochs) in config.MODEL_CODE

        assert config.CHECKPOINT_DIR is not None
        assert "model_checkpoints" in config.CHECKPOINT_DIR

        assert config.best_model_path is not None
        assert config.pretrained_path is not None
        assert config.BEST_MODEL_NAME in config.best_model_path
        assert config.PRETRAINED_MODEL_NAME in config.pretrained_path

    def test_checkpoint_dir_env_variable(self, temp_dir, monkeypatch):
        """Test that CHECKPOINT_BASE environment variable is respected."""
        test_base = str(temp_dir / "test_checkpoints")
        monkeypatch.setenv("CHECKPOINT_BASE", test_base)

        config = Config()

        assert test_base in config.CHECKPOINT_DIR
        assert "model_checkpoints" in config.CHECKPOINT_DIR

    def test_checkpoint_dir_default_fallback(self, monkeypatch):
        """Test that checkpoint directory falls back to default when no env var."""
        monkeypatch.delenv("CHECKPOINT_BASE", raising=False)

        config = Config()

        assert "./checkpoints" in config.CHECKPOINT_DIR or os.path.isabs(
            config.CHECKPOINT_DIR
        )

    def test_validation_split_values(self):
        """Test that data split fractions are reasonable."""
        config = Config()

        assert 0 < config.VAL_SPLIT < 1
        assert 0 < config.TEST_SPLIT < 1
        assert config.VAL_SPLIT + config.TEST_SPLIT < 1

    @pytest.mark.parametrize(
        "attr_name,expected_type",
        [
            ("ACCELERATOR", str),
            ("BATCH_SIZE", int),
            ("LR", float),
            ("N_LAYERS", int),
            ("D_MODEL", int),
            ("EMBEDDING_DIM", int),
            ("USE_MULTITASK", bool),
            ("enable_progress_bar", bool),
        ],
    )
    def test_config_attribute_types(self, attr_name, expected_type):
        """Test that config attributes have expected types."""
        config = Config()
        assert isinstance(getattr(config, attr_name), expected_type)
