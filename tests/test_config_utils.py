"""Tests for config utility functions."""

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from simba.utils.config_utils import (
    get_checkpoint_dir,
    get_model_code,
    get_model_paths,
    validate_paths,
)


@pytest.fixture
def hydra_config():
    """Load Hydra configuration for testing."""
    config_path = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(
        config_dir=str(config_path.absolute()), version_base=None
    ):
        cfg = compose(config_name="config")
    return cfg


def test_get_model_code(hydra_config):
    """Test model code generation."""
    model_code = get_model_code(hydra_config)

    # Check format
    assert "_units_" in model_code
    assert "_layers_" in model_code
    assert "_epochs_" in model_code
    assert "_lr_" in model_code
    assert "_bs" in model_code

    # Check values match config
    assert str(hydra_config.model.transformer.d_model) in model_code
    assert str(hydra_config.model.transformer.n_layers) in model_code
    assert str(hydra_config.training.epochs) in model_code


def test_get_checkpoint_dir_default(hydra_config):
    """Test checkpoint directory with default settings."""
    # Ensure checkpoint_dir is None
    hydra_config.paths.checkpoint_dir = None

    # Ensure no CHECKPOINT_BASE env var
    if "CHECKPOINT_BASE" in os.environ:
        del os.environ["CHECKPOINT_BASE"]

    checkpoint_dir = get_checkpoint_dir(hydra_config)

    # Should default to ./checkpoints/model_checkpoints_{MODEL_CODE}
    assert str(checkpoint_dir).startswith("checkpoints/model_checkpoints_")
    assert "256_units" in str(checkpoint_dir)  # d_model=256


def test_get_checkpoint_dir_env_var(hydra_config):
    """Test checkpoint directory with CHECKPOINT_BASE env var."""
    hydra_config.paths.checkpoint_dir = None

    # Set environment variable
    test_base = "/tmp/test_checkpoints"
    os.environ["CHECKPOINT_BASE"] = test_base

    try:
        checkpoint_dir = get_checkpoint_dir(hydra_config)
        assert str(checkpoint_dir).startswith(test_base)
        assert "model_checkpoints_" in str(checkpoint_dir)
    finally:
        del os.environ["CHECKPOINT_BASE"]


def test_get_checkpoint_dir_explicit(hydra_config):
    """Test checkpoint directory with explicit config value."""
    explicit_path = "/explicit/checkpoint/path"
    hydra_config.paths.checkpoint_dir = explicit_path

    checkpoint_dir = get_checkpoint_dir(hydra_config)

    # Should use explicit path
    assert str(checkpoint_dir) == explicit_path


def test_get_model_paths(hydra_config):
    """Test getting all model paths."""
    paths = get_model_paths(hydra_config)

    # Check all required keys exist
    assert "checkpoint_dir" in paths
    assert "best_model_path" in paths
    assert "pretrained_path" in paths

    # Check paths are Path objects
    assert isinstance(paths["checkpoint_dir"], Path)
    assert isinstance(paths["best_model_path"], Path)
    assert isinstance(paths["pretrained_path"], Path)

    # Check filenames match config
    assert paths["best_model_path"].name == hydra_config.checkpoints.best_model_name
    assert (
        paths["pretrained_path"].name == hydra_config.checkpoints.pretrained_model_name
    )

    # Check structure
    assert paths["best_model_path"].parent == paths["checkpoint_dir"]
    assert paths["pretrained_path"].parent == paths["checkpoint_dir"]


def test_validate_paths_nonexistent(hydra_config, tmp_path):
    """Test path validation with nonexistent paths."""
    # Set a nonexistent path
    nonexistent = tmp_path / "nonexistent"
    hydra_config.paths.spectra_path = str(nonexistent)

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="spectra_path does not exist"):
        validate_paths(hydra_config, create_dirs=False)


def test_validate_paths_create_dirs(hydra_config, tmp_path):
    """Test path validation with directory creation."""
    # Set a nonexistent preprocessing dir
    new_dir = tmp_path / "new_preprocessing"
    hydra_config.paths.preprocessing_dir = str(new_dir)
    hydra_config.paths.spectra_path = None  # Disable spectra path check

    # Should create directory
    validate_paths(hydra_config, create_dirs=True)
    assert new_dir.exists()
    assert new_dir.is_dir()


def test_model_code_deterministic(hydra_config):
    """Test that model code is deterministic."""
    code1 = get_model_code(hydra_config)
    code2 = get_model_code(hydra_config)

    assert code1 == code2


def test_model_code_changes_with_params(hydra_config):
    """Test that model code changes when parameters change."""
    code1 = get_model_code(hydra_config)

    # Change a parameter
    original_d_model = hydra_config.model.transformer.d_model
    hydra_config.model.transformer.d_model = 512

    code2 = get_model_code(hydra_config)

    assert code1 != code2
    assert "512_units" in code2

    # Restore
    hydra_config.model.transformer.d_model = original_d_model
