"""Utility functions for working with Hydra configurations.

This module provides helper functions to compute derived configuration parameters.
"""

import os
from pathlib import Path

from omegaconf import DictConfig


def get_config_path() -> Path:
    """Get the absolute path to the configs directory.

    Returns:
        Path to configs/ directory
    """
    # Get the simba package root directory
    package_root = Path(__file__).parent.parent.parent
    return package_root / "configs"


def get_model_code(cfg: DictConfig) -> str:
    """Generate model code string from config parameters.

    The model code is a unique identifier constructed from key hyperparameters.

    Format: {D_MODEL}_units_{N_LAYERS}_layers_{epochs}_epochs_{LR}_lr_{BATCH_SIZE}_bs{extra_info}

    Args:
        cfg: Hydra configuration object

    Returns:
        Model code string (e.g., "256_units_5_layers_1000_epochs_0.0001_lr_128_bs_multitasking")

    Example:
        >>> cfg = load_config()
        >>> model_code = get_model_code(cfg)
        >>> print(model_code)
        "256_units_5_layers_1000_epochs_0.0001_lr_128_bs_multitasking_mces20raw"
    """
    return (
        f"{cfg.model.transformer.d_model}_units_"
        f"{cfg.model.transformer.n_layers}_layers_"
        f"{cfg.training.epochs}_epochs_"
        f"{cfg.optimizer.lr}_lr_"
        f"{cfg.training.batch_size}_bs"
        f"{cfg.project.extra_info}"
    )


def get_checkpoint_dir(cfg: DictConfig) -> Path:
    """Get checkpoint directory path.

    Priority order:
    1. If cfg.paths.checkpoint_dir is set explicitly, use it
    2. Otherwise, generate from MODEL_CODE using:
       - Environment variable CHECKPOINT_BASE (if set)
       - Or default to ./checkpoints

    Directory structure: {CHECKPOINT_BASE}/model_checkpoints_{MODEL_CODE}

    Args:
        cfg: Hydra configuration object

    Returns:
        Path to checkpoint directory

    Example:
        >>> cfg = load_config()
        >>> checkpoint_dir = get_checkpoint_dir(cfg)
        >>> print(checkpoint_dir)
        PosixPath('./checkpoints/model_checkpoints_256_units_5_layers_...')

    Note:
        Set CHECKPOINT_BASE environment variable for cluster deployment:
        export CHECKPOINT_BASE=/scratch/user/data/model_checkpoints
    """
    # Check if explicitly set in config
    if cfg.paths.checkpoint_dir is not None:
        return Path(cfg.paths.checkpoint_dir)

    # Generate from MODEL_CODE
    checkpoint_base = os.environ.get("CHECKPOINT_BASE", "./checkpoints")
    model_code = get_model_code(cfg)
    checkpoint_dir = Path(checkpoint_base) / f"model_checkpoints_{model_code}"

    return checkpoint_dir


def get_model_paths(cfg: DictConfig) -> dict[str, Path]:
    """Get all model-related paths.

    Computes checkpoint directory and model file paths based on config.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dictionary with keys:
        - checkpoint_dir: Base directory for model checkpoints
        - best_model_path: Path to best model checkpoint
        - pretrained_path: Path to pretrained model checkpoint

    Example:
        >>> cfg = load_config()
        >>> paths = get_model_paths(cfg)
        >>> print(paths["best_model_path"])
        PosixPath('./checkpoints/model_checkpoints_.../best_model.ckpt')
    """
    checkpoint_dir = get_checkpoint_dir(cfg)

    return {
        "checkpoint_dir": checkpoint_dir,
        "best_model_path": checkpoint_dir / cfg.checkpoints.best_model_name,
        "pretrained_path": checkpoint_dir / cfg.checkpoints.pretrained_model_name,
    }


def validate_paths(cfg: DictConfig, create_dirs: bool = False) -> None:
    """Validate that required paths exist or create them.

    Args:
        cfg: Hydra configuration object
        create_dirs: If True, create missing directories

    Raises:
        FileNotFoundError: If required paths don't exist and create_dirs=False
    """
    paths_to_check = []

    # Check spectra path if set
    if cfg.paths.spectra_path is not None:
        paths_to_check.append(("spectra_path", Path(cfg.paths.spectra_path), False))

    # Check preprocessing directories
    if cfg.paths.preprocessing_dir is not None:
        paths_to_check.append(
            ("preprocessing_dir", Path(cfg.paths.preprocessing_dir), True)
        )

    for name, path, is_dir in paths_to_check:
        if not path.exists():
            if create_dirs and is_dir:
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"{name} does not exist: {path}")
