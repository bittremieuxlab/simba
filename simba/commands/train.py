"""Train command for SIMBA CLI."""

import sys
from pathlib import Path

import click


@click.command()
@click.option(
    "--checkpoint-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory where the trained model will be saved.",
)
@click.option(
    "--preprocessing-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory where preprocessing files are stored.",
)
@click.option(
    "--preprocessing-pickle",
    type=str,
    required=True,
    help="Filename of the mapping pickle file (e.g., mapping_unique_smiles.pkl).",
)
@click.option(
    "--epochs",
    type=int,
    default=10,
    show_default=True,
    help="Number of training epochs.",
)
@click.option(
    "--accelerator",
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="cpu",
    show_default=True,
    help="Hardware accelerator to use for training.",
)
@click.option(
    "--val-check-interval",
    type=int,
    default=10000,
    show_default=True,
    help="Validation check frequency (every N training steps).",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    show_default=True,
    help="Batch size for training and validation.",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    show_default=True,
    help="Number of data loading workers (0 = main thread only).",
)
@click.option(
    "--learning-rate",
    type=float,
    default=0.0001,
    show_default=True,
    help="Learning rate for the optimizer.",
)
def train(
    checkpoint_dir: Path,
    preprocessing_dir: Path,
    preprocessing_pickle: str,
    epochs: int,
    accelerator: str,
    val_check_interval: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
):
    """Train a SIMBA model on MS/MS spectral data.

    This command trains a custom SIMBA model using preprocessed MS/MS data.
    The model learns to predict structural similarity between molecules from
    their tandem mass spectra.

    Examples:

        # Basic training with CPU
        simba train --checkpoint-dir ./checkpoints \\
                    --preprocessing-dir ./preprocessed_data \\
                    --preprocessing-pickle mapping_unique_smiles.pkl \\
                    --epochs 10

        # Training with GPU and custom settings
        simba train --checkpoint-dir ./checkpoints \\
                    --preprocessing-dir ./preprocessed_data \\
                    --preprocessing-pickle mapping_unique_smiles.pkl \\
                    --epochs 50 \\
                    --accelerator gpu \\
                    --batch-size 64 \\
                    --learning-rate 0.001
    """
    click.echo("Starting SIMBA training...")
    click.echo(f"Checkpoint directory: {checkpoint_dir}")
    click.echo(f"Preprocessing directory: {preprocessing_dir}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Accelerator: {accelerator}")
    click.echo(f"Batch size: {batch_size}")

    # Lazy imports - only import heavy dependencies when actually training
    # This speeds up CLI help commands and reduces import time
    import dill

    import simba
    from simba.config import Config

    # Add project root to path to import training scripts
    _project_root = Path(__file__).parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    # Import training functions from existing script
    # TODO: Refactor these into simba.training module to make CLI independent
    # ruff: noqa: E402
    from training_scripts.final_training import (
        create_dataloaders,
        prepare_data,
        setup_callbacks,
        setup_model,
    )
    from training_scripts.final_training import (
        train as run_training,
    )

    try:
        # Setup configuration
        config = _setup_config(
            checkpoint_dir=checkpoint_dir,
            preprocessing_dir=preprocessing_dir,
            preprocessing_pickle=preprocessing_pickle,
            epochs=epochs,
            accelerator=accelerator,
            val_check_interval=val_check_interval,
            batch_size=batch_size,
            num_workers=num_workers,
            learning_rate=learning_rate,
            Config=Config,
        )

        # Create checkpoint directory
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        mapping_path = preprocessing_dir / preprocessing_pickle
        click.echo(f"Loading dataset from {mapping_path}...")
        (
            molecule_pairs_train,
            molecule_pairs_val,
            molecule_pairs_test,
            uniformed_molecule_pairs_test,
        ) = _load_dataset(mapping_path, simba, dill)

        # Prepare training data
        click.echo("Preparing training data...")

        (
            dataset_train,
            train_sampler,
            dataset_val,
            val_sampler,
            weights_ed,
            bins_ed,
        ) = prepare_data(
            molecule_pairs_train,
            molecule_pairs_val,
            molecule_pairs_test,
            uniformed_molecule_pairs_test,
            config,
        )

        # Create dataloaders
        dataloader_train, dataloader_val = create_dataloaders(
            config, dataset_train, train_sampler, dataset_val, val_sampler
        )

        # Adjust val_check_interval if dataset is too small
        num_train_batches = len(dataloader_train)
        if num_train_batches < config.VAL_CHECK_INTERVAL:
            click.echo(
                f"Warning: val_check_interval ({config.VAL_CHECK_INTERVAL}) is larger than number of training batches ({num_train_batches})"
            )
            config.VAL_CHECK_INTERVAL = max(1, num_train_batches // 2)
            click.echo(f"Adjusted val_check_interval to: {config.VAL_CHECK_INTERVAL}")

        # Setup model and callbacks
        click.echo("Initializing model...")
        checkpoint_callback, checkpoint_n_steps_callback, losscallback = (
            setup_callbacks(config)
        )

        # Get weights for MCES from first 100 batches (same as original script)
        click.echo("Computing MCES weights from training data...")

        import numpy as np

        from simba.train_utils import TrainUtils

        mces_sampled = []
        for i, batch in enumerate(dataloader_train):
            mces_sampled = mces_sampled + list(batch["mces"].reshape(-1))
            if i == 100:
                break

        mces_sampled = np.array(mces_sampled)
        counting_mces, bins_mces = TrainUtils.count_ranges(
            mces_sampled,
            number_bins=5,
            bin_sim_1=False,
            max_value=1,
        )

        # Calculate weights directly (same as original script)
        weights_mces = np.array(
            [np.sum(counting_mces) / c if c != 0 else 0 for c in counting_mces]
        )
        weights_mces = weights_mces / np.sum(weights_mces)

        model = setup_model(config, weights_mces)

        # Train model
        click.echo(f"Starting training for {epochs} epochs...")
        run_training(
            model,
            dataloader_train,
            dataloader_val,
            config,
            checkpoint_callback,
            checkpoint_n_steps_callback,
            losscallback,
        )

        click.echo("Training completed successfully!")
        click.echo(f"Model saved to: {checkpoint_dir}")

    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
        raise click.Abort() from e


def _setup_config(
    checkpoint_dir: Path,
    preprocessing_dir: Path,
    preprocessing_pickle: str,
    epochs: int,
    accelerator: str,
    val_check_interval: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    Config,
):
    """Setup configuration for training."""
    config = Config()

    # Paths
    config.CHECKPOINT_DIR = str(checkpoint_dir) + "/"
    config.PREPROCESSING_DIR_TRAIN = str(preprocessing_dir) + "/"
    config.PREPROCESSING_PICKLE_FILE = preprocessing_pickle

    # Training parameters
    config.epochs = epochs
    config.ACCELERATOR = accelerator
    config.VAL_CHECK_INTERVAL = val_check_interval
    config.TRAINING_BATCH_SIZE = batch_size
    config.VALIDATION_BATCH_SIZE = batch_size
    config.TRAINING_NUM_WORKERS = num_workers
    config.VALIDATION_NUM_WORKERS = num_workers
    config.LR = learning_rate

    # Inference settings
    config.bins_uniformise_INFERENCE = config.EDIT_DISTANCE_N_CLASSES - 1
    config.use_uniform_data_INFERENCE = True

    return config


def _load_dataset(mapping_path: Path, simba, dill):
    """Load training dataset from pickle file."""
    sys.modules["src"] = simba

    with open(mapping_path, "rb") as file:
        mapping = dill.load(file)

    return (
        mapping["molecule_pairs_train"],
        mapping["molecule_pairs_val"],
        mapping["molecule_pairs_test"],
        mapping["uniformed_molecule_pairs_test"],
    )
