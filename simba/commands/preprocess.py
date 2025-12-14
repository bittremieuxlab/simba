"""Preprocess command for SIMBA CLI."""

from pathlib import Path

import click


@click.command()
@click.option(
    "--spectra-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the input spectra file (.mgf format).",
)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory where preprocessed data will be saved.",
)
@click.option(
    "--max-spectra-train",
    type=int,
    default=10000,
    show_default=True,
    help="Maximum number of spectra to process for training. Set to large number to process all.",
)
@click.option(
    "--max-spectra-val",
    type=int,
    default=1000000,
    show_default=True,
    help="Maximum number of spectra to process for validation.",
)
@click.option(
    "--max-spectra-test",
    type=int,
    default=1000000,
    show_default=True,
    help="Maximum number of spectra to process for testing.",
)
@click.option(
    "--mapping-file-name",
    type=str,
    default="mapping_unique_smiles.pkl",
    show_default=True,
    help="Filename for the mapping file (saved in workspace directory).",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    show_default=True,
    help="Number of worker processes for parallel computation (0 = single process).",
)
@click.option(
    "--val-split",
    type=float,
    default=0.1,
    show_default=True,
    help="Fraction of data to use for validation (0.0-1.0).",
)
@click.option(
    "--test-split",
    type=float,
    default=0.1,
    show_default=True,
    help="Fraction of data to use for testing (0.0-1.0).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing preprocessing files in workspace directory.",
)
def preprocess(
    spectra_path: Path,
    workspace: Path,
    max_spectra_train: int,
    max_spectra_val: int,
    max_spectra_test: int,
    mapping_file_name: str,
    num_workers: int,
    val_split: float,
    test_split: float,
    overwrite: bool,
):
    """Preprocess MS/MS spectral data for SIMBA training.

    This command converts raw mass spectrometry data (.mgf format) into
    training-ready format by computing structural similarity metrics between
    molecules. The output includes numpy arrays with indexes and distances,
    plus a pickle file mapping spectra to molecular structures (SMILES).

    The preprocessing computes:
    - Edit distance between molecular structures
    - MCES (Maximum Common Edge Substructure) distance
    - Train/validation/test splits

    Examples:

        # Basic preprocessing with default settings
        simba preprocess \\
            --spectra-path data/spectra.mgf \\
            --workspace ./preprocessed_data

        # Process all spectra with 4 worker processes
        simba preprocess \\
            --spectra-path data/spectra.mgf \\
            --workspace ./preprocessed_data \\
            --max-spectra-train 1000000 \\
            --num-workers 4

        # Custom splits and mapping filename
        simba preprocess \\
            --spectra-path data/spectra.mgf \\
            --workspace ./preprocessed_data \\
            --val-split 0.15 \\
            --test-split 0.15 \\
            --mapping-file-name custom_mapping.pkl
    """
    click.echo("Starting SIMBA preprocessing...")
    click.echo(f"Input spectra: {spectra_path}")
    click.echo(f"Output workspace: {workspace}")
    click.echo(f"Max training spectra: {max_spectra_train}")
    click.echo(f"Number of workers: {num_workers}")

    # Validate splits
    if val_split + test_split >= 1.0:
        click.echo(
            "Error: val-split + test-split must be less than 1.0",
            err=True,
        )
        raise click.Abort()

    # Lazy imports - only import heavy dependencies when actually preprocessing
    import pickle

    from simba.config import Config
    from simba.core.data.preprocessing_simba import PreprocessingSimba
    from simba.logger_setup import logger
    from simba.core.chemistry.mces.mces_computation import MCES
    from simba.train_utils import TrainUtils

    try:
        # Setup configuration
        config = Config()
        config.SPECTRA_PATH = str(spectra_path)
        config.MOL_SPEC_MAPPING_FILE = mapping_file_name
        config.PREPROCESSING_DIR = str(workspace) + "/"
        config.PREPROCESSING_OVERWRITE = overwrite
        config.PREPROCESSING_NUM_WORKERS = num_workers
        config.MAX_SPECTRA_TRAIN = max_spectra_train
        config.MAX_SPECTRA_VAL = max_spectra_val
        config.MAX_SPECTRA_TEST = max_spectra_test
        config.VAL_SPLIT = val_split
        config.TEST_SPLIT = test_split

        # Create workspace directory
        workspace.mkdir(parents=True, exist_ok=True)

        # Clean workspace if overwrite is enabled
        if overwrite and workspace.exists():
            click.echo("Removing existing preprocessing files...")
            for file in workspace.iterdir():
                if file.is_file():
                    try:
                        file.unlink()
                    except Exception as e:
                        logger.error(f"Error deleting file {file}: {e}")

        # Load and preprocess spectra
        click.echo(f"Loading spectra from {spectra_path}...")
        all_spectra = PreprocessingSimba.load_spectra(
            config.SPECTRA_PATH,
            config,
            n_samples=700_000,
            use_gnps_format=False,
            use_only_protonized_adducts=config.USE_ONLY_PROTONIZED_ADDUCTS,
        )

        click.echo(f"Loaded {len(all_spectra)} spectra")

        # Split data into train, validation, and test sets
        click.echo("Splitting data into train/val/test sets...")
        all_spectra_train, all_spectra_val, all_spectra_test = (
            TrainUtils.train_val_test_split_bms(
                all_spectra,
                val_split=config.VAL_SPLIT,
                test_split=config.TEST_SPLIT,
            )
        )

        # Limit to max spectra
        all_spectra_train = all_spectra_train[0 : config.MAX_SPECTRA_TRAIN]
        all_spectra_val = all_spectra_val[0 : config.MAX_SPECTRA_VAL]
        all_spectra_test = all_spectra_test[0 : config.MAX_SPECTRA_TEST]

        click.echo(
            f"Split sizes - Train: {len(all_spectra_train)}, "
            f"Val: {len(all_spectra_val)}, Test: {len(all_spectra_test)}"
        )

        # Compute distances for each split
        molecule_pairs = {}
        for type_data, spectra in [
            ("_train", all_spectra_train),
            ("_val", all_spectra_val),
            ("_test", all_spectra_test),
        ]:
            split_name = type_data.replace("_", "").capitalize()
            click.echo(f"Computing distances for {split_name} set...")

            # First compute MCES, then compute Edit Distance reusing the same molecule pairs
            # This ensures both distance types are calculated and saved to files
            click.echo("  Computing MCES...")
            MCES.compute_all_mces_results_unique(
                spectra,
                max_combinations=10000000000000,
                num_workers=config.PREPROCESSING_NUM_WORKERS,
                random_sampling=config.RANDOM_MCES_SAMPLING,
                config=config,
                identifier=type_data,
                use_edit_distance=False,
                loaded_molecule_pairs=None,
            )

            click.echo("  Computing Edit Distance...")
            molecule_pairs[type_data] = MCES.compute_all_mces_results_unique(
                spectra,
                max_combinations=10000000000000,
                num_workers=config.PREPROCESSING_NUM_WORKERS,
                random_sampling=config.RANDOM_MCES_SAMPLING,
                config=config,
                identifier=type_data,
                use_edit_distance=True,
                loaded_molecule_pairs=None,
            )

        # Combine edit distance and MCES files
        click.echo("Combining edit distance and MCES files...")
        import os

        import numpy as np

        all_files = os.listdir(str(workspace))
        for partition in ["train", "val", "test"]:
            # Filter only the files ending with '.npy'
            npy_files = [f for f in all_files if f.endswith(".npy")]
            npy_files = [f for f in npy_files if partition in f]

            # Get unique indices
            indices = set()
            for file_loaded in npy_files:
                try:
                    index = file_loaded.split("_")[-1].split(".npy")[0]
                    indices.add(int(index))
                except (ValueError, IndexError):
                    continue

            for index in sorted(indices):
                # Define file names
                suffix = "indexes_tani_incremental_" + partition + "_"
                file_ed = workspace / f"edit_distance_{suffix}{index}.npy"
                file_mces = workspace / f"mces_{suffix}{index}.npy"
                file_output = workspace / f"ed_mces_{suffix}{index}.npy"

                if not file_ed.exists() or not file_mces.exists():
                    continue

                # Load data
                ed_data = np.load(file_ed)
                mces_data = np.load(file_mces)

                # Check that the indexes are the same
                if np.all(ed_data[:, 0] == mces_data[:, 0]) and np.all(
                    ed_data[:, 1] == mces_data[:, 1]
                ):
                    # Combine ED and MCES data
                    all_distance_data = np.column_stack((ed_data, mces_data[:, 2]))

                    # Filter out invalid values (NaN and 666)
                    ed = all_distance_data[:, config.COLUMN_EDIT_DISTANCE]
                    mces = all_distance_data[:, config.COLUMN_MCES20]
                    indexes_invalid = np.isnan(ed) | np.isnan(mces)
                    ed_exceeded = ed == 666
                    mces_exceeded = mces == 666
                    all_distance_data = all_distance_data[
                        (~indexes_invalid) & (~ed_exceeded) & (~mces_exceeded)
                    ]

                    np.save(file_output, all_distance_data)
                    click.echo(f"  Combined {partition} partition index {index}")

        # Save mapping file
        output_file = workspace / mapping_file_name
        click.echo(f"Saving mapping to {output_file}...")

        dataset = {
            "all_spectrums_train": None,
            "all_spectrums_val": None,
            "all_spectrums_test": None,
            "molecule_pairs_train": molecule_pairs["_train"],
            "molecule_pairs_val": molecule_pairs["_val"],
            "molecule_pairs_test": molecule_pairs["_test"],
            "uniformed_molecule_pairs_test": None,
        }

        with open(output_file, "wb") as file:
            pickle.dump(dataset, file)

        click.echo("Preprocessing completed successfully!")
        click.echo(f"Output saved to: {workspace}")
        click.echo(f"Mapping file: {output_file}")
        click.echo("\nNext step: Train a model using 'simba train' command")

    except Exception as e:
        click.echo(f"Error during preprocessing: {e}", err=True)
        raise click.Abort() from e
