"""Preprocessing workflow for SIMBA.

This module contains the main preprocessing logic for MS/MS spectral data.
It computes structural similarity metrics and prepares data for training.
"""

import os
import pickle
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from simba.core.chemistry.mces.mces_computation import MCES
from simba.core.data.preprocessing_simba import PreprocessingSimba
from simba.core.training.train_utils import TrainUtils
from simba.utils.logger_setup import logger


def write_data(
    file_path: str,
    molecule_pairs_train=None,
    molecule_pairs_val=None,
    molecule_pairs_test=None,
    uniformed_molecule_pairs_test=None,
    use_lightweight_format: bool = False,
    mgf_path: str = None,
):
    """Write preprocessed data to pickle file.

    Args:
        file_path: Path to output pickle file
        molecule_pairs_train: Training molecule pairs
        molecule_pairs_val: Validation molecule pairs
        molecule_pairs_test: Test molecule pairs
        uniformed_molecule_pairs_test: Uniformed test molecule pairs (optional)
        use_lightweight_format: If True, save lightweight format with only df_smiles and spectrum IDs
        mgf_path: Path to original MGF file (required for lightweight format)
    """
    if use_lightweight_format:
        logger.info(
            "Saving in lightweight format (df_smiles + spectrum indexes + MGF path)"
        )
        # Lightweight format: save only df_smiles, original MGF indexes, and mgf path
        # Spectra will be loaded at training time from mgf file using absolute indexes
        dataset = {
            "df_smiles_train": molecule_pairs_train.df_smiles
            if molecule_pairs_train is not None
            else None,
            "df_smiles_val": molecule_pairs_val.df_smiles
            if molecule_pairs_val is not None
            else None,
            "df_smiles_test": molecule_pairs_test.df_smiles
            if molecule_pairs_test is not None
            else None,
            "spectrum_indexes_train": [
                s.mgf_index for s in molecule_pairs_train.original_spectra
            ]
            if molecule_pairs_train is not None
            else None,
            "spectrum_indexes_val": [
                s.mgf_index for s in molecule_pairs_val.original_spectra
            ]
            if molecule_pairs_val is not None
            else None,
            "spectrum_indexes_test": [
                s.mgf_index for s in molecule_pairs_test.original_spectra
            ]
            if molecule_pairs_test is not None
            else None,
            "mgf_path": mgf_path,
            "format_version": "lightweight",
        }
    else:
        logger.info("Saving in full format (complete molecule pairs)")
        # Original format: save full MoleculePairs objects
        dataset = {
            "all_spectrums_train": None,
            "all_spectrums_val": None,
            "all_spectrums_test": None,
            "molecule_pairs_train": molecule_pairs_train,
            "molecule_pairs_val": molecule_pairs_val,
            "molecule_pairs_test": molecule_pairs_test,
            "uniformed_molecule_pairs_test": uniformed_molecule_pairs_test,
        }

    with open(file_path, "wb") as file:
        pickle.dump(dataset, file)


def preprocess(cfg: DictConfig) -> None:
    """Run the full preprocessing pipeline.

    This function:
    1. Loads spectra from MGF file
    2. Splits data into train/val/test sets
    3. Computes edit distance and MCES between molecules
    4. Saves preprocessed data to disk

    Args:
        cfg: Hydra configuration object with preprocessing parameters
    """
    logger.info("Starting SIMBA preprocessing workflow...")

    workspace = Path(cfg.paths.preprocessing_dir)

    # Create workspace directory
    workspace.mkdir(parents=True, exist_ok=True)

    # Clean workspace if overwrite is enabled
    if cfg.preprocessing.overwrite and workspace.exists():
        logger.info("Removing existing preprocessing files...")
        for file in workspace.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"Error deleting file {file}: {e}")

    # Load and preprocess spectra
    logger.info(f"Loading spectra from {cfg.paths.spectra_path}...")
    # Use max_spectra_load if specified, otherwise load all spectra (-1)
    n_samples = getattr(cfg.preprocessing, "max_spectra_load", -1)
    logger.info(
        f"Loading up to {n_samples if n_samples > 0 else 'all'} spectra from MGF file..."
    )
    all_spectra = PreprocessingSimba.load_spectra(
        str(cfg.paths.spectra_path),
        cfg,
        n_samples=n_samples,
        use_gnps_format=False,
        use_only_protonized_adducts=cfg.preprocessing.use_only_protonized_adducts,
    )

    logger.info(f"Loaded {len(all_spectra)} spectra")

    # Split data into train, validation, and test sets
    logger.info("Splitting data into train/val/test sets...")
    all_spectra_train, all_spectra_val, all_spectra_test = (
        TrainUtils.train_val_test_split_bms(
            all_spectra,
            val_split=cfg.preprocessing.val_split,
            test_split=cfg.preprocessing.test_split,
        )
    )

    # Limit to max spectra
    all_spectra_train = all_spectra_train[0 : cfg.preprocessing.max_spectra_train]
    all_spectra_val = all_spectra_val[0 : cfg.preprocessing.max_spectra_val]
    all_spectra_test = all_spectra_test[0 : cfg.preprocessing.max_spectra_test]

    logger.info(
        f"Split sizes - Train: {len(all_spectra_train)}, "
        f"Val: {len(all_spectra_val)}, Test: {len(all_spectra_test)}"
    )

    # Compute distances for each partition
    molecule_pairs = {}
    for type_data, spectra in [
        ("_train", all_spectra_train),
        ("_val", all_spectra_val),
        ("_test", all_spectra_test),
    ]:
        logger.info(f"Computing distances for {type_data[1:]} set...")
        molecule_pairs[type_data] = MCES.compute_all_mces_results_unique(
            spectra,
            max_combinations=10000000000000,
            num_workers=cfg.preprocessing.num_workers,
            random_sampling=cfg.preprocessing.random_mces_sampling,
            preprocessing_dir=str(workspace) + "/",
            batch_size=cfg.preprocessing.batch_size,
            num_nodes=cfg.preprocessing.num_nodes,
            current_node=cfg.preprocessing.current_node,
            compute_specific_pairs=cfg.data.similarity.compute_specific_pairs,
            format_file_specific_pairs=cfg.data.formats.format_file_specific_pairs,
            threshold_mces=cfg.model.tasks.mces.threshold,
            identifier=type_data,
            use_edit_distance=True,  # Ignored when compute_both_metrics=True
            loaded_molecule_pairs=None,
            compute_both_metrics=True,
        )

    # Combine edit distance and MCES files
    logger.info("Combining edit distance and MCES files...")

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
                ed = all_distance_data[:, cfg.model.data_columns.edit_distance]
                mces = all_distance_data[:, cfg.model.data_columns.mces20]
                indexes_invalid = np.isnan(ed) | np.isnan(mces)
                ed_exceeded = ed == 666
                mces_exceeded = mces == 666
                all_distance_data = all_distance_data[
                    (~indexes_invalid) & (~ed_exceeded) & (~mces_exceeded)
                ]

                np.save(file_output, all_distance_data)
                logger.info(f"  Combined {partition} partition index {index}")

    # Save mapping file
    output_file = workspace / cfg.paths.preprocessing_pickle_file
    logger.info(f"Saving mapping to {output_file}...")

    use_lightweight_format = getattr(cfg.preprocessing, "use_lightweight_format", False)
    if use_lightweight_format:
        logger.info("Saving in lightweight format")
    else:
        logger.info("Saving in full format")

    write_data(
        file_path=str(output_file),
        molecule_pairs_train=molecule_pairs["_train"],
        molecule_pairs_val=molecule_pairs["_val"],
        molecule_pairs_test=molecule_pairs["_test"],
        uniformed_molecule_pairs_test=None,
        use_lightweight_format=use_lightweight_format,
        mgf_path=cfg.paths.spectra_path,
    )

    logger.info("✅ Preprocessing completed successfully!")
