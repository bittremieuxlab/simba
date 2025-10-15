import argparse
import logging
import os
import pickle
import random
import sys
from datetime import datetime

import dill
import numpy as np

# from simba.load_data import LoadData
from sklearn.model_selection import train_test_split

from simba.config import Config
from simba.loader_saver import LoaderSaver
from simba.logger_setup import logger
from simba.mces.mces_computation import MCES
from simba.parser import Parser
from simba.preprocessor import Preprocessor
from simba.simba.preprocessing_simba import PreprocessingSimba
from simba.train_utils import TrainUtils


def setup_config():
    """
    Setup configuration by parsing command-line arguments and updating the Config object.

    Returns
    -------
    Config
        Updated configuration object.
    """
    parser = argparse.ArgumentParser(
        description="distance computation script."
    )
    parser.add_argument(
        f"--spectra_path",
        type=str,
        default=None,
        help="Path to the spectra file.",
    )
    parser.add_argument(
        f"--workspace",
        type=str,
        default=None,
        help="Directory where distances will be saved.",
    )
    parser.add_argument(
        "--PREPROCESSING_OVERWRITE",
        action="store_true",
        help="Whether to overwrite existing preprocessing files.",
    )
    parser.add_argument(
        f"--MAX_SPECTRA_TRAIN",
        type=int,
        default=10000000000,
        help="Maximum number of training spectra.",
    )
    parser.add_argument(
        f"--MAX_SPECTRA_VAL",
        type=int,
        default=1000000000000,
        help="Maximum number of validation spectra.",
    )
    parser.add_argument(
        f"--MAX_SPECTRA_TEST",
        type=int,
        default=100000000000,
        help="Maximum number of test spectra.",
    )
    parser.add_argument(
        f"--mapping_file_name",
        type=str,
        default=None,
        help="Name of the mapping file that maps SMILES to spectra (will be saved in <workspace>/).",
    )
    parser.add_argument(
        f"--PREPROCESSING_NUM_WORKERS",
        type=int,
        default=None,
        help="Number of workers for preprocessing.",
    )
    parser.add_argument(
        "--PREPROCESSING_NUM_NODES",
        type=int,
        default=1,
        help="Number of nodes for preprocessing.",
    )
    parser.add_argument(
        "--PREPROCESSING_CURRENT_NODE",
        type=int,
        default=None,
        help="Current node for preprocessing (0-indexed).",
    )
    parser.add_argument(
        f"--VAL_SPLIT", type=float, default=0.1, help="Validation split ratio."
    )
    parser.add_argument(
        f"--TEST_SPLIT", type=float, default=0.1, help="Test split ratio."
    )
    args = parser.parse_args()

    # Parsing arguments
    config = Config()
    config.SPECTRA_PATH = args.spectra_path
    config.MOL_SPEC_MAPPING_FILE = args.mapping_file_name
    config.PREPROCESSING_DIR = args.workspace
    config.PREPROCESSING_OVERWRITE = args.PREPROCESSING_OVERWRITE
    config.PREPROCESSING_NUM_WORKERS = args.PREPROCESSING_NUM_WORKERS
    config.PREPROCESSING_NUM_NODES = args.PREPROCESSING_NUM_NODES
    config.PREPROCESSING_CURRENT_NODE = args.PREPROCESSING_CURRENT_NODE

    config.MAX_SPECTRA_TRAIN = args.MAX_SPECTRA_TRAIN
    config.MAX_SPECTRA_VAL = args.MAX_SPECTRA_VAL
    config.MAX_SPECTRA_TEST = args.MAX_SPECTRA_TEST
    config.VAL_SPLIT = args.VAL_SPLIT
    config.TEST_SPLIT = args.TEST_SPLIT

    return config


def setup_paths(config: Config):
    """
    Setup output paths based on the configuration.

    Parameters
    ----------
    config : Config
        Configuration object containing paths and settings.

    Returns
    -------
    tuple
        A tuple containing paths for various output files.
    """
    if config.RANDOM_MCES_SAMPLING:
        subfix = ""
    else:
        subfix = "_exhaustive"

    # output filenames
    output_pairs_file = config.PREPROCESSING_DIR + config.MOL_SPEC_MAPPING_FILE
    output_np_indexes_train = (
        config.PREPROCESSING_DIR + f"indexes_tani_mces_train{subfix}.npy"
    )
    output_np_indexes_val = (
        config.PREPROCESSING_DIR + f"indexes_tani_mces_val{subfix}.npy"
    )
    output_np_indexes_test = (
        config.PREPROCESSING_DIR + f"indexes_tani_mces_test{subfix}.npy"
    )
    output_nist_file = config.PREPROCESSING_DIR + f"all_spectrums_nist.pkl"
    output_neurips_file = (
        config.PREPROCESSING_DIR + f"all_spectrums_neurips.pkl"
    )
    output_spectrums_file = (
        config.PREPROCESSING_DIR + f"all_spectrums_neurips_nist_20240814.pkl"
    )
    logging.info(f"Pairs will be saved to {output_pairs_file}")

    return (
        output_pairs_file,
        output_np_indexes_train,
        output_np_indexes_val,
        output_np_indexes_test,
        output_nist_file,
        output_neurips_file,
        output_spectrums_file,
    )


def compute_distances(config: Config, pairs_filename: str):
    """
    Compute distances between spectra using MCES and optionally edit distance.

    Parameters
    ----------
    config : Config
        Configuration object containing paths and settings.
    pairs_filename : str
        Filename to save the computed pairs.
    """
    # Load and preprocess spectra
    all_spectra = PreprocessingSimba.load_spectra(
        config.SPECTRA_PATH, config, use_gnps_format=False
    )
    logger.info(f"Read {len(all_spectra)} spectra from {config.SPECTRA_PATH}")

    # Split data into train, validation, and test sets
    all_spectra_train, all_spectra_val, all_spectra_test = (
        TrainUtils.train_val_test_split_bms(
            all_spectra,
            val_split=config.VAL_SPLIT,
            test_split=config.TEST_SPLIT,
        )
    )
    all_spectra_train = all_spectra_train[0 : config.MAX_SPECTRA_TRAIN]
    all_spectra_val = all_spectra_val[0 : config.MAX_SPECTRA_VAL]
    all_spectra_test = all_spectra_test[0 : config.MAX_SPECTRA_TEST]
    logger.info(
        f"Train: {len(all_spectra_train)}, Val: {len(all_spectra_val)}, Test: {len(all_spectra_test)}"
    )

    molecule_pairs = {}
    for type_data in ["_train", "_val", "_test"]:

        if type_data == "_train":
            spectra = all_spectra_train
            logger.info(f"Computing distances for training set...")
        elif type_data == "_val":
            spectra = all_spectra_val
            logger.info(f"Computing distances for validation set...")
        elif type_data == "_test":
            spectra = all_spectra_test
            logger.info(f"Computing distances for test set...")

        for use_edit_distance in [False, True]:
            molecule_pairs[type_data] = MCES.compute_all_mces_results_unique(
                spectra,
                max_combinations=10000000000000,
                num_workers=config.PREPROCESSING_NUM_WORKERS,
                random_sampling=config.RANDOM_MCES_SAMPLING,
                config=config,
                identifier=type_data,
                use_edit_distance=use_edit_distance,
                loaded_molecule_pairs=None,
            )

    # Write mapping to file
    write_data(
        pairs_filename,
        all_spectra_train=None,
        all_spectra_val=None,
        all_spectra_test=None,
        molecule_pairs_train=molecule_pairs["_train"],
        molecule_pairs_val=molecule_pairs["_val"],
        molecule_pairs_test=molecule_pairs["_test"],
        uniformed_molecule_pairs_test=None,
    )
    pass


def write_data(
    file_path,
    all_spectra_train=None,
    all_spectra_val=None,
    all_spectra_test=None,
    molecule_pairs_train=None,
    molecule_pairs_val=None,
    molecule_pairs_test=None,
    uniformed_molecule_pairs_test=None,
):
    """
    Write the molecule -> spectra mapping to a pickle file.

    Parameters
    ----------
    file_path : str
        The path to the output file.
    all_spectra_train : list, optional
        List of training spectra, by default None.
    all_spectra_val : list, optional
        List of validation spectra, by default None.
    all_spectra_test : list, optional
        List of test spectra, by default None.
    molecule_pairs_train : dict, optional
        Dictionary of training molecule pairs, by default None.
    molecule_pairs_val : dict, optional
        Dictionary of validation molecule pairs, by default None.
    molecule_pairs_test : dict, optional
        Dictionary of test molecule pairs, by default None.
    uniformed_molecule_pairs_test : dict, optional
        Dictionary of uniformed test molecule pairs, by default None.
    """
    dataset = {
        "all_spectrums_train": all_spectra_train,
        "all_spectrums_val": all_spectra_val,
        "all_spectrums_test": all_spectra_test,
        "molecule_pairs_train": molecule_pairs_train,
        "molecule_pairs_val": molecule_pairs_val,
        "molecule_pairs_test": molecule_pairs_test,
        "uniformed_molecule_pairs_test": uniformed_molecule_pairs_test,
    }
    logger.info(f"Writing molecule to spectra mapping to {file_path}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(dataset, file)


def combine_distances(config: Config):
    """
    Combine edit distance and MCES distances from precomputed files.

    Parameters
    ----------
    config : Config
        Configuration object containing paths and settings.
    """
    logger.info("Combining edit distance and MCES distances...")

    all_files = os.listdir(config.PREPROCESSING_DIR)
    for partition in ["train", "val", "test"]:
        # Filter only the files ending with '.npy'
        npy_files = [f for f in all_files if f.endswith(".npy")]
        npy_files = [f for f in npy_files if partition in f]
        for file_loaded in npy_files:
            # get the index
            index = file_loaded.split("_")[-1].split(".npy")[0]
            index = int(index)

            logger.info(f"Loading file index {index}...")

            # define file names
            sufix = "indexes_tani_incremental_" + partition + "_"
            file_ed = (
                config.PREPROCESSING_DIR
                + "edit_distance_"
                + sufix
                + str(index)
                + ".npy"
            )
            file_mces = (
                config.PREPROCESSING_DIR
                + "mces_"
                + sufix
                + str(index)
                + ".npy"
            )
            file_output = (
                config.PREPROCESSING_DIR
                + "ed_mces_"
                + sufix
                + str(index)
                + ".npy"
            )

            # load data
            ed_data = np.load(file_ed)
            try:
                mces_data = np.load(file_mces)
            except:
                logger.warning(f"No MCES distance file found at {file_mces}.")
                continue

            # check that the indexes are the same:
            if np.all(ed_data[:, 0] == mces_data[:, 0]) and np.all(
                ed_data[:, 1] == mces_data[:, 1]
            ):
                logger.info(
                    "The MCES and ED data corresponds to the same pairs"
                )

                all_distance_data = np.column_stack((ed_data, mces_data[:, 2]))
                all_distance_data = preprocess_distances(
                    all_distance_data, config
                )
                np.save(
                    file_output,
                    all_distance_data,
                )
            else:
                logger.error(
                    "The data loaded does not correspond to the same pairs"
                )


def preprocess_distances(array: np.ndarray, config: Config) -> np.ndarray:
    """
    Preprocess the input array by handling invalid values and filtering based on edit distance and MCES20.

    Parameters
    ----------
    array : np.ndarray
        Input array with shape (n_samples, n_features) where specific columns correspond to edit distance
        and MCES20 values.

    Returns
    -------
    np.ndarray
        Preprocessed array with invalid entries removed.
    """
    # extract ED and MCES20 columns
    ed = array[:, config.COLUMN_EDIT_DISTANCE]
    mces = array[:, config.COLUMN_MCES20]

    indexes_unvalid = np.isnan(ed) | np.isnan(mces)

    # identify rows with invalid values
    ed_exceeded = array[:, config.COLUMN_EDIT_DISTANCE] == 666
    mces_exceeded = array[:, config.COLUMN_MCES20] == 666

    # remove invalid values
    array = array[(~indexes_unvalid) & (~ed_exceeded) & (~mces_exceeded)]
    return array


if __name__ == "__main__":

    config = setup_config()
    output_paths = setup_paths(config)

    if config.PREPROCESSING_OVERWRITE:
        logger.info("Removing existing distance files...")
        if os.path.exists(config.PREPROCESSING_DIR):
            for file in os.listdir(config.PREPROCESSING_DIR):
                file_path = os.path.join(config.PREPROCESSING_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {e}")

    # to get more high similarity pairs
    USE_ONLY_LOW_RANGE = True
    high_tanimoto_range = 0 if USE_ONLY_LOW_RANGE else 0.5

    use_tqdm = config.enable_progress_bar

    compute_distances(config, output_paths[0])
    combine_distances(config)

    logger.info("All done!")
