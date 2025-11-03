import os
import random
import sys
from typing import List, Tuple

import dill
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.callbacks import ProgressBar
from torch.utils.data import DataLoader

import simba
from simba.config import Config
from simba.load_mces.load_mces import LoadMCES
from simba.logger_setup import logger
from simba.losscallback import LossCallback
from simba.ordinal_classification.embedder_multitask import EmbedderMultitask
from simba.ordinal_classification.load_data_multitasking import (
    LoadDataMultitasking,
)
from simba.parser import Parser
from simba.plotting import Plotting
from simba.sanity_checks import SanityChecks
from simba.train_utils import TrainUtils
from simba.transformers.embedder import Embedder
from simba.weight_sampling import WeightSampling
from simba.weight_sampling_tools.custom_weighted_random_sampler import (
    CustomWeightedRandomSampler,
)

# In case a MAC is being used:
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def setup_config():
    config = Config()
    parser = Parser()
    config = parser.update_config(config)
    config.bins_uniformise_INFERENCE = config.EDIT_DISTANCE_N_CLASSES - 1
    config.use_uniform_data_INFERENCE = True
    return config


def setup_paths(config):
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    mapping_path = (
        config.PREPROCESSING_DIR_TRAIN + config.PREPROCESSING_PICKLE_FILE
    )
    fig_path = config.CHECKPOINT_DIR + f"scatter_plot_{config.MODEL_CODE}.png"
    return mapping_path, fig_path


def load_dataset(mapping_path):
    logger.info(f"Loading mapping file from {mapping_path}")
    # Load the dataset from the pickle file
    sys.modules["src"] = simba
    with open(mapping_path, "rb") as file:
        mapping = dill.load(file)

    molecule_pairs_train = mapping["molecule_pairs_train"]
    molecule_pairs_val = mapping["molecule_pairs_val"]
    molecule_pairs_test = mapping["molecule_pairs_test"]
    uniformed_molecule_pairs_test = mapping["uniformed_molecule_pairs_test"]
    return (
        molecule_pairs_train,
        molecule_pairs_val,
        molecule_pairs_test,
        uniformed_molecule_pairs_test,
    )


def prepare_data(
    molecule_pairs_train,
    molecule_pairs_val,
    molecule_pairs_test,
    uniformed_molecule_pairs_test,
    config,
):
    logger.info("Loading pairs data ...")
    indexes_tani_multitasking_train = LoadMCES.merge_numpy_arrays(
        config.PREPROCESSING_DIR_TRAIN,
        prefix="ed_mces_indexes_tani_incremental_train",
        use_edit_distance=config.USE_EDIT_DISTANCE,
        use_multitask=config.USE_MULTITASK,
        add_high_similarity_pairs=config.ADD_HIGH_SIMILARITY_PAIRS,
        remove_percentage=0.0,
    )

    indexes_tani_multitasking_train = remove_duplicates_array(
        indexes_tani_multitasking_train
    )

    indexes_tani_multitasking_val = LoadMCES.merge_numpy_arrays(
        config.PREPROCESSING_DIR_TRAIN,
        prefix="ed_mces_indexes_tani_incremental_val",
        use_edit_distance=config.USE_EDIT_DISTANCE,
        use_multitask=config.USE_MULTITASK,
        add_high_similarity_pairs=config.ADD_HIGH_SIMILARITY_PAIRS,
    )

    indexes_tani_multitasking_val = remove_duplicates_array(
        indexes_tani_multitasking_val
    )

    # assign edit distance
    molecule_pairs_train.pair_distances = indexes_tani_multitasking_train[
        :, [0, 1, config.COLUMN_EDIT_DISTANCE]
    ]
    molecule_pairs_val.pair_distances = indexes_tani_multitasking_val[
        :, [0, 1, config.COLUMN_EDIT_DISTANCE]
    ]
    # add MCES
    molecule_pairs_train.extra_distances = indexes_tani_multitasking_train[
        :, config.COLUMN_MCES20
    ]
    molecule_pairs_val.extra_distances = indexes_tani_multitasking_val[
        :, config.COLUMN_MCES20
    ]
    logger.info(f"Number of pairs for train: {len(molecule_pairs_train)}")
    logger.info(f"Number of pairs for val: {len(molecule_pairs_val)}")

    # Sanity checks
    sanity_check_ids = SanityChecks.sanity_checks_ids(
        molecule_pairs_train,
        molecule_pairs_val,
        molecule_pairs_test,
        uniformed_molecule_pairs_test,
    )
    sanity_check_bms = SanityChecks.sanity_checks_bms(
        molecule_pairs_train,
        molecule_pairs_val,
        molecule_pairs_test,
        uniformed_molecule_pairs_test,
    )
    logger.info(f"Sanity check ids. Passed? {sanity_check_ids}")
    logger.info(f"Sanity check bms. Passed? {sanity_check_bms}")

    # calculate weights for the training set
    train_binned_list, ranges = TrainUtils.divide_data_into_bins_categories(
        molecule_pairs_train,
        config.EDIT_DISTANCE_N_CLASSES - 1,
        bin_sim_1=True,
    )

    weights_ed, bins_ed = WeightSampling.compute_weights(train_binned_list)
    weights_tr = WeightSampling.compute_sample_weights_categories(
        molecule_pairs_train, weights_ed
    )
    weights_val = WeightSampling.compute_sample_weights_categories(
        molecule_pairs_val, weights_ed
    )

    dataset_train = LoadDataMultitasking.from_molecule_pairs_to_dataset(
        molecule_pairs_train,
        max_num_peaks=int(config.TRANSFORMER_CONTEXT),
        training=True,
    )
    # dataset_test = LoadData.from_molecule_pairs_to_dataset(m_test)
    dataset_val = LoadDataMultitasking.from_molecule_pairs_to_dataset(
        molecule_pairs_val, max_num_peaks=int(config.TRANSFORMER_CONTEXT)
    )

    train_sampler = CustomWeightedRandomSampler(
        weights=weights_tr, num_samples=len(dataset_train), replacement=True
    )
    val_sampler = CustomWeightedRandomSampler(
        weights=weights_val, num_samples=len(dataset_val), replacement=True
    )
    return (
        dataset_train,
        train_sampler,
        dataset_val,
        val_sampler,
        weights_ed,
        bins_ed,
    )


# Initialize a set to track unique first two columns
def remove_duplicates_array(array):
    seen = set()
    filtered_rows = []

    for row in array:
        # Create a tuple of the first two columns to check uniqueness
        key = tuple(sorted(row[:2]))  # Sort to account for unordered pairs
        if key not in seen:
            seen.add(key)
            filtered_rows.append(row)

    # Convert the filtered rows back to a NumPy array
    result = np.array(filtered_rows)
    return result


def create_dataloaders(
    config, dataset_train, train_sampler, dataset_val, val_sampler
):

    logger.info("Creating the data loader for training")
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=config.TRAINING_NUM_WORKERS,
    )
    logger.info("Creating the data loader for validation")
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.BATCH_SIZE,
        sampler=val_sampler,
        worker_init_fn=worker_init_fn,
        num_workers=config.TRAINING_NUM_WORKERS,
    )
    return dataloader_train, dataloader_val


def worker_init_fn(
    worker_id,
):  # ensure the dataloader for validation is the same for every epoch
    seed = 42
    torch.manual_seed(seed)
    # Set the same seed for reproducibility in NumPy and Python's random module
    np.random.seed(seed)
    random.seed(seed)


def check_similarity_distribution(
    dataloader_train: DataLoader,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Check the distribution of similarities in the dataloader.

    Parameters
    ----------
    dataloader_train: DataLoader
        The dataloader to check.

    Returns
    -------
    ed_sampled: List[int]
        List of edit distances sampled from the dataloader.
    mces_sampled: List[float]
        List of MCES sampled from the dataloader.
    bins_mces_normalized: List[float]
        List of bins for MCES.
    weights_mces: List[float]
        List of weights for MCES.
    """
    # check that the distribution of the loader is balanced
    ed_sampled = []
    mces_sampled = []
    for i, batch in enumerate(dataloader_train):
        # ed = batch['ed']
        # ed = np.array(ed).reshape(-1)
        ed_sampled = ed_sampled + list(batch["ed"].reshape(-1))
        mces_sampled = mces_sampled + list(batch["mces"].reshape(-1))
        if i == 100:
            # for second similarity remove the sim=1 since it is the same task as the edit distance ==0
            mces_sampled = np.array(mces_sampled)
            # mces_sampled = mces_sampled[mces_sampled < 1]
            break

    # counting_ed, bins_ed, _ = plt.hist(ed_sampled, bins=6)

    # count the number of samples in each MCES bin
    counting_mces, bins_mces = TrainUtils.count_ranges(
        np.array(mces_sampled),
        number_bins=5,
        bin_sim_1=False,
        max_value=1,
    )

    weights_mces = np.array(
        [np.sum(counting_mces) / c if c != 0 else 0 for c in counting_mces]
    )
    weights_mces = weights_mces / np.sum(weights_mces)

    # save info about the weights of similarity 1
    bins_mces_normalized = [
        b if b > 0 else 0 for b in bins_mces
    ]  # the first bin has -inf as the lower range

    return (
        ed_sampled,
        mces_sampled,
        bins_mces_normalized,
        weights_mces,
    )


def plot_weights(weights_ed, bins_ed, weights_mces, bins_mces, config):
    Plotting.plot_weights(
        bins_ed,
        weights_ed,
        xlabel="weight bin similarity 1",
        filepath=config.CHECKPOINT_DIR + "weights_ed.png",
    )
    # plot MCES weights
    Plotting.plot_weights(
        bins_mces,
        weights_mces,
        xlabel="weight bin MCES",
        filepath=config.CHECKPOINT_DIR + "weights_mces.png",
    )


def plot_similarity_distribution(
    ed_sampled: List[int],
    mces_sampled: List[float],
    config: Config,
):
    """
    Plot the distribution of similarities in the dataloader.

    Parameters
    ----------
    ed_sampled: List[int]
        List of edit distances sampled from the dataloader.
    mces_sampled: List[float]
        List of MCES sampled from the dataloader.
    config: Config
        The configuration object.
    """
    # plot similarity distributions
    plt.figure()
    plt.xlabel("edit distance")
    plt.ylabel("freq")
    plt.hist(ed_sampled)
    plt.savefig(config.CHECKPOINT_DIR + "similarity_distribution_ed.png")

    plt.figure()
    plt.hist(mces_sampled)
    plt.xlabel("MCES")
    plt.ylabel("freq")
    plt.savefig(config.CHECKPOINT_DIR + "similarity_distribution_mces.png")


def setup_callbacks(config):
    # Define the ModelCheckpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        filename="best_model",
        monitor="validation_loss_epoch",
        mode="min",
        save_top_k=1,
    )

    checkpoint_n_steps_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        filename="best_model_n_steps",
        every_n_train_steps=1000,
        save_last=True,
        save_top_k=1,
    )
    # checkpoint_callback = SaveBestModelCallback(file_path=config.best_model_path)
    progress_bar_callback = ProgressBar()

    # loss callback
    losscallback = LossCallback(file_path=config.CHECKPOINT_DIR + f"loss.png")
    return checkpoint_callback, checkpoint_n_steps_callback, losscallback


def setup_model(config, weights_mces):
    logger.info("Setup model...")
    ## use or not use weights for the second similarity loss
    if config.USE_LOSS_WEIGHTS_SECOND_SIMILARITY:
        weights_mces = np.array(weights_mces)
    else:
        weights_mces = None

    model = EmbedderMultitask(
        d_model=int(config.D_MODEL),
        n_layers=int(config.N_LAYERS),
        n_classes=config.EDIT_DISTANCE_N_CLASSES,
        weights=None,
        lr=config.LR,
        use_cosine_distance=config.use_cosine_distance,
        use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
        weights_sim2=weights_mces,
        use_mces20_log_loss=config.USE_MCES20_LOG_LOSS,
        use_edit_distance_regresion=config.USE_EDIT_DISTANCE_REGRESSION,
        use_precursor_mz_for_model=config.USE_PRECURSOR_MZ_FOR_MODEL,
        tau_gumbel_softmax=config.TAU_GUMBEL_SOFTMAX,
        gumbel_reg_weight=config.GUMBEL_REG_WEIGHT,
    )

    # Create a model:
    if config.load_pretrained:
        # Try to load the full model, otherwise load the encoders only
        try:
            model = EmbedderMultitask.load_from_checkpoint(
                config.pretrained_path,
                d_model=int(config.D_MODEL),
                n_layers=int(config.N_LAYERS),
                n_classes=config.EDIT_DISTANCE_N_CLASSES,
                weights=None,
                lr=config.LR,
                use_cosine_distance=config.use_cosine_distance,
                use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
                weights_sim2=weights_mces,
                use_mces20_log_loss=config.USE_MCES20_LOG_LOSS,
                use_edit_distance_regresion=config.USE_EDIT_DISTANCE_REGRESSION,
                use_precursor_mz_for_model=config.USE_PRECURSOR_MZ_FOR_MODEL,
                tau_gumbel_softmax=config.TAU_GUMBEL_SOFTMAX,
                gumbel_reg_weight=config.GUMBEL_REG_WEIGHT,
            )
            logger.info("Loaded full model from checkpoint")
        except:
            model_pretrained = Embedder.load_from_checkpoint(
                config.pretrained_path,
                d_model=int(config.D_MODEL),
                n_layers=int(config.N_LAYERS),
                weights=None,
                lr=config.LR,
                use_cosine_distance=config.use_cosine_distance,
                use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
                weights_sim2=weights_mces,
                strict=False,
            )

            model.spectrum_encoder = model_pretrained.spectrum_encoder
            logger.info("Loaded encoder model from checkpoint")
    else:
        logger.info("No pretrained model loaded")

    return model


def train(
    model,
    dataloader_train,
    dataloader_val,
    config,
    checkpoint_callback,
    checkpoint_n_steps_callback,
    losscallback,
):
    trainer = pl.Trainer(
        # max_steps=100000,
        val_check_interval=config.VAL_CHECK_INTERVAL,
        max_epochs=config.epochs,
        callbacks=[
            checkpoint_callback,
            checkpoint_n_steps_callback,
            losscallback,
        ],
        enable_progress_bar=config.enable_progress_bar,
        accelerator=config.ACCELERATOR,
        # val_check_interval=config.validate_after_ratio,
        strategy="ddp_find_unused_parameters_true",
    )
    trainer.fit(
        model=model,
        train_dataloaders=(dataloader_train),
        val_dataloaders=dataloader_val,
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    logger.info("Training finished")


if __name__ == "__main__":
    config = setup_config()
    dataset_path, fig_path = setup_paths(config)

    # prepare data for training
    (
        molecule_pairs_train,
        molecule_pairs_val,
        molecule_pairs_test,
        uniformed_molecule_pairs_test,
    ) = load_dataset(dataset_path)
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
    dataloader_train, dataloader_val = create_dataloaders(
        config, dataset_train, train_sampler, dataset_val, val_sampler
    )
    (
        ed_sampled,
        mces_sampled,
        bins_mces,
        weights_mces,
    ) = check_similarity_distribution(dataloader_train)

    # make some plots
    plot_weights(weights_ed, bins_ed, weights_mces, bins_mces, config)
    plot_similarity_distribution(ed_sampled, mces_sampled, config)

    # setup training
    checkpoint_callback, checkpoint_n_steps_callback, losscallback = (
        setup_callbacks(config)
    )
    model = setup_model(config, weights_mces)
    train(
        model,
        dataloader_train,
        dataloader_val,
        config,
        checkpoint_callback,
        checkpoint_n_steps_callback,
        losscallback,
    )
