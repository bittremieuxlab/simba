"""Inference command for SIMBA CLI."""

import os
from pathlib import Path

import click


@click.command()
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing model checkpoints (e.g., best_model.ckpt or last.ckpt).",
)
@click.option(
    "--preprocessing-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing preprocessed data.",
)
@click.option(
    "--preprocessing-pickle",
    type=str,
    default="preprocessed_data.pkl",
    help="Name of the pickle file containing the preprocessed dataset.",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size for inference.",
)
@click.option(
    "--accelerator",
    type=click.Choice(["cpu", "gpu", "auto"], case_sensitive=False),
    default="auto",
    help="Hardware accelerator for inference (cpu, gpu, or auto).",
)
@click.option(
    "--use-last-model",
    is_flag=True,
    default=False,
    help="Use last.ckpt instead of best_model.ckpt.",
)
@click.option(
    "--uniformize-testing/--no-uniformize-testing",
    default=True,
    help="Whether to uniformize pairs across bins during testing.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Directory to save output plots and results. Defaults to checkpoint-dir.",
)
def inference(
    checkpoint_dir: Path,
    preprocessing_dir: Path,
    preprocessing_pickle: str,
    batch_size: int,
    accelerator: str,
    use_last_model: bool,
    uniformize_testing: bool,
    output_dir: Path | None,
) -> None:
    """Run inference on test data using a trained SIMBA model.

    This command loads a trained model from CHECKPOINT_DIR and runs inference on test data
    from the preprocessed dataset. It generates correlation metrics and visualization plots.

    Example:

        simba inference --checkpoint-dir ./checkpoints --preprocessing-dir ./preprocessed_data
    """
    # Lazy imports to speed up CLI
    import copy
    import sys

    import dill
    import lightning.pytorch as pl
    import numpy as np
    from scipy.stats import spearmanr
    from torch.utils.data import DataLoader

    import simba
    from simba.config import Config
    from simba.core.chemistry.mces_loader.load_mces import LoadMCES
    from simba.logger_setup import logger
    from simba.ordinal_classification.embedder_multitask import EmbedderMultitask
    from simba.ordinal_classification.load_data_multitasking import (
        LoadDataMultitasking,
    )
    from simba.train_utils import TrainUtils
    from simba.transformers.postprocessing import Postprocessing

    sys.modules["src"] = simba

    # Setup configuration (no Parser to avoid argparse conflict with Click)
    config = Config()

    # Override config with CLI parameters
    config.CHECKPOINT_DIR = str(checkpoint_dir) + os.sep
    config.PREPROCESSING_DIR = str(preprocessing_dir) + os.sep
    config.PREPROCESSING_DIR_TRAIN = (
        str(preprocessing_dir) + os.sep
    )  # Same as PREPROCESSING_DIR for inference
    config.PREPROCESSING_DIR_VAL_TEST = (
        str(preprocessing_dir) + os.sep
    )  # Same as PREPROCESSING_DIR for inference
    config.PREPROCESSING_PICKLE_FILE = preprocessing_pickle
    config.BATCH_SIZE = batch_size
    config.INFERENCE_USE_LAST_MODEL = use_last_model
    config.UNIFORMIZE_DURING_TESTING = uniformize_testing
    config.bins_uniformise_INFERENCE = config.EDIT_DISTANCE_N_CLASSES - 1
    config.use_uniform_data_INFERENCE = True

    # Set output directory
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_dir) + os.sep
    else:
        output_dir = config.CHECKPOINT_DIR

    # Validate paths
    dataset_path = config.PREPROCESSING_DIR + config.PREPROCESSING_PICKLE_FILE
    if not os.path.exists(dataset_path):
        raise click.UsageError(
            f"Dataset file not found: {dataset_path}\n"
            f"Please check that --preprocessing-dir and --preprocessing-pickle are correct."
        )

    # Determine model path
    if not use_last_model and os.path.exists(
        config.CHECKPOINT_DIR + config.BEST_MODEL_NAME
    ):
        best_model_path = config.CHECKPOINT_DIR + config.BEST_MODEL_NAME
        model_name = "best model"
    else:
        best_model_path = config.CHECKPOINT_DIR + "last.ckpt"
        model_name = "last checkpoint"

    if not os.path.exists(best_model_path):
        raise click.UsageError(
            f"Model checkpoint not found: {best_model_path}\n"
            f"Please ensure the checkpoint directory contains a valid model file."
        )

    click.echo(f"Using {model_name}: {best_model_path}")
    click.echo(f"Loading dataset from: {dataset_path}")

    # Load dataset
    logger.info("Loading the dataset...")
    with open(dataset_path, "rb") as file:
        dataset = dill.load(file)

    molecule_pairs = dataset["molecule_pairs_test"]
    molecule_pairs_ed = copy.deepcopy(molecule_pairs)
    molecule_pairs_mces = copy.deepcopy(molecule_pairs)

    # Prepare data
    logger.info("Preparing data...")
    pair_distances = LoadMCES.merge_numpy_arrays(
        config.PREPROCESSING_DIR_TRAIN,
        prefix="ed_mces_indexes_tani_incremental_test",
        use_edit_distance=config.USE_EDIT_DISTANCE,
        use_multitask=config.USE_MULTITASK,
    )
    pair_distances = _remove_duplicate_pairs(pair_distances)

    molecule_pairs_ed.pair_distances = pair_distances[:, 0:3]
    molecule_pairs_ed.extra_distances = pair_distances[:, 3]
    logger.info(f"{len(molecule_pairs_ed)} pairs remain after removing duplicates")

    molecule_pairs_mces.pair_distances = pair_distances[:, [0, 1, 3]]
    molecule_pairs_mces.extra_distances = pair_distances[:, 3]

    if config.UNIFORMIZE_DURING_TESTING:
        logger.info("Uniformize pairs across bins...")
        molecule_pairs_ed_uniform, _ = TrainUtils.uniformise(
            molecule_pairs_ed,
            number_bins=config.bins_uniformise_INFERENCE,
            return_binned_list=True,
            bin_sim_1=True,
            ordinal_classification=True,
        )
        molecule_pairs_mces_uniform, _ = TrainUtils.uniformise(
            molecule_pairs_mces,
            number_bins=config.bins_uniformise_INFERENCE,
            return_binned_list=True,
            bin_sim_1=False,
        )
    else:
        molecule_pairs_ed_uniform = molecule_pairs_ed
        molecule_pairs_mces_uniform = molecule_pairs_mces

    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataset_ed = LoadDataMultitasking.from_molecule_pairs_to_dataset(
        molecule_pairs_ed_uniform,
        max_num_peaks=int(config.TRANSFORMER_CONTEXT),
        use_adduct=config.USE_ADDUCT,
    )
    dataloader_ed = DataLoader(dataset_ed, batch_size=config.BATCH_SIZE, shuffle=False)

    dataset_mces = LoadDataMultitasking.from_molecule_pairs_to_dataset(
        molecule_pairs_mces_uniform,
        max_num_peaks=int(config.TRANSFORMER_CONTEXT),
        use_adduct=config.USE_ADDUCT,
    )
    dataloader_mces = DataLoader(
        dataset_mces, batch_size=config.BATCH_SIZE, shuffle=False
    )

    # Load model
    logger.info(f"Loading model from {best_model_path}...")
    load_kwargs = {
        "d_model": int(config.D_MODEL),
        "n_layers": int(config.N_LAYERS),
        "n_classes": config.EDIT_DISTANCE_N_CLASSES,
        "use_gumbel": config.EDIT_DISTANCE_USE_GUMBEL,
        "use_element_wise": True,
        "use_cosine_distance": config.use_cosine_distance,
        "use_edit_distance_regresion": config.USE_EDIT_DISTANCE_REGRESSION,
        "strict": False,
        "use_adduct": config.USE_ADDUCT,
        "categorical_adducts": config.CATEGORICAL_ADDUCTS,
        "adduct_mass_map": config.ADDUCT_MASS_MAP_CSV,
        "use_ce": config.USE_CE,
        "use_ion_activation": config.USE_ION_ACTIVATION,
        "use_ion_method": config.USE_ION_METHOD,
    }
    best_model = EmbedderMultitask.load_from_checkpoint(best_model_path, **load_kwargs)
    best_model.eval()

    # Check if dataloaders have data
    if len(dataloader_ed) == 0:
        raise click.UsageError(
            "No test data found! The dataloader is empty. "
            "This could mean:\n"
            "  1. No test data was created during preprocessing\n"
            "  2. All test pairs were filtered out during uniformization\n"
            "  3. The preprocessing-dir path is incorrect"
        )

    click.echo(
        f"Test data: {len(dataloader_ed)} batches for ED, {len(dataloader_mces)} batches for MCES"
    )

    # Run inference
    logger.info("Running inference...")
    trainer = pl.Trainer(
        max_epochs=2,
        enable_progress_bar=config.enable_progress_bar,
        devices=1,
        accelerator=accelerator,
    )

    pred_ed = trainer.predict(best_model, dataloader_ed)
    pred_mces = trainer.predict(best_model, dataloader_mces)

    # Check if predictions are valid
    if pred_ed is None or len(pred_ed) == 0:
        raise click.UsageError(
            "Inference failed: No predictions generated for edit distance. "
            "This could be a model or data issue."
        )
    if pred_mces is None or len(pred_mces) == 0:
        raise click.UsageError(
            "Inference failed: No predictions generated for MCES. "
            "This could be a model or data issue."
        )

    # Evaluate predictions
    logger.info("Evaluating predictions...")
    ed_true, _ = Postprocessing.get_similarities_multitasking(dataloader_ed)
    _, mces_true = Postprocessing.get_similarities_multitasking(dataloader_mces)

    # Flatten predictions
    pred_mces_mces_flat = [[p.item() for p in pred[1]] for pred in pred_mces]
    pred_mces_mces_flat = [item for sublist in pred_mces_mces_flat for item in sublist]
    pred_mces_mces_flat = np.array(pred_mces_mces_flat)

    pred_ed_ed_flat = [p[0] for p in pred_ed]
    pred_ed_ed_flat = [[_which_index(p) for p in p_list] for p_list in pred_ed_ed_flat]
    pred_ed_ed_flat = [item for sublist in pred_ed_ed_flat for item in sublist]
    pred_ed_ed_flat = np.array(pred_ed_ed_flat, dtype=float)

    # Clean data
    ed_true = np.array(ed_true)
    mces_true = np.array(mces_true)

    mask = ~np.isnan(pred_ed_ed_flat)
    ed_true_clean = ed_true[mask]
    pred_ed_ed_clean = pred_ed_ed_flat[mask]

    # Edit distance correlation
    corr_model_ed, _ = spearmanr(ed_true_clean, pred_ed_ed_clean)
    click.echo(f"\n✓ Edit distance correlation: {corr_model_ed:.4f}")

    # Plot confusion matrix
    _plot_cm(ed_true_clean, pred_ed_ed_clean, config, output_dir)

    # MCES evaluation
    counts, bins = TrainUtils.count_ranges(
        mces_true, number_bins=5, bin_sim_1=False, max_value=1
    )

    click.echo("\nMCES Statistics:")
    click.echo(f"  Max value: {max(mces_true):.4f}")
    click.echo(f"  Min value: {min(mces_true):.4f}")
    click.echo(f"  Samples per bin: {counts}")

    # Remove threshold values
    mces_true_original = mces_true.copy()
    mces_true = mces_true[mces_true_original != 0.5]
    pred_mces_mces_flat = pred_mces_mces_flat[mces_true_original != 0.5]
    corr_model_mces, _ = spearmanr(mces_true, pred_mces_mces_flat)

    click.echo(f"✓ MCES/Tanimoto correlation: {corr_model_mces:.4f}")

    # Denormalize if using MCES20
    if not config.USE_TANIMOTO:
        mces_true = config.MCES20_MAX_VALUE * (1 - mces_true)
        pred_mces_mces_flat = config.MCES20_MAX_VALUE * (1 - pred_mces_mces_flat)

    # Plot performance
    _plot_performance(mces_true, pred_mces_mces_flat, config, output_dir)

    click.echo(f"\n✓ Results saved to: {output_dir}")
    click.echo(f"  - Confusion matrix: {output_dir}cm.png")
    click.echo(f"  - Hexbin plot: {output_dir}hexbin_plot_{config.MODEL_CODE}.png")
    click.echo(f"  - Scatter plot: {output_dir}scatter_plot_{config.MODEL_CODE}.png")


def _remove_duplicate_pairs(array):
    """Remove duplicate pairs from array."""
    import numpy as np

    seen = set()
    filtered_rows = []

    for row in array:
        key = tuple(sorted(row[:2]))
        if key not in seen:
            seen.add(key)
            filtered_rows.append(row)

    return np.array(filtered_rows)


def _which_index(p) -> int:
    """Get the index of the maximum value."""
    import numpy as np

    return np.argmax(p)


def _plot_cm(
    true,
    preds,
    config,
    output_dir: str,
    file_name: str = "cm.png",
) -> None:
    """Plot confusion matrix."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix

    cm = confusion_matrix(true, preds)
    accuracy = accuracy_score(true, preds)

    # Normalize
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create plot
    plt.figure(figsize=(10, 7))
    labels = [">5", "4", "3", "2", "1", "0"]

    im = plt.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    cbar = plt.colorbar(im)
    cbar.set_label("Normalized frequency", fontsize=15)

    threshold = cm_normalized.max() / 2.0

    # Annotate cells
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            text_color = "white" if cm_normalized[i, j] > threshold else "black"
            plt.text(
                j,
                i,
                f"{cm_normalized[i, j]:.0%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=15,
            )

    plt.xticks(ticks=np.arange(len(labels)), labels=labels, fontsize=15)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=15)
    plt.xlabel("Substructure edit distance - Prediction", fontsize=15)
    plt.ylabel("Substructure edit distance - Ground truth", fontsize=15)
    plt.title(
        f"Confusion Matrix (Normalized), Acc: {accuracy:.2f}, Samples: {preds.shape[0]}",
        fontsize=15,
    )

    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()


def _plot_performance(mces_true, mces_pred, config, output_dir: str) -> None:
    """Plot performance metrics."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr

    corr_mces, _ = spearmanr(mces_true, mces_pred)
    sns.set_theme(style="ticks")

    # Hexbin plot
    plot = sns.jointplot(
        x=mces_true,
        y=mces_pred,
        kind="hex",
        color="#4CB391",
        joint_kws={"alpha": 1, "gridsize": 15},
    )
    plot.set_axis_labels("Ground truth Similarity", "Prediction", fontsize=12)
    plot.fig.suptitle(f"Spearman Correlation:{corr_mces:.4f}", fontsize=16)
    plot.ax_joint.set_xlim(0, 40)
    plot.ax_joint.set_ylim(0, 40)
    plt.tight_layout()
    plt.savefig(output_dir + f"hexbin_plot_{config.MODEL_CODE}.png")
    plt.close()

    # Scatter plot
    plt.figure()
    plt.scatter(mces_true, mces_pred, alpha=0.5)
    plt.xlabel("ground truth")
    plt.ylabel("prediction")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_dir + f"scatter_plot_{config.MODEL_CODE}.png")
    plt.close()
