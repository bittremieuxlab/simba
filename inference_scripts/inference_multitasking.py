import copy
import os
import sys
from typing import Tuple

import dill
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

import simba
from simba.config import Config
from simba.load_mces.load_mces import LoadMCES
from simba.logger_setup import logger
from simba.molecule_pairs_opt import MoleculePairsOpt
from simba.ordinal_classification.embedder_multitask import EmbedderMultitask
from simba.ordinal_classification.load_data_multitasking import (
    LoadDataMultitasking,
)
from simba.parser import Parser
from simba.train_utils import TrainUtils
from simba.transformers.postprocessing import Postprocessing

sys.modules["src"] = simba


def setup_config():
    config = Config()
    parser = Parser()
    config = parser.update_config(config)

    config.bins_uniformise_INFERENCE = config.EDIT_DISTANCE_N_CLASSES - 1
    config.use_uniform_data_INFERENCE = True

    return config


def setup_paths(config: Config):
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)

    dataset_path = config.PREPROCESSING_DIR + config.PREPROCESSING_PICKLE_FILE
    fig_path = config.CHECKPOINT_DIR + f"scatter_plot_{config.MODEL_CODE}.png"

    return dataset_path, fig_path


def load_dataset(dataset_path: str):

    logger.info("Loading the dataset...")
    # Load the dataset from the pickle file
    with open(dataset_path, "rb") as file:
        dataset = dill.load(file)

    molecule_pairs = dataset["molecule_pairs_test"]

    molecule_pairs_ed = copy.deepcopy(molecule_pairs)
    molecule_pairs_mces = copy.deepcopy(molecule_pairs)

    return molecule_pairs_ed, molecule_pairs_mces


def prepare_data(
    molecule_pairs_ed: MoleculePairsOpt,
    molecule_pairs_mces: MoleculePairsOpt,
    config: Config,
):
    pair_distances = LoadMCES.merge_numpy_arrays(
        config.PREPROCESSING_DIR_TRAIN,
        prefix="ed_mces_indexes_tani_incremental_test",
        use_edit_distance=config.USE_EDIT_DISTANCE,
        use_multitask=config.USE_MULTITASK,
    )
    pair_distances = remove_duplicate_pairs(pair_distances)

    molecule_pairs_ed.pair_distances = pair_distances[:, 0:3]
    molecule_pairs_ed.extra_distances = pair_distances[:, 3]
    logger.info(
        f"{len(molecule_pairs_ed)} pairs remain after removing duplicates"
    )

    molecule_pairs_mces.pair_distances = pair_distances[:, [0, 1, 3]]
    molecule_pairs_mces.extra_distances = pair_distances[:, 3]

    if config.UNIFORMIZE_DURING_TESTING:
        logger.info("Uniformize pairs across bins...")
        molecule_pairs_ed_uniform, binned_molecule_pairs_ed = (
            TrainUtils.uniformise(
                molecule_pairs_ed,
                number_bins=config.bins_uniformise_INFERENCE,
                return_binned_list=True,
                bin_sim_1=True,
                # bin_sim_1=False,
                ordinal_classification=True,
            )
        )  # do not treat sim==1 as another bin
        molecule_pairs_mces_uniform, binned_molecule_pairs_mces = (
            TrainUtils.uniformise(
                molecule_pairs_mces,
                number_bins=config.bins_uniformise_INFERENCE,
                return_binned_list=True,
                bin_sim_1=False,
                # bin_sim_1=False,
                # ordinal_classification=True,
            )
        )  # do not treat sim==1 as another bin
    else:
        molecule_pairs_ed_uniform = molecule_pairs_ed
        molecule_pairs_mces_uniform = molecule_pairs_mces
    return molecule_pairs_ed_uniform, molecule_pairs_mces_uniform


def remove_duplicate_pairs(array: np.ndarray) -> np.ndarray:
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
    molecule_pairs_ed_uniform,
    molecule_pairs_mces_uniform,
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    dataset_ed = LoadDataMultitasking.from_molecule_pairs_to_dataset(
        molecule_pairs_ed_uniform,
        max_num_peaks=int(config.TRANSFORMER_CONTEXT),
        use_extra_metadata=config.USE_EXTRA_METADATA_MODEL,
        use_categorical_adducts=config.USE_CATEGORICAL_ADDUCTS,
        adduct_info_csv= config.ADDUCT_INFO_CSV,
    )
    dataloader_ed = DataLoader(
        dataset_ed, batch_size=config.BATCH_SIZE, shuffle=False
    )

    dataset_mces = LoadDataMultitasking.from_molecule_pairs_to_dataset(
        molecule_pairs_mces_uniform,
        max_num_peaks=int(config.TRANSFORMER_CONTEXT),
        use_extra_metadata=config.USE_EXTRA_METADATA_MODEL,
    )
    dataloader_mces = DataLoader(
        dataset_mces, batch_size=config.BATCH_SIZE, shuffle=False
    )
    return dataloader_ed, dataloader_mces


def setup_model(config: Config):
    if not (config.INFERENCE_USE_LAST_MODEL) and (
        os.path.exists(config.CHECKPOINT_DIR + f"best_model.ckpt")
    ):
        best_model_path = config.CHECKPOINT_DIR + config.BEST_MODEL_NAME
    else:
        best_model_path = config.CHECKPOINT_DIR + f"last.ckpt"

    if config.USE_EXTRA_METADATA_MODEL:
        best_model = EmbedderMultitask.load_from_checkpoint(
            best_model_path,
            d_model=int(config.D_MODEL),
            n_layers=int(config.N_LAYERS),
            n_classes=config.EDIT_DISTANCE_N_CLASSES,
            use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
            use_element_wise=True,
            use_cosine_distance=config.use_cosine_distance,
            use_edit_distance_regresion=config.USE_EDIT_DISTANCE_REGRESSION,
            strict=False,
            use_extra_metadata=config.USE_EXTRA_METADATA_MODEL,
        )
    else:
        best_model = EmbedderMultitask.load_from_checkpoint(
            best_model_path,
            d_model=int(config.D_MODEL),
            n_layers=int(config.N_LAYERS),
            n_classes=config.EDIT_DISTANCE_N_CLASSES,
            use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
            use_element_wise=True,
            use_cosine_distance=config.use_cosine_distance,
            use_edit_distance_regresion=config.USE_EDIT_DISTANCE_REGRESSION,
            strict=False,
        )
    return best_model


def inference(dataloader_ed, dataloader_mces, model):
    model.eval()

    trainer = pl.Trainer(
        max_epochs=2,
        enable_progress_bar=config.enable_progress_bar,
        devices=1,
        accelerator="gpu",
    )
    # prediction of ed
    pred_ed = trainer.predict(
        model,
        dataloader_ed,
    )

    # prediction of mces
    pred_mces = trainer.predict(
        model,
        dataloader_mces,
    )
    return pred_ed, pred_mces


def evaluate_predictions(dataloader_ed, dataloader_mces, pred_ed, pred_mces):
    ed_true, mces_true = Postprocessing.get_similarities_multitasking(
        dataloader_ed
    )
    _, mces_true = Postprocessing.get_similarities_multitasking(
        dataloader_mces
    )

    # flat the results
    pred_mces_mces_flat = []
    pred_mces_mces_flat = [[p.item() for p in pred[1]] for pred in pred_mces]
    pred_mces_mces_flat = [
        item for sublist in pred_mces_mces_flat for item in sublist
    ]
    pred_mces_mces_flat = np.array(pred_mces_mces_flat)

    pred_ed_mces_flat = []
    pred_ed_mces_flat = [[p.item() for p in pred[1]] for pred in pred_ed]
    pred_ed_mces_flat = [
        item for sublist in pred_ed_mces_flat for item in sublist
    ]
    pred_ed_mces_flat = np.array(pred_ed_mces_flat)

    pred_ed_ed_flat = []
    pred_ed_ed_flat = [p[0] for p in pred_ed]
    pred_ed_ed_flat = [
        [which_index(p) for p in p_list] for p_list in pred_ed_ed_flat
    ]
    pred_ed_ed_flat = [item for sublist in pred_ed_ed_flat for item in sublist]
    pred_ed_ed_flat = np.array(pred_ed_ed_flat)

    # convert to numpy
    confident_pred_test1 = []
    confident_pred_test1 = np.array(confident_pred_test1)

    # get the results
    ed_true = np.array(ed_true)
    mces_true = np.array(mces_true)

    ed_true_clean = ed_true[~np.isnan(pred_ed_ed_flat)]
    pred_ed_ed_clean = pred_ed_ed_flat[~np.isnan(pred_ed_ed_flat)]

    # edit distance correlation
    corr_model_ed, _ = spearmanr(ed_true_clean, pred_ed_ed_clean)
    print(f"Correlation of edit distance model: {corr_model_ed}")

    plot_cm(ed_true_clean, pred_ed_ed_clean, config)

    # MCES
    counts, bins = TrainUtils.count_ranges(
        mces_true, number_bins=5, bin_sim_1=False, max_value=1
    )

    print("BEFORE BINING:")
    print(f"Max value of MCES: {max(mces_true)}")
    print(f"Min value of MCES: {min(mces_true)}")
    print(f"Number of samples per bin for MCES: {counts}")
    print(f"Bins for MCES: {bins}")

    min_bin = min([c for c in counts if c > 0])
    print(f"Min bin for MCES: {min_bin}")

    print("")
    print("AFTER BINING:")
    print(f"Max value of MCES: {max(mces_true)}")
    print(f"Min value of MCES: {min(mces_true)}")
    print(f"Number of samples per bin for MCES: {counts}")
    print(f"Bins for MCES: {bins}")

    min_bin = min([c for c in counts if c > 0])
    print(f"Min bin for MCES: {min_bin}")

    # Remove values correspoding to the threshold
    mces_true_original = mces_true.copy()
    mces_true = mces_true[mces_true_original != 0.5]
    pred_mces_mces_flat = pred_mces_mces_flat[mces_true_original != 0.5]
    corr_model_mces, _ = spearmanr(mces_true, pred_mces_mces_flat)
    # pred_ed_mces_flat = pred_ed_mces_flat[mces_true_original != 0.5]
    # corr_model_mces, _ = spearmanr(mces_true, pred_ed_mces_flat)
    print(f"Correlation of tanimoto model: {corr_model_mces}")

    if not config.USE_TANIMOTO:
        # if using mces20, apply de-normalization to obtain scalar value sof MCES20
        mces_true = config.MCES20_MAX_VALUE * (1 - mces_true)
        pred_mces_mces_flat = config.MCES20_MAX_VALUE * (
            1 - pred_mces_mces_flat
        )
        # pred_ed_mces_flat = config.MCES20_MAX_VALUE * (1 - pred_ed_mces_flat)

    plot_performance(mces_true, pred_mces_mces_flat)


def softmax(x):
    e_x = np.exp(x)  # Subtract max(x) for numerical stability
    return e_x / e_x.sum()


def which_index(p, threshold=0.5):
    return np.argmax(p)


def which_index_confident(p, threshold=0.50):
    # only predict confident predictions
    p_softmax = softmax(p)
    highest_pred = np.argmax(p_softmax)
    if p_softmax[highest_pred] > threshold:
        return np.argmax(p)
    else:
        return np.nan


def which_index_regression(p, max_index=5):
    ## the value of 0.2 must be the center of the second item

    index = np.round(p * max_index)
    # ad hoc solution
    # index=(-(np.round(p*max_index)))

    # index=np.clip(index, 0, 5)
    return index


def plot_cm(true, preds, config, file_name="cm.png", reverse_labels=True):

    # reverse the labels only for displaying:
    true = np.array(true)
    preds = np.array(preds)

    # Compute the confusion matrix and accuracy
    cm = confusion_matrix(true, preds)
    print("Confusion matrix")
    print(cm)
    accuracy = accuracy_score(true, preds)
    print(f"Accuracy of ED: {accuracy}")

    # Normalize the confusion matrix by the number of true instances per class
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create the plot
    plt.figure(figsize=(10, 7))
    labels = [">5", "4", "3", "2", "1", "0"]

    # Plot the heatmap using the 'Blues' colormap
    im = plt.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    cbar = plt.colorbar(im)
    cbar.set_label("Normalized frequency", fontsize=15)  # <-- add label

    # Compute a threshold to decide the annotation text color
    threshold = cm_normalized.max() / 2.0

    # Annotate each cell with the percentage, using white text if the background is dark
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            text_color = (
                "white" if cm_normalized[i, j] > threshold else "black"
            )
            plt.text(
                j,
                i,
                f"{cm_normalized[i, j]:.0%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=15,
            )

    # Set tick labels and increase font size for clarity
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, fontsize=15)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=15)
    plt.xlabel("Substructure edit distance - Prediction", fontsize=15)
    plt.ylabel("Substructure edit distance - Ground truth", fontsize=15)
    plt.title(
        f"Confusion Matrix (Normalized), Acc: {accuracy:.2f}, Samples: {preds.shape[0]}",
        fontsize=15,
    )

    # Save the plot
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, file_name))


@staticmethod
def divide_predictions_in_bins(
    list_elements1,
    list_elements2,
    number_bins=5,
    bin_sim_1=False,
    min_bin=0,
    max_value=0,
):
    # count the instances in the  bins from 0 to 1
    # Group the values into the corresponding bins, adding one for sim=1

    list_elements1 = list_elements1 / max_value
    list_elements2 = list_elements2 / max_value
    output_elements1 = np.array([])
    output_elements2 = np.array([])

    if bin_sim_1:
        number_bins_effective = number_bins + 1
    else:
        number_bins_effective = number_bins

    for p in range(int(number_bins_effective)):
        if p == 0:  # cover all the possible values equal or lower than 0
            low = -np.inf

        if bin_sim_1:
            high = (p + 1) * (1 / number_bins)
        else:
            if p == (number_bins_effective - 1):
                high = np.inf
            else:
                high = (p + 1) * (1 / number_bins)

        list_elements1_temp = list_elements1[
            (list_elements1 >= low) & (list_elements1 < high)
        ]
        list_elements2_temp = list_elements2[
            (list_elements1 >= low) & (list_elements1 < high)
        ]

        # randomize the arrays
        if len(list_elements1_temp) > 0:
            np.random.seed(42)
            random_indexes = np.random.randint(
                0, list_elements1_temp.shape[0], min_bin
            )
            output_elements1 = np.concatenate(
                (output_elements1, list_elements1_temp[random_indexes])
            )
            output_elements2 = np.concatenate(
                (output_elements2, list_elements2_temp[random_indexes])
            )

    return output_elements1, output_elements2


def plot_performance(mces_true, mces_pred):
    corr_mces, _ = spearmanr(mces_true, mces_pred)
    sns.set_theme(style="ticks")
    # hex plot
    plot = sns.jointplot(
        x=mces_true,
        y=mces_pred,
        kind="hex",
        color="#4CB391",
        joint_kws=dict(alpha=1, gridsize=15),
    )
    plot.set_axis_labels("Ground truth Similarity", "Prediction", fontsize=12)
    plot.fig.suptitle(f"Spearman Correlation:{corr_mces}", fontsize=16)
    plot.ax_joint.set_xlim(0, 40)
    plot.ax_joint.set_ylim(0, 40)
    plt.tight_layout()
    plt.savefig(config.CHECKPOINT_DIR + f"hexbin_plot_{config.MODEL_CODE}.png")

    # scatter plot
    plt.scatter(mces_true, mces_pred, alpha=0.5)
    plt.xlabel("ground truth")
    plt.ylabel("prediction")
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        config.CHECKPOINT_DIR + f"scatter_plot_{config.MODEL_CODE}.png"
    )


if __name__ == "__main__":
    config = setup_config()
    dataset_path, fig_path = setup_paths(config)

    molecule_pairs_ed, molecule_pairs_mces = load_dataset(dataset_path)
    molecule_pairs_ed_uniform, molecule_pairs_mces_uniform = prepare_data(
        molecule_pairs_ed, molecule_pairs_mces, config
    )
    dataloader_ed, dataloader_mces = create_dataloaders(
        molecule_pairs_ed_uniform, molecule_pairs_mces_uniform, config
    )
    model = setup_model(config)
    pred_ed, pred_mces = inference(dataloader_ed, dataloader_mces, model)
    evaluate_predictions(dataloader_ed, dataloader_mces, pred_ed, pred_mces)
