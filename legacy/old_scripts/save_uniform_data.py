import argparse
import os
import sys

import dill
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.callbacks import ProgressBar
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from simba.config import Config
from simba.deterministic_similarity import DetSimilarity
from simba.parser import Parser
from simba.plotting import Plotting
from simba.train_utils import TrainUtils
from simba.transformers.embedder import Embedder
from simba.transformers.embedder_fingerprint import EmbedderFingerprint
from simba.transformers.load_data import LoadData
from simba.transformers.postprocessing import Postprocessing

# parse arguments
config = Config()
parser = Parser()
config = parser.update_config(config)
dataset_path = config.dataset_path
best_model_path = config.best_model_path
use_uniform_data = config.use_uniform_data_INFERENCE
bins_uniformise = config.bins_uniformise_INFERENCE
fig_path = config.CHECKPOINT_DIR + f"scatter_plot_{config.MODEL_CODE}.png"
roc_file_path = config.CHECKPOINT_DIR + f"roc_curve_{config.MODEL_CODE}.png"
enable_progress_bar = config.enable_progress_bar
write_uniform_test_data = True
uniformed_molecule_pairs_test_path = (
    "/scratch/antwerpen/209/vsc20939/data/uniformed_molecule_pairs_test.pkl"
)

if not os.path.exists(config.CHECKPOINT_DIR):
    os.makedirs(config.CHECKPOINT_DIR)

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")

    # Get the name of the current GPU
    current_gpu = torch.cuda.get_device_name(
        0
    )  # assuming you have at least one GPU
    print(f"Current GPU: {current_gpu}")

    # Check if PyTorch is currently using GPU
    current_device = torch.cuda.current_device()
    print(f"PyTorch is using GPU: {torch.cuda.is_initialized()}")

    # Print CUDA version
    print(f"CUDA version: {torch.version.cuda}")

    # Additional information about the GPU
    print(torch.cuda.get_device_properties(current_device))

else:
    print("CUDA (GPU support) is not available.")


print("loading file")
# Load the dataset from the pickle file
with open(dataset_path, "rb") as file:
    dataset = dill.load(file)

molecule_pairs_test = dataset["molecule_pairs_test"]
print(f"Number of molecule pairs: {len(molecule_pairs_test)}")
print("Uniformize the data")
uniformed_molecule_pairs_test, _ = TrainUtils.uniformise(
    molecule_pairs_test,
    number_bins=bins_uniformise,
    return_binned_list=True,
    bin_sim_1=False,
)  # do not treat sim==1 as another bin

# write uniform data
uniformed_molecule_pairs_test_dict = {
    "uniformed_molecule_pairs_test": uniformed_molecule_pairs_test
}
with open(uniformed_molecule_pairs_test_path, "wb") as file:
    dill.dump(uniformed_molecule_pairs_test_dict, file)
