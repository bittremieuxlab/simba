import os
import random

import dill
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pytorch_lightning.callbacks import ProgressBar
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler

from simba.config import Config
from simba.load_mces.load_mces import LoadMCES
from simba.losscallback import LossCallback
from simba.molecular_pairs_set import MolecularPairsSet
from simba.ordinal_classification.embedder_ordinal import EmbedderOrdinal
from simba.ordinal_classification.load_data_ordinal import LoadDataOrdinal
from simba.parser import Parser
from simba.sanity_checks import SanityChecks
from simba.train_utils import TrainUtils
from simba.transformers.postprocessing import Postprocessing
from simba.weight_sampling import WeightSampling
from simba.weight_sampling_tools.custom_weighted_random_sampler import (
    CustomWeightedRandomSampler,
)

# In[268]:


# parameters
config = Config()
parser = Parser()
config = parser.update_config(config)
config.USE_GUMBEL = False
config.N_CLASSES = 6
config.bins_uniformise_INFERENCE = config.N_CLASSES - 1
config.use_uniform_data_INFERENCE = True

# In[281]:
if not os.path.exists(config.CHECKPOINT_DIR):
    os.makedirs(config.CHECKPOINT_DIR)


# parameters
dataset_path = config.PREPROCESSING_DIR + config.PREPROCESSING_PICKLE_FILE
epochs = config.epochs
bins_uniformise_inference = config.bins_uniformise_INFERENCE
enable_progress_bar = config.enable_progress_bar
fig_path = config.CHECKPOINT_DIR + f"scatter_plot_{config.MODEL_CODE}.png"
model_code = config.MODEL_CODE


print("loading file")
# Load the dataset from the pickle file
with open(dataset_path, "rb") as file:
    dataset = dill.load(file)

molecule_pairs_train = dataset["molecule_pairs_train"]
molecule_pairs_val = dataset["molecule_pairs_val"]
molecule_pairs_test = dataset["molecule_pairs_test"]
uniformed_molecule_pairs_test = dataset["uniformed_molecule_pairs_test"]


# In[283]:
print("Loading pairs data ...")

print(
    f"Number of original spectra: {len(molecule_pairs_train.spectrums_original)}"
)


def get_distribution(directory_path, prefix):
    """
    load np arrays containing data as well as apply normalization
    """
    distribution = {k: 0 for k in range(0, 6)}

    # find all np arrays
    files = LoadMCES.find_file(directory_path, prefix)

    # load np files
    print("Loading the partitioned files of the pairs")
    for i, f in enumerate(files):
        print(f"Processing batch {i}")
        np_array = np.load(f)

        for k in distribution:
            number_samples = np_array[np_array == k].shape[0]
            distribution[k] = distribution[k] + number_samples
            if k == 5:
                number_samples = np_array[np_array > k].shape[0]
                distribution[k] = distribution[k] + number_samples

    return distribution


distribution = get_distribution(
    config.PREPROCESSING_DIR, prefix="indexes_tani_incremental_train"
)

for k in distribution:
    print(f"For {k}, samples= {distribution[k]}")
