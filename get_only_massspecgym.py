## Parameters
data_folder = "/scratch/antwerpen/209/vsc20939/data/"
preprocessing_folder = data_folder + "preprocessing_ed_mces_20250123/"
molecular_file = (
    preprocessing_folder + "edit_distance_neurips_nist_exhaustive.pkl"
)

new_preprocessing_folder = data_folder + "preprocessing_massspecgym_20250415/"
molecular_file_output = new_preprocessing_folder + "mapping.pkl"
USE_TRAINING = True


# In[283]:
import pickle
import sys

import numpy as np
import pandas as pd

import simba

sys.modules["src"] = simba


# In[284]:


from simba.config import Config
from simba.load_mces.load_mces import LoadMCES

# In[285]:


config = Config()


config.PREPROCESSING_DIR_TRAIN = preprocessing_folder
# In[287]:


with open(molecular_file, "rb") as f:
    dataset = pickle.load(f)


# In[288]:


molecule_pairs_train = dataset["molecule_pairs_train"]
molecule_pairs_val = dataset["molecule_pairs_val"]
molecule_pairs_test = dataset["molecule_pairs_test"]
uniformed_molecule_pairs_test = dataset["uniformed_molecule_pairs_test"]


# In[289]:


## Load molecular file
## Based on the pairs numpy array, filter only those pairs where both molecules are from massspecgym
## Change the indexes to the new mapping


# In[290]:


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


# In[293]:


from simba.filtering_massspecgym.filtering_massspecgym import (
    FilteringMassSpecGym,
)

# In[304]:

# In[ ]:

if USE_TRAINING:
    molecule_pairs_train_filtered, indexes_tani_multitasking_train_filtered = (
        FilteringMassSpecGym.filter_massspecgym(
            molecule_pairs_train,
            path_pairs=config.PREPROCESSING_DIR_TRAIN,
            config=config,
            prefix="ed_mces_indexes_tani_incremental_train",
        )
    )

else:
    molecule_pairs_train_filtered = molecule_pairs_train
    indexes_tani_multitasking_train_filtered = indexes_tani_multitasking_train
# In[295]:


molecule_pairs_test_filtered, indexes_tani_multitasking_test_filtered = (
    FilteringMassSpecGym.filter_massspecgym(
        molecule_pairs_test,
        path_pairs=config.PREPROCESSING_DIR_TRAIN,
        config=config,
        prefix="ed_mces_indexes_tani_incremental_test",
    )
)


# In[ ]:


molecule_pairs_val_filtered, indexes_tani_multitasking_val_filtered = (
    FilteringMassSpecGym.filter_massspecgym(
        molecule_pairs_val,
        path_pairs=config.PREPROCESSING_DIR_TRAIN,
        config=config,
        prefix="ed_mces_indexes_tani_incremental_val",
    )
)


import os

if not (os.path.exists(new_preprocessing_folder)):
    os.mkdir(new_preprocessing_folder)


# In[296]:


def write_data(
    file_path,
    all_spectrums_train=None,
    all_spectrums_val=None,
    all_spectrums_test=None,
    molecule_pairs_train=None,
    molecule_pairs_val=None,
    molecule_pairs_test=None,
    uniformed_molecule_pairs_test=None,
):
    dataset = {
        "all_spectrums_train": None,
        "all_spectrums_val": None,
        "all_spectrums_test": None,
        "molecule_pairs_train": molecule_pairs_train_filtered,
        "molecule_pairs_val": molecule_pairs_val_filtered,
        "molecule_pairs_test": molecule_pairs_test_filtered,
        "uniformed_molecule_pairs_test": None,
    }
    with open(file_path, "wb") as file:
        pickle.dump(dataset, file)


# ## Write data


# In[297]:
def save_array_in_chunks(array, output_folder, base_filename, n_parts=10):
    """
    Splits a NumPy array or list into n_parts and saves each part to a .npy file.

    Parameters:
    - array: list or np.ndarray – the array to split and save
    - output_folder: str – directory to save the files (must include trailing slash if needed)
    - base_filename: str – base filename without the index or extension
    - n_parts: int – number of chunks to split the array into
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure folder exists

    array = np.array(array)
    chunks = np.array_split(array, n_parts)

    for i, chunk in enumerate(chunks):
        filename = f"{base_filename}_{i}.npy"
        file_path = os.path.join(output_folder, filename)
        np.save(file_path, chunk)


## save results
save_array_in_chunks(
    indexes_tani_multitasking_train_filtered,
    output_folder=new_preprocessing_folder,
    base_filename="ed_mces_indexes_tani_incremental_train",
    n_parts=10,
)

save_array_in_chunks(
    indexes_tani_multitasking_val_filtered,
    output_folder=new_preprocessing_folder,
    base_filename="ed_mces_indexes_tani_incremental_val",
    n_parts=10,
)

save_array_in_chunks(
    indexes_tani_multitasking_test_filtered,
    output_folder=new_preprocessing_folder,
    base_filename="ed_mces_indexes_tani_incremental_test",
    n_parts=10,
)


# In[301]:


write_data(
    molecular_file_output,
    all_spectrums_train=None,
    all_spectrums_val=None,
    all_spectrums_test=None,
    molecule_pairs_train=molecule_pairs_train_filtered,
    molecule_pairs_val=molecule_pairs_val_filtered,
    molecule_pairs_test=molecule_pairs_test_filtered,
    uniformed_molecule_pairs_test=None,
)


# In[ ]:
