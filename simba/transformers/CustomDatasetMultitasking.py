import torch
from torch.utils.data import Dataset
import random
from simba.transformers.augmentation import Augmentation
import numpy as np
from tqdm import tqdm


class CustomDatasetMultitasking(Dataset):
    def __init__(
        self,
        your_dict,
        training=False,
        prob_aug=1.0,
        # prob_aug=0.2,
        mz=None,
        intensity=None,
        precursor_mass=None,
        precursor_charge=None,
        df_smiles=None,
        use_fingerprints=False,
        fingerprint_0=None,
        fingerprint_1=None,
        max_num_peaks=None,
    ):
        self.data = your_dict
        self.keys = list(your_dict.keys())
        self.training = training
        self.prob_aug = prob_aug

        self.mz = mz
        self.intensity = intensity
        self.precursor_mass = precursor_mass
        self.precursor_charge = precursor_charge
        self.df_smiles = df_smiles  ### df with rows smiles, indexes
        self.use_fingerprints = use_fingerprints
        if self.use_fingerprints:
            self.fingerprint_0 = fingerprint_0
            self.fingerprint_1 = fingerprint_1
        self.max_num_peaks = max_num_peaks

    def __len__(self):
        return len(self.data[self.keys[0]])
        # return len(self.keys)



    def __getitem__(self, idx):
        # key = self.keys[idx]
        # sample = self.data[key]
        # print(idx)
        sample_unique = {k: self.data[k][idx] for k in self.keys}

        # indexes_unique_0 = list(sample_unique['index_unique_0'])
        # indexes_unique_1 = list(sample_unique['index_unique_1'])

        indexes_unique_0 = sample_unique["index_unique_0"]
        indexes_unique_1 = sample_unique["index_unique_1"]

        # for each unique value 0 sample from the distribution
        # indexes_original_0 = [random.choice(self.df_smiles.loc[int(index),'indexes']) for index in indexes_unique_0]
        # indexes_original_1 = [random.choice(self.df_smiles.loc[int(index),'indexes']) for index in indexes_unique_1]

        if self.training:
            # select random samples
            indexes_original_0 = random.choice(
                self.df_smiles.loc[int(indexes_unique_0[0]), "indexes"]
            )
            indexes_original_1 = random.choice(
                self.df_smiles.loc[int(indexes_unique_1[0]), "indexes"]
            )
        else:
            # select the first index
            indexes_original_0 = self.df_smiles.loc[
                int(indexes_unique_0[0]), "indexes"
            ][0]
            # select the last index
            indexes_original_1 = self.df_smiles.loc[
                int(indexes_unique_1[0]), "indexes"
            ][-1]

        ## now get an original spectra based on indexes
        sample = {}

        sample["mz_0"] = self.mz[indexes_original_0].astype(np.float32)
        sample["intensity_0"] = self.intensity[indexes_original_0].astype(np.float32)

        sample["mz_1"] = self.mz[indexes_original_1].astype(np.float32)
        sample["intensity_1"] = self.intensity[indexes_original_1].astype(np.float32)
        sample["precursor_mass_0"] = self.precursor_mass[indexes_original_0].astype(
            np.float32
        )
        sample["precursor_mass_1"] = self.precursor_mass[indexes_original_1].astype(
            np.float32
        )
        sample["precursor_charge_0"] = self.precursor_charge[indexes_original_0].astype(
            np.float32
        )
        sample["precursor_charge_1"] = self.precursor_charge[indexes_original_1].astype(
            np.float32
        )
        sample["similarity"] = sample_unique["similarity"].astype(np.float32)
        sample["similarity2"] = sample_unique["similarity2"].astype(np.float32)

        if self.use_fingerprints:

            ind_0 =  int(indexes_unique_0[0])
            ind_1 =  int(indexes_unique_1[0])

            if self.training:
                if (ind_0%2)==0:
                    sample["fingerprint_0"] = self.fingerprint_0[
                        ind_0
                    ].astype(np.float32)
                else:
                    # return 0s
                    sample["fingerprint_0"] = 0*self.fingerprint_0[
                        ind_0
                    ].astype(np.float32)

                if (ind_1%2)==0:
                    sample["fingerprint_1"] = self.fingerprint_1[
                        ind_1
                    ].astype(np.float32)
                else:
                    # return 0s
                    sample["fingerprint_1"] = 0*self.fingerprint_1[
                        ind_1
                    ].astype(np.float32)

                
                if (ind_0%4)==0:
                     sample["mz_0"]= 0* sample["mz_0"] 
                     sample["intensity_0"] = 0* sample["intensity_0"] 
                if (ind_1%4)==0:
                     sample["mz_1"]= 0* sample["mz_1"] 
                     sample["intensity_1"] = 0* sample["intensity_1"] 
            else:
                sample["fingerprint_0"] = self.fingerprint_0[
                        ind_0
                    ].astype(np.float32)
                sample["fingerprint_1"] = 0*self.fingerprint_1[
                        ind_1
                    ].astype(np.float32)
                    
        # print(sample["mz_0"]).shape
        # print(sample["intensity_0"].shape)
        # print(sample["precursor_charge_0"].shape)
        # print(sample["precursor_mass_0"].shape)

        # Convert your sample to PyTorch tensors if needed
        # e.g., use torch.tensor(sample) if sample is a numpy array

        if self.training:
            if random.random() < self.prob_aug:
                # augmentation
                sample = Augmentation.augment(sample, max_num_peaks=self.max_num_peaks)

        # normalize
        sample = Augmentation.normalize_intensities(sample)
        return sample
