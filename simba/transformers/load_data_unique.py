import copy

import numpy as np
from tqdm import tqdm

from simba.molecule_pairs_opt import MoleculePairsOpt
from simba.preprocessor import Preprocessor
from simba.transformers.CustomDatasetUnique import CustomDatasetUnique


class LoadDataUnique:
    """
    using unique identifiers
    """

    @staticmethod
    def from_molecule_pairs_to_dataset(
        molecule_pairs_input,
        max_num_peaks=100,
        training=False,  # shuffle the spectrum 0 and 1 for data augmentation
    ):
        """
        preprocess the spectra and convert it for being used in Pytorch
        """
        # copy spectrums to avoid overwriting
        molecule_pairs = MoleculePairsOpt(
            original_spectra=[
                copy.copy(s) for s in molecule_pairs_input.spectrums_original
            ],
            unique_spectra=molecule_pairs_input.spectrums,
            df_smiles=molecule_pairs_input.df_smiles,
            pair_distances=molecule_pairs_input.indexes_tani,
        )

        ## Preprocess the data
        pp = Preprocessor()
        print("Preprocessing all the data ...")
        molecule_pairs.original_spectra = pp.preprocess_all_spectra(
            molecule_pairs.original_spectra
        )

        print("Finished preprocessing ")

        ## Get the mz, intensity values and precursor data
        mz = np.zeros(
            (len(molecule_pairs.original_spectra), max_num_peaks),
            dtype=np.float32,
        )
        intensity = np.zeros(
            (len(molecule_pairs.original_spectra), max_num_peaks),
            dtype=np.float32,
        )
        precursor_mass = np.zeros(
            (len(molecule_pairs.original_spectra), 1), dtype=np.float32
        )
        precursor_charge = np.zeros(
            (len(molecule_pairs.original_spectra), 1), dtype=np.int32
        )

        print("loading data")
        for i, l in enumerate(molecule_pairs.original_spectra):
            # check for maximum length
            length = len(l.mz) if len(l.mz) <= max_num_peaks else max_num_peaks

            # assign the values to the array
            mz[i, 0:length] = np.array(l.mz[0:length])
            intensity[i, 0:length] = np.array(l.intensity[0:length])

            precursor_mass[i] = l.precursor_mz
            precursor_charge[i] = l.precursor_charge

        print("Normalizing intensities")
        # Normalize the intensity array
        intensity = intensity / np.sqrt(
            np.sum(intensity**2, axis=1, keepdims=True)
        )

        print("Creating dictionaries")
        dictionary_data = {
            "index_unique_0": molecule_pairs_input.indexes_tani[:, 0].reshape(
                -1, 1
            ),
            "index_unique_1": molecule_pairs_input.indexes_tani[:, 1].reshape(
                -1, 1
            ),
            "similarity": molecule_pairs_input.indexes_tani[:, 2].reshape(
                -1, 1
            ),
            # "fingerprint": fingerprints,
        }

        return CustomDatasetUnique(
            dictionary_data,
            training=training,
            mz=mz,
            intensity=intensity,
            precursor_mass=precursor_mass,
            precursor_charge=precursor_charge,
            df_smiles=molecule_pairs_input.df_smiles,
        )
