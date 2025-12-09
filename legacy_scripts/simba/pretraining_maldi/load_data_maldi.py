import copy

import numpy as np
from tqdm import tqdm

from simba.molecular_pairs_set import MolecularPairsSet
from simba.preprocessor import Preprocessor
from simba.pretraining_maldi.CustomDataset_Maldi import CustomDatasetMaldi


class LoadDataMaldi:

    @staticmethod
    def from_spectra_to_dataset(
        input_spectra,
        max_num_peaks=100,
        training=False,  # shuffle the spectrum 0 and 1 for data augmentation,
        min_intensity=0.00,
    ):
        """
        preprocess the spectra and convert it for being used in Pytorch
        """
        spectra = [copy.copy(s) for s in input_spectra]

        ## Preprocess the data
        pp = Preprocessor()
        spectra = pp.preprocess_all_spectra(
            spectra, min_intensity=min_intensity
        )

        # basic features
        mz_0 = np.zeros((len(spectra), max_num_peaks), dtype=np.float32)
        number_of_peaks = np.zeros((len(spectra), 1), dtype=np.float32)
        intensity_0 = np.zeros((len(spectra), max_num_peaks), dtype=np.float32)
        precursor_mass_0 = np.zeros((len(spectra), 1), dtype=np.float32)
        precursor_charge_0 = np.zeros((len(spectra), 1), dtype=np.int32)

        # for self supervising approach
        sampled_mz = np.zeros((len(spectra), max_num_peaks), dtype=np.float32)
        sampled_intensity = np.zeros(
            (len(spectra), max_num_peaks), dtype=np.float32
        )
        flips = np.zeros((len(spectra), max_num_peaks), dtype=np.int32)

        # fill arrays
        for i, s in enumerate(spectra):
            # check for maximum length
            length_0 = (
                len(s.mz) if len(s.mz) <= max_num_peaks else max_num_peaks
            )
            # assign the values to the array
            mz_0[i, 0:length_0] = np.array(s.mz[0:length_0])
            intensity_0[i, 0:length_0] = np.array(s.intensity[0:length_0])

            number_of_peaks[i] = length_0
            precursor_mass_0[i] = s.precursor_mz
            precursor_charge_0[i] = s.precursor_charge

        # Normalize the intensity array
        intensity_0 = intensity_0 / np.sqrt(
            np.sum(intensity_0**2, axis=1, keepdims=True)
        )

        dictionary_data = {
            "mz_0": mz_0,
            "intensity_0": intensity_0,
            "precursor_mass_0": precursor_mass_0,
            "precursor_charge_0": precursor_charge_0,
            "number_peaks": number_of_peaks.astype(int),
            "sampled_mz": sampled_mz,
            "sampled_intensity": sampled_intensity,
            "flips": flips,
        }

        return CustomDatasetMaldi(dictionary_data, training=training)
