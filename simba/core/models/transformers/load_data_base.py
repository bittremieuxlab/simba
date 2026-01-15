import copy

import numpy as np

from simba.core.data.preprocessor import Preprocessor


class LoadDataBase:
    @staticmethod
    def load_spectrum_data(
        input_spectra,
        max_num_peaks=100,
    ):
        ## Preprocess the data
        pp = Preprocessor()
        spectra = [copy.deepcopy(s) for s in input_spectra]
        spectra = pp.preprocess_all_spectra(spectra, max_num_peaks=max_num_peaks)
        # spectrums = pp.preprocess_all_spectrums_variable_max_peaks(spectrums, max_num_peaks=max_num_peaks)

        ## Get the mz, intensity values and precursor data
        mz = np.zeros((len(spectra), max_num_peaks), dtype=np.float32)
        intensity = np.zeros((len(spectra), max_num_peaks), dtype=np.float32)
        precursor_mass = np.zeros((len(spectra), 1), dtype=np.float32)
        precursor_charge = np.zeros((len(spectra), 1), dtype=np.int32)

        for i, spectrum in enumerate(spectra):
            # check for maximum length
            length = (
                len(spectrum.mz) if len(spectrum.mz) <= max_num_peaks else max_num_peaks
            )

            # assign the values to the array
            mz[i, 0:length] = np.array(spectrum.mz[0:length])
            intensity[i, 0:length] = np.array(spectrum.intensity[0:length])

            precursor_mass[i] = spectrum.precursor_mz
            precursor_charge[i] = spectrum.precursor_charge

        # Normalize the intensity array
        intensity = intensity / np.sqrt(np.sum(intensity**2, axis=1, keepdims=True))

        return {
            "mz": mz,
            "intensity": intensity,
            "precursor_mass": precursor_mass,
            "precursor_charge": precursor_charge,
        }
