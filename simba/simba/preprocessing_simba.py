import copy
from typing import List, Optional

from simba.config import Config
from simba.load_data import LoadData
from simba.loader_saver import LoaderSaver
from simba.logger_setup import logger
from simba.preprocessor import Preprocessor
from simba.spectrum_ext import SpectrumExt


class PreprocessingSimba:

    def load_spectra(
        file_name: str,
        config: Config,
        min_peaks: int = 6,
        n_samples: int = 500000,
        use_gnps_format: bool = False,
        use_only_protonized_adducts: bool = True,
    ) -> List[SpectrumExt]:
        """Load and preprocess spectra from a file.
        Parameters
        ----------
        file_name : str
            The path to the file containing the spectra.
        config : Config
            Configuration object containing parameters.
        min_peaks : int, optional
            The minimum number of peaks a spectrum must have to be included, by default 6.
        n_samples : int, optional
            The number of samples to load, by default 500000.
        use_gnps_format : bool, optional
            Whether to use GNPS format for loading, by default False.
        use_only_protonized_adducts : bool, optional
            Whether to use only protonized adducts, by default True.

        Returns
        -------
        List[SpectrumExt]
            A list of preprocessed SpectrumExt objects."""
        # load
        if file_name.endswith(".mgf"):
            loader_saver = LoaderSaver(
                block_size=100,
                pickle_nist_path=None,
                pickle_gnps_path=None,
                pickle_janssen_path=None,
            )
            all_spectrums = loader_saver.get_all_spectrums(
                file_name,
                n_samples,
                use_tqdm=True,
                use_nist=False,
                config=config,
                use_janssen=not (use_gnps_format),
                use_only_protonized_adducts=use_only_protonized_adducts,
            )
        elif file_name.endswith(".pkl"):
            all_spectrums = LoadData.get_all_spectrums_casmi(
                file_name,
                config=config,
            )
        else:
            logger.error("Error: unrecognized file extension")
        # preprocess
        all_spectrums_processed = [copy.deepcopy(s) for s in all_spectrums]

        pp = Preprocessor()
        ### remove extra peaks in janssen
        all_spectrums_processed = [
            pp.preprocess_spectrum(
                s,
                fragment_tol_mass=10,
                fragment_tol_mode="ppm",
                min_intensity=0.01,
                max_num_peaks=1000,
                scale_intensity=None,
            )
            for s in all_spectrums_processed
        ]

        # remove spectra that does not have at least min peaks
        filtered_spectra = [
            s_original
            for s_original, s_processed in zip(
                all_spectrums, all_spectrums_processed
            )
            if len(s_processed.mz) >= min_peaks
        ]

        return filtered_spectra
