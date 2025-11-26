import copy

import numpy as np
from tqdm import tqdm

from simba.logger_setup import logger
from simba.molecule_pairs_opt import MoleculePairsOpt
from simba.ordinal_classification.ordinal_classification import (
    OrdinalClassification,
)
from simba.preprocessor import Preprocessor
from simba.tanimoto import Tanimoto
from simba.transformers.CustomDatasetMultitasking import (
    CustomDatasetMultitasking,
)


class LoadDataMultitasking:
    """
    using unique identifiers
    """

    @staticmethod
    def from_molecule_pairs_to_dataset(
        molecule_pairs_input: MoleculePairsOpt,
        max_num_peaks,
        training=False,  # shuffle the spectrum 0 and 1 for data augmentation
        N_classes=6,
        use_fingerprints=False,
        use_adduct=False,
    ):
        """
        preprocess the spectra and convert it for being used in Pytorch
        """
        # copy spectrums to avoid overwriting
        molecule_pairs = MoleculePairsOpt(
            original_spectra=[
                copy.copy(s) for s in molecule_pairs_input.original_spectra
            ],
            unique_spectra=molecule_pairs_input.spectra,
            df_smiles=molecule_pairs_input.df_smiles,
            pair_distances=molecule_pairs_input.pair_distances,
            extra_distances=molecule_pairs_input.extra_distances,
        )

        # Preprocess the spectra
        pp = Preprocessor()
        logger.info("Preprocess all spectra ...")
        molecule_pairs.original_spectra = pp.preprocess_all_spectra(
            molecule_pairs.original_spectra,
            max_num_peaks=max_num_peaks,
            training=training,
        )

        # Get the mz, intensity values and precursor data
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
        if use_adduct:
            ionmode = np.zeros(
                (len(molecule_pairs.original_spectra), 1), dtype=np.float32
            )
            adduct_mass = np.zeros(
                (len(molecule_pairs.original_spectra), 1), dtype=np.float32
            )

        logger.info("Loading mz, intensity and precursor data ...")
        for i, spec in enumerate(molecule_pairs.original_spectra):
            # check for maximum length
            length = (
                len(spec.mz)
                if len(spec.mz) <= max_num_peaks
                else max_num_peaks
            )

            # assign the values to the array
            mz[i, 0:length] = np.array(spec.mz[0:length])
            intensity[i, 0:length] = np.array(spec.intensity[0:length])

            precursor_mass[i] = spec.precursor_mz
            precursor_charge[i] = spec.precursor_charge

            if use_adduct:
                if spec.ionmode == "none":
                    ionmode[i] = np.nan
                else:
                    ionmode[i] = 1.0 if spec.ionmode == "positive" else -1.0
                adduct_mass[i] = spec.adduct_mass

        # logger.info("Normalizing intensities")
        # Normalize the intensity array
        # intensity = intensity / np.sqrt(np.sum(intensity**2, axis=1, keepdims=True))

        # Adjust ED towards a N classification problem
        ed = OrdinalClassification.from_float_to_class(
            molecule_pairs_input.pair_distances[:, 2].reshape(-1, 1),
            N_classes=N_classes,
        )

        if molecule_pairs.extra_distances is None:
            raise ValueError(
                "extra_distances must be provided for multitask training."
            )
        mces = molecule_pairs.extra_distances.reshape(-1, 1)

        if use_fingerprints:
            logger.info("Computing molecular fingerprints...")
            fingerprint_0 = np.array(
                [
                    np.array(Tanimoto.compute_fingerprint(s.params["smiles"]))
                    for s in molecule_pairs_input.spectra
                ]
            )
        else:
            fingerprint_0 = np.array([0 for m in molecule_pairs_input.spectra])

        dictionary_data = {
            "index_unique_0": molecule_pairs_input.pair_distances[
                :, 0
            ].reshape(-1, 1),
            "index_unique_1": molecule_pairs_input.pair_distances[
                :, 1
            ].reshape(-1, 1),
            "ed": ed,
            "mces": mces,
            # "fingerprint_0": fingerprint_0,
        }

        return CustomDatasetMultitasking(
            dictionary_data,
            training=training,
            mz=mz,
            intensity=intensity,
            precursor_mass=precursor_mass,
            precursor_charge=precursor_charge,
            df_smiles=molecule_pairs_input.df_smiles,
            use_fingerprints=use_fingerprints,
            fingerprint_0=fingerprint_0,
            max_num_peaks=max_num_peaks,
            use_extra_metadata=use_adduct,
            ionization_mode_precursor=(ionmode if use_adduct else None),
            adduct_mass_precursor=(adduct_mass if use_adduct else None),
        )
