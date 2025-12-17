import numpy as np
import pandas as pd

from simba.core.data.molecular_pairs import MolecularPairsSet
from simba.core.data.molecule_pair import MoleculePair
from simba.core.data.spectrum import SpectrumExt


class MoleculePairsOpt(MolecularPairsSet):
    """
    optimized version of molecule pairs set with the possiblitiy of working over unique smiles
    """

    def __init__(
        self,
        original_spectra: list[SpectrumExt],
        unique_spectra: list[SpectrumExt],
        df_smiles: pd.DataFrame,
        pair_distances: np.ndarray,
        extra_distances: np.ndarray | None = None,
    ):
        """
        Initialize the optimized molecule pairs.

        Parameters
        ----------
        original_spectra: List[SpectrumExt]
            list of all the spectra, including repetitions of the same compound
        unique_spectra: List[SpectrumExt]
            list of unique spectra, one per compound
        df_smiles: pd.DataFrame
            dataframe containing the mapping from unique smiles to original spectra
        pair_distances: np.ndarray
            array of shape (num_pairs, 3) with the indexes of the pairs and their distance
            - first column: index of the first spectrum in the pair
            - second column: index of the second spectrum in the pair
            - third column: distance (e.g., substructure edit distance) between the two compounds
        extra_distances: Optional[np.ndarray]
            array of shape (num_pairs, 1) with an extra distance metric (e.g., MCES)
        """
        self.original_spectra = original_spectra
        self.spectra = unique_spectra
        self.df_smiles = df_smiles  # table containing the indexes to map unique to repetitions of the same smiles
        # treat the first 2 columns as int and the 3 column as float
        # self.indexes_tani = MolecularPairsSet.adjust_data_format(
        #    np.array(indexes_tani_unique)
        # )
        self.pair_distances = pair_distances
        self.extra_distances = extra_distances

    def __add__(self, other):
        # only to be used when the spectrums are the same

        if self.spectra_equal(self.original_spectra, other.original_spectra):
            new_indexes_tani = np.concatenate(
                (self.pair_distances, other.pair_distances), axis=0
            )
            if (self.extra_distances is not None) and (
                other.extra_distances is not None
            ):
                extra_distances = np.concatenate(
                    (self.extra_distances, other.extra_distances), axis=0
                )
            else:
                extra_distances = None
            return MoleculePairsOpt(
                unique_spectra=self.spectra,
                original_spectra=self.original_spectra,
                pair_distances=new_indexes_tani,
                df_smiles=self.df_smiles,
                extra_distances=extra_distances,
            )
        else:
            print("ERROR: Attempting to add 2 set of spectrums with different content")
            return 0

    def get_molecular_pair(self, index: int) -> MoleculePair:
        """
        get a molecular pair.
        For the first molecule of the pair, retrieve the first element, for the second element retrieve the last index
        this is to avoid to retrieve the same spectrum when the indexes are the same : sim=1
        """
        # i,j,tani = self.indexes_tani[index]
        i = int(self.pair_distances[index, 0])
        j = int(self.pair_distances[index, 1])
        dist = self.pair_distances[index, 2]

        molecule_pair = MoleculePair(
            vector_0=None,
            vector_1=None,
            smiles_0=self.spectra[i].smiles,
            smiles_1=self.spectra[j].smiles,
            similarity=dist,
            global_feats_0=MolecularPairsSet.get_global_variables(self.spectra[i]),
            global_feats_1=MolecularPairsSet.get_global_variables(self.spectra[j]),
            index_in_spectrum_0=self.get_original_index_from_unique_index(
                i, 0
            ),  # index in the spectrum list used as input
            index_in_spectrum_1=self.get_original_index_from_unique_index(j, 1),
            spectrum_object_0=self.get_original_spectrum_from_unique_index(i, 0),
            spectrum_object_1=self.get_original_spectrum_from_unique_index(j, 1),
            params_0=self.get_original_spectrum_from_unique_index(i, 0).params,
            params_1=self.get_original_spectrum_from_unique_index(j, 1).params,
        )

        return molecule_pair

    def get_original_spectrum_from_unique_index(self, unique_index, pair):
        return self.original_spectra[
            self.get_original_index_from_unique_index(unique_index, pair)
        ]

    def get_original_index_from_unique_index(self, index, pair):
        """
        obtain the mapped spectrum from index computed in the unique compound space
        if pair=0, return the first index, else return the last index
        """
        if pair == 0:
            return self.df_smiles.loc[index, "indexes"][0]
        else:
            return self.df_smiles.loc[index, "indexes"][-1]

    def get_spectrums_from_indexes(self, pair_index):
        # pair index refers if it is 0 or 1 in the pair
        indexes = list(self.pair_distances[:, pair_index])
        original_indexes = [
            self.get_original_index_from_unique_index(index, pair_index)
            for index in indexes
        ]
        return [self.original_spectra[index] for index in original_indexes]

    def get_sampled_spectrums(self):
        """
        retrieve the sampled spectrums for the first and second molecule of the pairs
        """
        spectrums_index_0 = self.get_spectrums_from_indexes(0)
        spectrums_index_1 = self.get_spectrums_from_indexes(1)
        return spectrums_index_0, spectrums_index_1
