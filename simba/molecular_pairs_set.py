import numpy as np

from simba.molecule_pair import MoleculePair


class MolecularPairsSet:
    """
    class that encapsulates the indexes and the spectra from where they are retrieved
    """

    def __init__(self, spectra, pair_distances):
        """
        it receives a list of spectra, and a 2D array with the indexes of the spectra
        and the distances between them
        """
        self.spectra = spectra
        self.pair_distances = pair_distances

    @staticmethod
    def adjust_data_format(indexes_tani):
        # Extracting the first two columns and changing their data type to int
        # int_columns = indexes_tani[:, 0:2].astype(np.int32)

        # Extracting the last column and changing its data type to float
        # float_column = indexes_tani[:, 2].astype(np.float16)

        # Combining the modified columns to create a new array
        # new_indexes_tani = np.column_stack((int_columns, float_column))
        return indexes_tani

    def __len__(self):
        return len(self.pair_distances)

    def spectra_equal(self, spectra_0, spectra_1):
        spectra_hash_0 = [s.spectrum_hash for s in spectra_0]
        spectra_hash_1 = [s.spectrum_hash for s in spectra_1]
        return all(
            [s0 == s1 for s0, s1 in zip(spectra_hash_0, spectra_hash_1)]
        )

    def __add__(self, other):
        # only to be used when the spectra are the same

        if self.spectra_equal(self.spectra, other.spectra):
            new_spectra = self.spectra
            new_pair_distances = np.concatenate(
                (self.pair_distances, other.pair_distances), axis=0
            )
            return MolecularPairsSet(
                spectra=new_spectra, pair_distances=new_pair_distances
            )
        else:
            print(
                "ERROR: Attempting to add 2 set of spectra with different content"
            )
            return 0

    def __getitem__(self, index):
        return self.get_molecular_pair(index)

    @staticmethod
    def get_global_variables(spectrum):
        """
        get global variables from a spectrum such as precursor mass
        """
        list_global_variables = [
            spectrum.precursor_mz,
            spectrum.precursor_charge,
        ]
        return np.array(list_global_variables)

    def get_molecular_pair(self, index):
        # i,j,tani = self.indexes_tani[index]
        i = int(self.pair_distances[index, 0])
        j = int(self.pair_distances[index, 1])
        tani = self.pair_distances[index, 2]

        molecule_pair = MoleculePair(
            vector_0=None,
            vector_1=None,
            smiles_0=self.spectra[i].smiles,
            smiles_1=self.spectra[j].smiles,
            similarity=tani,
            global_feats_0=MolecularPairsSet.get_global_variables(
                self.spectra[i]
            ),
            global_feats_1=MolecularPairsSet.get_global_variables(
                self.spectra[j]
            ),
            index_in_spectrum_0=i,  # index in the spectrum list used as input
            index_in_spectrum_1=j,
            spectrum_object_0=self.spectra[i],
            spectrum_object_1=self.spectra[j],
            params_0=self.spectra[i].params,
            params_1=self.spectra[j].params,
        )

        return molecule_pair

    def get_molecular_pairs(self, indexes):
        # create dataset
        molecule_pairs = []

        if indexes is None:
            iterator = self.pair_distances
        else:
            iterator = self.pair_distances[indexes]

        for i, j, tani in iterator:
            molecule_pair = MoleculePair(
                vector_0=None,
                vector_1=None,
                smiles_0=self.spectra[i].smiles,
                smiles_1=self.spectra[j].smiles,
                similarity=tani,
                global_feats_0=MolecularPairsSet.get_global_variables(
                    self.spectra[i]
                ),
                global_feats_1=MolecularPairsSet.get_global_variables(
                    self.spectra[j]
                ),
                index_in_spectrum_0=i,  # index in the spectrum list used as input
                index_in_spectrum_1=j,
                spectrum_object_0=self.spectra[i],
                spectrum_object_1=self.spectra[j],
                params_0=self.spectra[i].params,
                params_1=self.spectra[j].params,
            )
            molecule_pairs.append(molecule_pair)

        return molecule_pairs

    def remove_duplicates(self):
        self.pair_distances = np.unique(self.pair_distances, axis=0)
        return self

    def get_janssen_pairs(self):
        """
        filter our pairs that are not from janssen
        """
        indexes_tani = []
        for i, m in enumerate([mol for mol in self]):
            if (m.spectrum_object_0.library == "janssen") and (
                m.spectrum_object_1.library == "janssen"
            ):
                # molecule_pairs.append(m)
                indexes_tani.append(self.pair_distances[i])

        molecule_pairs = MolecularPairsSet(
            spectra=self.spectra, pair_distances=np.array(indexes_tani)
        )
        return molecule_pairs

    def get_gnps_pairs(self):
        """
        filter only pairs that have exclusively gnps data
        """
        indexes_tani = []
        for i, m in enumerate([mol for mol in self]):
            if (
                "spectrumid" in m.params_0.keys()
                and "spectrumid" in m.params_1.keys()
            ):
                if m.params_0["spectrumid"].startswith(
                    "CCMSLIB"
                ) and m.params_1["spectrumid"].startswith("CCMSLIB"):
                    # molecule_pairs.append(m)
                    indexes_tani.append(self.pair_distances[i])

        molecule_pairs = MolecularPairsSet(
            spectra=self.spectra, pair_distances=np.array(indexes_tani)
        )
        return molecule_pairs

    def get_no_gnps_pairs(self):
        """
        filter any of the gnps data out
        """
        indexes_tani = []
        for i, m in enumerate([mol for mol in self]):
            if "spectrumid" in m.params_0.keys():
                if m.params_0["spectrumid"].startswith("CCMSLIB"):
                    pass
                else:
                    indexes_tani.append(self.pair_distances[i])
            elif "spectrumid" in m.params_1.keys():
                if m.params_1["spectrumid"].startswith("CCMSLIB"):
                    pass
                else:
                    indexes_tani.append(self.pair_distances[i])
            else:
                indexes_tani.append(self.pair_distances[i])

        molecule_pairs = MolecularPairsSet(
            spectra=self.spectra, pair_distances=np.array(indexes_tani)
        )
        return molecule_pairs

    # remove janssen pairs from training and validation
    def remove_library_pairs(self, library):
        spectrums = self.spectra
        indexes_tani = self.pair_distances
        new_indexes_tani = [
            row
            for row in indexes_tani
            if (
                (spectrums[int(row[0])].library != library)
                and (spectrums[int(row[1])].library != library)
            )
        ]
        return MolecularPairsSet(
            spectra=spectrums, pair_distances=new_indexes_tani
        )

    def filter_by_similarity(self, min_sim, max_sim):
        new_indexes_tani = self.pair_distances[
            (self.pair_distances[:, 2] >= min_sim)
            & (self.pair_distances[:, 2] <= max_sim)
        ]
        new_mols = MolecularPairsSet(
            spectra=self.spectra, pair_distances=new_indexes_tani
        )
        return new_mols
