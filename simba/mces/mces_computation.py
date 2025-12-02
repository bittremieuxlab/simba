import itertools
import multiprocessing
import os
import subprocess
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from simba.config import Config
from simba.edit_distance import edit_distance
from simba.load_mces.load_mces import LoadMCES
from simba.logger_setup import logger
from simba.molecular_pairs_set import MolecularPairsSet
from simba.molecule_pairs_opt import MoleculePairsOpt
from simba.spectrum_ext import SpectrumExt
from simba.train_utils import TrainUtils


class MCES:
    @staticmethod
    def compute_all_mces_results_unique(
        original_spectra: List[SpectrumExt],
        max_combinations: int = 1000000,
        num_workers: int = 15,
        random_sampling: bool = True,
        config: Config = None,
        identifier: str = "",
        use_edit_distance: bool = False,
        loaded_molecule_pairs: Optional[MolecularPairsSet] = None,
    ) -> MoleculePairsOpt:
        """
        Compute MCES or edit distance for all pairs of spectra using multiprocessing.

        Parameters
        ----------
        original_spectra : List[SpectrumExt]
            List of all spectrum objects.
        max_combinations : int, optional
            Maximum number of combinations to compute, by default 1000000.
        num_workers : int, optional
            Number of worker processes to use, by default 15.
        random_sampling : bool, optional
            If True, use random sampling; otherwise, compute all pairs, by default True.
        config : Config, optional
            Configuration object containing parameters, by default None.
        identifier : str, optional
            Identifier for the computation run, by default "".
        use_edit_distance : bool, optional
            If True, compute edit distance instead of MCES, by default False.
        loaded_molecule_pairs : Optional[MolecularPairsSet]
            Precomputed molecule pairs to use instead of computing new ones, by default None.

        Returns
        -------
        MoleculePairsOpt
            An object containing the spectra and their computed pair distances.
        """

        logger.info("Computing MCES results based on unique smiles")

        if loaded_molecule_pairs is None:
            # get dummy spectra for unique smiles and associated metadata
            spectra_unique, df_smiles = TrainUtils.get_unique_spectra(
                original_spectra
            )

            molecular_pairs = MCES.compute_all_mces_results_exhaustive(
                spectra_unique,
                max_combinations=max_combinations,
                num_workers=num_workers,
                random_sampling=random_sampling,
                config=config,
                identifier=identifier,
                use_edit_distance=use_edit_distance,
            )
        else:
            molecular_pairs = loaded_molecule_pairs
            df_smiles = molecular_pairs.df_smiles
            spectra_unique = molecular_pairs.spectra
            original_spectra = molecular_pairs.original_spectra

        return MoleculePairsOpt(
            original_spectra=original_spectra,
            unique_spectra=spectra_unique,
            df_smiles=df_smiles,
            pair_distances=molecular_pairs.pair_distances,
        )

    @staticmethod
    def create_combinations(all_spectra):

        logger.info(f"Number of unique spectra:{len(all_spectra)}")
        indexes = [i for i in range(0, len(all_spectra))]
        combinations = list(itertools.combinations(indexes, 2))
        return combinations

    @staticmethod
    def create_input_df(smiles, indexes_0, indexes_1):
        df = pd.DataFrame()
        print(f"Length of smiles array: {len(smiles)}")
        print(f"Max value of indexes_0:{max(indexes_0)}")
        print(f"Max value of indexes_0:{max(indexes_1)}")
        df["smiles_0"] = [smiles[int(r)] for r in indexes_0]
        df["smiles_1"] = [smiles[int(r)] for r in indexes_1]

        return df

    @staticmethod
    def normalize_mces(mces, max_mces=5):
        # asuming series
        # normalize mces. the higher the mces the lower the similarity
        # mces_normalized = mces.apply(lambda x:x if x<=max_mces else max_mces)
        # return mces_normalized.apply(lambda x:(1-(x/max_mces)))

        ## asuming numpy
        print(f"Example of input mces: {mces}")
        mces_normalized = mces.copy()
        mces_normalized[mces_normalized >= max_mces] = max_mces
        mces_normalized = 1 - (mces_normalized / max_mces)
        print(f"Example of normalized mces: {mces_normalized}")
        return mces_normalized

    def compute_mces_myopic(
        smiles,
        sampled_index,
        size_batch,
        id,
        random_sampling,
        config,
        split_group="train",  # if it is train, val or test
        to_compute_indexes_np=None,  # the indexes to be computed
    ):

        # where to save results

        # initialize randomness

        if config.COMPUTE_SPECIFIC_PAIRS:
            size_batch_effective = to_compute_indexes_np.shape[0]
            indexes_np = np.zeros(
                (int(size_batch_effective), 3),
            )
            indexes_np[:, 0:2] = to_compute_indexes_np[:, 0:2]
        else:
            indexes_np = np.zeros(
                (int(size_batch), 3),
            )
            if random_sampling:
                np.random.seed(id)
                indexes_np[:, 0] = np.random.randint(
                    0, len(smiles), int(size_batch)
                )
                indexes_np[:, 1] = np.random.randint(
                    0, len(smiles), int(size_batch)
                )
            else:
                indexes_np[:, 0] = sampled_index
                indexes_np[:, 1] = np.arange(0, size_batch)

        # indexes_np[:,0]=sampled_index*np.ones((size_batch,))
        # indexes_np[:,1]= np.arange(0,len(all_spectrums))

        print("Creating df for computation")
        # print('ground truth')
        # print(random_first_index[index:index+size_batch])
        df = MCES.create_input_df(smiles, indexes_np[:, 0], indexes_np[:, 1])

        print("Saving df ...")
        df[["smiles_0", "smiles_1"]].to_csv(
            f"{config.PREPROCESSING_DIR}input_{str(id)}.csv", header=False
        )

        # compute mces
        # command = 'myopic_mces  ./input.csv ./output.csv'
        print("Running myopic ...")
        command = ["myopic_mces"]
        # command = ['myopic_mces', f'./input_{str(id)}.csv', f'./output_{str(id)}.csv']

        # Add the argument --num_jobs 15
        # command.extend(['--num_jobs', '32'])
        command.extend([f"{config.PREPROCESSING_DIR}input_{str(id)}.csv"])
        command.extend([f"{config.PREPROCESSING_DIR}output_{str(id)}.csv"])
        command.extend(["--num_jobs", "1"])
        # command.extend(['--solver','CPLEX_CMD'])
        # Define threshold
        command.extend(["--threshold", str(int(config.THRESHOLD_MCES))])

        command.extend(["--solver_onethreaded"])
        command.extend(["--solver_no_msg"])
        command.extend(["--choose_bound_dynamically"])

        # x = subprocess.run(command,capture_output=True)
        subprocess.run(command)
        print("Finished myopic")

        # read results
        print("reading csv")
        results = pd.read_csv(
            f"{config.PREPROCESSING_DIR}output_{str(id)}.csv", header=None
        )
        os.system(f"rm {config.PREPROCESSING_DIR}input_{str(id)}.csv")
        os.system(f"rm {config.PREPROCESSING_DIR}output_{str(id)}.csv")
        df["mces"] = results[2]  # the column 2 is the mces result

        # normalize mces. the higher the mces the lower the similarity
        # df['mces_normalized'] = MCES.normalize_mces(df['mces'])
        df["mces_normalized"] = df["mces"]

        # print('saving intermediate results')
        indexes_np[:, 2] = df["mces_normalized"].values
        return indexes_np

    def get_samples(
        all_spectra: List[SpectrumExt],
        random_sampling: bool,
        max_combinations: int,
        batch_size_random: int = 100,
    ):
        """
        Get sample indices. Supports random and deterministic sampling.
        Returns all spectra if random_sampling is False.

        Parameters
        ----------
        all_spectra : List[SpectrumExt]
            List of all spectrum objects.
        random_sampling : bool
            If True, use random sampling; otherwise, use deterministic sampling.
        max_combinations : int
            Maximum number of combinations to sample.
        batch_size_random : int, optional
            Size of each batch when using random sampling, by default 100.

        Returns
        -------
        sample_idx : np.ndarray
            Array of sampled spectrum indices.
        batch_size : int
            Size of each batch.
        """
        if random_sampling:
            logger.info("Random sampling")
            batch_size = batch_size_random
            num_samples = np.floor(max_combinations / batch_size)
            sample_idx = np.random.randint(
                0, len(all_spectra), int(num_samples)
            )
        else:
            logger.info("No random sampling")
            batch_size = len(all_spectra)
            sample_idx = np.arange(0, len(all_spectra))
        return sample_idx, batch_size

    def compute_all_mces_results_exhaustive(
        all_spectra: List[SpectrumExt],
        max_combinations: int = 1000000,
        num_workers: int = 15,
        random_sampling: bool = True,
        config: Config = None,
        identifier: str = "",
        use_edit_distance=False,
    ) -> MolecularPairsSet:
        """
        Compute MCES or edit distance for all pairs of spectra using multiprocessing.

        Parameters
        ----------
        all_spectra : List[SpectrumExt]
            List of all spectrum objects.
        max_combinations : int, optional
            Maximum number of combinations to compute, by default 1000000.
        num_workers : int, optional
            Number of worker processes to use, by default 15.
        random_sampling : bool, optional
            If True, use random sampling; otherwise, compute all pairs, by default True.
        config : Config, optional
            Configuration object containing parameters, by default None.
        identifier : str, optional
            Identifier for the computation run, by default "".
        use_edit_distance : bool, optional
            If True, compute edit distance instead of MCES, by default False.

        Returns
        -------
        MolecularPairsSet
            An object containing the spectra and their computed pair distances.
        """
        num_workers = num_workers if num_workers > 0 else 1

        logger.info(
            f"Starting computation of molecule pairs with {num_workers} workers"
        )

        # sample spectrum indices
        sample_idx, batch_size = MCES.get_samples(
            all_spectra, random_sampling, max_combinations
        )

        all_smiles = [s.params["smiles"] for s in all_spectra]

        if config.COMPUTE_SPECIFIC_PAIRS:  ## specifically for mces
            directory_path = config.PREPROCESSING_DIR
            prefix = config.FORMAT_FILE_SPECIFIC_PAIRS + identifier
            indices_np_loaded = LoadMCES.load_raw_data(directory_path, prefix)

            print(
                f"Size of the pairs loaded for computing specific pairs: {indices_np_loaded.shape[0]}"
            )

            chunk_size = config.PREPROCESSING_BATCH_SIZE * num_workers
            num_chunks = int(np.ceil(indices_np_loaded.shape[0] / chunk_size))
            chunks = np.array_split(indices_np_loaded, num_chunks)

        else:
            # Split the sample idx array into chunks of size chunk_size
            chunk_size = num_workers * 10
            num_chunks = int(np.ceil(len(sample_idx) / chunk_size))
            chunks = np.array_split(sample_idx, num_chunks)

        if num_chunks == 0:
            logger.info(
                "No pairs to process; returning empty MolecularPairsSet."
            )
            return MolecularPairsSet(
                spectra=all_spectra,
                pair_distances=np.empty((0, 3)),
            )

        logger.info(f"Number of chunks: {num_chunks}")
        logger.info(f"Size of each chunk: {chunks[0].shape[0]}")
        logger.info(
            f"Size of each sub-chunk: {np.array_split(chunks[0], num_workers)[0].shape[0]}"
        )

        computed_pair_distances = np.empty((0, 3))

        for chunk_idx, chunk in enumerate(chunks):
            # select chunks to compute based on node number
            if (
                (chunk_idx % config.PREPROCESSING_NUM_NODES)
                == config.PREPROCESSING_CURRENT_NODE
            ) or (config.PREPROCESSING_CURRENT_NODE is None):
                prefix_file = (
                    "edit_distance_" if use_edit_distance else "mces_"
                )
                filename = (
                    f"{config.PREPROCESSING_DIR}"
                    + prefix_file
                    + f"indexes_tani_incremental{identifier}_{str(chunk_idx)}.npy"
                )

                if not (
                    os.path.exists(filename)
                ):  # do not overwrite existing files
                    logger.info(f"Processing split {chunk_idx}/{len(chunks)}")

                    pool = multiprocessing.Pool(processes=num_workers)

                    mols = [Chem.MolFromSmiles(s) for s in all_smiles]
                    fpgen = AllChem.GetRDKitFPGenerator(maxPath=3, fpSize=512)
                    fps = [fpgen.GetFingerprint(m) for m in mols]

                    if config.COMPUTE_SPECIFIC_PAIRS:
                        logger.info(
                            "Computing specific pairs from loaded indexes ..."
                        )
                        logger.info(f"Size of each array {chunk.shape[0]}")
                        sub_arrays = np.array_split(chunk, num_workers)
                        logger.info(
                            f"Size of each sub-array {sub_arrays[0].shape[0]}"
                        )

                        results = [
                            pool.apply_async(
                                edit_distance.compute_ed_or_mces,  # TODO: fix this
                                args=(
                                    all_smiles,
                                    None,
                                    batch_size,
                                    (chunk_idx * chunks[0].shape[0])
                                    + sub_index,
                                    None,
                                    config,
                                    fps,
                                    mols,
                                    use_edit_distance,
                                ),
                            )
                            for sub_index, sampled_array in enumerate(
                                sub_arrays
                            )
                        ]
                    else:
                        results = [
                            pool.apply_async(
                                edit_distance.compute_ed_or_mces,
                                args=(
                                    all_smiles,
                                    sampled_index,
                                    batch_size,
                                    (chunk_idx * len(chunks[0])) + sub_index,
                                    random_sampling,
                                    config,
                                    fps,
                                    mols,
                                    use_edit_distance,
                                ),
                            )
                            for sub_index, sampled_index in enumerate(chunk)
                        ]

                    # Close the pool and wait for all processes to finish
                    pool.close()
                    pool.join()

                    # Get results from async objects
                    computed_pair_distances = np.concatenate(
                        [result.get() for result in results], axis=0
                    )
                    # create parent directory if it does not exist
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    # save distances per chunk
                    np.save(arr=computed_pair_distances, file=filename)

        logger.info(
            f"Computed distances for {computed_pair_distances.shape[0]} pairs."
        )

        molecular_pair_set = MolecularPairsSet(
            spectra=all_spectra, pair_distances=computed_pair_distances
        )

        return molecular_pair_set

    @staticmethod
    def compute_all_mces_results_exhaustive_single_thread(
        all_spectrums,
        max_combinations=1000000,
        limit_low_tanimoto=True,
        max_low_pairs=0.5,
        use_tqdm=True,
        max_mass_diff=None,  # maximum number of elements in which we stop adding new items
        min_mass_diff=0,
        num_workers=15,
        MIN_SIM=0.8,
        MAX_SIM=1,
        high_tanimoto_range=0.5,
    ):

        logger.info("Starting computation of molecule pairs")

        print("Initializing big array")
        size_batch = len(all_spectrums)
        number_sampled_spectrums = np.floor(max_combinations / size_batch)
        indexes_np = np.zeros(
            (int(number_sampled_spectrums * size_batch), 3),
        )
        random_samples = np.random.randint(
            0, len(all_spectrums), int(number_sampled_spectrums)
        )

        # for index in tqdm(range(0,max_combinations, size_batch)):
        for i, sampled_index in tqdm(enumerate(random_samples)):
            # for index in tqdm(range(0, size_combinations,size_batch)):
            # compute the consecutive pairs
            indexes_np[i : i + size_batch, 0] = sampled_index * np.ones(
                (size_batch,)
            )
            indexes_np[i : i + size_batch, 1] = np.arange(
                0, len(all_spectrums)
            )

            print(f"{sampled_index}")
            print("Creating df for computation")

            # print('ground truth')
            # print(random_first_index[index:index+size_batch])
            df = MCES.create_input_df(
                all_spectrums,
                indexes_np[:, 0][i : i + size_batch],
                indexes_np[:, 1][i : i + size_batch],
            )

            print("Saving df ...")
            df[["smiles_0", "smiles_1"]].to_csv("./input.csv", header=False)

            # compute mces
            # command = 'myopic_mces  ./input.csv ./output.csv'
            print("Running myopic ...")
            command = ["myopic_mces", "./input.csv", "./output.csv"]

            # Add the argument --num_jobs 15
            command.extend(["--num_jobs", "1"])
            # command.extend(['--num_jobs', '32'])
            # command.extend(['--num_jobs', '32'])

            # Define threshold
            command.extend(["--threshold", "5"])

            # command.extend(['--solver_onethreaded'])
            command.extend(["--solver_no_msg"])

            # x = subprocess.run(command,capture_output=True)
            subprocess.run(command)
            print("Finished myopic")

            # read results
            print("reading csv")
            results = pd.read_csv("./output.csv", header=None)
            # os.system('rm ./input.csv')
            # os.system('rm ./output.csv')
            df["mces"] = results[2]  # the column 2 is the mces result

            # normalize mces. the higher the mces the lower the similarity
            df["mces_normalized"] = MCES.normalize_mces(df["mces"])

            # print('saving intermediate results')
            indexes_np[i : i + size_batch, 2] = df["mces_normalized"].values

        print(f"Number of effective pairs retrieved: {indexes_np.shape[0]} ")

        print("Example of pairs retreived:")
        print(indexes_np[0:1000])
        # molecular_pair_set= MolecularPairsSet(spectrums=all_spectrums,indexes_tani= indexes)
        molecular_pair_set = MolecularPairsSet(
            spectra=all_spectrums, pair_distances=indexes_np
        )

        print(datetime.now())
        return molecular_pair_set, df

    def exp_normalize_mces20(x, scale=20, low_threshold=0.20):
        """
        normalize the input np array

        values of 0 are set to 1
        values of infinite are set to 0

        if we set scale=2, the mces distance of 20 corresponds to ~0.1
        """
        mces_normalized = 1 / (1 + (x / scale))

        mces_normalized[mces_normalized <= low_threshold] = 0

        return mces_normalized

    def inverse_exp_normalize_mces20(
        mces_normalized, scale, epsilon=0.000000000001
    ):

        # add epsilon to avoid divide by 0
        mces_normalized_epsilon = mces_normalized + epsilon

        return scale * ((1 / mces_normalized_epsilon) - 1)

    def compute_mces_list_smiles(smiles_0, smiles_1, threshold_mces=20):

        if not (os.path.exists(f"temp")):
            os.mkdir(f"temp")

        input_csv_file = f"temp/smiles_myopic_input.csv"
        output_csv_file = f"temp/smiles_myopic_output.csv"

        df = pd.DataFrame()

        df["smiles_0"] = smiles_0
        df["smiles_1"] = smiles_1

        df[["smiles_0", "smiles_1"]].to_csv(input_csv_file, header=False)

        # compute mces
        # command = 'myopic_mces  ./input.csv ./output.csv'
        print("Running myopic ...")
        command = ["myopic_mces"]
        # command = ['myopic_mces', f'./input_{str(id)}.csv', f'./output_{str(id)}.csv']

        # Add the argument --num_jobs 15
        # command.extend(['--num_jobs', '32'])
        command.extend([input_csv_file])
        command.extend([output_csv_file])
        command.extend(["--num_jobs", "1"])

        # Define threshold
        command.extend(["--threshold", str(int(threshold_mces))])

        command.extend(["--solver_onethreaded"])
        command.extend(["--solver_no_msg"])

        # x = subprocess.run(command,capture_output=True)
        subprocess.run(command)
        print("Finished myopic")

        # read results
        print("reading csv")
        results = pd.read_csv(output_csv_file, header=None)
        os.system(f"rm {input_csv_file}")
        os.system(f"rm {output_csv_file}")
        df["mces"] = results[2]  # the column 2 is the mces result

        os.system(f"rm -r temp")
        return df
