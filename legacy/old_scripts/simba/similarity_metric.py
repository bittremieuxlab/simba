from matchms import calculate_scores
from matchms.similarity import FingerprintSimilarity
from tqdm import tqdm


class SimilarityMetric:

    @staticmethod
    def compute_scores_tanimoto(
        molecule_pairs,
        preprocessed_spectra,
        target_hashes_subset,
        similarity,
        compute_tanimoto=False,  # if to retrieve the similarity from the molecular pairs
    ):
        tanimotos = []
        scores = []

        # Build hash lookup dict
        spectrum_hash_dict = {
            hash_val: spectrum
            for spectrum, hash_val in zip(
                preprocessed_spectra, target_hashes_subset
            )
        }
        for m in tqdm(molecule_pairs):
            hash_0 = m.spectrum_object_0.spectrum_hash
            hash_1 = m.spectrum_object_1.spectrum_hash

            # get right spectra
            spectrum_found_0_ms = spectrum_hash_dict.get(hash_0, None)
            spectrum_found_1_ms = spectrum_hash_dict.get(hash_1, None)

            # calculate scores
            if (spectrum_found_0_ms is not None) and (
                spectrum_found_1_ms is not None
            ):
                score = calculate_scores(
                    [spectrum_found_0_ms], [spectrum_found_1_ms], similarity
                )

                score = score.scores_by_query(spectrum_found_1_ms, sort=True)[
                    0
                ][1]

                if compute_tanimoto:
                    tanimoto_measure = FingerprintSimilarity(
                        similarity_measure="jaccard"
                    )
                    tani = tanimoto_measure.pair(
                        spectrum_found_0_ms, spectrum_found_1_ms
                    )
                else:
                    tani = m.similarity
                tanimotos.append(tani)
                # tanimotos.append(m.similarity)
                scores.append(score)
            else:
                tanimotos.append(None)
                scores.append(None)
        return tanimotos, scores
