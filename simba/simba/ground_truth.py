import numpy as np

from simba.edit_distance.edit_distance import EditDistance
from simba.mces.mces_computation import MCES
from simba.tanimoto import Tanimoto
from myopic_mces.myopic_mces import MCES as MCES2


class GroundTruth:

    def compute_edit_distance(spectra0, spectra1, max_value=5):
        ground_truth_ed = np.zeros((len(spectra0), len(spectra1)))
        smiles0 = [s.params["smiles"] for s in spectra0]
        smiles1 = [s.params["smiles"] for s in spectra1]

        for i, s0 in enumerate(smiles0):
            for j, s1 in enumerate(smiles1):
                ground_truth_ed[i, j] = EditDistance.get_edit_distance_from_smiles(
                    s0, s1, return_nans=True
                )

        ground_truth_ed[np.isnan(ground_truth_ed)] = max_value
        ground_truth_ed[ground_truth_ed >= max_value] = max_value
        return ground_truth_ed

    def compute_mces(spectra0, spectra1, threshold=20):

        ground_truth_mces = np.zeros((len(spectra0), len(spectra1)))
        smiles0 = [s.params["smiles"] for s in spectra0]
        smiles1 = [s.params["smiles"] for s in spectra1]

        for j, s1 in enumerate(smiles1):
            for i, s0 in enumerate(smiles0):

                    #df_results = MCES.compute_mces_list_smiles([s0] * len(smiles1), smiles1)
                    #mces_result = df_results["mces"]

                    result = MCES2(
                            s0,
                            s1,
                            threshold=threshold,
                            i=0,
                            # solver='CPLEX_CMD',       # or another fast solver you have installed
                            solver="PULP_CBC_CMD",
                            solver_options={
                                "threads": 1,
                                "msg": False,
                                "timeLimit": 10,  # Stop CBC after 1 seconds
                            },
                            no_ilp_threshold=False,  # allow the ILP to stop early once the threshold is exceeded
                            always_stronger_bound=False,  # use dynamic bounding for speed
                            catch_errors=False,  # typically raise exceptions if something goes wrong
                        )
                    distance = result[1]
                    time_taken = result[2]
                    exact_answer = result[3]
                    mces_result = distance

                    ground_truth_mces[i, j] = mces_result
                    
        return ground_truth_mces

    def compute_tanimoto(spectra0, spectra1):

        ground_tanimoto = np.zeros((len(spectra0), len(spectra1)))
        smiles0 = [s.params["smiles"] for s in spectra0]
        smiles1 = [s.params["smiles"] for s in spectra1]

        for i, s0 in enumerate(smiles0):
            for j, s1 in enumerate(smiles1):
                ground_tanimoto[i, j] = Tanimoto.compute_tanimoto_from_smiles(
                    s0,
                    s1,
                )

        return ground_tanimoto
