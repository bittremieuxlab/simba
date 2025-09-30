# Extract the bracketed list using regex
import re
import ast
from rdkit import Chem
import pickle
from simba.simba.ground_truth import GroundTruth
import matplotlib.pyplot as plt
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdRascalMCES


#MIN_SUBSTRUCTURE_SIZE=1000000000
#MIN_SUBSTRUCTURE_SIZE=10
MIN_SUBSTRUCTURE_SIZE=10

def get_python_list_from_string(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        python_list = ast.literal_eval(match.group(0))
    else:
        python_list=None
    return python_list



# simba
def compute_coverage(input_data, predicted_substructure, use_mcs=True, queries_to_process=None):

    if queries_to_process is not None:
        queries = queries_to_process
    else:
        queries= list(predicted_substructure)
    coverage= {}
    for index,target_query in  enumerate(queries):
            print(f'*** Processing query: {index}')
            list_substructures= [m for m in predicted_substructure[target_query] if m is not None]
            cov = substructure_average_coverage(input_data['data'][target_query]['smiles_query'], list_substructures, use_mcs=use_mcs,)
            coverage[target_query] = cov
    return coverage


from rdkit import Chem
from rdkit.Chem import rdFMCS

def get_opts_rdkit():
    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = 0
    opts.allBestMCESs = True
    opts.returnEmptyMCES = True
    opts.singleLargestFrag = True
    return opts
def substructure_average_coverage(parent_smiles, sub_smiles_list, use_mcs=True, use_chirality=False, max_matches=10000):
    """
    Returns (average_coverage, per_structure_coverages)
    - average_coverage: mean fraction of atoms covered by each substructure
    - per_structure_coverages: list of coverage fractions per substructure
    Notes:
      * Each substructure is evaluated independently (no greedy union).
      * Coverage = unique atoms covered by substructure / total atoms in parent.
      * Invalid substructures are skipped.
    """
    # mces options
    opts = get_opts_rdkit()

    print(f'Parent smiles: {parent_smiles}')
    mol = Chem.MolFromSmiles(parent_smiles,sanitize=False)
    total_atoms = mol.GetNumAtoms()
    coverages_parent = []
    coverages_sub = []
    substructures= []
    all_occurrences = []

    precision = 0
    for index, s in enumerate(sub_smiles_list):
        try:
            if use_mcs:
                mol2 = Chem.MolFromSmiles(s,  sanitize=False)
                if (mol2.GetNumAtoms()>MIN_SUBSTRUCTURE_SIZE):
                    #res  = rdRascalMCES.FindMCES(mol, mol2, opts)
                    #smarts= res[0].smartsString
                    res = rdFMCS.FindMCS([mol, mol2])
                    smarts = res.smartsString
                    sub = Chem.MolFromSmarts(smarts)
                    print(f'index: {index}, subtructure: {Chem.MolToSmiles(sub)}')

                    # if the common substructure is less than 10 atoms take the original motif as the substructure
                    if sub.GetNumAtoms()<MIN_SUBSTRUCTURE_SIZE:
                        sub = mol2
                else:
                    sub = mol2
            else:
                sub = Chem.MolFromSmiles(s, sanitize=False)
            substructures.append(Chem.MolToSmiles(sub))
        except Exception as e:
            print(f'Error processing substructure: {s}: {e}')
            coverages_parent.append(0)
            coverages_sub.append(0)
            continue
        
        if sub is None:
            coverages_parent.append(0)
            coverages_sub.append(0)
            continue

        
        if mol.HasSubstructMatch(sub):
            precision = precision+1
            matches_parent = mol.GetSubstructMatches(sub, useChirality=use_chirality, maxMatches=max_matches, uniquify=0)
        else:
            matches_parent= []
        

        #if not matches_parent:
        #    coverages_parent.append(0)


        # Union of all atom indices matched by this substructure
        covered_atoms = set()

        for m in matches_parent:
            covered_atoms.update(m)
        
        
        coverage_fraction_parent = len(covered_atoms) / total_atoms

        if (sub is not None) or (sub.GetNumAtoms()>0):
            coverage_fraction_sub =  sub.GetNumAtoms()/mol2.GetNumAtoms()
            print(coverage_fraction_sub)
        else:
            coverage_fraction_sub =  0

        coverages_parent.append(coverage_fraction_parent)
        coverages_sub.append(coverage_fraction_sub)

        all_occurrences.extend([tuple(m) for m in matches_parent])

    if len(sub_smiles_list)>0:
        precision = precision/len(sub_smiles_list)
    else:
        precision=0
    # Greedy maximum coverage over occurrences
    covered = set()
    picked = []
    # Convert to sets for fast set ops
    occ_sets = [set(m) for m in all_occurrences]

    while True:
        best_i = -1
        best_gain = 0
        for i, occ in enumerate(occ_sets):
            gain = len(occ - covered)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_gain == 0:
            break
        covered |= occ_sets[best_i]
        picked.append(all_occurrences[best_i])

    total_coverage = len(covered) / total_atoms


    if len(coverages_parent)==0:
        coverages_parent=[0]
    if len(coverages_sub) == 0:
        coverages_sub= [0]
    return {'total_coverage':total_coverage , 
                'coverages_parent':coverages_parent, 
                'coverages_sub':coverages_sub, 
                'substructures': substructures,
                'precision':precision}


def substructure_list_coverage(parent_smiles, sub_smiles_list, use_mcs=True, use_chirality=False, max_matches=10000, ):
    """
    Returns (coverage_fraction, covered_atom_indices, picked_matches)
    - coverage_fraction: unique atoms covered / total atoms in parent
    - covered_atom_indices: set of atom indices covered
    - picked_matches: list of tuples of atom indices (the chosen occurrences)
    Notes:
      * Counts heavy atoms (default RDKit behavior). 
      * Overlaps don't double-count. Nested patterns don't add extra coverage.
      * Greedy selection over all match occurrences for maximum coverage.
    """
    # mces options
    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = 0
    opts.allBestMCESs = True
    opts.returnEmptyMCES = True
    opts.singleLargestFrag = True

    mol = Chem.MolFromSmiles(parent_smiles, sanitize=False)
    if mol is None:
        raise ValueError("Invalid parent SMILES")

    total_atoms = mol.GetNumAtoms()
    if total_atoms == 0:
        return 0.0, set(), []

    # Collect every match occurrence from every substructure
    all_occurrences = []
    for index, s in enumerate(sub_smiles_list):
        try:
            if use_mcs:
                mol2 = Chem.MolFromSmiles(s,sanitize=False)
                if (mol2.GetNumAtoms()>MIN_SUBSTRUCTURE_SIZE):
                    #res  = rdRascalMCES.FindMCES(mol, mol2, opts)
                    #smarts= res[0].smartsString
                    res = rdFMCS.FindMCS([mol, mol2])
                    smarts = res.smartsString
                    sub = Chem.MolFromSmarts(smarts)
                    print(f'index: {index}, subtructure: {Chem.MolToSmiles(sub)}')
                else:
                    sub = mol2 
            else:
                sub = Chem.MolFromSmiles(s,sanitize=False)
        except Exception as e:
            print(f'Error {e}')
            sub=None
        if sub is None:
            continue  # skip invalid substructure
        matches = mol.GetSubstructMatches(sub, useChirality=use_chirality, maxMatches=max_matches, uniquify=0)


        all_occurrences.extend([tuple(m) for m in matches])

    if not all_occurrences:
        return 0.0, set(), []

    # Greedy maximum coverage over occurrences
    covered = set()
    picked = []
    # Convert to sets for fast set ops
    occ_sets = [set(m) for m in all_occurrences]


    while True:
        best_i = -1
        best_gain = 0
        for i, occ in enumerate(occ_sets):
            gain = len(occ - covered)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_gain == 0:
            break
        covered |= occ_sets[best_i]
        picked.append(all_occurrences[best_i])

    coverage = len(covered) / total_atoms

    print(f'total_atoms: {total_atoms}')
    return coverage, covered, picked


def get_predicted_substructures(results_agent):
    predicted_substructure = {}
    for target_query in results_agent:
        python_list= get_python_list_from_string(results_agent[target_query]['result_list_substructures']) 
        predicted_substructure[target_query] = python_list
    return predicted_substructure