<p align="center">
  <img src="docs/simba_logo.png" width="200" />
</p>

# SIMBA: Spectral Identification of Molecule Bio-Analogues

**SIMBA** is a transformer-based neural network that accurately predicts chemical structural similarity from tandem mass spectrometry (MS/MS) spectra. Unlike traditional methods relying on heuristic metrics (e.g., modified cosine similarity), SIMBA directly models structural differences, enabling precise analog identification in metabolomics.

SIMBA predicts two interpretable metrics:

1. **Substructure Edit Distance**: Number of molecular graph edits required to convert one molecule into another.
2. **Maximum Common Edge Substructure (MCES) Distance**: Number of bond modifications required to achieve molecular equivalence.

---

## 🚀 Quickstart

### Requirements
- Python 3.11.7
- [Conda](https://docs.conda.io/en/latest/)

### Installation (10–20 minutes)

Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate simba
```

Install the module:
```bash
pip install -e .
```

**Note for macOS users:**
```bash
brew install xz
```

---

## 🔎 Computing Structural Similarities

We provide a pretrained SIMBA model trained on spectra from  **MassSpecGym**. The model operates in positive ionization mode for protonated adducts.

### Usage Example

Follow the [Run Inference Notebook](https://github.com/bittremieux-lab/simba/tree/main/notebooks/final_tutorials/run_inference.ipynb) for a comprehensive tutorial:

- **Runtime:** < 10 minutes (including model/data download)
- **Example data:** data folder.
- **Supported format:** `.mgf`

### Performance

Using an Apple M3 Pro (36 GB RAM):
- **Embedding computation:** ~100,000 spectra in ~1 minute
- **Similarity computation:** 1 query vs. 100,000 spectra in ~10 seconds

SIMBA caches computed embeddings, significantly speeding repeated library searches.

<p align="center">
  <img src="docs/nn_architecture.png" width="600" />
</p>

---
## Analog discovery using SIMBA

Modern metabolomics relies on tandem mass spectrometry (MS/MS) to identify unknown compounds by comparing their spectra against large reference libraries. SIMBA enables analog discovery—finding structurally related molecules—by predicting the 2 complementary, interpretable metrics directly from spectra.

The notebook [Run Analog Discovery Notebook](https://github.com/bittremieux-lab/simba/tree/main/notebooks/final_tutorials/run_analog_discovery.ipynb) presents an analog discovery task based on the MassSpecGym dataset and CASMI2022 dataset.

The notebook shows how to:

* Load a pretrained SIMBA model and MS/MS data.

* Compute distance matrices between query and reference spectra.

* Extract top analogs for a given query.

* Compare predictions against ground truth and visualize the best match. 

## 📚 Training Your Custom SIMBA Model

SIMBA supports training custom models using your own MS/MS datasets in `.mgf` format.

### Step 1: Generate Training Data

Run the script below to generate training data:

```bash
python preprocessing_scripts/final_generation_data.py  \
   --spectra_path=/path/to/your/spectra.mgf   \
   --workspace=/path/to/preprocessed_data/  \
   --MAX_SPECTRA_TRAIN=10000 \
   --mapping_file_name=mapping_unique_smiles.pkl  \
   --PREPROCESSING_NUM_WORKERS=0

```
Spectra_path: Location of spectra
Workspace: Location where the calculated distances are going to be saved
MAX_SPECTRA_TRAIN: 10000 the maximum number of spectra to be processed. Set to a large number to avoid removing spectra
Mapping_file_name: name of the file that saves the mapping of the spectra from spectra to unique compounds.
PROCESSING_NUM_WORKERS: Number of processors to be used. By default 0.

This script will generate a file 'mapping_unique_smiles.pkl' with the specific mapping information between unique compounds and corresponding spectra. As known, each compound can have several spectra and this file saves information about this mapping.

### Output
- Numpy arrays with indexes and structural similarity metrics
- Pickle file (`mapping_unique_smiles.pkl`) mapping spectra indexes to SMILES structures

### Accessing Data Mapping
```python
import pickle

with open('/path/to/output_dir/mapping_unique_smiles.pkl', 'rb') as f:
    data = pickle.load(f)

mol_train = data['molecule_pairs_train']
print(mol_train.df_smiles) 
```
The dataframe df_smiles contains the mapping from indexes of unique compounds to the original spectra loaded.

### Step 2: Model Training

Train your SIMBA model using the following commnad:

```bash

python training_scripts/final_training.py  \
  --CHECKPOINT_DIR=/path/to/checkpoints/ \
  --PREPROCESSING_PICKLE_FILE=mapping_unique_smiles.pkl \
  --PREPROCESSING_DIR_TRAIN=/path/to/preprocessed_data/ \
  --TRAINING_NUM_WORKERS=0  \
  --ACCELERATOR=cpu  \
  --epochs=10 \
  --VAL_CHECK_INTERVAL=10000 
```
CHECKPOINT_DIR: Place where the trained model will be saved
PREPROCESSING_DIR_TRAIN: Folder where the preprocessing files are saved
PREPROCESSING_PICKLE_FILE: File name with the mapping
ACCELERATOR: cpu or gpu
Epochs: Number of epochs to be trained
VAL_CHECK_INTERVAL: Used to check validation performance every N steps. 

The code uses the mapping file produced in the last step and the preprocessing dir folder `PREPROCESSING_DIR_TRAIN` must be the same where the preprocessing files are generated. The best-performing model (lowest validation loss) is saved in `CHECKPOINT_DIR`. 

---

### Step 3: Testing

To test the SIMBA model use the following command:

```bash
 python inference_scripts/inference_multitasking.py \
   --CHECKPOINT_DIR=/path/to/checkpoints/  \
   --PREPROCESSING_DIR=/path/to/preprocessed_data/ \
   --PREPROCESSING_DIR_TRAIN=/path/to/preprocessed_data/  \
   --PREPROCESSING_PICKLE_FILE=mapping_unique_smiles.pkl \
   --UNIFORMIZE_DURING_TESTING=1
```
CHECKPOINT_DIR: Folder where the trained model is saved an testing results will be saved
PREPROCESSING_DIR and PREPROCESSING_DIR_TRAIN: Location where the preprocessing files are saved
PREPROCESSING_PICKLE_FILE: Mapping file
UNIFORMIZE_DURING_TESTING: If to balance the edit distance classes or not.

## 📬 Contact & Support

- **Code repository**: [SIMBA GitHub](https://github.com/bittremieux-lab/simba)
- For questions, issues, or feature requests, please [open an issue](https://github.com/bittremieux-lab/simba/issues).

---

## 📦 Data Availability

- Training and testing datasets available at: [https://zenodo.org/records/15275257].

---

