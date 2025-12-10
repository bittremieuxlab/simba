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
- [UV](https://docs.astral.sh/uv/) (recommended) or [Conda](https://docs.conda.io/en/latest/)

### Installation

#### Option 1: UV (Recommended - Fastest ⚡)

**Install UV:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Setup SIMBA (~2-5 minutes):**
```bash
# Clone the repository
git clone https://github.com/bittremieux-lab/simba.git
cd simba

# Create virtual environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

**For Jupyter notebooks:**
```bash
# Install notebook dependencies
uv sync --extra notebooks

# Register the kernel
python -m ipykernel install --user --name=simba --display-name="SIMBA (UV)"
```

**To use notebooks in VS Code:**
1. Open any `.ipynb` file in the `notebooks/` folder
2. Click "Select Kernel" in the top-right corner
3. Choose "SIMBA (UV)" or "Python 3.11 (.venv: venv)"
4. If the kernel doesn't appear, reload VS Code window (Cmd+Shift+P → "Developer: Reload Window")

#### Option 2: Conda (Alternative)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate simba

# Install the module
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

Preprocess your MS/MS spectral data using one of the following methods:

#### Option 1: CLI Command (Recommended)

```bash
simba preprocess \
  --spectra-path /path/to/your/spectra.mgf \
  --workspace /path/to/preprocessed_data \
  --max-spectra-train 10000 \
  --mapping-file-name mapping_unique_smiles.pkl \
  --num-workers 0
```

**Parameters:**
* `--spectra-path`: Path to input spectra file (.mgf format)
* `--workspace`: Directory where preprocessed data will be saved
* `--max-spectra-train`: Maximum number of spectra to process for training (default: 10000). Set to large number to process all
* `--max-spectra-val`: Maximum number of spectra for validation (default: 1000000)
* `--max-spectra-test`: Maximum number of spectra for testing (default: 1000000)
* `--mapping-file-name`: Filename for the mapping file (default: mapping_unique_smiles.pkl)
* `--num-workers`: Number of worker processes for parallel computation (default: 0)
* `--val-split`: Fraction of data for validation (default: 0.1)
* `--test-split`: Fraction of data for testing (default: 0.1)
* `--overwrite`: Overwrite existing preprocessing files

#### Option 2: Python Script (Legacy)

```bash
python preprocessing_scripts/final_generation_data.py  \
   --spectra_path=/path/to/your/spectra.mgf   \
   --workspace=/path/to/preprocessed_data/  \
   --MAX_SPECTRA_TRAIN=10000 \
   --mapping_file_name=mapping_unique_smiles.pkl  \
   --PREPROCESSING_NUM_WORKERS=0
```

**Parameters:**
* `spectra_path`: Location of spectra
* `workspace`: Location where the calculated distances are going to be saved
* `MAX_SPECTRA_TRAIN`: Maximum number of spectra to be processed. Set to large number to avoid removing spectra
* `mapping_file_name`: Name of the file that saves the mapping of the spectra from spectra to unique compounds
* `PREPROCESSING_NUM_WORKERS`: Number of processors to be used (default: 0)

---

**Note:** Both methods produce identical results. The preprocessing computes:
- Edit distance between molecular structures
- MCES (Maximum Common Edge Substructure) distance
- Train/validation/test splits

The output includes a file `mapping_unique_smiles.pkl` with mapping information between unique compounds and corresponding spectra. Each compound can have several spectra and this file saves information about this mapping.

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

Train your SIMBA model using one of the following methods:

#### Option 1: CLI Command (Recommended)

```bash
simba train \
  --checkpoint-dir /path/to/checkpoints/ \
  --preprocessing-dir /path/to/preprocessed_data/ \
  --preprocessing-pickle mapping_unique_smiles.pkl \
  --epochs 10 \
  --accelerator cpu \
  --batch-size 32 \
  --num-workers 0 \
  --learning-rate 0.0001 \
  --val-check-interval 10000
```

**Parameters:**
* `--checkpoint-dir`: Directory where the trained model will be saved
* `--preprocessing-dir`: Directory where preprocessing files are stored
* `--preprocessing-pickle`: Filename of the mapping pickle file
* `--epochs`: Number of training epochs (default: 10)
* `--accelerator`: Hardware accelerator: `cpu` or `gpu` (default: cpu)
* `--batch-size`: Batch size for training and validation (default: 32)
* `--num-workers`: Number of data loading workers (default: 0)
* `--learning-rate`: Learning rate for the optimizer (default: 0.0001)
* `--val-check-interval`: Validation check frequency in training steps (default: 10000)

#### Option 2: Python Script (Legacy)

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

**Parameters:**
* `CHECKPOINT_DIR`: Place where the trained model will be saved
* `PREPROCESSING_DIR_TRAIN`: Folder where the preprocessing files are saved
* `PREPROCESSING_PICKLE_FILE`: File name with the mapping
* `ACCELERATOR`: cpu or gpu
* `epochs`: Number of epochs to be trained
* `VAL_CHECK_INTERVAL`: Used to check validation performance every N steps

---

**Note:** Both methods produce identical results and use the mapping file produced in Step 1. The preprocessing directory `PREPROCESSING_DIR_TRAIN` / `--preprocessing-dir` must be the same where the preprocessing files were generated. The best-performing model (lowest validation loss) is saved in `CHECKPOINT_DIR` / `--checkpoint-dir`.

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
* CHECKPOINT_DIR: Folder where the trained model is saved an testing results will be saved.
* PREPROCESSING_DIR and PREPROCESSING_DIR_TRAIN: Location where the preprocessing files are saved.
* PREPROCESSING_PICKLE_FILE: Mapping file.
* UNIFORMIZE_DURING_TESTING: If to balance the edit distance classes or not.

---

## 🛠️ Development & Contributing

### Setting Up Development Environment

```bash
# Clone and install with dev dependencies
git clone https://github.com/bittremieux-lab/simba.git
cd simba
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=simba --cov-report=html --cov-report=term-missing

# View coverage report
open htmlcov/index.html
```

### Code Quality

The project uses:
- **Ruff** for linting and formatting
- **pytest** for testing
- **pre-commit hooks** for automated checks

```bash
# Run linter
uv run ruff check simba/

# Format code
uv run ruff format simba/

# Run pre-commit on all files
uv run pre-commit run --all-files
```

Pre-commit hooks automatically run on every commit and check:
- Code formatting (Ruff)
- Linting (Ruff)
- Tests (pytest)
- File formatting (trailing whitespace, line endings)
- YAML/TOML syntax

---

## 📬 Contact & Support

- **Code repository**: [SIMBA GitHub](https://github.com/bittremieux-lab/simba)
- For questions, issues, or feature requests, please [open an issue](https://github.com/bittremieux-lab/simba/issues).

---

## 📦 Data Availability

- Training and testing datasets available at: [https://zenodo.org/records/15275257].

---
