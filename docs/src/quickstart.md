# Quick Start

This guide will get you up and running with SIMBA.

## Overview

SIMBA provides a pretrained model trained on spectra from **MassSpecGym**. The model operates in positive ionization mode for protonated adducts.

A typical SIMBA workflow consists of:

1. **Computing Structural Similarities**: Predict edit distance and MCES between spectra
2. **Analog Discovery**: Find structurally similar molecules in a reference library
3. **Training Custom Models**: Train SIMBA on your own MS/MS data (optional)

## Computing Structural Similarities

Follow the [Run Inference Notebook](https://github.com/bittremieux-lab/simba/tree/main/notebooks/final_tutorials/run_inference.ipynb) for a comprehensive tutorial:

- **Runtime:** < 10 minutes (including model/data download)
- **Example data:** data folder
- **Supported format:** `.mgf`

### Performance

Using an Apple M3 Pro (36 GB RAM):

- **Embedding computation:** ~100,000 spectra in ~1 minute
- **Similarity computation:** 1 query vs. 100,000 spectra in ~10 seconds

SIMBA caches computed embeddings, significantly speeding repeated library searches.

## Analog Discovery

Perform analog discovery to find structurally similar molecules:

```bash
simba analog-discovery \
  --model-path /path/to/model.ckpt \
  --query-spectra /path/to/query.mgf \
  --reference-spectra /path/to/reference_library.mgf \
  --output-dir /path/to/output \
  --query-index 0 \
  --top-k 10 \
  --device cpu \
  --compute-ground-truth
```

**Parameters:**

- `--model-path`: Path to trained SIMBA model checkpoint (.ckpt file)
- `--query-spectra`: Path to query spectra file (.mgf or .pkl format)
- `--reference-spectra`: Path to reference library spectra file (.mgf or .pkl format)
- `--output-dir`: Directory where results will be saved
- `--query-index`: Index of the query spectrum to analyze (default: 0)
- `--top-k`: Number of top matches to return (default: 10)
- `--device`: Hardware device: `cpu` or `gpu` (default: cpu)
- `--batch-size`: Batch size for processing (default: 32)
- `--cache-embeddings` / `--no-cache-embeddings`: Cache embeddings for faster repeated searches (default: True)
- `--use-gnps-format` / `--no-use-gnps-format`: Whether spectra files use GNPS format (default: False)
- `--compute-ground-truth`: Compute ground truth edit distance and MCES for validation
- `--save-rankings`: Save complete ranking matrix to file

**Output:**

The command generates several files in the output directory:

- `results.json`: Summary of top matches with predictions and ground truth
- `matches.csv`: Detailed table of all matches
- `query_molecule.png`: Structure of the query molecule
- `match_N_molecule.png`: Structures of matched molecules
- `mirror_plot_match_N.png`: Mirror plots comparing query and matched spectra
- `rankings.npy`: Complete ranking matrix (if `--save-rankings` is used)

For interactive exploration, use the [Run Analog Discovery Notebook](https://github.com/bittremieux-lab/simba/tree/main/notebooks/final_tutorials/run_analog_discovery.ipynb).

## Training Custom Models

### Step 1: Preprocess Data

```bash
simba preprocess \
  --spectra-path /path/to/your/spectra.mgf \
  --workspace /path/to/preprocessed_data \
  --max-spectra-train 10000 \
  --mapping-file-name mapping_unique_smiles.pkl \
  --num-workers 0
```

### Step 2: Train Model

```bash
simba train \
  --checkpoint-dir checkpoints/ \
  --preprocessing-dir preprocessing/ \
  --preprocessing-pickle preprocessed_data.pkl \
  --epochs 50 \
  --accelerator gpu \
  --batch-size 64
```

### Step 3: Run Inference

```bash
simba inference \
  --checkpoint-dir checkpoints/ \
  --preprocessing-dir preprocessing/ \
  --preprocessing-pickle test_data.pkl \
  --batch-size 128 \
  --accelerator gpu
```
