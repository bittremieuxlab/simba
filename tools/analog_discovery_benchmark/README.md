# Analog-discovery benchmark (SIMBA paper, Figure 2)

Reproduces the analog-discovery evaluation of the SIMBA paper exactly as
Sebastian's notebooks do it (branch `simba_v1`:
`notebooks/analog_discovery_casmi_msnlib.ipynb`,
`notebooks/precision_candidates.ipynb`). The ground truth is the **normalized
MCES**, computed from the actual maximum common edge subgraph with his
`calculate_mces_sim_exact`.

Two different MCES quantities appear in this pipeline:

- **Raw MCES distance:** the number of bond edits between two molecules. It
  grows with molecule size. SIMBA is trained to predict it, as the
  similarity `s = 1 − MCES/40` (`normalize_mces20` in `simba/load_mces/`).
- **Normalized MCES** (used for evaluation): the Johnson similarity
  `(E_c + V_c)² / ((E₁ + V₁)(E₂ + V₂))`, which lies in [0, 1] and does not
  depend on molecule size. Here E_c and V_c are the common subgraph's bonds
  and atoms, and E₁, V₁, E₂, V₂ are each molecule's bonds and atoms. The
  normalized MCES distance is 1 − that similarity. RASCAL returns it as
  `result.similarity`.

"Exact" in `calculate_mces_sim_exact` means the value comes from the actual
MCES. His older `calculate_mces_sim` returned RASCAL's faster tier estimates
instead. It does **not** mean the raw MCES.

Each CASMI 2022 query spectrum ranks a reference library, and we measure how
structurally close the top-ranked library molecules really are, for SIMBA and
the baselines (Spec2Vec, MS2DeepScore, DreaMS).

| | queries | library spectra | library molecules |
|---|---|---|---|
| Search A: NIST20 + MassSpecGym (SIMBA's training pool) | 132 | 328,555 | 42,774 |
| Search B: GNPS | 148 | 53,618 | 16,523 |

## Steps

| step | script | input | output | compute |
|---|---|---|---|---|
| 1 | `01_build_sets.py` | dataset pickle, CASMI pickle, GNPS MGF | `sets/search_{A,B}/` | CPU, ~10 min |
| 2 | `02_mces_ground_truth.py` | `sets/` | `gt/search_{A,B}/` | CPU, ~1,300 CPU-hours |
| 3 | `03_export_baseline_spectra.py` | the 3 data files, `sets/` | `baseline_spectra/` | CPU, ~10 min |
| 4 | `04_score_spec2vec_aligned.py` + baseline tools | `baseline_spectra/`, baseline checkpoints | `baseline_scores/<name>/search_{A,B}/` | GPU, minutes per checkpoint |
| 5 | `05_score_simba.py` | the 3 data files, `sets/`, SIMBA checkpoint | `simba_scores/<name>/search_{A,B}/` | GPU, < 1 h |
| 6 | `06_evaluate.py` | `gt/`, `sets/`, scores of steps 4 and 5 | `figures/summary.json`, `figures/analog_discovery.png` | CPU, ~1 min per method |

`casmi_sets.py` holds the set definitions shared by steps 1, 3 and 5. Every
matrix uses the same order:
- rows: the queries in `sets/search_X/queries.txt`;
- score-matrix columns: library spectra in `reference_spectra_smiles.txt`;
- ground-truth columns: library molecules in `molecules.txt`;
- spectrum to molecule: `spectrum_mol_idx.npy`.

Run all commands from the repository root. Example settings, as used on sofia:

```bash
DATA=/sofia/projects/2026_053/simba_project/data/paper
WORK=$DATA/analog_discovery_sebastian       # all outputs of this pipeline
DATASET=$DATA/preprocessing_ed_mces_20250123/edit_distance_neurips_nist_exhaustive.pkl
CASMI=data/casmi/casmi2022_spec_df.pkl      # Zenodo 15275257
GNPS=data/gnps/ALL_GNPS_NO_PROPOGATED_sebastian.mgf
INPUTS="--dataset_pkl $DATASET --casmi_pkl $CASMI --gnps_mgf $GNPS"
```

### 1. Query and library sets

```bash
.venv/bin/python tools/analog_discovery_benchmark/01_build_sets.py $INPUTS --out_dir $WORK/gt_inputs
```

This follows his loading code; see the `casmi_sets.py` docstring:
- Queries are CASMI spectra with at least 6 peaks after preprocessing.
- Queries whose molecule is in that search's own library are excluded, so
  Search B excludes against GNPS and has 148 queries.
- Library A is all spectra of the SIMBA training dataset.
- Library B is GNPS after his `use_gnps_format` filter.

The script stops if the counts differ from his matrices.

### 2. Ground truth: normalized MCES similarity for every query × library molecule

`02_mces_ground_truth.py` is standalone; it needs only Python, numpy < 2 and
`rdkit==2023.9.4`. Build that environment on any cluster:

```bash
python -m venv gt-venv && gt-venv/bin/pip install "numpy<2" rdkit==2023.9.4
gt-venv/bin/python 02_mces_ground_truth.py selftest     # must print "selftest passed"
```

**Metric:** his snippet, RASCAL `FindMCES` with `similarityThreshold=0`,
`returnEmptyMCES=True`, timeout 10 s, returning `result.similarity`. It has
two deviations from his code:
- **Retry:** a pair that times out with tier-1 similarity ≥ 0.7 (a possible
  close analog) is retried once with 60 s.
- **Bailouts:** RASCAL's "Too many matching bond pairs" bailout returns 0
  matched bonds without setting `timedOut`. His code would report similarity 0;
  here it is detected and stored as NaN.

**Running it:** the work is split into 4,236 blocks of 2,000 pairs. Task `t`
of `N` computes the blocks with index `% N == t`, using all cores of its
machine. Run `N` tasks in parallel, e.g. one scheduler array task per node,
with `--task_id` set to the array index and `--n_tasks` to the array size.
Finished blocks are skipped, so tasks can be resubmitted.

```bash
G="--inputs_dir $WORK/gt_inputs --out_dir $WORK/gt_results"
gt-venv/bin/python 02_mces_ground_truth.py run $G --task_id 0 --n_tasks 100 --workers 64   # ... task_id 0..99
gt-venv/bin/python 02_mces_ground_truth.py status $G                 # until "total missing blocks: 0"
gt-venv/bin/python 02_mces_ground_truth.py rerun $G --status hard_fail --mem_per_worker_gb 32 --workers 7 \
    --task_id 0 --n_tasks 1                                           # memory blow-ups, more memory each
gt-venv/bin/python 02_mces_ground_truth.py combine $G
```

**Output** in `gt_results/search_X/`:
- `gt_sim.npy`: float32 (queries × molecules), normalized MCES similarity
  (Johnson), NaN if not computed;
- `gt_tier1.npy`, `gt_tier2.npy`: RASCAL's upper bounds;
- `gt_status.npy`: why a value is missing, with status codes in the script
  docstring;
- `gt_summary.json`.

### 3. Spectra for the baseline tools

```bash
.venv/bin/python tools/analog_discovery_benchmark/03_export_baseline_spectra.py $INPUTS \
    --sets_dir $WORK/gt_inputs --out_dir $WORK/baseline_spectra
```

These are the original spectra, with intensities scaled to max = 1. Each tool
applies its own preprocessing. The output files are
`search_{A,B}_{query,library}.pkl`, in the format the scoring scripts of
`simba-paper-other-tools` read.

### 4. Baseline scores

The baselines use the package versions of the SIMBA paper: Spec2Vec 0.8.0,
MS2DeepScore 2.5.0, DreaMS 1.0.0. Each tool has its own environment, built
from the freeze files in `simba-paper-other-tools-data/`
(`spec2vec-0.8.0-freeze.txt`, `ms2deepscore-2.5.0-freeze.txt`,
`dreams-freeze.txt`).

| name | tool | checkpoint |
|---|---|---|
| `spec2vec_paper` | Spec2Vec | `spec2vec-pretrained/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model` (GNPS 2020) |
| `spec2vec_ours` | Spec2Vec | `spec2vec_v0_8_0_data/spec2vec_simba_iter15.model` (trained on SIMBA data) |
| `ms2deepscore_paper` | MS2DeepScore | `ms2deepscore-pretrained-zenodo13897744/ms2deepscore_model.pt` (Zenodo 13897744, the model of 2.5.0) |
| `ms2deepscore_ours` | MS2DeepScore | `ms2deepscore_v2_5_0_data/models_paper_settings/ms2deepscore_model.pt` (SIMBA data, official architecture) |
| `dreams_paper_embedding` | DreaMS | `dreams-pretrained/embedding_model.ckpt` (+ `ssl_model.ckpt` backbone) |
| `dreams_ours` | DreaMS | `checkpoints/SimbaContrastiveFineTune/simba_mona_recipe_lr8.66e-6_bs96_margin0.1/last.ckpt` |

Checkpoint paths are relative to `simba-paper-other-tools-data/`. Each run
writes `similarity.npy` (queries × library spectra, higher = more similar)
and the SMILES lists. Example for Search A; repeat with `search_B`:

```bash
TOOLS=/sofia/projects/2026_053/simba_project/simba-paper-other-tools
CKPTS=/sofia/projects/2026_053/simba_project/simba-paper-other-tools-data
VENVS=$CKPTS/venvs
IO="--query-pkl $WORK/baseline_spectra/search_A_query.pkl --library-pkl $WORK/baseline_spectra/search_A_library.pkl"

# Spec2Vec: our aligned script (spectra rejected by matchms' default filters get NaN, nothing is dropped)
$VENVS/spec2vec-0.8.0/bin/python tools/analog_discovery_benchmark/04_score_spec2vec_aligned.py $IO \
    --model-path $CKPTS/spec2vec_v0_8_0_data/spec2vec_simba_iter15.model --output-dir $WORK/baseline_scores/spec2vec_ours/search_A

# MS2DeepScore (GPU)
$VENVS/ms2deepscore-2.5.0/bin/python $TOOLS/ms2deepscore/experiments/simba_training/05_score_analog_discovery.py $IO \
    --model-path $CKPTS/ms2deepscore_v2_5_0_data/models_paper_settings/ms2deepscore_model.pt \
    --output-dir $WORK/baseline_scores/ms2deepscore_ours/search_A

# DreaMS (GPU); for the paper checkpoint use --checkpoint-type paper_embedding --backbone-ckpt <ssl_model.ckpt>
$VENVS/dreams/bin/python $TOOLS/DreaMS/experiments/simba_finetune/05_score_analog_discovery.py $IO \
    --checkpoint-type finetuned \
    --checkpoint-path $CKPTS/checkpoints/SimbaContrastiveFineTune/simba_mona_recipe_lr8.66e-6_bs96_margin0.1/last.ckpt \
    --output-dir $WORK/baseline_scores/dreams_ours/search_A
```

### 5. SIMBA scores (GPU)

```bash
.venv/bin/python tools/analog_discovery_benchmark/05_score_simba.py $INPUTS --sets_dir $WORK/gt_inputs \
    --checkpoint /sofia/projects/2026_053/best_model_paper_SIMBA.ckpt --out_dir $WORK/simba_scores/paper_checkpoint
```

This is his notebook code: SIMBA preprocessing plus L2-normalised intensities.
His ranking first orders by the predicted **raw** MCES distance in whole bonds,
`round(40·s) = 40 − MCES`, where `s` is SIMBA's predicted MCES similarity. Ties
are broken by the predicted edit-distance class, then by `s` itself. The output
`ranking.npy` (higher = better) is what gets evaluated. `sim_mces.npy` and
`ed_class.npy` are kept for inspection.

### 6. Metrics and figure

```bash
.venv/bin/python tools/analog_discovery_benchmark/06_evaluate.py \
    --gt_dir $WORK/gt_results --sets_dir $WORK/gt_inputs --out_dir $WORK/figures \
    --methods "SIMBA=$WORK/simba_scores/paper_checkpoint" \
              "Spec2Vec (paper)=$WORK/baseline_scores/spec2vec_paper" "Spec2Vec (ours)=$WORK/baseline_scores/spec2vec_ours" \
              "MS2DeepScore (paper)=$WORK/baseline_scores/ms2deepscore_paper" "MS2DeepScore (ours)=$WORK/baseline_scores/ms2deepscore_ours" \
              "DreaMS (paper)=$WORK/baseline_scores/dreams_paper_embedding" "DreaMS (ours)=$WORK/baseline_scores/dreams_ours"
```

Distance is the normalized MCES distance from step 2 (1 − Johnson
similarity); lower is better.
- **Retrieved-analog distance:** for each query, the smallest distance among
  the top-10 ranked library spectra, as in his analog notebook. Reported as
  median, q05, q25, q75 and q95 over queries, next to the true best analog
  ("oracle").
- **Precision@K**, K = 5, 10, 20, 40, 80, 100, from his
  `obtain_precision_k_candidates`:
  - the true best analog is the argmax of the ground-truth row, taking the
    first spectrum on ties;
  - a query is a hit when that molecule is among the top-K ranked spectra;
  - queries whose best similarity is 0 are skipped.
- **ROC AUC:** pooled over all (query, library spectrum) pairs that have a
  ground truth. A pair is positive when its distance is < 0.3; the score is
  minus the spectrum's rank within its query.

## Notes

- **Missing ground truth.** About 4.5% of pairs have no value: RASCAL
  returned nothing, timed out, bailed out, or ran out of memory. They are
  ignored, as his code does with `None`. For 102 of 132 queries (A) and 98 of
  148 (B), no missing pair can beat the query's best value, judged by its
  RASCAL upper bound. For the rest, the true best analog might be among the
  missing pairs.
- **Our run** (October 2026), all outputs in `$WORK` on sofia:
  - **Ground truth:** computed on CalcUA, 6,400 cores, ~1,270 CPU-hours,
    script md5 `8ce7a775e160dc8e14806b88861e9179`. The counts per status are
    in `gt_results/search_X/gt_summary.json`.
  - **MS2DeepScore paper checkpoint:** Zenodo 13897744 (October 2024), not
    the December 2025 release 17826815. The newer one is built for
    MS2DeepScore > 2.6 and cannot be loaded by 2.5.0.
