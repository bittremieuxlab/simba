# Analog discovery (paper Figure 2 reproduction)

Reproduces the paper's analog-discovery evaluation: given a CASMI query
spectrum, rank a reference library by predicted structural closeness, and
check how close the top candidates really are (exact ground-truth MCES).
Compares SIMBA against 6 baseline checkpoints (Spec2Vec/MS2DeepScore/DreaMS,
each "ours" retrained-on-SIMBA's-data and the paper's own official
checkpoint), under two independent normalized-MCES ground-truth metrics
(the paper's own Sebastian/tier-similarity formula, and the exact
Johnson-similarity formula from the paper's Methods text).

Two searches: **A** (NIST20 + MassSpecGym library) and **B** (GNPS library),
132 CASMI queries each.

## Data

- **CASMI queries**: CASMI 2022 benchmark spectra, official paper data
  (`data/casmi/casmi2022_spec_df.pkl`, from Zenodo record 15275257).
  Filtered to M+H adduct only, then run through SIMBA's preprocessing
  (precursor-peak removal, intensity filter min 1% + cap 100 peaks);
  spectra left with < 6 peaks are dropped. This reproduces the paper's
  reported count exactly: 170 M+H spectra → 158 after filtering.
- **Exclusion**: queries that are themselves members of SIMBA's training
  set are excluded. The exclusion set is the literal 42,774-compound pool
  SIMBA's own train/val/test pairs were built from — this is what
  reproduces the paper's reported 158 → 132 exactly.
- **Search A library**: that same 42,774-compound NIST20+MassSpecGym pool
  (the model's own training/val/test molecules).
- **Search B library**: GNPS (`ALL_GNPS_NO_PROPOGATED_sebastian.mgf`) —
  the actual snapshot Sebastian used (shared directly with us, dated
  2024-01-31; 504,840 spectra → 25,445 unique molecules after filtering).
  We originally used our own freshly-downloaded copy of
  `ALL_GNPS_NO_PROPOGATED.mgf` from https://external.gnps2.org/gnpslibrary/
  instead (956,358 spectra → 66,815 unique molecules — GNPS is a living,
  continuously-growing community library, so a fresh download is
  substantially larger than a 2024 snapshot). Switching to his actual file
  did NOT close the gap to the paper's reported numbers — see "Known
  deviation from paper" below; that archived run's data is kept at
  `analog_discovery/archive_old_large_gnps/` on the cluster for reference.
- **Checkpoint**: `best_model_paper_SIMBA.ckpt` — the paper's checkpoint.
- Both CASMI and GNPS store raw absolute instrument intensities; rescaled
  to relative-to-base-peak (max=1.0) before scoring, matching the
  convention the training data already uses.
- **Raw ground-truth MCES**: exact (not approximate) MCES, threshold=20,
  via the `myopic_mces` ILP solver (`PULP_CBC_CMD`), exhaustive over every
  query x library pair in each search. Used only to pick each query's
  single TRUE best analog molecule, for panel (c)'s ranking-performance
  curve.
- **Normalized MCES — two parallel metrics, computed side by side**:
  - *paper metric*: what Sebastian's own notebook actually computes,
    `1 - tier_sim` (RASCAL's fast pre-screening similarity estimate).
  - *exact metric*: the paper Methods text's own stated formula,
    `1 - (E_c+V_c)^2 / ((E1+V1)(E2+V2))`, from RASCAL's actual found MCES.
  - See `simba/analog_discovery/paper_metrics.py`'s module docstring for
    why these differ and by how much.

## Methodology notes

- **No molecule-level aggregation**: library spectra are NOT deduplicated
  to one-per-molecule when ranking — matching Sebastian's own
  `AnalogDiscovery.get_k_candidates`, which operates on the raw
  per-spectrum similarity matrix directly. A molecule with many replicate
  spectra can occupy multiple top-K slots.
- **Ranking**: primarily by predicted MCES rounded to the nearest integer
  (not continuous — this creates real ties, which is what makes the next
  tiebreak matter), tiebroken by predicted edit-distance class (higher
  class = closer analog, confirmed from Sebastian's own sanity-check
  code), final tiebreak by continuous MCES. Baseline models have no
  edit-distance prediction, so their ranking is plain descending
  similarity.
- **Panel (c) "K candidates" means K raw ranked spectrum slots**, duplicates
  allowed, no molecule-level re-ranking — verified directly from Sebastian's
  own simba_v1 source (`PerformanceAnalysis.obtain_precision_k_candidates`):
  a query counts as "found within top-K" iff ANY of its true best
  molecule's own spectra has per-spectrum rank < K. An earlier version of
  this collapsed to a per-molecule dense rank first instead — that was
  never his actual method, just something that happened to land closer to
  his reported numbers; implementing his real method moves recall@K
  *away* from his reported values, not closer (see "Known deviation from
  paper" below) — a genuine, unresolved discrepancy, not something this
  fix was expected to close.
- **Panel (c) K-range is `[5, 10, 20, 40, 80, 100]`**, not a dense curve to
  1000 — matches Sebastian's own `precision_candidates.ipynb` comparison
  plot exactly (star markers, `xlim=(0, 100)`), per reviewer comment
  (relayed by Wout). We previously plotted a dense curve to K=1000; the
  real recall@K numbers computed at his actual K-values are smaller than
  what that longer curve's K=1000 endpoint showed, since precision keeps
  climbing well past K=100 (see "Known deviation from paper").

## Approach

Three scripts, matching the three pipeline stages:

1. **`analog_discovery_prepare.py`** — everything upstream of applying any
   model: builds the two search MGFs from CASMI+GNPS, then the exact raw
   ground-truth MCES, then both normalized-MCES caches. One file, seven
   subcommands (`--help` for the full list): `build_mgf`, then
   `exact_mces_prepare`/`exact_mces_compute_block`/`exact_mces_status`/
   `exact_mces_combine`, then `normalize_shard`/`normalize_combine` (run
   twice per search, once per metric, via `--cache_name`). The two
   exhaustive, millions-of-pairs steps (exact MCES, normalization) are
   each split into independent blocks/shards — split
   `--task_id`/`--shard_id` across as many parallel workers/machines as
   you have.
2. **`analog_discovery_score_simba.py`** — applies SIMBA, two
   subcommands: `dedup` (one spectrum/molecule, produces the canonical
   library ordering ground truth and baseline alignment are keyed
   against) and `no_mol_agg` (every library spectrum, duplicates allowed
   — the actual per-spectrum ranking the final figure uses for SIMBA).
3. **`analog_discovery_plot.py`** — combines (1)'s ground truth + (2)'s
   SIMBA scores + the 6 baseline-model score matrices (see "Baseline
   models" below) into the two final figures (paper metric / exact
   metric), written to `tools/analog_discovery_figures/`.

## Baseline models (Spec2Vec, MS2DeepScore, DreaMS)

Each baseline is scored outside this repo, in
`simba-paper-other-tools` (see that repo's own README for the underlying
`NN_score_analog_discovery.py` scripts, their batch-scoring APIs, and
per-checkpoint benchmark numbers). Final comparison uses 6 checkpoints —
"ours" (retrained/fine-tuned on SIMBA's own data) and "paper" (the
official checkpoint each tool's own authors released) for each of the 3
tools. Output: `similarity.npy` (n_query x n_library, NOT deduplicated by
molecule) + `smiles_query.txt`/`smiles_library.txt`, one directory per
checkpoint under `<baseline_scores_dir>/<name>/search_{A,B}/`.

Run each via SLURM (GPU, shared cluster — never raw `ssh`+background).
These sbatch scripts aren't checked into this repo (operational, cluster-
specific, and easy to regenerate from this); run via `sbatch <<'EOF' ... EOF`
or save locally outside the repo.

**Spec2Vec** (CPU only, `one_hour` partition is plenty):
```bash
#!/bin/bash
#SBATCH --job-name=baseline_spec2vec
#SBATCH --partition=one_hour
#SBATCH --nodelist=asimov2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
PY=/mnt/data/nkubrakov/simba-paper-other-tools/ms2deepscore-venv/bin/python
SCRIPT=/home/nkubrakov/simba-paper-other-tools/spec2vec/experiments/simba_training/03_score_analog_discovery.py
SPECTRA_DIR=/mnt/data2/nkubrakov/analog_discovery/baseline_spectra
for SEARCH in A B; do
  "$PY" "$SCRIPT" \
    --query-pkl "$SPECTRA_DIR/search_${SEARCH}_query.pkl" \
    --library-pkl "$SPECTRA_DIR/search_${SEARCH}_library.pkl" \
    --model-path "$CKPT_PATH" \
    --output-dir "$OUT_BASE/search_${SEARCH}"
done
```
Launch with `CKPT_PATH`/`OUT_BASE` set for each of the 2 checkpoints:
- ours: `/mnt/data/nkubrakov/simba-paper-other-tools/spec2vec_data/spec2vec_simba_iter15.model` → `.../baseline_scores/spec2vec_ours`
- paper: `/mnt/data/nkubrakov/simba-paper-other-tools/spec2vec-pretrained/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model` → `.../baseline_scores/spec2vec_paper`

**MS2DeepScore** (1 GPU, `one_hour` partition is plenty):
```bash
#!/bin/bash
#SBATCH --job-name=baseline_ms2ds
#SBATCH --partition=one_hour
#SBATCH --nodelist=asimov2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
PY=/mnt/data/nkubrakov/simba-paper-other-tools/ms2deepscore-venv/bin/python
SCRIPT=/home/nkubrakov/simba-paper-other-tools/ms2deepscore/experiments/simba_training/05_score_analog_discovery.py
SPECTRA_DIR=/mnt/data2/nkubrakov/analog_discovery/baseline_spectra
for SEARCH in A B; do
  "$PY" "$SCRIPT" \
    --query-pkl "$SPECTRA_DIR/search_${SEARCH}_query.pkl" \
    --library-pkl "$SPECTRA_DIR/search_${SEARCH}_library.pkl" \
    --model-path "$CKPT_PATH" \
    --output-dir "$OUT_BASE/search_${SEARCH}"
done
```
- ours: `/mnt/data/nkubrakov/simba-paper-other-tools/ms2deepscore_data/models_production_paper_settings/ms2deepscore_model.pt` → `.../baseline_scores/ms2deepscore_ours`
  (retrained with the paper's own architecture settings — `base_dims=10000`,
  `embedding_dim=500`, `patience=20` — not the ms2deepscore package's
  smaller defaults; see `simba-paper-other-tools/ms2deepscore/.../README.md`)
- paper: `/mnt/data/nkubrakov/simba-paper-other-tools/ms2deepscore-pretrained/ms2deepscore_model.pt` → `.../baseline_scores/ms2deepscore_paper`

**DreaMS** (1 GPU; run per search, not looped, since the "paper" checkpoints
run ~1.6x slower per-pair than "ours" and can exceed `one_hour`'s limit
on Search B's larger library if both searches share one job):
```bash
#!/bin/bash
#SBATCH --job-name=baseline_dreams
#SBATCH --partition=one_hour
#SBATCH --nodelist=asimov2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
set -euo pipefail
PY=/mnt/data/nkubrakov/simba-paper-other-tools/dreams-venv/bin/python
SCRIPT=/home/nkubrakov/simba-paper-other-tools/DreaMS/experiments/simba_finetune/05_score_analog_discovery.py
SPECTRA_DIR=/mnt/data2/nkubrakov/analog_discovery/baseline_spectra
"$PY" "$SCRIPT" \
  --query-pkl "$SPECTRA_DIR/search_${SEARCH}_query.pkl" \
  --library-pkl "$SPECTRA_DIR/search_${SEARCH}_library.pkl" \
  --checkpoint-path "$CKPT_PATH" \
  --checkpoint-type "$CKPT_TYPE" \
  --output-dir "$OUT_BASE/search_${SEARCH}" \
  "${EXTRA_ARGS[@]}"
```
Launch once per `SEARCH in {A,B}`, with:
- ours: `CKPT_TYPE=finetuned`, `CKPT_PATH=/mnt/data/nkubrakov/simba-paper-other-tools/checkpoints/SimbaContrastiveFineTune/simba_mona_recipe_lr8.66e-6_bs96_margin0.1/last.ckpt`, `OUT_BASE=.../baseline_scores/dreams_ours`
- paper: `CKPT_TYPE=paper_embedding`, `CKPT_PATH=/mnt/data/nkubrakov/simba-paper-other-tools/dreams-pretrained/embedding_model.ckpt`, `EXTRA_ARGS=(--backbone-ckpt /mnt/data/nkubrakov/simba-paper-other-tools/dreams-pretrained/ssl_model.ckpt)`, `OUT_BASE=.../baseline_scores/dreams_paper_embedding`

Note: DreaMS also ships a raw `ssl_model.ckpt` self-supervised backbone
(`--checkpoint-type raw_ssl`, no `--backbone-ckpt` needed). We scored it
too during development but excluded it from the final comparison: its
curve on our analog-discovery task tracked the paper's published DreaMS
curve noticeably worse than `embedding_model.ckpt` did (on every one of
median/AUC/recall@1000, both searches) — consistent with
`embedding_model.ckpt` being the one DreaMS's own library-matching
tutorial actually recommends, not the raw backbone.

## How to run

```bash
# 1a. assemble data (once, builds both searches' MGFs)
python tools/analog_discovery_prepare.py build_mgf \
  --casmi_pkl     data/casmi/casmi2022_spec_df.pkl \
  --training_pkl  /mnt/data2/nkubrakov/preprocessing_ed_mces_20250123/edit_distance_neurips_nist_exhaustive.pkl \
  --gnps_mgf      data/gnps/ALL_GNPS_NO_PROPOGATED_sebastian.mgf \
  --output_dir    /mnt/data2/nkubrakov/analog_discovery

# 1b. exact raw ground-truth MCES (once per search)
python tools/analog_discovery_prepare.py exact_mces_prepare \
  --output_dir /mnt/data2/nkubrakov/analog_discovery/search_A_exact_mces_final \
  --mgf /mnt/data2/nkubrakov/analog_discovery/search_A_nist_msg.mgf --n_blocks 100
for i in $(seq 0 99); do
  python tools/analog_discovery_prepare.py exact_mces_compute_block \
    --output_dir /mnt/data2/nkubrakov/analog_discovery/search_A_exact_mces_final --task_id $i
done
python tools/analog_discovery_prepare.py exact_mces_combine \
  --output_dir /mnt/data2/nkubrakov/analog_discovery/search_A_exact_mces_final
# (repeat for search_B)

# 2a. apply SIMBA -- dedup library (once per search)
python tools/analog_discovery_score_simba.py dedup \
  --checkpoint  /mnt/data2/nkubrakov/paper_checkpoint/best_model_paper_SIMBA.ckpt \
  --mgf         /mnt/data2/nkubrakov/analog_discovery/search_A_nist_msg.mgf \
  --output_dir  /mnt/data2/nkubrakov/analog_discovery/scores/search_A
# (repeat for search_B_gnps.mgf -> scores/search_B)

# 2b. apply SIMBA -- no molecule aggregation (once per search)
python tools/analog_discovery_score_simba.py no_mol_agg \
  --checkpoint  /mnt/data2/nkubrakov/paper_checkpoint/best_model_paper_SIMBA.ckpt \
  --mgf         /mnt/data2/nkubrakov/analog_discovery/search_A_nist_msg.mgf \
  --scores_dir  /mnt/data2/nkubrakov/analog_discovery/scores/search_A \
  --library a --training_pkl /mnt/data2/nkubrakov/preprocessing_ed_mces_20250123/edit_distance_neurips_nist_exhaustive.pkl \
  --label "Search A (NIST20 + MassSpecGym)" \
  --output_dir /mnt/data2/nkubrakov/analog_discovery/results_final_no_mol_agg
# (repeat for search B with --library b --gnps_mgf data/gnps/ALL_GNPS_NO_PROPOGATED_sebastian.mgf --label "Search B (GNPS)")

# 1c. normalized MCES, sharded, once per metric (independent, run however many in parallel you like)
for CACHE in normalized_mces_cache.npy normalized_mces_cache_sebastian.npy; do
  for i in $(seq 0 1999); do
    python tools/analog_discovery_prepare.py normalize_shard \
      --scores_dir /mnt/data2/nkubrakov/analog_discovery/scores/search_A \
      --exact_mces_dir /mnt/data2/nkubrakov/analog_discovery/search_A_exact_mces_final \
      --shard_id $i --n_shards 2000 --cache_name $CACHE \
      --output_dir /mnt/data2/nkubrakov/analog_discovery/shards_search_A
  done
done
# (repeat for search_B)

# 1d. combine + clean, once per metric (once per search)
for CACHE in normalized_mces_cache.npy normalized_mces_cache_sebastian.npy; do
  python tools/analog_discovery_prepare.py normalize_combine \
    --scores_dir      /mnt/data2/nkubrakov/analog_discovery/scores/search_A \
    --exact_mces_dir  /mnt/data2/nkubrakov/analog_discovery/search_A_exact_mces_final \
    --shard_dir       /mnt/data2/nkubrakov/analog_discovery/shards_search_A \
    --n_shards 2000 --cache_name $CACHE
done
# (repeat for search_B)

# -- baseline models: see "Baseline models" above, 6 SLURM jobs into
#    <baseline_scores_dir>/<name>/search_{A,B}/similarity.npy

# 3. build the final figures
python tools/analog_discovery_plot.py \
  --search_a_scores_dir     /mnt/data2/nkubrakov/analog_discovery/scores/search_A \
  --search_a_exact_mces_dir /mnt/data2/nkubrakov/analog_discovery/search_A_exact_mces_final \
  --search_b_scores_dir     /mnt/data2/nkubrakov/analog_discovery/scores/search_B \
  --search_b_exact_mces_dir /mnt/data2/nkubrakov/analog_discovery/search_B_exact_mces_final \
  --no_mol_agg_dir          /mnt/data2/nkubrakov/analog_discovery/results_final_no_mol_agg \
  --baseline_scores_dir     /mnt/data2/nkubrakov/analog_discovery/baseline_scores \
  --output_dir              tools/analog_discovery_figures
```

Output: `combined_all_methods_paper_metric.png` + `combined_all_methods_exact_metric.png`
(+ one `summary_*.json` each) in `tools/analog_discovery_figures/`.

## Results

Figures: `tools/analog_discovery_figures/combined_all_methods_{paper,exact}_metric.png`.
Values below are from `summary_combined_all_methods_{paper,exact}_metric.json` next to them.

**Paper metric** (Sebastian's tier-similarity normalized MCES):

| Method | Search A median / AUC / recall@100 | Search B median / AUC / recall@100 |
|---|---|---|
| Best candidate MCES (oracle) | 0.067 / — / — | 0.076 / — / — |
| Spec2Vec (ours) | 0.204 / 0.71 / 0.242 | 0.271 / 0.60 / 0.205 |
| Spec2Vec (paper) | 0.255 / 0.68 / 0.220 | 0.281 / 0.59 / 0.189 |
| MS2DeepScore (ours) | 0.232 / 0.84 / 0.220 | 0.269 / 0.71 / 0.174 |
| MS2DeepScore (paper) | 0.204 / 0.82 / 0.235 | 0.238 / 0.78 / 0.220 |
| DreaMS (ours) | 0.157 / 0.74 / 0.295 | 0.191 / 0.68 / **0.318** |
| DreaMS (paper embedding) | 0.147 / 0.75 / 0.280 | 0.167 / 0.65 / 0.265 |
| **SIMBA** | **0.171 / 0.92 / 0.341** | **0.204 / 0.80 / 0.212** |
| Paper's own reported values | ~0.175 / ~0.93 / ~0.39 | ~0.18 / ~0.90 / ~0.39 |

**Exact metric** (paper Methods text's Johnson-similarity formula): same
recall@100 column as above (panel (c) depends only on the raw MCES ground
truth, not on which normalized metric is plotted).

| Method | Search A median / AUC / recall@100 | Search B median / AUC / recall@100 |
|---|---|---|
| Best candidate MCES (oracle) | 0.162 / — / — | 0.173 / — / — |
| Spec2Vec (ours) | 0.432 / 0.73 / 0.242 | 0.468 / 0.59 / 0.205 |
| Spec2Vec (paper) | 0.452 / 0.74 / 0.220 | 0.482 / 0.58 / 0.189 |
| MS2DeepScore (ours) | 0.462 / 0.93 / 0.220 | 0.469 / 0.79 / 0.174 |
| MS2DeepScore (paper) | 0.446 / 0.91 / 0.235 | 0.463 / 0.87 / 0.220 |
| DreaMS (ours) | 0.405 / 0.83 / 0.295 | 0.399 / 0.74 / **0.318** |
| DreaMS (paper embedding) | 0.351 / 0.82 / 0.280 | 0.405 / 0.68 / 0.265 |
| **SIMBA** | **0.407 / 0.95 / 0.341** | **0.437 / 0.87 / 0.212** |

SIMBA still leads every baseline on median and AUC in both metrics and both
searches; MS2DeepScore (ours, retrained with the paper's own architecture
settings) is consistently the strongest baseline on those two. But on
panel (c)'s recall@100 — now matching Sebastian's literal algorithm (flat
top-K spectrum-slot hit-check over his own K-range, no molecule-level
re-ranking) — SIMBA no longer clearly dominates: for Search B, **DreaMS
(ours and paper embedding) both beat SIMBA** (0.318/0.265 vs. 0.212), and
for Search A SIMBA still leads clearly (0.341 vs. DreaMS's 0.295/0.280).
See "Known deviation from paper" below — this fix moved recall@K *away*
from the paper's reported values, not closer, which is itself an
unresolved finding, not a bug in the fix.

## Known deviation from paper

Even after matching Sebastian's actual ranking/selection/aggregation logic
(not just the paper's Methods-text description) AND switching Search B to
his own actual GNPS snapshot, a real gap to the paper's reported numbers
remains for SIMBA.

**GNPS library drift was tested directly and ruled out as the explanation**:
switching from our own fresh GNPS download (956,358 spectra / 66,815
unique molecules) to Sebastian's actual 2024-01-31 snapshot (504,840 /
25,445) made SIMBA's Search B numbers *worse* on median/AUC, not better
(paper metric: median 0.178→0.204, AUC 0.84→0.80) — still well short of
the paper's ~0.18/~0.90/~0.39. (The pre-switch run predates the K=100
panel-c fix, so there's no directly comparable recall@100 before/after
figure — only median/AUC are compared here.) DreaMS also doesn't explain
SIMBA's remaining gap: it now beats SIMBA on recall@100 for Search B
(0.318/0.265 vs. 0.212) regardless of which GNPS snapshot is used.

Remaining plausible causes, still unconfirmed: (1) no way to confirm our
SIMBA checkpoint or exact 132-query set is byte-identical to the one
behind the published figure, (2) the paper's panel-(c) curve has no
printed numeric value to compare against (only a static plot), so
"paper's ~X" values are our own visual reads, not published numbers, (3)
his own ground-truth artifact for panel (c) (`mces_similarity_casmi_NEURIPS`)
is a precomputed file we've never seen — we substitute our own exact-MCES
computation and assume it picks the same "true best" molecule.
