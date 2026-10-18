# Training-time diagnostics, TensorBoard, and the contrastive/retrieval loss

Adds three independent, off-by-default-where-applicable pieces to
`training_scripts/final_training.py`, all controlled by `Config`/CLI flags
so the default training command behaves exactly as before:

1. **TensorBoard logging** (`--USE_TENSORBOARD_LOGGING 1`): points Lightning's
   logger at `<CHECKPOINT_DIR>/tb_logs/` instead of the default
   `lightning_logs/`. When off, `pl.Trainer(logger=True)` still auto-creates
   a default TensorBoardLogger if the package is installed -- the flag only
   controls the *directory*, not whether scalars get logged at all.

2. **In-batch InfoNCE contrastive ("retrieval") loss**
   (`--USE_CONTRASTIVE_LOSS 1`): an additional, optional loss term computed
   from same-molecule/different-spectrum pairs ("self-pairs") that land in a
   batch, ported from `bittremieuxlab/simba`'s `exp/triplet-loss` branch
   (commits `82d7e77`/`1392e82`). Self-pairs are detected via
   `mol_idx_0 == mol_idx_1` (threaded through `CustomDatasetMultitasking` --
   see `simba/transformers/CustomDatasetMultitasking.py`), L2-normalized,
   optionally passed through a small 2-layer MLP projection head, then
   scored via symmetric cross-entropy over the in-batch cosine-similarity
   matrix (true match on the diagonal). Added to the main loss as
   `loss + contrastive_loss_weight * loss_contrastive`. Config flags:
   - `USE_CONTRASTIVE_LOSS` (default `False`)
   - `CONTRASTIVE_LOSS_WEIGHT` (default `0.1` -- **note:** the reference
     branch's own default is `1.0`; we did not match it exactly on the
     first two runs below, see "Known issues")
   - `CONTRASTIVE_LOSS_TEMPERATURE` (default `0.1`; reference default is `0.07`)
   - `USE_CONTRASTIVE_PROJECTION_HEAD` (default `False`, matches reference)

3. **Periodic training-time diagnostics** (`--LOG_DIAGNOSTICS_DURING_TRAINING 1`,
   via `simba/training_diagnostics_callback.py`'s `TrainingDiagnosticsCallback`,
   run on every validation epoch end): reproduces, live during training,
   the same plots/numbers `inference_scripts/inference_multitasking.py`
   and `tools/analog_discovery_plot.py` produce offline --
   - MCES regression hexbin (Spearman r) + edit-distance confusion matrix
     (accuracy), on BOTH the test split and the val split, each a fixed
     (seeded) class-balanced subsample built once via `TrainUtils.uniformise`
     (same mechanism as the original `final_training.py`/
     `inference_multitasking.py`, self-contained per split -- **not**
     affected by the validation-weights bug below).
   - The full analog-discovery combined plot (oracle + 6 baseline
     checkpoints + live SIMBA), for Search A and B, for BOTH
     normalized-MCES ground-truth metrics (paper/Sebastian and
     exact/Johnson) -- exactly reusing `tools/analog_discovery_plot.py`'s
     `prepare_search_data`/`build_figure`/`METHODS`. The 6 baselines + raw
     GT are cached once at construction (same on-disk cache as the offline
     pipeline); only SIMBA's own rank is rescored live each validation,
     using the full no-mol-agg (every library spectrum, duplicates
     allowed) library -- not a deduplicated shortcut, so numbers are
     directly comparable to the offline figures.
   Config flags: `LOG_DIAGNOSTICS_DURING_TRAINING` (default `False`),
   `ANALOG_DISCOVERY_DATA_DIR`, `ANALOG_DISCOVERY_SEARCHES`,
   `ANALOG_DISCOVERY_K_VALUES`.

## The three launched experiments

All three ran on `/mnt/data/nkubrakov/preprocessing_ed_mces_20250123/` +
`/mnt/data/nkubrakov/analog_discovery/` (copied once from the canonical
`/mnt/data2/nkubrakov/...` on asimov2, so training reads/writes stay local
to one node instead of crossing the network every batch -- asimov had free
GPUs while asimov2 was contended). Outputs are periodically rsynced back to
`/mnt/data2/nkubrakov/simba_paper_repro/experiments/` for the canonical copy.

**Baseline** (Sebastian's exact paper command, our paths):
```
python training_scripts/final_training.py \
  --CHECKPOINT_DIR=/mnt/data/nkubrakov/simba_paper_repro/experiments/paper_run/ \
  --PREPROCESSING_PICKLE_FILE=edit_distance_neurips_nist_exhaustive.pkl \
  --PREPROCESSING_DIR_TRAIN=/mnt/data/nkubrakov/preprocessing_ed_mces_20250123/ \
  --ANALOG_DISCOVERY_DATA_DIR=/mnt/data/nkubrakov/analog_discovery \
  --TRAINING_NUM_WORKERS=10 \
  --USE_LEARNABLE_MULTITASK=0 \
  --USE_EXTRA_METADATA_MODEL=0 \
  --ACCELERATOR=gpu \
  --epochs=1000 \
  --VAL_CHECK_INTERVAL=10000 \
  --USE_TENSORBOARD_LOGGING=1 \
  --LOG_DIAGNOSTICS_DURING_TRAINING=1
```

**Contrastive, underweighted** (weight/temperature not yet corrected to
match the reference branch -- kept for the record, see "Known issues"):
same as above plus
```
  --USE_CONTRASTIVE_LOSS=1 \
  --CONTRASTIVE_LOSS_WEIGHT=0.1 \
  --CONTRASTIVE_LOSS_TEMPERATURE=0.1
```
`--CHECKPOINT_DIR=/mnt/data/nkubrakov/simba_paper_repro/experiments/paper_run_contrastive/`

**Contrastive, weight=1.0 + projection head** (ran on asimov2 instead --
asimov was fully allocated by the other two runs; same
`ANALOG_DISCOVERY_DATA_DIR`/`PREPROCESSING_DIR_TRAIN` but pointed at
`/mnt/data2/nkubrakov/...` directly, since that's asimov2's own local copy
of the same data -- and `--TRAINING_NUM_WORKERS=2`/`--cpus-per-task=2`,
reduced to fit the CPU headroom actually free on that shared node):
same as baseline plus
```
  --USE_CONTRASTIVE_LOSS=1 \
  --CONTRASTIVE_LOSS_WEIGHT=1.0 \
  --USE_CONTRASTIVE_PROJECTION_HEAD=1
```
`--CHECKPOINT_DIR=/mnt/data2/nkubrakov/simba_paper_repro/experiments/paper_run_contrastive_w1_projhead/`

All three were stopped (not crashed) once the diagnostics were confirmed
working and the comparison below was judged inconclusive enough to not
justify further compute. Checkpoints/logs remain on disk for anyone who
wants to resume or inspect further.

## Comparison (best checkpoint per run, selected by diag_val/spearman_mces)

`ModelCheckpoint`'s own selection (lowest `validation_loss_epoch`) is
**not reliable** here -- see "Known issues" -- so checkpoints below are
instead the step with the highest `diag_val/spearman_mces` per run (the
diagnostics callback's own, correctly-balanced validation metric).

| metric | paper checkpoint (offline, 520K steps) | baseline (610K) | contrastive w0.1/t0.1 (560K) | contrastive w1.0+proj (350K) |
|---|---|---|---|---|
| spearman (test) | 0.9275 | 0.9308 | 0.9307 | 0.9347 |
| edit-distance accuracy | 0.4955 | 0.4741 | 0.4705 | 0.4774 |
| Search A median (lower better) | 0.1710 | 0.1587 | 0.1726 | 0.1567 |
| Search A ROC AUC | 0.9162 | 0.9268 | 0.9150 | 0.9269 |
| Search A precision@10 | 0.1212 | 0.0833 | 0.0985 | 0.0909 |
| Search A precision@100 | 0.3409 | 0.2652 | 0.2424 | 0.2424 |
| Search B median (lower better) | 0.2036 | 0.1860 | 0.1926 | 0.1932 |
| Search B ROC AUC | 0.7968 | 0.7810 | 0.7824 | 0.7915 |
| Search B precision@10 | 0.0758 | 0.0758 | 0.0606 | 0.0833 |
| Search B precision@100 | 0.2121 | 0.1894 | 0.1742 | 0.1742 |

No clean win for the contrastive/retrieval loss over the baseline at
either weight -- mixed, mostly-flat-to-slightly-worse on analog discovery,
a small edge on spearman/accuracy for the fully-weighted+projection-head
variant. Not conclusive either way given how early all three checkpoints
still are relative to the paper's full run (which itself only reached
520K steps / epoch 0).

## Known issues (pre-existing in `final_training.py`, not introduced here)

**Validation uses a mismatched, over-aggressive class-balancing weight,
making `validation_loss_epoch` unreliable for checkpoint selection.**
`weights_val = WeightSampling.compute_sample_weights_categories(molecule_pairs_val, weights)`
reuses `weights`, the per-edit-distance-bin inverse-frequency weight array
computed from **train's** bin frequencies, instead of computing its own
from validation's (differently-shaped) bin frequencies. Train's
"identical molecule" bin is far rarer relative to its pool (1-in-8,475)
than validation's own (1-in-829), so applying train's more aggressive
correction over-shoots: instead of the intended ~1/6 balanced share, that
bin ends up at ~64% of every validation batch. Consequences:
- `validation_loss_epoch` (what `ModelCheckpoint` monitors) is dominated
  by the easiest possible examples (a spectrum vs. a different measurement
  of itself, cosine~1, edit distance 0), not representative of overall
  task difficulty.
- The sampler also redraws fresh with `replacement=True` on every
  validation pass (no per-call seed), so on top of the systematic skew,
  `validation_loss_epoch` is noisy round to round -- e.g. our baseline
  run's validation-loss minimum (step 210K, 6.10) never recurs afterward
  (step 610K: 9.0), which is why it was not used for the comparison table.
- This affects every run through this script unconditionally, including
  the original published `best_model_paper_SIMBA.ckpt` (Sebastian's exact
  training command does not override any weighting-related flag).
- **Not** affected: the diagnostics callback's own regression/classification
  plots (hexbin + confusion matrix), which use `TrainUtils.uniformise` --
  a self-contained, per-split balancing mechanism that never references
  the other split's weights.

**Contrastive loss weight/temperature on the first two runs don't match
the reference branch's own defaults** (`0.1`/`0.1` used here vs. `1.0`/`0.07`
on `exp/triplet-loss`) -- not caught until after those two jobs were
already running. The third run (`w1.0+proj`) corrects the weight (and adds
the projection head) but was not re-tested with `temperature=0.07`.

**Self-pairs are ~15% of training batches, not an oddity.** Batch size is
128; `contrastive_n_pairs` averages ~20/128 in real training steps (the
`CustomWeightedRandomSampler`'s inverse-frequency balancing deliberately
oversamples the otherwise 1-in-8,475-rare "identical molecule" bin up to
its balanced ~1/6 share). A naive read of the TensorBoard aggregate for
this tag will show ~58%/batch instead -- that's an artifact of
`log_every_n_steps=50` (Lightning's default, never overridden here)
throttling training-step logging 50x while validation logs every batch
uncapped, so the aggregate is mostly validation batches (at their own,
much-higher, buggy ~64% rate), not training batches.
