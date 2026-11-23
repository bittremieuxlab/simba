"""SIMBA acquisition-regime stratified inference and figures.


Run with the same Hydra overrides as the existing inference script.
Optional: analysis.training_abundance_csv=/path/to/training_counts.csv
CSV columns: field,value,n_train_spectra (field: adduct, ionmode, ce_interval,
ion_activation, ionization_method). Without it, abundance uses TEST support
and is explicitly labeled as such, NOT as training-data abundance.
"""
import os
import sys
from pathlib import Path


import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from torch.utils.data import DataLoader


import simba.core.data.molecule_pairs
import simba.core.data.spectrum
from simba.core.data.datasets.multitask_dataset_builder import MultitaskDataBuilder
from simba.core.training.train_utils import TrainUtils
from simba.workflows.inference import (
   _get_ground_truth, _which_index, load_inference_data,
   load_model_for_inference, run_inference as core_run_inference,
)


sys.modules['simba.molecule_pairs_opt'] = simba.core.data.molecule_pairs
sys.modules['simba.molecular_pairs'] = simba.core.data.molecule_pairs
sys.modules['simba.spectrum'] = simba.core.data.spectrum
sys.modules['simba.spectrum_ext'] = simba.core.data.spectrum


FIELDS = ('adduct', 'ionmode', 'ce_interval', 'ion_activation', 'ionization_method')
CE_BINS = (0, 10, 20, 30, 40, 60, 100, float('inf'))  # numeric CE, instrument-specific units
MIN_PAIRS = 10




def prepare(cfg, pairs_ed, pairs_mces):
   if cfg.inference.uniformize_testing:
       n_bins = cfg.data.edit_distance_n_classes - 1
       pairs_ed, _ = TrainUtils.uniformise(pairs_ed, number_bins=n_bins,
           return_binned_list=True, bin_sim_1=True, ordinal_classification=True)
       pairs_mces, _ = TrainUtils.uniformise(pairs_mces, number_bins=n_bins,
           return_binned_list=True, bin_sim_1=False)
   def make(pairs):
       dataset = MultitaskDataBuilder.from_molecule_pairs_to_dataset(
           pairs, training=False,
           max_num_peaks=int(cfg.model.transformer.context_length),
           use_adduct=cfg.model.features.use_adduct,
           use_ce=cfg.model.features.use_ce,
           use_ion_activation=cfg.model.features.use_ion_activation,
           use_ion_method=cfg.model.features.use_ion_method,
           use_ion_mode=cfg.model.features.use_ion_mode)
       return DataLoader(dataset, batch_size=cfg.inference.batch_size, shuffle=False)
   return make(pairs_ed), make(pairs_mces), pairs_ed, pairs_mces




def flatten_predictions(pred_ed, pred_mces):
   ed = np.asarray([_which_index(row) for batch in pred_ed for row in batch[0]], dtype=float)
   mc = np.asarray([v for batch in pred_mces
                    for v in np.asarray(batch[1].detach().cpu()).reshape(-1)], dtype=float)
   return ed, mc




def safe_value(v):
   if v is None or isinstance(v, float) and not np.isfinite(v):
       return 'Missing'
   s = str(v).strip()
   return s if s and s.lower() not in ('nan', 'none', 'null') else 'Missing'




def ce_group(value):
   if isinstance(value, (list, tuple, np.ndarray)):
       value = value[0] if len(value) == 1 else None
   try:
       val = float(value)
   except (TypeError, ValueError):
       # Do not interpret NCE strings or ranges as absolute CE.
       return 'Missing/unparsed'
   if not np.isfinite(val) or val < 0:
       return 'Missing/unparsed'
   for lo, hi in zip(CE_BINS[:-1], CE_BINS[1:]):
       if lo <= val < hi:
           return f'{lo:g}–{hi:g}' if np.isfinite(hi) else f'{lo:g}+'
   return 'Missing/unparsed'




def spectrum_metadata(spec):
   p = spec.params
   return {
       'adduct': safe_value(p.get('adduct')),
       'ionmode': safe_value(p.get('ionmode')),
       'ce_interval': ce_group(p.get('ce')),
       'ion_activation': safe_value(p.get('ion_activation')),
       'ionization_method': safe_value(p.get('ionization_method')),
   }




def metadata_for_pairs(pairs, n_expected):
   # pair_distances columns 0 and 1 are unique spectrum indices in supplied code.
   if len(pairs.pair_distances) != n_expected:
       raise ValueError(f'Pair/prediction alignment mismatch: {len(pairs.pair_distances)} vs {n_expected}')
   cache = {}
   records = []
   for i, row in enumerate(pairs.pair_distances):
       ids = (int(row[0]), int(row[1]))
       for side, idx in enumerate(ids):
           key = (side, idx)
           if key not in cache:
               spec = pairs.get_original_spectrum_from_unique_index(idx, pair=side)
               cache[key] = spectrum_metadata(spec)
       records.append({f'{field}_{side}': cache[(side, idx)][field]
                       for side, idx in enumerate(ids) for field in FIELDS})
   return pd.DataFrame.from_records(records)




def metrics(y, p, kind):
   y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
   mask = np.isfinite(y) & np.isfinite(p)
   y, p = y[mask], p[mask]
   result = {'n_pairs': len(y), 'mae': float(np.mean(np.abs(y-p))) if len(y) else np.nan,
             'spearman': np.nan, 'accuracy': np.nan}
   if len(y) >= 2 and np.unique(y).size > 1 and np.unique(p).size > 1:
       result['spearman'] = float(spearmanr(y, p).statistic)
   if kind == 'ed' and len(y):
       result['accuracy'] = float(np.mean(y == p))
   return result




def make_rows(meta, y, pred, task, min_pairs):
   rows = []
   for field in FIELDS:
       a, b = meta[f'{field}_0'], meta[f'{field}_1']
       same = (a == b) & (~a.isin(['Missing', 'Missing/unparsed']))
       for value in sorted(a[same].unique()):
           mask = same & (a == value)
           if mask.sum() < min_pairs:
               continue
           result = metrics(y[mask.to_numpy()], pred[mask.to_numpy()], task)
           rows.append({'task': task, 'field': field, 'regime': value,
                        'pair_rule': 'both spectra same category', **result})
   return rows




def add_abundance(meta, y, pred, task, counts, min_pairs):
   # Counts are based on *distinct spectrum identities* when supplied from training.
   # The fallback is the number of test spectra per category, not training abundance.
   rows = []
   for field in FIELDS:
       col0, col1 = meta[f'{field}_0'], meta[f'{field}_1']
       same = (col0 == col1) & (~col0.isin(['Missing', 'Missing/unparsed']))
       if counts is None:
           # Test pair frequency is only a proxy; not independent spectra counts.
           freq = col0[same].value_counts().to_dict()
           source = 'test_pair_frequency_proxy'
       else:
           sub = counts[counts['field'] == field]
           freq = dict(zip(sub['value'].astype(str), sub['n_train_spectra'].astype(float)))
           source = 'training_spectrum_counts'
       for label, lo, hi in [('Low', 0, 100), ('Medium', 100, 1000), ('High', 1000, np.inf)]:
           allowed = {v for v, count in freq.items() if lo <= count < hi}
           mask = same & col0.isin(allowed)
           if mask.sum() < min_pairs:
               continue
           r = metrics(y[mask.to_numpy()], pred[mask.to_numpy()], task)
           rows.append({'task': task, 'field': f'abundance_{field}',
                        'regime': label, 'pair_rule': f'both same category; {source}', **r})
   return rows




# Five acquisition fields; one row per field in stratified figures.
DISPLAY_FIELDS = (
   ('ionmode', 'Ion mode'),
   ('adduct', 'Adduct'),
   ('ce_interval', 'Collision energy'),
   ('ion_activation', 'Activation method'),
   ('ionization_method', 'Ionization method'),
)
FEATURE_FLAGS = (
   ('use_ion_mode', 'Ion mode'),
   ('use_adduct', 'Adduct'),
   ('use_ce', 'Collision energy'),
   ('use_ion_activation', 'Activation method'),
   ('use_ion_method', 'Ionization method'),
)
PAPER_DIR = Path('/data/paper_graphs')
METADATA_MODEL_NAME = 'ms2_merged_ref_auto_stratified'
BASELINE_MODEL_NAME = 'ms2_merged_ref_auto_stratified_baseline'
COLORS = {'Metadata model': '#3478aa', 'Baseline model': '#e49b47'}




def set_flags(cfg, active):
   for flag, _ in FEATURE_FLAGS:
       cfg.model.features[flag] = int(flag in active)




def checkpoint_file(cfg, directory):
   name = 'last.ckpt' if cfg.inference.use_last_model else cfg.checkpoints.best_model_name
   path = Path(directory) / name
   if not path.is_file():
       raise FileNotFoundError(f'Missing checkpoint: {path}')
   return path




def infer_variant(cfg, model, pairs_ed, pairs_mces, label):
   # Reconstruct dataloaders so metadata flags affect dataset encoding.
   dl_ed, dl_mces, used_ed, used_mces = prepare(cfg, pairs_ed, pairs_mces)
   ed_batches, mc_batches = core_run_inference(cfg, model, dl_ed, dl_mces)
   ed_pred, mc_pred = flatten_predictions(ed_batches, mc_batches)
   ed_true, _ = _get_ground_truth(dl_ed)
   _, mc_true = _get_ground_truth(dl_mces)
   ed_true = np.asarray(ed_true, dtype=float).reshape(-1)
   mc_true = np.asarray(mc_true, dtype=float).reshape(-1)
   if len(ed_true) != len(ed_pred) or len(mc_true) != len(mc_pred):
       raise ValueError(f'Prediction/ground truth mismatch for {label}')
   if not cfg.data.use_tanimoto:
       scale = float(cfg.data.mces20_max_value)
       mc_true, mc_pred = scale * (1 - mc_true), scale * (1 - mc_pred)
   result = {
       'ed': {'true': ed_true, 'pred': ed_pred,
              'meta': metadata_for_pairs(used_ed, len(ed_pred))},
       'mces': {'true': mc_true, 'pred': mc_pred,
                'meta': metadata_for_pairs(used_mces, len(mc_pred))},
   }
   for task in ('ed', 'mces'):
       result[task]['metrics'] = metrics(result[task]['true'], result[task]['pred'], task)
   print(f'{label}: MCES MAE={result["mces"]["metrics"]["mae"]:.4f}; '
         f'ED MAE={result["ed"]["metrics"]["mae"]:.4f}', flush=True)
   return result




def check_same_test_pairs(metadata_result, baseline_result):
   for task in ('ed', 'mces'):
       a, b = metadata_result[task], baseline_result[task]
       if not np.array_equal(a['true'], b['true'], equal_nan=True):
           raise ValueError(f'{task}: test ground truth differs between models')
       if not a['meta'].equals(b['meta']):
           raise ValueError(f'{task}: test metadata/order differs between models')




def collect_strata(result, model_name, min_pairs):
   rows = []
   for task in ('mces', 'ed'):
       part = result[task]
       for r in make_rows(part['meta'], part['true'], part['pred'], task, min_pairs):
           rows.append({'model': model_name, **r})
   return rows




def paired_bar(ax, categories, mces, ed, title, show_counts=None):
   """Side-by-side bars with separate y axes for unlike MAE scales."""
   x = np.arange(len(categories))
   width = .38
   right = ax.twinx()
   b1 = ax.bar(x - width / 2, mces, width, color='#3478aa', label='MCES MAE')
   b2 = right.bar(x + width / 2, ed, width, color='#e49b47', label='Edit-distance MAE')
   ax.set_xticks(x, categories, rotation=25, ha='right')
   ax.set_ylabel('MCES MAE (distance units)', color='#3478aa')
   right.set_ylabel('Edit-distance MAE', color='#b4772d')
   ax.tick_params(axis='y', labelcolor='#3478aa')
   right.tick_params(axis='y', labelcolor='#b4772d')
   ax.set_ylim(0, max([v for v in mces if np.isfinite(v)] + [1]) * 1.22)
   right.set_ylim(0, max([v for v in ed if np.isfinite(v)] + [0.2]) * 1.22)
   ax.set_title(title)
   ax.grid(axis='y', alpha=.17)
   ax.set_axisbelow(True)
   ax.legend([b1, b2], ['MCES MAE', 'Edit-distance MAE'], loc='upper right', fontsize=8)
   if show_counts:
       for i, value in enumerate(show_counts):
           ax.text(x[i], -.22, value, transform=ax.get_xaxis_transform(),
                   ha='center', va='top', fontsize=7)
   return ax, right




def plot_acquisition_mae(df, output):
   fig, axes = plt.subplots(len(DISPLAY_FIELDS), 1, figsize=(13, 21))
   for ax, (field, title) in zip(axes, DISPLAY_FIELDS):
       sub = df[df.field == field]
       if sub.empty:
           ax.text(.5, .5, 'No eligible pairs', ha='center', transform=ax.transAxes)
           continue
       pivot = sub.pivot(index='regime', columns='task', values='mae')
       pivot = pivot.sort_index()
       paired_bar(ax, pivot.index.astype(str).tolist(),
                  pivot.get('mces', pd.Series(np.nan, index=pivot.index)).to_numpy(),
                  pivot.get('ed', pd.Series(np.nan, index=pivot.index)).to_numpy(), title)
   fig.suptitle('Acquisition-regime performance — metadata model\n'
                'Both spectra share the plotted acquisition category', fontsize=15)
   fig.tight_layout(rect=[0, 0, 1, .96], h_pad=3)
   fig.savefig(output / 'acquisition_regime_mae.svg', bbox_inches='tight')
   plt.close(fig)




def plot_sensitivity_mae(rows, output):
   df = pd.DataFrame(rows)
   fig, ax = plt.subplots(figsize=(12, 5.8))
   paired_bar(ax, df['variant'].tolist(), df['mces_mae'].to_numpy(),
              df['ed_mae'].to_numpy(), 'Metadata sensitivity: no metadata vs one field enabled')
   fig.tight_layout()
   fig.savefig(output / 'metadata_flag_sensitivity_mae.svg', bbox_inches='tight')
   plt.close(fig)




def plot_overall_comparison(overall, output):
   fig, ax = plt.subplots(figsize=(8, 5.2))
   paired_bar(ax, overall['model'].tolist(), overall['mces_mae'].to_numpy(),
              overall['ed_mae'].to_numpy(), 'Overall MAE: metadata vs spectrum-only baseline')
   fig.tight_layout()
   fig.savefig(output / 'overall_model_comparison_mae.svg', bbox_inches='tight')
   plt.close(fig)




def plot_stratified_comparison(df, task, output):
   """Five stacked panels, one y-axis per panel, grouped by model."""
   fig, axes = plt.subplots(len(DISPLAY_FIELDS), 1, figsize=(13, 21))
   for ax, (field, title) in zip(axes, DISPLAY_FIELDS):
       sub = df[(df.field == field) & (df.task == task)]
       if sub.empty:
           ax.text(.5, .5, 'No eligible pairs', ha='center', transform=ax.transAxes)
           continue
       pivot = sub.pivot(index='regime', columns='model', values='mae').sort_index()
       x = np.arange(len(pivot))
       width = .38
       for shift, name in [(-.5, 'Metadata model'), (.5, 'Baseline model')]:
           vals = pivot[name].to_numpy() if name in pivot else np.full(len(pivot), np.nan)
           ax.bar(x + shift * width, vals, width, label=name, color=COLORS[name])
       ax.set_xticks(x, pivot.index.astype(str), rotation=25, ha='right')
       ax.set_ylabel('MCES MAE (distance units)' if task == 'mces' else 'Edit-distance MAE')
       ax.set_title(title)
       ax.grid(axis='y', alpha=.18)
       ax.set_axisbelow(True)
       ax.legend(loc='upper right', fontsize=8)
       vals = pivot.to_numpy(dtype=float)
       if np.isfinite(vals).any():
           ax.set_ylim(0, np.nanmax(vals) * 1.2)
   fig.suptitle(('MCES' if task == 'mces' else 'Edit-distance') +
                ' MAE by acquisition regime — both models', fontsize=15)
   fig.tight_layout(rect=[0, 0, 1, .97], h_pad=3)
   fig.savefig(output / f'{task}_stratified_model_comparison_mae.svg', bbox_inches='tight')
   plt.close(fig)




def run_stratified_inference(cfg: DictConfig):
   PAPER_DIR.mkdir(parents=True, exist_ok=True)
   min_pairs = int(OmegaConf.select(cfg, 'analysis.min_pairs', default=MIN_PAIRS))
   metadata_dir = Path(cfg.paths.checkpoint_dir)
   # By default baseline is a sibling of the supplied metadata checkpoint.
   baseline_dir = Path(OmegaConf.select(cfg, 'analysis.baseline_checkpoint_dir', default=None)
                       or metadata_dir.parent / BASELINE_MODEL_NAME)
   metadata_ckpt = checkpoint_file(cfg, metadata_dir)
   baseline_ckpt = checkpoint_file(cfg, baseline_dir)
   pairs_ed, pairs_mces, _ = load_inference_data(cfg)
   original_flags = {name: cfg.model.features[name] for name, _ in FEATURE_FLAGS}
   try:
       # Load each checkpoint with the feature configuration it was trained with.
       set_flags(cfg, {name for name, _ in FEATURE_FLAGS})
       metadata_model = load_model_for_inference(cfg, str(metadata_ckpt))
       set_flags(cfg, set())
       baseline_model = load_model_for_inference(cfg, str(baseline_ckpt))


       # Spectrum-only checkpoint: metadata fields must all be zero.
       set_flags(cfg, set())
       baseline_result = infer_variant(cfg, baseline_model, pairs_ed, pairs_mces, 'Baseline model')


       # Full-metadata checkpoint: used for both overall and stratified comparisons.
       set_flags(cfg, {name for name, _ in FEATURE_FLAGS})
       full_result = infer_variant(cfg, metadata_model, pairs_ed, pairs_mces, 'Metadata model')
       check_same_test_pairs(full_result, baseline_result)


       # Sensitivity of the metadata checkpoint only: no fields vs each single field.
       variants = [('No metadata', set())] + [
           (f'{label} only', {flag}) for flag, label in FEATURE_FLAGS
       ]
       sensitivity = []
       for label, active in variants:
           set_flags(cfg, active)
           result = infer_variant(cfg, metadata_model, pairs_ed, pairs_mces, label)
           check_same_test_pairs(full_result, result)
           sensitivity.append({'variant': label,
                               'mces_mae': result['mces']['metrics']['mae'],
                               'ed_mae': result['ed']['metrics']['mae'],
                               'mces_n_pairs': result['mces']['metrics']['n_pairs'],
                               'ed_n_pairs': result['ed']['metrics']['n_pairs']})
   finally:
       for name, value in original_flags.items():
           cfg.model.features[name] = value


   metadata_rows = collect_strata(full_result, 'Metadata model', min_pairs)
   baseline_rows = collect_strata(baseline_result, 'Baseline model', min_pairs)
   # Only show categories meeting the minimum sample count for BOTH models.
   comparison = pd.DataFrame(metadata_rows + baseline_rows)
   if comparison.empty:
       raise ValueError('No eligible acquisition regimes; lower analysis.min_pairs')
   counts = comparison.groupby(['task', 'field', 'regime'])['model'].nunique()
   eligible = counts[counts == 2].index
   comparison = comparison.set_index(['task', 'field', 'regime']).loc[eligible].reset_index()
   if comparison.empty:
       raise ValueError('No categories shared by both model results')
   metadata_df = comparison[comparison.model == 'Metadata model'].copy()


   overall = pd.DataFrame([
       {'model': name,
        'mces_mae': res['mces']['metrics']['mae'],
        'ed_mae': res['ed']['metrics']['mae'],
        'mces_n_pairs': res['mces']['metrics']['n_pairs'],
        'ed_n_pairs': res['ed']['metrics']['n_pairs']}
       for name, res in [('Metadata model', full_result), ('Baseline model', baseline_result)]
   ])
   pd.DataFrame(sensitivity).to_csv(PAPER_DIR / 'metadata_flag_sensitivity_mae.csv', index=False)
   metadata_df.to_csv(PAPER_DIR / 'acquisition_regime_mae.csv', index=False)
   overall.to_csv(PAPER_DIR / 'overall_model_comparison_mae.csv', index=False)
   comparison.to_csv(PAPER_DIR / 'stratified_model_comparison_mae.csv', index=False)


   plot_acquisition_mae(metadata_df, PAPER_DIR)
   plot_sensitivity_mae(sensitivity, PAPER_DIR)
   plot_overall_comparison(overall, PAPER_DIR)
   plot_stratified_comparison(comparison, 'mces', PAPER_DIR)
   plot_stratified_comparison(comparison, 'ed', PAPER_DIR)
   print(f'Saved 5 SVG figures and 4 CSV tables to {PAPER_DIR}', flush=True)
   return {
       'ed_correlation': full_result['ed']['metrics']['spearman'],
       'mces_correlation': full_result['mces']['metrics']['spearman'],
       'ed_mae': full_result['ed']['metrics']['mae'],
       'ed_accuracy': full_result['ed']['metrics']['accuracy'],
       'mces_mae': full_result['mces']['metrics']['mae'],
       'output_dir': str(PAPER_DIR),
   }




def run_metadata_sensitivity_analysis(cfg: DictConfig):
   """Existing simba metadata_analysis CLI entry point."""
   return run_stratified_inference(cfg)




def run_inference(cfg: DictConfig):
   """One-argument workflow wrapper (not the underlying inference function)."""
   return run_stratified_inference(cfg)




@hydra.main(version_base=None, config_path='/home/spiedrahita/simba/simba/configs', config_name='config')
def main(cfg: DictConfig):
   return run_stratified_inference(cfg)




if __name__ == '__main__':
   main()



