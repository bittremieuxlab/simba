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
import re
import warnings


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
   load_model_for_inference, run_inference,
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




def plot_tables(df, output):
   for field, part in df.groupby('field', sort=False):
       for metric, title in [('mae', 'Mean absolute error'), ('spearman', 'Spearman correlation'),
                             ('accuracy', 'Edit-distance accuracy')]:
           subset = part if metric != 'accuracy' else part[part.task == 'ed']
           subset = subset[np.isfinite(subset[metric])]
           if subset.empty:
               continue
           fig, ax = plt.subplots(figsize=(max(7, len(subset.regime.unique()) * 1.0), 4.8))
           pivot = subset.pivot_table(index='regime', columns='task', values=metric, aggfunc='first')
           pivot = pivot.reindex(columns=[x for x in ('mces', 'ed') if x in pivot.columns])
           pivot.plot.bar(ax=ax, width=0.75, rot=35)
           ax.set_title(f'{field.replace("_", " ").title()} — {title}')
           ax.set_xlabel('Acquisition category (both spectra)')
           ax.set_ylabel(title + (' (MCES distance units)' if metric == 'mae' else ''))
           ax.grid(axis='y', alpha=0.2)
           ax.set_axisbelow(True)
           ax.legend(title='Task')
           fig.tight_layout()
           name = re.sub(r'[^a-zA-Z0-9_-]', '_', field)
           fig.savefig(output / f'{name}_{metric}.png', dpi=220, bbox_inches='tight')
           fig.savefig(output / f'{name}_{metric}.pdf', bbox_inches='tight')
           plt.close(fig)




def run_stratified_inference(cfg: DictConfig):
   ckpt = os.path.join(cfg.paths.checkpoint_dir,
       'last.ckpt' if cfg.inference.use_last_model else cfg.checkpoints.best_model_name)
   base = cfg.paths.get('output_dir') or cfg.paths.checkpoint_dir
   out = Path(base) / 'acquisition_stratified'
   out.mkdir(parents=True, exist_ok=True)
   pairs_ed, pairs_mces, _ = load_inference_data(cfg)
   dl_ed, dl_mces, pairs_ed, pairs_mces = prepare(cfg, pairs_ed, pairs_mces)
   model = load_model_for_inference(cfg, ckpt)
   pred_ed_batches, pred_mc_batches = run_inference(cfg, model, dl_ed, dl_mces)
   ed_pred, mc_pred = flatten_predictions(pred_ed_batches, pred_mc_batches)
   ed_true, _ = _get_ground_truth(dl_ed)
   _, mc_true = _get_ground_truth(dl_mces)
   ed_true, mc_true = np.asarray(ed_true, dtype=float).reshape(-1), np.asarray(mc_true, dtype=float).reshape(-1)
   if not (len(ed_true) == len(ed_pred) and len(mc_true) == len(mc_pred)):
       raise ValueError('Ground truth and predictions are not aligned')
   # The supplied workflow uses normalized MCES similarity, converting to distance here.
   if not cfg.data.use_tanimoto:
       scale = float(cfg.data.mces20_max_value)
       mc_true, mc_pred = scale * (1-mc_true), scale * (1-mc_pred)
   ed_meta = metadata_for_pairs(pairs_ed, len(ed_pred))
   mc_meta = metadata_for_pairs(pairs_mces, len(mc_pred))
   min_pairs = int(OmegaConf.select(cfg, 'analysis.min_pairs', default=MIN_PAIRS))
   counts_path = OmegaConf.select(cfg, 'analysis.training_abundance_csv', default=None)
   counts = pd.read_csv(counts_path) if counts_path else None
   if counts is not None and not {'field','value','n_train_spectra'}.issubset(counts.columns):
       raise ValueError('Training abundance CSV must contain field,value,n_train_spectra')
   if counts is None:
       warnings.warn('Training abundance counts unavailable: abundance plots use TEST PAIR FREQUENCY proxy, not training abundance.')
   rows = []
   for task, meta, y, p in [('ed', ed_meta, ed_true, ed_pred),
                            ('mces', mc_meta, mc_true, mc_pred)]:
       rows.extend(make_rows(meta, y, p, task, min_pairs))
       rows.extend(add_abundance(meta, y, p, task, counts, min_pairs))
       overall = metrics(y, p, task)
       print(f'{task.upper()} overall: {overall}')
   df = pd.DataFrame(rows)
   if df.empty:
       raise ValueError('No regimes met minimum pair count. Reduce analysis.min_pairs.')
   df.to_csv(out / 'stratified_metrics.csv', index=False)
   ed_meta.to_csv(out / 'ed_pair_metadata.csv', index=False)
   mc_meta.to_csv(out / 'mces_pair_metadata.csv', index=False)
   plot_tables(df, out)
   print(f'Saved metrics and graphs to: {out}')
   return df




@hydra.main(version_base=None, config_path='/home/spiedrahita/simba/simba/configs', config_name='config')
def main(cfg: DictConfig):
   return run_stratified_inference(cfg)




if __name__ == '__main__':
   main()




