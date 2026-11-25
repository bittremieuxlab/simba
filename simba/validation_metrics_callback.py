"""Checkpoint-selection metrics during training.

Every n training steps (at a validation run) the model is scored on fixed,
class-balanced val and test pairs (uniformised as in
inference_scripts/inference_multitasking.py, resolved once at start):

  spearman_mces                     Spearman r of predicted vs true MCES distance
  mae_mces                          mean |predicted - true| MCES distance, in bonds
  mae_mces_balanced                 mean of the MAEs per 2-bond bin of the true distance
  edit_distance_accuracy            6-class edit-distance accuracy
  edit_distance_balanced_accuracy   mean recall per edit-distance class

Each run appends one row to <out_dir>/val_metrics.tsv and saves
<out_dir>/step_<step>.ckpt; tools/select_checkpoint.py picks the checkpoint
with the best mean rank of val Spearman and val balanced MAE.
"""

import copy
import sys
from pathlib import Path

import dill
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader

import simba
from simba.load_mces.load_mces import LoadMCES
from simba.ordinal_classification.load_data_multitasking import LoadDataMultitasking
from simba.train_utils import TrainUtils

MAE_BIN_WIDTH = 2  # bonds, bins of the ground-truth MCES distance for the balanced MAE
METRICS = ("spearman_mces", "mae_mces", "mae_mces_balanced", "edit_distance_accuracy", "edit_distance_balanced_accuracy")


def remove_duplicate_pairs(pairs):
    """First occurrence of each unordered (index0, index1) pair, in the original order."""
    i0 = pairs[:, 0].astype(np.int64)
    i1 = pairs[:, 1].astype(np.int64)
    lo, hi = np.minimum(i0, i1), np.maximum(i0, i1)
    _, first = np.unique(lo * (int(hi.max()) + 1) + hi, return_index=True)
    return pairs[np.sort(first)]


def selection_metrics(gt_ed, pred_ed, gt_mces_norm, pred_mces_norm, mces20_max_value=40.0):
    """Metrics of METRICS from ground-truth / predicted edit-distance classes and
    normalized MCES similarities (1 - MCES / mces20_max_value)."""
    mask = gt_mces_norm != 0.5  # drop threshold-censored pairs
    x = mces20_max_value * (1 - gt_mces_norm[mask])
    y = mces20_max_value * (1 - pred_mces_norm[mask])
    error = np.abs(y - x)
    bins = np.floor(np.round(x, 6) / MAE_BIN_WIDTH).astype(int)
    return {
        "spearman_mces": float(spearmanr(x, y)[0]),
        "mae_mces": float(error.mean()),
        "mae_mces_balanced": float(np.mean([error[bins == b].mean() for b in np.unique(bins)])),
        "edit_distance_accuracy": float(accuracy_score(gt_ed, pred_ed)),
        "edit_distance_balanced_accuracy": float(balanced_accuracy_score(gt_ed, pred_ed)),
    }


class ValidationMetricsCallback(Callback):
    def __init__(self, preprocessing_dir, dataset_path, out_dir, every_n_steps=0,
                 mces20_max_value=40.0, n_classes=6, max_num_peaks=100, splits=("val", "test")):
        """every_n_steps: 0 = at every validation run; N = at most once per N training steps."""
        self.out_dir = Path(out_dir)
        self.every_n_steps = every_n_steps
        self.mces20_max_value = mces20_max_value
        self.n_classes = n_classes
        self.max_num_peaks = max_num_peaks
        self._last_run_step = None

        sys.modules["src"] = simba
        with open(dataset_path, "rb") as f:
            dataset = dill.load(f)
        print("[ValidationMetricsCallback] Preparing the fixed class-balanced pairs ...")
        self.pairs = {split: self._prepare_pairs(preprocessing_dir, dataset[f"molecule_pairs_{split}"], split)
                      for split in splits}

    def _prepare_pairs(self, preprocessing_dir, molecule_pairs, split):
        indexes_tani = LoadMCES.merge_numpy_arrays(
            preprocessing_dir, prefix=f"ed_mces_indexes_tani_incremental_{split}",
            use_edit_distance=True, use_multitask=True,
        )
        indexes_tani = remove_duplicate_pairs(indexes_tani)

        pairs_ed = copy.deepcopy(molecule_pairs)
        pairs_ed.indexes_tani = indexes_tani[:, 0:3]
        pairs_ed.tanimotos = indexes_tani[:, 3]
        pairs_mces = copy.deepcopy(molecule_pairs)
        pairs_mces.indexes_tani = indexes_tani[:, [0, 1, 3]]
        pairs_mces.tanimotos = indexes_tani[:, 3]

        pairs_ed, _ = TrainUtils.uniformise(
            pairs_ed, number_bins=self.n_classes - 1,
            return_binned_list=True, bin_sim_1=True, ordinal_classification=True,
        )
        pairs_mces, _ = TrainUtils.uniformise(
            pairs_mces, number_bins=self.n_classes - 1,
            return_binned_list=True, bin_sim_1=False,
        )
        print(f"  [{split}] {len(pairs_ed)} edit-distance pairs, {len(pairs_mces)} MCES pairs")
        return {"ed": pairs_ed, "mces": pairs_mces}

    def _predict(self, pl_module, molecule_pairs):
        dataset = LoadDataMultitasking.from_molecule_pairs_to_dataset(molecule_pairs, max_num_peaks=self.max_num_peaks)
        gt1, gt2, pred1, pred2 = [], [], [], []
        for batch in DataLoader(dataset, batch_size=256, shuffle=False):
            batch_dev = {k: v.to(pl_module.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits1, logits2 = pl_module(batch_dev)
            gt1.append(batch["similarity"].numpy().reshape(-1))
            gt2.append(batch["similarity2"].numpy().reshape(-1))
            pred1.append(logits1.argmax(dim=-1).cpu().numpy().reshape(-1))
            pred2.append(logits2.detach().cpu().numpy().reshape(-1))
        return np.concatenate(gt1), np.concatenate(pred1), np.concatenate(gt2), np.concatenate(pred2)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        step = trainer.global_step
        if self.every_n_steps and self._last_run_step is not None and step - self._last_run_step < self.every_n_steps:
            return
        self._last_run_step = step

        row = {"step": step}
        pl_module.eval()
        with torch.no_grad():
            for split, pairs in self.pairs.items():
                gt_ed, pred_ed, _, _ = self._predict(pl_module, pairs["ed"])
                _, _, gt_mces, pred_mces = self._predict(pl_module, pairs["mces"])
                for name, value in selection_metrics(gt_ed, pred_ed, gt_mces, pred_mces, self.mces20_max_value).items():
                    row[f"{split}_{name}"] = value
                    pl_module.log(f"val_metrics/{split}_{name}", value, on_step=False, on_epoch=True)
        pl_module.train()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self.out_dir / f"step_{step:07d}.ckpt"
        trainer.save_checkpoint(checkpoint)
        row["checkpoint"] = checkpoint.name
        tsv = self.out_dir / "val_metrics.tsv"
        new_file = not tsv.exists()
        with open(tsv, "a") as f:
            if new_file:
                f.write("\t".join(row) + "\n")
            f.write("\t".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row.values()) + "\n")
