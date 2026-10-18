"""Periodic in-training diagnostics, logged every time validation runs
(on_validation_epoch_end), gated by config.LOG_DIAGNOSTICS_DURING_TRAINING:

  (1) Supplementary-Figure-1-style plots: MCES regression hexbin (+ Spearman
      r) and edit-distance 6-class confusion matrix (+ accuracy). Computed on
      BOTH the test split and the val split (same source data as
      inference_scripts/inference_multitasking.py: the sharded
      ed_mces_indexes_tani_incremental_{test,val}_*.npy files + the training
      pickle's spectra, uniform-binned the same way), each resolved once at
      construction time.

  (2) The full analog-discovery combined plot (oracle + 6 baseline
      checkpoints + SIMBA, box/ROC/precision-at-K), for Search A and/or B,
      for BOTH normalized-MCES ground-truth metrics (paper/Sebastian and
      exact/Johnson) -- exactly reusing tools/analog_discovery_plot.py's own
      prepare_search_data/build_figure/METHODS, the same pipeline and the
      same combined figure simba-paper's own offline results use (no
      "training-only" simplified version). The 6 baseline checkpoints +
      ground truth are prepared/cached ONCE at construction (nothing about
      them changes during training); only SIMBA's own rank is rescored live
      with the CURRENT in-memory model at every validation, using the exact
      same no-mol-agg (every library spectrum, duplicates allowed) scoring
      as tools/analog_discovery_score_simba.py's `no_mol_agg` subcommand --
      not the smaller deduplicated (one spectrum/molecule) library, so
      numbers here are directly comparable to simba-paper's own published
      figures, not an approximation of them.

Both pieces are logged as TensorBoard scalars (via pl_module.log, so they
also land in whatever other logger is attached) and as figures (via
tb_logger.experiment.add_figure), only when a TensorBoardLogger is actually
attached to the trainer.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, accuracy_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# normalized-MCES ground-truth cache filename -> (metric label, key used in
# metric-qualified tensorboard tags). Matches tools/analog_discovery_plot.py's
# run() exactly (same two files, same two combined PNGs it produces offline).
ANALOG_METRICS = [
    ("normalized_mces_cache_sebastian.npy", "paper", "paper metric"),
    ("normalized_mces_cache.npy", "exact", "exact metric"),
]


def _tb_logger(trainer):
    for lg in trainer.loggers:
        if isinstance(lg, TensorBoardLogger):
            return lg
    return None


class TrainingDiagnosticsCallback(Callback):
    def __init__(
        self,
        preprocessing_dir_train: str,
        dataset_path: str,
        mces20_max_value: float = 40.0,
        n_regression_classes: int = 6,
        analog_discovery_data_dir: str = "/mnt/data2/nkubrakov/analog_discovery",
        analog_discovery_searches: list[str] | None = None,
        analog_discovery_k_values: list[int] | None = None,
        gnps_mgf: str | None = None,
    ):
        self.mces20_max_value = mces20_max_value
        self.n_regression_classes = n_regression_classes
        self.analog_discovery_data_dir = Path(analog_discovery_data_dir)
        self.analog_discovery_searches = analog_discovery_searches or ["A", "B"]
        self.k_values = analog_discovery_k_values or [1, 5, 10, 20, 40, 80, 100]
        self.training_pkl = dataset_path
        self.gnps_mgf = gnps_mgf or str(REPO_ROOT / "data/gnps/ALL_GNPS_NO_PROPOGATED_sebastian.mgf")

        print("[TrainingDiagnosticsCallback] Preparing fixed regression/classification sets (test + val) ...")
        self._regression = {
            split: self._prepare_regression_set(preprocessing_dir_train, dataset_path, split)
            for split in ("test", "val")
        }

        print("[TrainingDiagnosticsCallback] Preparing analog-discovery ground truth + baseline "
              "checkpoints (cached once, exact offline no-mol-agg pipeline) ...")
        self._analog = {key: self._prepare_analog_discovery(key) for key in self.analog_discovery_searches}

    # ------------------------------------------------------------------
    # One-time setup: (1) regression/classification, test + val
    # ------------------------------------------------------------------

    def _prepare_regression_set(self, preprocessing_dir_train, dataset_path, split: str):
        import dill
        import simba
        sys.modules["src"] = simba
        from simba.load_mces.load_mces import LoadMCES
        from simba.train_utils import TrainUtils

        with open(dataset_path, "rb") as f:
            dataset = dill.load(f)
        molecule_pairs_split = dataset[f"molecule_pairs_{split}"]

        indexes_tani_split = LoadMCES.merge_numpy_arrays(
            preprocessing_dir_train, prefix=f"ed_mces_indexes_tani_incremental_{split}",
            use_edit_distance=True, use_multitask=True,
        )
        # Deduplicate once, up front -- this is the slow part (loading/
        # merging sharded arrays), never repeated per validation. No random
        # pre-subsample here: TrainUtils.uniformise below balances classes
        # by taking the rarest class's count, so subsampling first would
        # starve rare classes before balancing ever sees them (confirmed:
        # this used to leave as few as 18 total edit-distance pairs).
        seen, rows = set(), []
        for row in indexes_tani_split:
            key = tuple(sorted(row[:2]))
            if key not in seen:
                seen.add(key)
                rows.append(row)
        indexes_tani_split = np.asarray(rows)

        pairs_ed = copy.deepcopy(molecule_pairs_split)
        pairs_ed.indexes_tani = indexes_tani_split[:, 0:3]
        pairs_ed.tanimotos = indexes_tani_split[:, 3]
        pairs_mces = copy.deepcopy(molecule_pairs_split)
        pairs_mces.indexes_tani = indexes_tani_split[:, [0, 1, 3]]
        pairs_mces.tanimotos = indexes_tani_split[:, 3]

        pairs_ed, _ = TrainUtils.uniformise(
            pairs_ed, number_bins=self.n_regression_classes - 1,
            return_binned_list=True, bin_sim_1=True, ordinal_classification=True,
        )
        pairs_mces, _ = TrainUtils.uniformise(
            pairs_mces, number_bins=self.n_regression_classes - 1,
            return_binned_list=True, bin_sim_1=False,
        )
        print(f"  [{split}] {len(pairs_ed)} edit-distance pairs, {len(pairs_mces)} MCES pairs (fixed, resolved once)")
        return {"pairs_ed": pairs_ed, "pairs_mces": pairs_mces}

    # ------------------------------------------------------------------
    # One-time setup: (2) analog discovery -- ground truth + 6 baselines,
    # cached exactly like the offline pipeline; SIMBA itself is NOT cached
    # here (rescored live every validation).
    # ------------------------------------------------------------------

    def _prepare_analog_discovery(self, key):
        from tools.analog_discovery_plot import prepare_search_data, load_scores
        from tools.analog_discovery_score_simba import load_library_a_all, load_library_b_all

        search_key = {"A": "Search A", "B": "Search B"}[key]
        label = {"A": "Search A (NIST20 + MassSpecGym)", "B": "Search B (GNPS)"}[key]
        exact_dir = self.analog_discovery_data_dir / f"search_{key}_exact_mces_final"
        scores_dir = self.analog_discovery_data_dir / "scores" / f"search_{key}"
        no_mol_agg_dir = self.analog_discovery_data_dir / "results_final_no_mol_agg"
        baseline_dir = self.analog_discovery_data_dir / "baseline_scores"
        cache_dir = no_mol_agg_dir / "plot_cache"

        # Baselines + ground truth: identical call to the offline script, so
        # this transparently reuses its on-disk cache (same cache_dir) --
        # the only "SIMBA" entry it contains is the stale paper-checkpoint
        # one, which _log_analog_discovery overwrites with a live rank
        # before ever calling build_figure.
        prepared = prepare_search_data(search_key, scores_dir, exact_dir, no_mol_agg_dir,
                                        baseline_dir, label, cache_dir, refresh=False)

        # Everything needed to rescore SIMBA live, no-mol-agg style (every
        # library spectrum, duplicates allowed -- matches
        # analog_discovery_score_simba.py's run_no_mol_agg exactly).
        from tools.analog_discovery_score_simba import load_spectra
        mgf = self.analog_discovery_data_dir / {"A": "search_A_nist_msg.mgf", "B": "search_B_gnps.mgf"}[key]
        _, q_spectra = load_spectra(str(mgf), "query")
        lib_smiles_dedup = prepared["lib_smiles_dedup"]
        lib_pos = {s: i for i, s in enumerate(lib_smiles_dedup)}
        if key == "A":
            lib_smiles_all, lib_spectra_all = load_library_a_all(self.training_pkl, lib_pos)
        else:
            lib_smiles_all, lib_spectra_all = load_library_b_all(self.gnps_mgf, lib_pos)
        lib_mol_idx_live = np.array([lib_pos[s] for s in lib_smiles_all], dtype=np.int32)
        print(f"  [{key}] live no-mol-agg library: {len(lib_spectra_all)} spectra "
              f"({len(set(lib_smiles_all))} unique molecules), {len(q_spectra)} queries")

        return {
            "search_key": search_key, "label": label, "exact_dir": exact_dir,
            "prepared": prepared, "q_spectra": q_spectra,
            "lib_spectra_all": lib_spectra_all, "lib_mol_idx_live": lib_mol_idx_live,
        }

    # ------------------------------------------------------------------
    # Per-validation hook
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer, pl_module):
        # NOT on_validation_end: Lightning forbids pl_module.log(...) there
        # (validation metrics are already closed out by that point).
        tb_logger = _tb_logger(trainer)
        step = trainer.global_step
        pl_module.eval()
        with torch.no_grad():
            for split in ("test", "val"):
                self._log_regression_diagnostics(pl_module, tb_logger, step, split)
            self._log_analog_discovery(pl_module, tb_logger, step)
        pl_module.train()

    # ------------------------------------------------------------------
    # (1) Supp Fig 1 style, test + val
    # ------------------------------------------------------------------

    def _embed_and_predict(self, pl_module, molecule_pairs, device):
        from simba.ordinal_classification.load_data_multitasking import LoadDataMultitasking
        from torch.utils.data import DataLoader

        dataset = LoadDataMultitasking.from_molecule_pairs_to_dataset(molecule_pairs, max_num_peaks=100)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        gt1, gt2, pred1, pred2 = [], [], [], []
        for batch in loader:
            batch_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits1, logits2 = pl_module(batch_dev)
            gt1.append(batch["similarity"].numpy().reshape(-1))
            gt2.append(batch["similarity2"].numpy().reshape(-1))
            pred1.append(logits1.argmax(dim=-1).cpu().numpy().reshape(-1))
            pred2.append(logits2.detach().cpu().numpy().reshape(-1))
        return (np.concatenate(gt1), np.concatenate(pred1),
                np.concatenate(gt2), np.concatenate(pred2))

    def _log_regression_diagnostics(self, pl_module, tb_logger, step, split: str):
        device = pl_module.device
        d = self._regression[split]
        gt_ed, pred_ed, _, _ = self._embed_and_predict(pl_module, d["pairs_ed"], device)
        _, _, gt_mces_norm2, pred_mces_norm2 = self._embed_and_predict(pl_module, d["pairs_mces"], device)

        # --- panel (b): MCES hexbin ---
        mask = gt_mces_norm2 != 0.5  # drop threshold-censored pairs
        x = self.mces20_max_value * (1 - gt_mces_norm2[mask])
        y = self.mces20_max_value * (1 - pred_mces_norm2[mask])
        rho, _ = spearmanr(x, y)
        pl_module.log(f"diag_{split}/spearman_mces", float(rho), on_step=False, on_epoch=True)

        sns.set_theme(style="ticks")
        g = sns.jointplot(x=x, y=y, kind="hex", color="#4CB391", joint_kws=dict(alpha=1, gridsize=20))
        g.set_axis_labels("Ground truth similarity", "Prediction", fontsize=12)
        g.fig.suptitle(f"MCES prediction, {split} (Spearman r={rho:.2f})", fontsize=14)
        g.ax_joint.set_xlim(0, self.mces20_max_value)
        g.ax_joint.set_ylim(0, self.mces20_max_value)
        if tb_logger is not None:
            tb_logger.experiment.add_figure(f"diag_plots/mces_hexbin_{split}", g.fig, global_step=step)
        plt.close(g.fig)

        # --- panel (c): edit-distance confusion matrix ---
        true = self.n_regression_classes - 1 - gt_ed
        preds = self.n_regression_classes - 1 - pred_ed
        cm = confusion_matrix(true, preds, labels=list(range(self.n_regression_classes)))
        accuracy = accuracy_score(true, preds)
        pl_module.log(f"diag_{split}/edit_distance_accuracy", float(accuracy), on_step=False, on_epoch=True)
        cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

        fig_cm = plt.figure(figsize=(8, 6))
        labels = [">5", "4", "3", "2", "1", "0"][: self.n_regression_classes]
        im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im)
        threshold = cm_norm.max() / 2.0
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                color = "white" if cm_norm[i, j] > threshold else "black"
                plt.text(j, i, f"{cm_norm[i, j]:.0%}", ha="center", va="center", color=color)
        plt.xticks(ticks=np.arange(len(labels)), labels=labels)
        plt.yticks(ticks=np.arange(len(labels)), labels=labels)
        plt.xlabel("Predicted"); plt.ylabel("Ground truth")
        plt.title(f"Edit-distance confusion matrix, {split}, Acc: {accuracy:.2f}")
        plt.tight_layout()
        if tb_logger is not None:
            tb_logger.experiment.add_figure(f"diag_plots/edit_distance_confusion_{split}", fig_cm, global_step=step)
        plt.close(fig_cm)

    # ------------------------------------------------------------------
    # (2) Full analog-discovery combined plot (oracle + 6 baselines +
    # live SIMBA), both normalized-MCES metrics, exact no-mol-agg parity
    # with the offline pipeline.
    # ------------------------------------------------------------------

    def _score_simba_live(self, pl_module, d, device):
        from tools.analog_discovery_score_simba import embed_spectra, score_pairs
        from simba.analog_discovery.paper_metrics import rank_by_mces_then_edit_distance

        q_embs = embed_spectra(pl_module, d["q_spectra"], batch_size=256, device=device)
        lib_embs = embed_spectra(pl_module, d["lib_spectra_all"], batch_size=256, device=device)
        pred_mces, pred_edit = score_pairs(pl_module, q_embs, lib_embs, device, pair_chunk_size=20000)
        rank = rank_by_mces_then_edit_distance(pred_mces, pred_edit)  # (n_q, n_lib_spec), 0=best
        return rank

    def _log_analog_discovery(self, pl_module, tb_logger, step):
        from tools.analog_discovery_plot import build_figure, flat_rank_of_true_best
        from sklearn.metrics import auc, roc_curve

        device = pl_module.device
        rank_live, tag_by_key = {}, {}
        for key in self.analog_discovery_searches:
            d = self._analog[key]
            rank = self._score_simba_live(pl_module, d, device)
            d["prepared"]["methods"]["SIMBA"] = (rank, d["lib_mol_idx_live"])
            rank_live[key] = rank
            tag_by_key[key] = f"analog_{key}"

        for cache_filename, metric_key, metric_label in ANALOG_METRICS:
            for key in self.analog_discovery_searches:
                d = self._analog[key]
                rank = rank_live[key]
                gt_raw = d["prepared"]["gt_raw"]
                n_q = gt_raw.shape[0]
                tag = tag_by_key[key]
                gt_norm = np.load(d["exact_dir"] / cache_filename)

                # best-in-top-10 by normalized metric (SIMBA only, for scalars)
                best_norm = []
                for i in range(n_q):
                    top = np.argsort(rank[i])[:10]
                    cand_mol = d["lib_mol_idx_live"][top]
                    vals = gt_norm[i, cand_mol]
                    ok = ~np.isnan(vals)
                    if ok.any():
                        best_norm.append(float(np.min(vals[ok])))
                best_norm = np.asarray(best_norm)
                if len(best_norm):
                    for q, name in [(0.05, "q05"), (0.25, "q25"), (0.5, "median"), (0.75, "q75"), (0.95, "q95")]:
                        pl_module.log(f"{tag}_{metric_key}/{name}", float(np.quantile(best_norm, q)),
                                      on_step=False, on_epoch=True)

                # ROC @ normalized-MCES < 0.3 (SIMBA only)
                gt_norm_per_spec = gt_norm[np.arange(n_q)[:, None], d["lib_mol_idx_live"][None, :]]
                valid = ~np.isnan(gt_norm_per_spec)
                labels_roc = (gt_norm_per_spec[valid] < 0.3).astype(np.int32)
                pred = -rank[valid]
                fpr, tpr, _ = roc_curve(labels_roc, pred)
                roc_auc = auc(fpr, tpr)
                tpr_at_05 = float(np.interp(0.05, fpr, tpr))
                pl_module.log(f"{tag}_{metric_key}/roc_auc", float(roc_auc), on_step=False, on_epoch=True)
                pl_module.log(f"{tag}_{metric_key}/tpr_at_fpr_0.05", tpr_at_05, on_step=False, on_epoch=True)

                # precision@K, Sebastian's literal flat-rank algorithm (SIMBA only)
                found_ranks = flat_rank_of_true_best(rank, d["lib_mol_idx_live"], gt_raw)
                for k in self.k_values:
                    p = float((found_ranks <= k).mean()) if len(found_ranks) else float("nan")
                    pl_module.log(f"{tag}_{metric_key}/precision_at_{k}", p, on_step=False, on_epoch=True)

            if tb_logger is None:
                continue
            # Full combined figure, both searches as rows: oracle + 6
            # baselines + live SIMBA, same function/METHODS list/styling as
            # the offline pipeline -- no separate "training-only" plot.
            searches = [(self._analog[key]["search_key"], self._analog[key]["label"], None, None)
                        for key in self.analog_discovery_searches]
            prepared_by_key = {self._analog[key]["search_key"]: self._analog[key]["prepared"]
                                for key in self.analog_discovery_searches}
            exact_dirs_by_key = {self._analog[key]["search_key"]: self._analog[key]["exact_dir"]
                                  for key in self.analog_discovery_searches}
            out_path = Path(f"/tmp/_training_diag_{metric_key}_{step}.png")
            build_figure(searches, prepared_by_key, exact_dirs_by_key, metric_label, cache_filename, out_path)
            fig = plt.imread(out_path)
            tb_logger.experiment.add_image(f"diag_plots/analog_{metric_key}", fig, global_step=step, dataformats="HWC")
            out_path.unlink(missing_ok=True)
            out_path.parent.joinpath(f"summary_{out_path.stem}.json").unlink(missing_ok=True)
