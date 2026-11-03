"""Periodic in-training diagnostics, run at validation epoch end (gated by
config.LOG_DIAGNOSTICS_DURING_TRAINING, at most once per every_n_steps):

  (1) Supplementary-Figure-1-style plots: MCES regression hexbin (+ Spearman
      r, MAE and balanced MAE in bonds) and edit-distance 6-class confusion
      matrix (+ accuracy, balanced accuracy), on the test
      and the val split (sharded ed_mces_indexes_tani_incremental_{test,val}
      files + the training pickle's spectra, uniform-binned as in
      inference_scripts/inference_multitasking.py), resolved once at start.

  (2) The analog-discovery benchmark of tools/analog_discovery_benchmark/
      (Sebastian's query/library sets, normalized-MCES ground truth, his
      metrics), scored with the CURRENT model: all spectra preprocessed with
      sqrt + L2-normalised intensities as in his notebook (and in training),
      embedded with the spectrum encoder,
      pairs scored with the model's own heads (compute_from_embeddings), ranked
      with his compute_ranking. Logged as analog_{A,B}/{median, q05, q25, q75,
      q95, roc_auc, tpr_at_fpr_0.05, precision_at_K} plus one figure with the
      fixed baselines; with msnlib_benchmark_dir also with the MSnLib queries
      (msnlib_{A,B}/...). Scoring runs on the GPU here; the reported benchmark
      numbers of a checkpoint come from steps 5-6 of the benchmark, which run
      his CPU path, so in-training values can differ in the last digits.

Optionally saves a checkpoint (diag_step_<step>.ckpt) every time the
diagnostics run, so any diagnostics step can be re-scored offline.
Scalars go through pl_module.log; figures need a TensorBoardLogger.
"""
import copy
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lightning.pytorch.callbacks import Callback  # noqa: E402
from lightning.pytorch.loggers import TensorBoardLogger  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_LABELS = {
    "spec2vec_ours": "Spec2Vec (ours)", "spec2vec_paper": "Spec2Vec (paper)",
    "ms2deepscore_ours": "MS2DeepScore (ours)", "ms2deepscore_paper": "MS2DeepScore (paper)",
    "dreams_ours": "DreaMS (ours)", "dreams_paper_embedding": "DreaMS (paper)",
}


def _tb_logger(trainer):
    for lg in trainer.loggers:
        if isinstance(lg, TensorBoardLogger):
            return lg
    return None


MAE_BIN_WIDTH = 2  # bonds, bins of the ground-truth MCES distance for the balanced MAE


def regression_diagnostics(gt_ed, pred_ed, gt_mces_norm2, pred_mces_norm2, label, mces20_max_value=40.0,
                           n_classes=6, figures=True):
    """Supplementary-Figure-1-style metrics on fixed pairs: Spearman r of the predicted
    raw MCES distance (bonds; threshold-censored pairs dropped), its MAE in bonds
    (overall, and balanced: mean of the MAEs per MAE_BIN_WIDTH-bond bin of the true
    distance), edit-distance accuracy and balanced accuracy (mean recall per class),
    plus (optionally) the hexbin and confusion-matrix figures.
    Figure failures are reported, never raised (they must not stop training)."""
    mask = gt_mces_norm2 != 0.5  # drop threshold-censored pairs
    x = mces20_max_value * (1 - gt_mces_norm2[mask])
    y = mces20_max_value * (1 - pred_mces_norm2[mask])
    rho, _ = spearmanr(x, y)
    error = np.abs(y - x)
    bins = np.floor(np.round(x, 6) / MAE_BIN_WIDTH).astype(int)
    mae_per_bin = {f"{b * MAE_BIN_WIDTH}-{(b + 1) * MAE_BIN_WIDTH}": (float(error[bins == b].mean()), int((bins == b).sum()))
                   for b in np.unique(bins)}
    true = n_classes - 1 - gt_ed
    preds = n_classes - 1 - pred_ed
    cm = confusion_matrix(true, preds, labels=list(range(n_classes)))
    accuracy = accuracy_score(true, preds)
    result = {"spearman_mces": float(rho), "mae_mces": float(error.mean()),
              "mae_mces_balanced": float(np.mean([mae for mae, _ in mae_per_bin.values()])),
              "mae_mces_per_bin": mae_per_bin,  # "lo-hi" bonds -> (MAE, pairs)
              "edit_distance_accuracy": float(accuracy),
              "edit_distance_balanced_accuracy": float(balanced_accuracy_score(true, preds)),
              "n_mces_pairs": int(mask.sum()), "n_edit_distance_pairs": int(len(true))}
    if not figures:
        return result
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        hb = ax.hexbin(x, y, gridsize=20, mincnt=1, cmap="Greens", extent=(0, mces20_max_value, 0, mces20_max_value))
        fig.colorbar(hb, ax=ax, label="pairs")
        ax.set(xlabel="Ground truth MCES distance", ylabel="Predicted MCES distance",
               xlim=(0, mces20_max_value), ylim=(0, mces20_max_value),
               title=f"MCES prediction, {label} (Spearman r={rho:.2f})")
        result["mces_hexbin"] = fig

        cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = [">5", "4", "3", "2", "1", "0"][:n_classes]
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        threshold = cm_norm.max() / 2.0
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j, i, f"{cm_norm[i, j]:.0%}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > threshold else "black")
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set(xlabel="Predicted", ylabel="Ground truth",
               title=f"Edit-distance confusion matrix, {label}, Acc: {accuracy:.2f}")
        fig.tight_layout()
        result["edit_distance_confusion"] = fig
    except Exception as e:
        plt.close("all")
        print(f"[regression_diagnostics] figures for {label} skipped: {e!r}")
    return result


class TrainingDiagnosticsCallback(Callback):
    def __init__(
        self,
        preprocessing_dir_train: str,
        dataset_path: str,
        benchmark_dir: str,
        casmi_pkl: str,
        gnps_mgf: str,
        mces20_max_value: float = 40.0,
        n_regression_classes: int = 6,
        searches: list[str] | None = None,
        baselines: list[str] | None = None,
        every_n_steps: int = 0,
        checkpoint_dir: str | None = None,
        msnlib_benchmark_dir: str | None = None,
        msnlib_mgf: str | None = None,
        analog_every_n_steps: int = 0,
        precursor_mass_mode: str = "measured",
    ):
        """benchmark_dir: work dir of tools/analog_discovery_benchmark (gt_inputs/,
        gt_results/, baseline_scores/<name>/). baselines: names under baseline_scores/.
        every_n_steps: 0 = every validation; N = at most once per N training steps.
        checkpoint_dir: if set, save diag_step_<step>.ckpt there at every run.
        msnlib_benchmark_dir, msnlib_mgf: also run the benchmark with the MSnLib queries
        (work dir of the --queries msnlib benchmark), logged as msnlib_{A,B}/...
        analog_every_n_steps: run the analog-discovery part at most once per N steps
        (0 = at every diagnostics run); the regression part runs at every run.
        precursor_mass_mode: precursor m/z of the val/test regression pairs, "measured"
        or "theoretical" (as the run's training data); the analog searches keep the measured one."""
        from tools.analog_discovery_benchmark.benchmark_sets import SetLoader

        self.every_n_steps = every_n_steps
        self._last_run_step = None
        self.analog_every_n_steps = analog_every_n_steps
        self.precursor_mass_mode = precursor_mass_mode
        self._last_analog_step = None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.mces20_max_value = mces20_max_value
        self.n_regression_classes = n_regression_classes
        self.benchmark_dir = Path(benchmark_dir)
        self.searches = searches or ["A", "B"]
        self.baselines = baselines if baselines is not None else ["spec2vec_ours", "ms2deepscore_ours", "dreams_ours"]

        print("[TrainingDiagnosticsCallback] Preparing fixed regression/classification sets (test + val) ...")
        self._regression = {
            split: self._prepare_regression_set(preprocessing_dir_train, dataset_path, split)
            for split in ("test", "val")
        }
        print("[TrainingDiagnosticsCallback] Preparing the analog-discovery benchmark (sets, ground truth, baselines) ...")
        benchmarks = {"casmi": (SetLoader(dataset_path, casmi_pkl, gnps_mgf), self.benchmark_dir)}
        if msnlib_benchmark_dir:
            benchmarks["msnlib"] = (SetLoader(dataset_path, casmi_pkl, gnps_mgf, "msnlib", msnlib_mgf),
                                    Path(msnlib_benchmark_dir))
        self._analog = {name: {search: self._prepare_analog_discovery(loader, search, benchmark_dir)
                               for search in self.searches}
                        for name, (loader, benchmark_dir) in benchmarks.items()}

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
    # One-time setup: (2) analog-discovery benchmark
    # ------------------------------------------------------------------

    def _prepare_analog_discovery(self, loader, search, benchmark_dir=None):
        from simba.transformers.load_data_base import LoadDataBase
        from tools.analog_discovery_benchmark.benchmark_metrics import GroundTruth
        from tools.analog_discovery_benchmark.benchmark_sets import check_order

        benchmark_dir = Path(benchmark_dir or self.benchmark_dir)
        sets_dir = benchmark_dir / "gt_inputs"
        queries, library = loader.load(search)
        check_order(sets_dir, search, queries, library)
        gt = GroundTruth(benchmark_dir / "gt_results", sets_dir, search)
        # his notebook's input path, once: SIMBA preprocessing + L2 (LoadDataBase), then the
        # per-spectrum sqrt + L2 that CustomDatasetEncoder applies (as training does)
        inputs = {"queries": LoadDataBase.load_spectrum_data(queries, max_num_peaks=100),
                  "library": LoadDataBase.load_spectrum_data(library, max_num_peaks=100)}
        for data in inputs.values():
            intensity = np.sqrt(data["intensity"])
            data["intensity"] = intensity / np.sqrt(np.sum(intensity ** 2, axis=1, keepdims=True))
        baselines = {}
        for name in self.baselines:  # fixed scores: evaluated once
            baselines[BASELINE_LABELS.get(name, name)] = gt.evaluate(gt.load_scores(benchmark_dir / "baseline_scores" / name))
        print(f"  [{loader.query_set} {search}] {gt.n_queries} queries x {gt.n_spectra} library spectra, baselines: {list(baselines)}")
        return {"gt": gt, "inputs": inputs, "baselines": baselines}

    # ------------------------------------------------------------------
    # Per-validation hook
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer, pl_module):
        # NOT on_validation_end: Lightning forbids pl_module.log(...) there
        # (validation metrics are already closed out by that point).
        if trainer.sanity_checking:
            return
        step = trainer.global_step
        if self.every_n_steps and self._last_run_step is not None and step - self._last_run_step < self.every_n_steps:
            return
        self._last_run_step = step
        tb_logger = _tb_logger(trainer)
        pl_module.eval()
        with torch.no_grad():
            for split in ("test", "val"):
                self._log_regression_diagnostics(pl_module, tb_logger, step, split)
            if (not self.analog_every_n_steps or self._last_analog_step is None
                    or step - self._last_analog_step >= self.analog_every_n_steps):
                self._last_analog_step = step
                for name in self._analog:
                    self._log_analog_discovery(pl_module, tb_logger, step, name)
        pl_module.train()
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(self.checkpoint_dir / f"diag_step_{step:07d}.ckpt")

    # ------------------------------------------------------------------
    # (1) Supp Fig 1 style, test + val
    # ------------------------------------------------------------------

    def _embed_and_predict(self, pl_module, molecule_pairs, device):
        from simba.ordinal_classification.load_data_multitasking import LoadDataMultitasking
        from torch.utils.data import DataLoader

        dataset = LoadDataMultitasking.from_molecule_pairs_to_dataset(
            molecule_pairs, max_num_peaks=100, precursor_mass_mode=getattr(self, "precursor_mass_mode", "measured"))
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

        result = regression_diagnostics(gt_ed, pred_ed, gt_mces_norm2, pred_mces_norm2, split,
                                        self.mces20_max_value, self.n_regression_classes,
                                        figures=tb_logger is not None)
        for name in ("spearman_mces", "mae_mces", "mae_mces_balanced", "edit_distance_accuracy",
                     "edit_distance_balanced_accuracy"):
            pl_module.log(f"diag_{split}/{name}", result[name], on_step=False, on_epoch=True)
        for name in ("mces_hexbin", "edit_distance_confusion"):
            if result.get(name) is not None:
                tb_logger.experiment.add_figure(f"diag_plots/{name}_{split}", result[name], global_step=step)
                plt.close(result[name])

    # ------------------------------------------------------------------
    # (2) Analog-discovery benchmark with the current model
    # ------------------------------------------------------------------

    @staticmethod
    def _embed(pl_module, data, device, batch_size=1024):
        """Encoder.forward of his notebook: spectrum_encoder, first token, ReLU."""
        embeddings = []
        for start in range(0, len(data["mz"]), batch_size):
            part = {k: torch.from_numpy(np.asarray(v[start:start + batch_size])).to(device).float() for k, v in data.items()}
            emb, _ = pl_module.spectrum_encoder(mz_array=part["mz"], intensity_array=part["intensity"],
                                                precursor_mass=part["precursor_mass"],
                                                precursor_charge=part["precursor_charge"])
            embeddings.append(torch.relu(emb[:, 0, :]))
        return torch.cat(embeddings)

    @staticmethod
    def _score_pairs(pl_module, emb_q, emb_lib, chunk_size=65536):
        """compute_emb_from_existing_embeddings for every (query, library) pair:
        -> MCES similarity (n_q, n_lib) and predicted edit-distance class."""
        n_q, n_lib = len(emb_q), len(emb_lib)
        sim_mces = np.empty((n_q, n_lib), dtype=np.float64)
        ed_class = np.empty((n_q, n_lib), dtype=np.int64)
        lib = torch.relu(emb_lib)
        for i in range(n_q):
            for start in range(0, n_lib, chunk_size):
                part = lib[start:start + chunk_size]
                logits, sim = pl_module.compute_from_embeddings(torch.relu(emb_q[i]).expand_as(part), part)
                sim_mces[i, start:start + len(part)] = sim.reshape(-1).double().cpu().numpy()
                ed_class[i, start:start + len(part)] = logits.argmax(dim=-1).cpu().numpy()
        return sim_mces, ed_class

    def _log_analog_discovery(self, pl_module, tb_logger, step, benchmark="casmi"):
        from tools.analog_discovery_benchmark.benchmark_metrics import K_VALUES, draw_search, simba_ranking

        device = pl_module.device
        fig, axes = (plt.subplots(len(self.searches), 3, figsize=(18, 5.5 * len(self.searches)), squeeze=False)
                     if tb_logger is not None else (None, None))
        for row, search in enumerate(self.searches):
            d = self._analog[benchmark][search]
            emb_q = self._embed(pl_module, d["inputs"]["queries"], device)
            emb_lib = self._embed(pl_module, d["inputs"]["library"], device)
            sim_mces, ed_class = self._score_pairs(pl_module, emb_q, emb_lib)
            result = d["gt"].evaluate(simba_ranking(sim_mces, ed_class))

            tag = f"analog_{search}" if benchmark == "casmi" else f"{benchmark}_{search}"
            for name, value in result["best_of_10_distance"].items():
                pl_module.log(f"{tag}/{name}", value, on_step=False, on_epoch=True)
            pl_module.log(f"{tag}/roc_auc", result["roc_auc"], on_step=False, on_epoch=True)
            pl_module.log(f"{tag}/tpr_at_fpr_0.05", result["tpr_at_fpr_0.05"], on_step=False, on_epoch=True)
            for k in K_VALUES:
                pl_module.log(f"{tag}/precision_at_{k}", result["precision_at_k"][k], on_step=False, on_epoch=True)
            if fig is not None:
                try:
                    draw_search(axes[row], d["gt"], {**d["baselines"], f"SIMBA (step {step})": result})
                except Exception as e:
                    print(f"[TrainingDiagnosticsCallback] analog figure for search {search} skipped: {e!r}")
        if fig is not None:
            fig.tight_layout()
            suffix = "" if benchmark == "casmi" else f"_{benchmark}"
            tb_logger.experiment.add_figure(f"diag_plots/analog_discovery{suffix}", fig, global_step=step)
            plt.close(fig)
