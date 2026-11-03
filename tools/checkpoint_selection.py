"""Score every saved checkpoint of a run, to choose a checkpoint-selection rule:
regression metrics on the val (and test) split, and the CASMI (and optionally
MSnLib) analog-discovery benchmark, with one analog figure per checkpoint.

Per checkpoint and split (training diagnostics' regression_diagnostics, same
fixed class-balanced pairs for all checkpoints):
  spearman_mces                     Spearman r of predicted vs true MCES distance
  mae_mces                          mean |predicted - true| MCES distance, in bonds
  mae_mces_balanced                 mean of the MAEs per 2-bond bin of the true distance
  edit_distance_accuracy            6-class edit-distance accuracy
  edit_distance_balanced_accuracy   mean recall per edit-distance class
Analog discovery (Search A and B, the GPU path of the training diagnostics):
best-of-10 distance quantiles, ROC AUC, precision@K; columns casmi_A/..., msnlib_A/...

Output in <out_dir>:
  metrics.tsv              one row per checkpoint; val ranks (1 = best) of Spearman,
                           balanced MAE and balanced ED accuracy, plus their mean ranks
  summary.json             everything, incl. MAE per distance bin
  <casmi|msnlib>_<label>.png/.svg  figure: baselines, --reference (dashed), this checkpoint

Usage (needs a GPU):
  .venv/bin/python tools/checkpoint_selection.py --out_dir <dir> \
      --checkpoint_dir <run>/diag_checkpoints --reference paper=<paper ckpt> \
      --preprocessing_dir <data>/preprocessing_ed_mces_20250123/ \
      --dataset_pkl <data>/preprocessing_ed_mces_20250123/edit_distance_neurips_nist_exhaustive.pkl \
      --benchmark_dir <data>/analog_discovery_sebastian \
      [--msnlib_benchmark_dir <data>/analog_discovery_msnlib --msnlib_mgf <enamdisc_diverse_500_nonconsensus.mgf>]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from simba.config import Config  # noqa: E402
from simba.ordinal_classification.embedder_multitask import EmbedderMultitask  # noqa: E402
from simba.training_diagnostics_callback import TrainingDiagnosticsCallback, regression_diagnostics  # noqa: E402
from tools.analog_discovery_benchmark.benchmark_metrics import draw_search, simba_ranking  # noqa: E402
from tools.analog_discovery_benchmark.benchmark_sets import SetLoader  # noqa: E402

RANKED = {"spearman_mces": False, "mae_mces_balanced": True, "edit_distance_balanced_accuracy": False}  # True = lower is better


def checkpoint_list(args):
    checkpoints = []
    if args.checkpoint_dir:
        for path in sorted(Path(args.checkpoint_dir).glob("diag_step_*.ckpt"))[args.stride - 1::args.stride]:
            step = int(re.search(r"diag_step_(\d+)", path.name).group(1))
            checkpoints.append((f"step {round(step / 1000)}K", str(path), step))
    for item in args.checkpoints:
        label, path = item.split("=", 1)
        checkpoints.append((label, path, None))
    return checkpoints


def load_model(ckpt, config):
    return EmbedderMultitask.load_from_checkpoint(
        ckpt, d_model=int(config.D_MODEL), n_layers=int(config.N_LAYERS), weights=None,
        n_classes=config.EDIT_DISTANCE_N_CLASSES, use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL, lr=config.LR,
        use_cosine_distance=config.use_cosine_distance, strict=False, map_location="cuda").eval()


def score(model, helper, pairs, analog, config):
    """Regression metrics per split and analog results per search for one model."""
    out = {}
    with torch.no_grad():
        for split, p in pairs.items():
            gt_ed, pred_ed, _, _ = helper._embed_and_predict(model, p["pairs_ed"], "cuda")
            _, _, gt_mces, pred_mces = helper._embed_and_predict(model, p["pairs_mces"], "cuda")
            out[split] = regression_diagnostics(gt_ed, pred_ed, gt_mces, pred_mces, split, config.MCES20_MAX_VALUE,
                                                config.EDIT_DISTANCE_N_CLASSES, figures=False)
        for benchmark, searches in analog.items():
            for search, d in searches.items():
                emb_q = helper._embed(model, d["inputs"]["queries"], "cuda")
                emb_lib = helper._embed(model, d["inputs"]["library"], "cuda")
                out[f"{benchmark}_{search}"] = d["gt"].evaluate(simba_ranking(*helper._score_pairs(model, emb_q, emb_lib)))
    return out


def public(result):
    """Drop the arrays kept for drawing."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def row(label, step, result, splits, analog_keys, reference=False):
    r = {"label": label, "step": step, "reference": reference}
    for split in splits:
        for name in ("spearman_mces", "mae_mces", "mae_mces_balanced", "edit_distance_accuracy",
                     "edit_distance_balanced_accuracy"):
            r[f"{split}/{name}"] = result[split][name]
    for key in analog_keys:
        a = result[key]
        for q in ("median", "q75", "q95"):
            r[f"{key}/{q}"] = a["best_of_10_distance"][q]
        r[f"{key}/roc_auc"] = a["roc_auc"]
        for k in (5, 10, 100):
            r[f"{key}/P@{k}"] = a["precision_at_k"][k]
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", default=None, help="run's diag_checkpoints/ (all diag_step_*.ckpt)")
    ap.add_argument("--stride", type=int, default=1, help="with --checkpoint_dir: every N-th checkpoint")
    ap.add_argument("--checkpoints", nargs="*", default=[], help="extra label=checkpoint pairs")
    ap.add_argument("--reference", default=None, help="label=checkpoint, drawn dashed in every figure, not ranked")
    ap.add_argument("--preprocessing_dir", required=True)
    ap.add_argument("--dataset_pkl", required=True)
    ap.add_argument("--benchmark_dir", required=True, help="analog-discovery work dir (gt_inputs, gt_results, baseline_scores)")
    ap.add_argument("--msnlib_benchmark_dir", default=None, help="work dir of the --queries msnlib benchmark")
    ap.add_argument("--msnlib_mgf", default=None)
    ap.add_argument("--casmi_pkl", default=str(REPO / "data/casmi/casmi2022_spec_df.pkl"))
    ap.add_argument("--gnps_mgf", default=str(REPO / "data/gnps/ALL_GNPS_NO_PROPOGATED_sebastian.mgf"))
    ap.add_argument("--baselines", nargs="*", default=["spec2vec_ours", "ms2deepscore_ours", "dreams_ours"])
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--searches", nargs="+", default=["A", "B"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--precursor_mass_mode", default="measured", choices=["measured", "theoretical"],
                    help="precursor m/z of the val/test pairs; use the run's own PRECURSOR_MASS_MODE")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = Config()

    helper = TrainingDiagnosticsCallback.__new__(TrainingDiagnosticsCallback)  # only its helpers
    helper.precursor_mass_mode = args.precursor_mass_mode
    helper.n_regression_classes, helper.mces20_max_value = config.EDIT_DISTANCE_N_CLASSES, config.MCES20_MAX_VALUE
    helper.benchmark_dir, helper.baselines = Path(args.benchmark_dir), args.baselines
    pairs = {split: helper._prepare_regression_set(args.preprocessing_dir, args.dataset_pkl, split) for split in args.splits}
    benchmarks = {"casmi": (SetLoader(args.dataset_pkl, args.casmi_pkl, args.gnps_mgf), args.benchmark_dir)}
    if args.msnlib_benchmark_dir:
        benchmarks["msnlib"] = (SetLoader(args.dataset_pkl, args.casmi_pkl, args.gnps_mgf, "msnlib", args.msnlib_mgf),
                                args.msnlib_benchmark_dir)
    analog = {name: {search: helper._prepare_analog_discovery(loader, search, benchmark_dir) for search in args.searches}
              for name, (loader, benchmark_dir) in benchmarks.items()}
    analog_keys = [f"{name}_{search}" for name in analog for search in args.searches]

    reference = None
    summary, rows = {}, []
    if args.reference:
        ref_label, ref_ckpt = args.reference.split("=", 1)
        reference = (f"SIMBA ({ref_label})", score(load_model(ref_ckpt, config), helper, pairs, analog, config))
        summary[ref_label] = {"checkpoint": ref_ckpt, "reference": True,
                              **{k: public(v) for k, v in reference[1].items()}}
        rows.append(row(ref_label, None, reference[1], args.splits, analog_keys, reference=True))

    for label, ckpt, step in checkpoint_list(args):
        result = score(load_model(ckpt, config), helper, pairs, analog, config)
        summary[label] = {"checkpoint": ckpt, "step": step, **{k: public(v) for k, v in result.items()}}
        rows.append(row(label, step, result, args.splits, analog_keys))
        print(json.dumps(rows[-1]), flush=True)

        for benchmark, searches in analog.items():
            fig, axes = plt.subplots(len(searches), 3, figsize=(18, 5.5 * len(searches)), squeeze=False)
            for i, (search, d) in enumerate(searches.items()):
                methods = {**d["baselines"], f"SIMBA ({label})": result[f"{benchmark}_{search}"]}
                if reference:
                    methods[reference[0]] = reference[1][f"{benchmark}_{search}"]
                draw_search(axes[i], d["gt"], methods)
            fig.suptitle(f"{benchmark.upper()} queries, SIMBA {label}", fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            for ext in ("png", "svg"):
                fig.savefig(out / f"{benchmark}_{label.replace(' ', '_')}.{ext}", dpi=150)
            plt.close(fig)
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")  # partial results survive

    table = pd.DataFrame(rows)
    ranked = ~table["reference"]
    if "val" in args.splits:
        for name, ascending in RANKED.items():
            table.loc[ranked, f"rank_val/{name}"] = table.loc[ranked, f"val/{name}"].rank(ascending=ascending)
        table["mean_rank(spearman,mae_bal)"] = table[["rank_val/spearman_mces", "rank_val/mae_mces_balanced"]].mean(axis=1)
        table["mean_rank(spearman,mae_bal,ed_bal)"] = table[[f"rank_val/{n}" for n in RANKED]].mean(axis=1)
    table.to_csv(out / "metrics.tsv", sep="\t", index=False, float_format="%.5f")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
