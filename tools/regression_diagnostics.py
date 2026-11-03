"""Supplementary-Figure-1-style diagnostics for SIMBA checkpoints: MCES regression
hexbin (+ Spearman r of the predicted raw MCES distance) and edit-distance
confusion matrix (+ accuracy), on the test and/or val split.

Uses the training diagnostics' own code (simba/training_diagnostics_callback.py):
the split's class-balanced pairs are built ONCE, so all checkpoints are scored on
exactly the same pairs. Needs a GPU.

Output in <out_dir>: <label>_<split>_mces_hexbin.{png,svg},
<label>_<split>_edit_distance_confusion.{png,svg}, summary.json

Usage:
  .venv/bin/python tools/regression_diagnostics.py --out_dir <dir> \
      --preprocessing_dir <data>/preprocessing_ed_mces_20250123/ \
      --dataset_pkl <data>/preprocessing_ed_mces_20250123/edit_distance_neurips_nist_exhaustive.pkl \
      --checkpoints paper_checkpoint=<ckpt> run_step62k=<ckpt> [--splits test val]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simba.config import Config  # noqa: E402
from simba.ordinal_classification.embedder_multitask import EmbedderMultitask  # noqa: E402
from simba.training_diagnostics_callback import TrainingDiagnosticsCallback, regression_diagnostics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="label=checkpoint pairs")
    ap.add_argument("--preprocessing_dir", required=True, help="dir with ed_mces_indexes_tani_incremental_{split}_* shards")
    ap.add_argument("--dataset_pkl", required=True)
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["test", "val"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--precursor_mass_mode", default="measured", choices=["measured", "theoretical"],
                    help="precursor m/z of the val/test pairs; use the run's own PRECURSOR_MASS_MODE")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = Config()

    helper = TrainingDiagnosticsCallback.__new__(TrainingDiagnosticsCallback)  # only its pair/prediction helpers
    helper.precursor_mass_mode = args.precursor_mass_mode
    helper.n_regression_classes, helper.mces20_max_value = config.EDIT_DISTANCE_N_CLASSES, config.MCES20_MAX_VALUE
    pairs = {split: helper._prepare_regression_set(args.preprocessing_dir, args.dataset_pkl, split) for split in args.splits}

    summary = {}
    for label, ckpt in (c.split("=", 1) for c in args.checkpoints):
        model = EmbedderMultitask.load_from_checkpoint(
            ckpt, d_model=int(config.D_MODEL), n_layers=int(config.N_LAYERS), weights=None,
            n_classes=config.EDIT_DISTANCE_N_CLASSES, use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL, lr=config.LR,
            use_cosine_distance=config.use_cosine_distance, strict=False, map_location="cuda").eval()
        summary[label] = {"checkpoint": ckpt}
        with torch.no_grad():
            for split in args.splits:
                gt_ed, pred_ed, _, _ = helper._embed_and_predict(model, pairs[split]["pairs_ed"], "cuda")
                _, _, gt_mces, pred_mces = helper._embed_and_predict(model, pairs[split]["pairs_mces"], "cuda")
                result = regression_diagnostics(gt_ed, pred_ed, gt_mces, pred_mces, f"{split}, {label}",
                                                config.MCES20_MAX_VALUE, config.EDIT_DISTANCE_N_CLASSES)
                for name in ("mces_hexbin", "edit_distance_confusion"):
                    fig = result.pop(name, None)
                    if fig is not None:
                        for ext in ("png", "svg"):
                            fig.savefig(out / f"{label}_{split}_{name}.{ext}", dpi=150)
                        plt.close(fig)
                summary[label][split] = result
                print(f"{label} {split}: Spearman MCES {result['spearman_mces']:.4f}, "
                      f"edit-distance accuracy {result['edit_distance_accuracy']:.4f}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
