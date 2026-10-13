"""Re-runs ONE real validation pass for a CORN-equipped checkpoint to
produce the CORN-corrected counterpart of ValMetricsCallback's own
mces_binned_box plot. Raw per-example predictions aren't saved anywhere
during training (only the final PNG), so reproducing a corrected-values
version of that same plot requires actually re-running the real validation
set through the real checkpoint -- this reuses the exact same data-loading
functions training.py itself uses (load_dataset/prepare_data/
create_dataloaders) and the exact same plotting code
(ValMetricsCallback._plot_binned_box), just computing CORN-corrected values
instead of raw ones before calling it.

Usage:
    uv run python tools/rerun_corn_corrected_plot.py \\
        --checkpoint .../full_run_diagnostics/checkpoint-epoch=22-step=228000.ckpt \\
        --hparams_path .../full_run_diagnostics/tb_logs/version_0/hparams.yaml \\
        --output_dir .../full_run_diagnostics \\
        --step 228000
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir


sys.path.insert(0, str(Path(__file__).parent))
from simba_retrieval import model_kwargs_from_hparams  # noqa: E402

from simba.core.models.similarity_models import SimilarityModelMultitask  # noqa: E402
from simba.core.training.callbacks import ValMetricsCallback, _bin_index  # noqa: E402
from simba.utils.config_utils import get_config_path  # noqa: E402
from simba.workflows.training import (  # noqa: E402
    create_dataloaders,
    load_dataset,
    prepare_data,
)


# Mirrors full_run_diagnostics/train.slurm.sh's own overrides (minus
# training-only settings that don't affect what dataset_val contains).
OVERRIDES = [
    "paths.preprocessing_dir=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2",
    "paths.preprocessing_dir_train=/sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2",
    "paths.preprocessing_pickle_file=mapping.pkl",
    "paths.checkpoint_dir=/tmp/rerun_corn_corrected_scratch",
    "paths.mgf_path=/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf",
    "training.batch_size=2048",
    "sampling.add_identity_pairs=true",
    "sampling.use_resampling=true",
    "sampling.precursor_mass_mode=theoretical",
    "sampling.precursor_noise_mode=mist_cf",
    "hardware.num_workers=8",
    "model.features.use_adduct=false",
    "model.features.use_ce=false",
    "model.features.use_ion_mode=false",
    "model.tasks.mces_bucket.enabled=true",
    "model.tasks.mces_bucket.use_mlp=true",
]


def main(checkpoint: str, hparams_path: str, output_dir: str, step: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config_path = get_config_path()
    with initialize_config_dir(
        config_dir=str(config_path.absolute()), version_base=None
    ):
        cfg = compose(config_name="config", overrides=OVERRIDES)

    print("Loading dataset ...")
    mp_train, mp_val, mp_test, mp_uniform_test = load_dataset(cfg)
    print("Preparing data ...")
    ds_train, train_sampler, ds_val, val_sampler = prepare_data(
        mp_train, mp_val, mp_test, mp_uniform_test, cfg
    )
    print("Creating dataloaders ...")
    _, dataloader_val = create_dataloaders(
        cfg, ds_train, train_sampler, ds_val, val_sampler
    )
    print(f"  {len(dataloader_val)} validation batches")

    print(f"Loading model from {checkpoint} ...")
    model_kwargs = model_kwargs_from_hparams(hparams_path)
    model_kwargs.setdefault("weights", None)
    model_kwargs.setdefault("lr", 1e-4)
    model = SimilarityModelMultitask.load_from_checkpoint(
        checkpoint, map_location=device, strict=False, **model_kwargs
    )
    model.eval()
    model.to(device)
    assert model.use_mces_bucket_head, (
        "checkpoint has no mces_bucket head -- nothing to correct"
    )

    mces_preds, mces_targets, bucket_preds = [], [], []
    print("Running validation forward passes ...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader_val):
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits_list, *_ = model._forward_with_embeddings(batch)
            logits2, logits3 = logits_list
            target2 = batch["mces"].to(dtype=torch.float32, device=device).view(-1)
            bucket_pred = model._corn_decode_bin_generic(logits3)
            mces_preds.append(logits2.view(-1).cpu().numpy())
            mces_targets.append(target2.cpu().numpy())
            bucket_preds.append(bucket_pred.cpu().numpy())
            if i % 20 == 0:
                print(f"  batch {i}/{len(dataloader_val)}")

    mces_pred_sim = np.concatenate(mces_preds)
    mces_target_sim = np.concatenate(mces_targets)
    bucket_pred = np.concatenate(bucket_preds)
    print(f"Total validation pairs: {len(mces_pred_sim)}")

    pred_mces = (1.0 - mces_pred_sim) * model.mces_max_value
    gt_mces = (1.0 - mces_target_sim) * model.mces_max_value
    is_self = gt_mces <= 1e-6

    mae_bin_edges = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=float)
    bin_idx = _bin_index(gt_mces, is_self, mae_bin_edges)

    edges = model.mces_bucket_bin_edges.cpu().numpy()
    from simba_retrieval_iceberg import _corn_corrected_mces

    corrected_mces = _corn_corrected_mces(pred_mces, bucket_pred, edges)

    abs_err_raw = np.abs(pred_mces - gt_mces)
    abs_err_corrected = np.abs(corrected_mces - gt_mces)
    print(f"Raw MAE (this eval):       {abs_err_raw.mean():.4f}")
    print(f"Corrected MAE (this eval): {abs_err_corrected.mean():.4f}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    callback = ValMetricsCallback(
        output_dir=str(out_dir),
        mae_bin_edges=mae_bin_edges,
        mces_max_value=model.mces_max_value,
    )

    # Raw version first (sanity check against the real training-time PNG).
    callback._plot_binned_box(pred_mces, bin_idx, step, None)
    raw_path = out_dir / f"mces_binned_box_step{step:06d}.png"
    raw_renamed = out_dir / f"mces_binned_box_step{step:06d}_RAW_rerun.png"
    shutil.move(raw_path, raw_renamed)
    print(f"Wrote {raw_renamed}")

    # Corrected version -- the one actually requested.
    callback._plot_binned_box(corrected_mces, bin_idx, step, None)
    corrected_path = out_dir / f"mces_binned_box_step{step:06d}.png"
    corrected_renamed = out_dir / f"mces_binned_box_step{step:06d}_CORRECTED.png"
    shutil.move(corrected_path, corrected_renamed)
    print(f"Wrote {corrected_renamed}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hparams_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--step", type=int, required=True)
    args = p.parse_args()
    main(args.checkpoint, args.hparams_path, args.output_dir, args.step)
