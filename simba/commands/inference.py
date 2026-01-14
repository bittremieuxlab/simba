"""Inference command for SIMBA CLI."""

import sys
from pathlib import Path

import click
from hydra import compose, initialize_config_dir

import simba


# Fix import paths for old checkpoints
sys.modules["src"] = simba


@click.command()
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing model checkpoints (e.g., best_model.ckpt or last.ckpt).",
)
@click.option(
    "--preprocessing-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing preprocessed data.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Directory to save output plots and results. Defaults to checkpoint-dir.",
)
@click.argument("overrides", nargs=-1)
def inference(
    checkpoint_dir: Path,
    preprocessing_dir: Path,
    output_dir: Path | None,
    overrides: tuple[str, ...],
) -> None:
    """Run inference on test data using a trained SIMBA model.

    This command loads a trained model from CHECKPOINT_DIR and runs inference on test data
    from the preprocessed dataset. It generates correlation metrics and visualization plots.

    Examples:

        # Basic inference with best model
        simba inference --checkpoint-dir ./checkpoints --preprocessing-dir ./preprocessed_data

        # Use last checkpoint instead of best model
        simba inference --checkpoint-dir ./checkpoints --preprocessing-dir ./preprocessed_data \\
            inference.use_last_model=true

        # Fast dev inference with smaller batch size
        simba inference --checkpoint-dir ./checkpoints --preprocessing-dir ./preprocessed_data \\
            inference=fast_dev

        # Custom batch size and no uniformization
        simba inference --checkpoint-dir ./checkpoints --preprocessing-dir ./preprocessed_data \\
            inference.batch_size=32 inference.uniformize_testing=false
    """
    import os
    import platform

    from simba.logger_setup import logger
    from simba.workflows.inference import inference as run_inference

    # Enable MPS fallback on macOS for unsupported ops
    if platform.system() == "Darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Validate paths
    checkpoint_dir = checkpoint_dir.resolve()
    preprocessing_dir = preprocessing_dir.resolve()

    if not checkpoint_dir.exists():
        raise click.UsageError(f"Checkpoint directory not found: {checkpoint_dir}")

    if not preprocessing_dir.exists():
        raise click.UsageError(
            f"Preprocessing directory not found: {preprocessing_dir}"
        )

    # Check for model checkpoint
    best_model_path = checkpoint_dir / "best_model.ckpt"
    last_model_path = checkpoint_dir / "last.ckpt"

    if not best_model_path.exists() and not last_model_path.exists():
        raise click.UsageError(
            f"No model checkpoint found in {checkpoint_dir}\n"
            f"Expected either 'best_model.ckpt' or 'last.ckpt'"
        )

    # Set output directory
    if output_dir:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = checkpoint_dir

    click.echo(f"Loading config from: {checkpoint_dir}")
    click.echo(f"Preprocessing data: {preprocessing_dir}")
    click.echo(f"Output directory: {output_dir}")

    # Initialize Hydra with absolute path to config directory
    config_dir = Path(__file__).parent.parent.parent / "configs"
    config_dir = str(config_dir.resolve())

    # Build overrides list
    override_list = [
        f"paths.checkpoint_dir={checkpoint_dir}",
        f"paths.preprocessing_dir={preprocessing_dir}",
        f"paths.preprocessing_dir_train={preprocessing_dir}",
        f"paths.output_dir={output_dir}",
    ]
    override_list.extend(overrides)

    try:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=override_list)

        # Validate dataset exists
        dataset_path = Path(preprocessing_dir) / cfg.inference.preprocessing_pickle
        if not dataset_path.exists():
            raise click.UsageError(
                f"Dataset file not found: {dataset_path}\n"
                f"Expected '{cfg.inference.preprocessing_pickle}' in preprocessing directory.\n"
                f"You can override with: inference.preprocessing_pickle=<filename>"
            )

        click.echo(f"\nDataset: {dataset_path}")
        click.echo(f"Batch size: {cfg.inference.batch_size}")
        click.echo(f"Accelerator: {cfg.inference.accelerator}")
        click.echo(f"Use last model: {cfg.inference.use_last_model}")
        click.echo(f"Uniformize testing: {cfg.inference.uniformize_testing}\n")

        # Run inference workflow
        metrics = run_inference(cfg)

        # Display results
        click.echo(f"\n{'=' * 60}")
        click.echo("INFERENCE RESULTS")
        click.echo(f"{'=' * 60}")
        click.echo(f"✓ Edit distance correlation: {metrics['ed_correlation']:.4f}")
        click.echo(f"✓ MCES/Tanimoto correlation: {metrics['mces_correlation']:.4f}")
        click.echo(f"\nResults saved to: {output_dir}")
        click.echo(f"  - Confusion matrix: {output_dir}/cm.png")
        click.echo(
            f"  - Hexbin plot: {output_dir}/hexbin_plot_{cfg.project.extra_info}.png"
        )
        click.echo(
            f"  - Scatter plot: {output_dir}/scatter_plot_{cfg.project.extra_info}.png"
        )

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise click.ClickException(str(e)) from e
