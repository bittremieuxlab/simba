"""Analog discovery command for SIMBA CLI."""

from pathlib import Path

import click


@click.command(name="analog-discovery")
@click.option(
    "--model-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the trained SIMBA model checkpoint (e.g., best_model.ckpt).",
)
@click.option(
    "--query-spectra",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to query spectra file (.mgf or .pkl format).",
)
@click.option(
    "--reference-spectra",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to reference library spectra file (.mgf or .pkl format).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory to save analog discovery results and plots.",
)
@click.option(
    "--query-index",
    type=int,
    default=None,
    help="Index of specific query spectrum to analyze (0-based). If not provided, processes all queries.",
)
@click.option(
    "--top-k",
    type=int,
    default=10,
    help="Number of top matches to retrieve for each query.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "gpu", "auto"], case_sensitive=False),
    default="cpu",
    help="Device to run inference on (cpu, gpu, or auto).",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for processing spectra.",
)
@click.option(
    "--cache-embeddings/--no-cache-embeddings",
    default=True,
    help="Cache computed embeddings to speed up repeated searches.",
)
@click.option(
    "--use-gnps-format/--no-use-gnps-format",
    default=False,
    help="Whether the input files use GNPS format.",
)
@click.option(
    "--compute-ground-truth/--no-compute-ground-truth",
    default=False,
    help="Compute ground truth MCES and edit distance for evaluation (requires SMILES in spectra).",
)
@click.option(
    "--save-rankings/--no-save-rankings",
    default=True,
    help="Save full ranking matrices to output directory.",
)
def analog_discovery(
    model_path: Path,
    query_spectra: Path,
    reference_spectra: Path,
    output_dir: Path,
    query_index: int | None,
    top_k: int,
    device: str,
    batch_size: int,
    cache_embeddings: bool,
    use_gnps_format: bool,
    compute_ground_truth: bool,
    save_rankings: bool,
) -> None:
    """Find structural analogs in a reference library using SIMBA.

    This command performs analog discovery by:
    1. Loading query and reference spectra
    2. Computing SIMBA predictions (edit distance and MCES)
    3. Ranking matches based on structural similarity
    4. Saving top matches and visualizations

    Example:

        simba analog-discovery \\
            --model-path ./models/best_model.ckpt \\
            --query-spectra ./data/casmi2022.mgf \\
            --reference-spectra ./data/massspecgym.mgf \\
            --output-dir ./results/analog_discovery \\
            --top-k 10
    """
    # Lazy imports to speed up CLI
    import json
    import os

    import numpy as np

    from simba.config import Config
    from simba.simba.analog_discovery import AnalogDiscovery
    from simba.simba.preprocessing_simba import PreprocessingSimba
    from simba.simba.simba import Simba

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir) + os.sep

    click.echo("=" * 70)
    click.echo("SIMBA Analog Discovery")
    click.echo("=" * 70)

    # Setup configuration
    config = Config()
    config.USE_LEARNABLE_MULTITASK = True
    config.USE_FINGERPRINT = False
    config.BATCH_SIZE = batch_size

    # Load spectra
    click.echo(f"\n📂 Loading query spectra from: {query_spectra}")
    all_spectrums_query = PreprocessingSimba.load_spectra(
        str(query_spectra), config, use_gnps_format=use_gnps_format
    )
    click.echo(f"✓ Loaded {len(all_spectrums_query)} query spectra")

    click.echo(f"\n📂 Loading reference spectra from: {reference_spectra}")
    all_spectrums_reference = PreprocessingSimba.load_spectra(
        str(reference_spectra), config, use_gnps_format=use_gnps_format
    )
    click.echo(f"✓ Loaded {len(all_spectrums_reference)} reference spectra")

    # Validate inputs
    if len(all_spectrums_query) == 0:
        raise click.UsageError("No query spectra found! Check your input file.")
    if len(all_spectrums_reference) == 0:
        raise click.UsageError("No reference spectra found! Check your input file.")

    # Initialize SIMBA model
    click.echo(f"\n🧠 Loading SIMBA model from: {model_path}")
    simba_model = Simba(
        str(model_path), config=config, device=device, cache_embeddings=cache_embeddings
    )
    click.echo("✓ Model loaded successfully")

    # Run predictions
    click.echo(
        f"\n🔬 Computing predictions ({len(all_spectrums_query)} queries × {len(all_spectrums_reference)} references)..."
    )
    sim_ed, sim_mces = simba_model.predict(
        all_spectrums_query,
        all_spectrums_reference,
    )
    click.echo(f"✓ Predictions complete! Shape: {sim_ed.shape}")

    # Compute rankings
    click.echo("\n📊 Computing similarity rankings...")
    ranking = AnalogDiscovery.compute_ranking(sim_mces, sim_ed)
    click.echo(f"✓ Rankings computed! Shape: {ranking.shape}")

    # Save full rankings if requested
    if save_rankings:
        click.echo("\n💾 Saving ranking matrices...")
        np.save(output_path + "similarity_ed.npy", sim_ed)
        np.save(output_path + "similarity_mces.npy", sim_mces)
        np.save(output_path + "rankings.npy", ranking)
        click.echo(f"✓ Saved to: {output_path}")

    # Generate distribution plots
    click.echo("\n📈 Generating distribution plots...")
    _plot_distributions(sim_ed, sim_mces, ranking, output_path)
    click.echo(f"✓ Plots saved to: {output_path}")

    # Process specific query or all queries
    results_summary = []

    if query_index is not None:
        # Process single query
        if query_index >= len(all_spectrums_query):
            raise click.UsageError(
                f"Query index {query_index} is out of range! "
                f"Available indices: 0-{len(all_spectrums_query) - 1}"
            )

        click.echo(f"\n🔍 Analyzing query spectrum at index {query_index}...")
        result = _process_single_query(
            query_index,
            all_spectrums_query,
            all_spectrums_reference,
            sim_ed,
            sim_mces,
            ranking,
            top_k,
            compute_ground_truth,
            output_path,
        )
        results_summary.append(result)
        click.echo(f"✓ Analysis complete for query {query_index}")

    else:
        # Process all queries
        click.echo(
            f"\n🔍 Processing all {len(all_spectrums_query)} queries (top-{top_k} matches each)..."
        )

        for idx in range(len(all_spectrums_query)):
            if idx % 10 == 0:
                click.echo(f"  Processing query {idx}/{len(all_spectrums_query)}...")

            result = _process_single_query(
                idx,
                all_spectrums_query,
                all_spectrums_reference,
                sim_ed,
                sim_mces,
                ranking,
                top_k,
                compute_ground_truth,
                output_path,
                save_plots=False,  # Don't save individual plots for all queries
            )
            results_summary.append(result)

        click.echo("✓ Processed all queries")

    # Save results summary
    click.echo("\n💾 Saving results summary...")
    results_file = output_path + "analog_discovery_results.json"
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    click.echo(f"✓ Results saved to: {results_file}")

    # Print summary statistics
    _print_summary_statistics(results_summary, compute_ground_truth)

    click.echo("\n" + "=" * 70)
    click.echo("✓ Analog discovery complete!")
    click.echo(f"📁 All results saved to: {output_dir}")
    click.echo("=" * 70)


def _process_single_query(
    query_index: int,
    all_spectrums_query,
    all_spectrums_reference,
    sim_ed,
    sim_mces,
    ranking,
    top_k: int,
    compute_ground_truth: bool,
    output_path: str,
    save_plots: bool = True,
) -> dict:
    """Process a single query spectrum and find its top matches."""
    import numpy as np

    from simba.simba.ground_truth import GroundTruth

    spectra_query = all_spectrums_query[query_index]

    # Get top K matches
    best_matches_indices = np.argsort(ranking[query_index])[-top_k:][::-1]
    spectra_matches = [all_spectrums_reference[ind] for ind in best_matches_indices]

    # Get predictions for top matches
    top_rankings = ranking[query_index, best_matches_indices]
    top_ed_predictions = sim_ed[query_index, best_matches_indices]
    top_mces_predictions = sim_mces[query_index, best_matches_indices]

    # Build result dictionary
    result = {
        "query_index": int(query_index),
        "query_smiles": spectra_query.params.get("smiles", "N/A"),
        "query_precursor_mz": float(spectra_query.precursor_mz),
        "top_matches": [],
    }

    # Compute ground truth if requested
    ground_truth_mces = None
    ground_truth_ed = None

    if compute_ground_truth and "smiles" in spectra_query.params:
        try:
            ground_truth_mces = GroundTruth.compute_mces(
                [spectra_query], spectra_matches
            )
            ground_truth_ed = GroundTruth.compute_edit_distance(
                [spectra_query], spectra_matches
            )
            ground_truth_mces = ground_truth_mces[0]  # Extract first row
            ground_truth_ed = ground_truth_ed[0]
        except Exception:
            # If ground truth computation fails, continue without it
            pass

    # Add match information
    for i, match_idx in enumerate(best_matches_indices):
        match_info = {
            "rank": i + 1,
            "reference_index": int(match_idx),
            "ranking_score": float(top_rankings[i]),
            "predicted_edit_distance": float(top_ed_predictions[i]),
            "predicted_mces_distance": float(top_mces_predictions[i]),
            "reference_smiles": spectra_matches[i].params.get("smiles", "N/A"),
            "reference_precursor_mz": float(spectra_matches[i].precursor_mz),
        }

        if ground_truth_mces is not None:
            match_info["ground_truth_mces_distance"] = float(ground_truth_mces[i])
            match_info["ground_truth_edit_distance"] = float(ground_truth_ed[i])

        result["top_matches"].append(match_info)

    # Save plots if requested
    if save_plots:
        _save_query_plots(
            query_index,
            spectra_query,
            spectra_matches,
            best_matches_indices,
            top_rankings,
            top_ed_predictions,
            top_mces_predictions,
            ground_truth_mces,
            ground_truth_ed,
            output_path,
        )

    return result


def _plot_distributions(sim_ed, sim_mces, ranking, output_path: str) -> None:
    """Plot distributions of predictions and rankings."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Sample random predictions
    flat_ed = sim_ed.ravel()
    flat_mces = sim_mces.ravel()
    flat_ranking = ranking.ravel()

    n_samples = min(10000, len(flat_ed))
    idx = np.random.choice(len(flat_ed), size=n_samples, replace=False)

    samples_ed = flat_ed[idx]
    samples_mces = flat_mces[idx]
    samples_ranking = flat_ranking[idx]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Edit distance distribution
    axes[0].hist(samples_ed, bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Substructure Edit Distance")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Edit Distance Distribution")
    axes[0].grid(True, alpha=0.3)

    # MCES distribution
    axes[1].hist(samples_mces, bins=20, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("MCES Distance")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("MCES Distance Distribution")
    axes[1].grid(True, alpha=0.3)

    # Ranking distribution
    axes[2].hist(samples_ranking, bins=20, edgecolor="black", alpha=0.7)
    axes[2].set_xlabel("Ranking Score")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Ranking Score Distribution")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path + "distributions.png", dpi=300)
    plt.close()


def _save_query_plots(
    query_index: int,
    spectra_query,
    spectra_matches,
    best_matches_indices,
    top_rankings,
    top_ed_predictions,
    top_mces_predictions,
    ground_truth_mces,
    ground_truth_ed,
    output_path: str,
) -> None:
    """Save visualization plots for a specific query."""
    import matplotlib.pyplot as plt
    import spectrum_utils.plot as sup
    from rdkit import Chem

    query_dir = output_path + f"query_{query_index}/"
    import os

    os.makedirs(query_dir, exist_ok=True)

    # Save query molecule structure
    if "smiles" in spectra_query.params:
        mol_query = Chem.MolFromSmiles(spectra_query.params["smiles"])
        if mol_query is not None:
            from rdkit.Chem import Draw

            img = Draw.MolToImage(mol_query, size=(400, 400))
            img.save(query_dir + "query_molecule.png")

    # Save query spectrum
    fig, ax = plt.subplots(figsize=(10, 6))
    sup.spectrum(spectra_query, ax=ax)
    plt.title(f"Query Spectrum (Index: {query_index})")
    plt.tight_layout()
    plt.savefig(query_dir + "query_spectrum.png", dpi=300)
    plt.close()

    # Save top match molecules
    for i, match_spectrum in enumerate(spectra_matches[:5]):  # Save top 5
        if "smiles" in match_spectrum.params:
            mol_match = Chem.MolFromSmiles(match_spectrum.params["smiles"])
            if mol_match is not None:
                from rdkit.Chem import Draw

                img = Draw.MolToImage(mol_match, size=(400, 400))
                img.save(query_dir + f"match_{i + 1}_molecule.png")

        # Save mirror plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sup.mirror(spectra_query, match_spectrum, ax=ax)
        title = f"Match {i + 1} - Ranking: {top_rankings[i]:.3f}, "
        title += f"ED: {top_ed_predictions[i]:.1f}, MCES: {top_mces_predictions[i]:.1f}"
        if ground_truth_mces is not None:
            title += f"\nGT ED: {ground_truth_ed[i]:.1f}, GT MCES: {ground_truth_mces[i]:.1f}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(query_dir + f"match_{i + 1}_mirror.png", dpi=300)
        plt.close()


def _print_summary_statistics(
    results_summary: list, compute_ground_truth: bool
) -> None:
    """Print summary statistics of analog discovery results."""
    import numpy as np

    click.echo("\n" + "=" * 70)
    click.echo("SUMMARY STATISTICS")
    click.echo("=" * 70)

    # Average ranking scores
    all_rankings = []
    all_ed_predictions = []
    all_mces_predictions = []

    for result in results_summary:
        for match in result["top_matches"]:
            all_rankings.append(match["ranking_score"])
            all_ed_predictions.append(match["predicted_edit_distance"])
            all_mces_predictions.append(match["predicted_mces_distance"])

    click.echo(f"\nTotal queries processed: {len(results_summary)}")
    click.echo(f"Total matches found: {len(all_rankings)}")
    click.echo("\nRanking scores:")
    click.echo(f"  Mean: {np.mean(all_rankings):.4f}")
    click.echo(f"  Median: {np.median(all_rankings):.4f}")
    click.echo(f"  Min: {np.min(all_rankings):.4f}")
    click.echo(f"  Max: {np.max(all_rankings):.4f}")

    click.echo("\nPredicted Edit Distance:")
    click.echo(f"  Mean: {np.mean(all_ed_predictions):.2f}")
    click.echo(f"  Median: {np.median(all_ed_predictions):.2f}")

    click.echo("\nPredicted MCES Distance:")
    click.echo(f"  Mean: {np.mean(all_mces_predictions):.2f}")
    click.echo(f"  Median: {np.median(all_mces_predictions):.2f}")

    if compute_ground_truth:
        all_gt_mces = []
        all_gt_ed = []

        for result in results_summary:
            for match in result["top_matches"]:
                if "ground_truth_mces_distance" in match:
                    all_gt_mces.append(match["ground_truth_mces_distance"])
                    all_gt_ed.append(match["ground_truth_edit_distance"])

        if all_gt_mces:
            click.echo("\nGround Truth Edit Distance:")
            click.echo(f"  Mean: {np.mean(all_gt_ed):.2f}")
            click.echo(f"  Median: {np.median(all_gt_ed):.2f}")

            click.echo("\nGround Truth MCES Distance:")
            click.echo(f"  Mean: {np.mean(all_gt_mces):.2f}")
            click.echo(f"  Median: {np.median(all_gt_mces):.2f}")
