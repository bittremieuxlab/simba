"""Select the checkpoint of a training run from the metrics of ValidationMetricsCallback.

Checkpoints are ranked by val Spearman r of the predicted MCES distance (higher is
better) and by val balanced MAE (lower is better); the checkpoint with the lowest
mean of the two ranks is selected (ties: the earlier step).

Usage:
  python tools/select_checkpoint.py <CHECKPOINT_DIR>/val_metrics/val_metrics.tsv [--top 10]
"""

import argparse
from pathlib import Path

import pandas as pd

RANKED = {"val_spearman_mces": False, "val_mae_mces_balanced": True}  # column -> ascending


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics_tsv")
    parser.add_argument("--top", type=int, default=10, help="rows to print")
    args = parser.parse_args()

    table = pd.read_csv(args.metrics_tsv, sep="\t")
    table["step"] = table["step"].astype(int)
    for column, ascending in RANKED.items():
        table[f"rank_{column}"] = table[column].rank(ascending=ascending)
    table["mean_rank"] = table[[f"rank_{c}" for c in RANKED]].mean(axis=1)
    table = table.sort_values(["mean_rank", "step"], kind="stable")

    columns = ["step", "mean_rank", *RANKED, "val_edit_distance_accuracy", "checkpoint"]
    print(table[columns].head(args.top).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = table.iloc[0]
    print(f"\nSelected: step {int(best['step'])},{Path(args.metrics_tsv).parent / best['checkpoint']}")


if __name__ == "__main__":
    main()
