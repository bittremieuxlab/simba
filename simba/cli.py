"""SIMBA Command Line Interface."""

import click

from simba.commands.train import train


@click.group()
@click.version_option(package_name="simba")
def cli():
    """SIMBA: Spectral Identification of Molecule Bio-Analogues.

    A transformer-based neural network for predicting chemical structural
    similarity from tandem mass spectrometry (MS/MS) spectra.
    """
    pass


# Register commands
cli.add_command(train)


if __name__ == "__main__":
    cli()
