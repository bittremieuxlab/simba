"""Integration tests for SIMBA CLI commands.

These are smoke tests to ensure CLI commands can be invoked without errors.
They do NOT test actual training functionality.
"""

import shutil
import subprocess

import pytest
from click.testing import CliRunner

from simba.cli import cli


class TestCLI:
    """Test suite for SIMBA CLI commands using Click's test runner."""

    def test_cli_help(self):
        """Test that the CLI shows help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output or "simba" in result.output.lower()

    def test_preprocess_command_help(self):
        """Test that the preprocess command shows help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess", "--help"])
        assert result.exit_code == 0
        assert "preprocess" in result.output.lower()
        assert "--spectra-path" in result.output
        assert "--workspace" in result.output
        assert "--mapping-file-name" in result.output

    def test_preprocess_command_missing_required_args(self):
        """Test that preprocess command fails without required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess"])
        assert result.exit_code != 0
        assert "Error" in result.output or "Missing" in result.output

    def test_preprocess_command_invalid_spectra_path(self):
        """Test that preprocess command fails with invalid spectra path."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "preprocess",
                "--spectra-path",
                "/nonexistent/spectra.mgf",
                "--workspace",
                "/tmp/test_workspace",
            ],
        )
        assert result.exit_code != 0

    @pytest.mark.parametrize(
        "option",
        [
            "--max-spectra-train",
            "--max-spectra-val",
            "--max-spectra-test",
            "--num-workers",
            "--val-split",
            "--test-split",
            "--overwrite",
        ],
    )
    def test_preprocess_command_has_expected_options(self, option):
        """Test that preprocess command has expected options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess", "--help"])
        assert result.exit_code == 0
        assert option in result.output

    def test_train_command_help(self):
        """Test that the train command shows help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "train" in result.output.lower()
        assert "--checkpoint-dir" in result.output
        assert "--preprocessing-dir" in result.output
        assert "--preprocessing-pickle" in result.output

    def test_train_command_missing_required_args(self):
        """Test that train command fails without required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train"])
        assert result.exit_code != 0
        assert "Error" in result.output or "Missing" in result.output

    def test_train_command_invalid_preprocessing_dir(self):
        """Test that train command fails with invalid preprocessing directory."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "train",
                "--checkpoint-dir",
                "/tmp/test_checkpoints",
                "--preprocessing-dir",
                "/nonexistent/directory",
                "--preprocessing-pickle",
                "mapping.pkl",
            ],
        )
        assert result.exit_code != 0
        # Click should complain about the path not existing

    @pytest.mark.parametrize(
        "option",
        [
            "--epochs",
            "--batch-size",
            "--learning-rate",
            "--num-workers",
            "--accelerator",
            "--val-check-interval",
        ],
    )
    def test_train_command_has_expected_options(self, option):
        """Test that train command has expected options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert option in result.output


class TestCLIEntryPoint:
    """Test suite for CLI entry point (simba command).

    These tests verify that the 'simba' command is properly installed
    and accessible from PATH. They will FAIL if the package is not
    installed in editable mode (uv sync / pip install -e .).
    """

    def test_cli_entry_point_installed(self):
        """Test that simba CLI entry point is installed."""
        # Try to find simba command in PATH
        simba_path = shutil.which("simba")
        assert simba_path is not None, (
            "simba command not found in PATH. "
            "Run 'uv sync' or 'pip install -e .' to install the CLI entry point."
        )

    def test_cli_entry_point_help(self):
        """Test that simba CLI entry point shows help."""
        try:
            result = subprocess.run(
                ["simba", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            assert "Usage" in result.stdout or "simba" in result.stdout.lower()
        except subprocess.TimeoutExpired:
            pytest.fail("simba --help command timed out after 10 seconds")

    def test_cli_preprocess_entry_point_help(self):
        """Test that simba preprocess entry point shows help."""
        try:
            result = subprocess.run(
                ["simba", "preprocess", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            assert "preprocess" in result.stdout.lower()
        except subprocess.TimeoutExpired:
            pytest.fail("simba preprocess --help command timed out after 10 seconds")

    def test_cli_train_entry_point_help(self):
        """Test that simba train entry point shows help."""
        try:
            result = subprocess.run(
                ["simba", "train", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            assert "train" in result.stdout.lower()
        except subprocess.TimeoutExpired:
            pytest.fail("simba train --help command timed out after 10 seconds")
