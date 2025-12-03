"""Shared pytest configuration and fixtures for SIMBA tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_mgf(fixtures_dir):
    """Path to sample MGF file (standard format)."""
    return str(fixtures_dir / "sample_spectra.mgf")


@pytest.fixture
def sample_mgf_casmi(fixtures_dir):
    """Path to sample MGF file (CASMI2022 format with SMILES)."""
    return str(fixtures_dir / "sample_spectra_casmi.mgf")
