"""Shared pytest configuration and fixtures for SIMBA tests."""

import tempfile
from pathlib import Path

import pytest

from simba.config import Config


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


@pytest.fixture
def mini_training_config(tmp_path):
    """Create a minimal config for training tests."""
    config = Config()
    config.PREPROCESSING_DIR = str(tmp_path) + "/"
    config.CHECKPOINT_DIR = str(tmp_path / "checkpoints") + "/"
    config.SPECTRA_PATH = str(tmp_path / "spectra.mgf")
    config.MOL_SPEC_MAPPING_FILE = "mapping_test.pkl"
    config.PREPROCESSING_OVERWRITE = True
    config.PREPROCESSING_NUM_WORKERS = 1
    config.PREPROCESSING_NUM_NODES = 1
    config.PREPROCESSING_CURRENT_NODE = 0
    config.MAX_SPECTRA_TRAIN = 10
    config.MAX_SPECTRA_VAL = 2
    config.MAX_SPECTRA_TEST = 2
    config.VAL_SPLIT = 0.2
    config.TEST_SPLIT = 0.2
    config.RANDOM_MCES_SAMPLING = True
    config.USE_ONLY_PROTONIZED_ADDUCTS = False
    return config


@pytest.fixture
def sample_training_spectra(tmp_path):
    """Create a small MGF file with SMILES for training tests."""
    mgf_path = tmp_path / "spectra.mgf"

    spectra_data = []
    for i in range(14):
        if i % 2 == 0:
            smiles = "CCO"
            formula = "C2H6O"
            inchikey = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
            mz_values = [45.034, 46.041, 47.049, 48.052, 49.055, 50.058]
            intensities = [100.0, 50.0, 25.0, 20.0, 15.0, 10.0]
        else:
            smiles = "CC(C)O"
            formula = "C3H8O"
            inchikey = "KFZMGEQAYNKOFK-UHFFFAOYSA-N"
            mz_values = [43.018, 45.034, 60.058, 61.062, 62.065, 63.068]
            intensities = [80.0, 60.0, 100.0, 40.0, 30.0, 20.0]

        spectrum = f"""BEGIN IONS
PEPMASS=100.0
RT=100.0
IONMODE=Positive
ADDUCT=[M+H]+
CHARGE=1
ID={i}
SMILES={smiles}
FORMULA={formula}
INCHIKEY={inchikey}
SCAN NUMBER={i}
NUM_PEAKS={len(mz_values)}
{mz_values[0]} {intensities[0]}
{mz_values[1]} {intensities[1]}
{mz_values[2]} {intensities[2]}
{mz_values[3]} {intensities[3]}
{mz_values[4]} {intensities[4]}
{mz_values[5]} {intensities[5]}
END IONS

"""
        spectra_data.append(spectrum)

    mgf_path.write_text("".join(spectra_data))
    return mgf_path


@pytest.fixture
def mock_model(mocker):
    """Create a mocked SIMBA model with random weights for testing."""
    from simba.ordinal_classification.embedder_multitask import EmbedderMultitask

    config = Config()

    model = EmbedderMultitask(
        d_model=int(config.D_MODEL),
        n_layers=int(config.N_LAYERS),
        n_classes=config.EDIT_DISTANCE_N_CLASSES,
        use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
        use_element_wise=True,
        use_cosine_distance=config.use_cosine_distance,
        use_edit_distance_regresion=config.USE_EDIT_DISTANCE_REGRESSION,
        use_fingerprints=config.USE_FINGERPRINT,
        USE_LEARNABLE_MULTITASK=config.USE_LEARNABLE_MULTITASK,
    )
    model.eval()

    mocker.patch(
        "simba.ordinal_classification.embedder_multitask.EmbedderMultitask.load_from_checkpoint",
        return_value=model,
    )

    return model
