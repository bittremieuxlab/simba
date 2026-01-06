# Installation

## Requirements

- Python 3.11.x (tested with 3.11.7)
- [UV](https://docs.astral.sh/uv/) (recommended) or [Conda](https://docs.conda.io/en/latest/)

## Option 1: UV (Recommended - Fastest ⚡)

### Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup SIMBA

Installation takes approximately 2-5 minutes:

```bash
# Clone the repository
git clone https://github.com/bittremieux-lab/simba.git
cd simba

# Create virtual environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

### For Jupyter notebooks

```bash
# Install notebook dependencies
uv sync --extra notebooks

# Register the kernel
python -m ipykernel install --user --name=simba --display-name="SIMBA (UV)"
```

**To use notebooks in VS Code:**

1. Open any `.ipynb` file in the `notebooks/` folder
2. Click "Select Kernel" in the top-right corner
3. Choose "SIMBA (UV)" or "Python 3.11 (.venv: venv)"
4. If the kernel doesn't appear, reload VS Code window (Cmd+Shift+P → "Developer: Reload Window")

## Option 2: Conda (Alternative)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate simba

# Install the module
pip install -e .
```

**Note for macOS users:**

```bash
brew install xz
```

## Verify Installation

```bash
simba --version
```
