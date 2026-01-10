# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-10

### Added

- PyPI package publication ready: `pip install simba-ms`
- Initial release of SIMBA
- CLI commands: `simba train`, `simba inference`, `simba preprocess`, `simba analog-discovery`
- Transformer-based neural network for MS/MS structural similarity prediction
- Prediction of substructure edit distance and MCES distance
- Pretrained model trained on MassSpecGym dataset
- Support for MGF spectral format
- UV package manager integration
- Comprehensive test suite with pytest
- CI/CD pipelines with GitHub Actions (tests and code quality checks)
- Pre-commit hooks for code quality (Ruff formatting and linting)
- Community files: CONTRIBUTING.md, CODE_OF_CONDUCT.md, CHANGELOG.md
- Version management infrastructure (`__version__.py`)
- Documentation infrastructure with Sphinx and ReadTheDocs configuration
- Hydra configuration for managing complex training, inference, and preprocessing workflows

### Changed

- **Major refactoring:** Restructured entire codebase into logical modules
  - Organized code into `core/`, `legacy/`, and `tools/` directories
  - Created clear module hierarchy: `core/models/`, `core/data/`, `core/chemistry/`, `core/training/`
  - Fixed `simba/simba/` nested folder structure
  - Moved all legacy training scripts to `legacy/` with documentation
  - All imports updated and validated
- **Code quality improvements:** Applied Ruff formatting to entire `simba/` package
  - Added ignore rules for acceptable patterns (B020, B905, N812, N999)
  - Removed `simba/` from Ruff exclusions - now enforcing consistent code style
  - All code follows PEP 8 and modern Python best practices
  - CLI commands in `simba/commands/` now use workflow functions from `simba/workflows/`
  - Clean separation of concerns: CLI handles argument parsing, workflows handle business logic

[unreleased]: https://github.com/bittremieuxlab/simba/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bittremieuxlab/simba/releases/tag/v0.1.0
