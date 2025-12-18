# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Changed

- **Major refactoring:** Restructured entire codebase into logical modules
  - Organized code into `core/`, `legacy/`, and `tools/` directories
  - Created clear module hierarchy: `core/models/`, `core/data/`, `core/chemistry/`
  - Fixed `simba/simba/` nested folder structure
  - Moved all legacy training scripts to `legacy/` with documentation
  - All imports updated and validated
- **Code quality improvements:** Applied Ruff formatting to entire `simba/` package
  - Added ignore rules for acceptable patterns (B020, B905, N812, N999)
  - Removed `simba/` from Ruff exclusions - now enforcing consistent code style
  - All code follows PEP 8 and modern Python best practices

[unreleased]: https://github.com/bittremieuxlab/simba/compare/HEAD
