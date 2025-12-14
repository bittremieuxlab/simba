# Contributing to SIMBA

First off, thank you for taking the time to contribute!

The following document provides guidelines for contributing to the documentation and the code of SIMBA. **No contribution is too small!** Even fixing a simple typo in the documentation is immensely helpful.

## Contributing to the Documentation

We appreciate improvements to our documentation! The README and documentation files are written in Markdown.

### Editing Documents

The easiest way to edit a document is by clicking the "Edit on GitHub" link. You'll be taken to GitHub where you can click on the pencil to edit the document.

You can then make your changes directly on GitHub. Once you're finished, fill in a description of what you changed and click the "Propose Changes" button.

Alternatively, these documents can be edited like code. See [Contributing to the Code](#contributing-to-the-code) below for more details on contributing this way.

## Contributing to the Code

We welcome contributions to the source code of SIMBA—particularly ones that address discussed [issues](https://github.com/bittremieuxlab/simba/issues).

Contributions to SIMBA follow a standard GitHub contribution workflow:

1. Create your own fork of the SIMBA repository on GitHub.

2. Clone your forked SIMBA repository to work on locally.

3. Create a new branch with a descriptive name for your changes:

```bash
git checkout -b fix_x
```

4. Make your changes (make sure to read below first).

5. Add, commit, and push your changes to your forked repository.

6. On the GitHub page for your forked repository, click "Pull request" to propose adding your changes to SIMBA.

7. We'll review, discuss, and help you make any revisions that are required. If all goes well, your changes will be added to SIMBA in the next release!

### Python Code Style

The SIMBA project follows the [PEP 8 guidelines](https://www.python.org/dev/peps/pep-0008/) for Python code style. More specifically, we use [Ruff](https://docs.astral.sh/ruff/) to format and lint Python code in SIMBA.

We highly recommend setting up a pre-commit hook for Ruff. This will run Ruff on all of the Python source files before the changes can be committed. Because we run Ruff for code linting as part of our tests, setting up this hook can save you from having to revise code formatting. Take the following steps to set up the pre-commit hook:

1. Verify that Ruff and pre-commit are installed. If not, you can install them with UV:

```bash
# Install with dev dependencies
uv sync --all-extras
```

2. Navigate to your local copy of the SIMBA repository and activate the hook:

```bash
pre-commit install
```

Once the hook is installed, Ruff will be run before any commit is made. If a file is changed by Ruff, then you need to `git add` the file again before finishing the commit.

### Running Tests

We use pytest for testing. Please write tests for any new features or bug fixes:

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=simba --cov-report=html --cov-report=term-missing
```

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/simba.git
   cd simba
   ```

2. **Set Up Environment with UV**
   ```bash
   # Install UV if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment and install dependencies
   uv sync --all-extras

   # Activate the environment
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install Pre-commit Hooks**
   ```bash
   uv run pre-commit install
   ```

## Questions?

For questions, issues, or feature requests, please [open an issue](https://github.com/bittremieuxlab/simba/issues) on GitHub.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
