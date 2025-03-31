# Contributing to ERCOT Price Forecasting

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Development Environment Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ercot_price_forecasting.git
   cd ercot_price_forecasting
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```
4. Install the development dependencies:
   ```bash
   pip install -e ".[dev,notebook]"
   ```
5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Quality Tools

This project uses several code quality tools to maintain high standards. These are automatically run through pre-commit hooks when you make a commit.

### Pre-commit Hooks

The following checks are run on commit:

- **trailing-whitespace**: Trims trailing whitespace
- **end-of-file-fixer**: Ensures files end with a newline
- **check-yaml**: Validates YAML files
- **check-added-large-files**: Prevents large files from being committed
- **debug-statements**: Checks for debugger imports and py37+ `breakpoint()` calls
- **check-merge-conflict**: Checks for merge conflict strings
- **isort**: Sorts imports
- **black**: Formats Python code
- **ruff**: Lints Python code
- **mypy**: Checks type annotations
- **nbqa**: Applies code quality tools to Jupyter notebooks
- **nbstripout**: Removes Jupyter notebook outputs

### Running Checks Manually

You can run all pre-commit checks on all files with:

```bash
pre-commit run --all-files
```

Or run a specific check:

```bash
pre-commit run black --all-files
```

## Pull Request Process

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes. Be sure to:
   - Add or update tests as necessary
   - Update documentation as needed
   - Follow the existing code style

3. Commit your changes:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```
   Note: The pre-commit hooks will automatically run and may modify files to meet style guidelines.

4. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a Pull Request from your fork to the main repository

## Testing

We use pytest for testing. Make sure your changes don't break existing functionality:

```bash
pytest
```

For coverage information:

```bash
pytest --cov=src
```

## Documentation

- Update the README.md if you change user-facing functionality
- Add docstrings to new functions, classes, and methods
- Follow Google-style docstrings

## Type Annotations

All new code should include proper type annotations. We use mypy to check types:

```bash
mypy src
```

## Continuous Integration

The project has GitHub Actions workflows that will automatically run tests and quality checks on your pull requests.

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 