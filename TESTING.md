# Testing Guide for DISENT-KWS

## Overview

This project uses **pytest** for unit testing, **GitHub Actions** for CI-CD, and **pre-commit hooks** for local code quality enforcement.

All dependencies are managed via **`uv`** (fast Python package installer). Make sure `uv` is installed before running tests.

---

## Quick Start

### Run All Tests Locally

```bash
# Option 1: Using make
make test

# Option 2: Using uv directly
uv run pytest tests/ -v

# Option 3: Using the Python runner
python scripts/test.py
```

### Run Tests with Coverage

```bash
# Using make
make test-cov

# Using uv directly
uv run pytest tests/ -v --cov=data --cov=models --cov=training --cov-report=html

# Using the Python runner
python scripts/test.py --coverage
```

### Run Specific Tests

```bash
# Run only dataloader tests
make test-dataloaders
# or
uv run pytest tests/test_dataloaders.py -v

# Run specific test by name
uv run pytest tests/test_dataloaders.py::TestLFBETransform::test_transform_1d_input -v

# Run tests matching a pattern
uv run pytest tests/ -k "lfbe" -v
```

---

## Test Structure

```
tests/
├── __init__.py
├── test_dataloaders.py          # Data loader tests
└── test_models.py               # Model tests

scripts/
├── run_tests.sh                 # Shell script for local testing
└── test.py                      # Python runner with detailed output
```

### Test Files

#### `tests/test_dataloaders.py`
Tests for data loading and feature extraction:
- `TestLFBETransform`: Log-mel filterbank feature extraction
- `TestGSCDataset`: Google Speech Commands dataset
- `TestVoxCelebDataset`: VoxCeleb speaker dataset
- `TestLibriPhraseDataset`: LibriPhrase triplet dataset
- `TestDataLoadersIntegration`: Integration tests

---

## Running Tests Locally

### Prerequisites

```bash
# Install dependencies using uv
uv sync --all-extras
```

### Available Commands

#### Using Makefile

```bash
make help              # Show all available commands
make test              # Run all tests
make test-cov          # Run with coverage report
make test-v            # Verbose output
make test-dataloaders  # Run only dataloader tests
make lint              # Run flake8
make format            # Auto-format code
make install           # Sync dependencies with uv
make pre-commit-install # Setup pre-commit hooks
make clean             # Remove build artifacts
```

#### Using uv directly

```bash
uv run pytest tests/ -v                              # Run all tests
uv run pytest tests/test_dataloaders.py -v           # Run specific test file
uv run pytest tests/ -k test_lfbe -v                 # Run tests matching pattern
uv run pytest tests/ -m "not slow" -v                # Skip slow tests
uv run pytest tests/ -v --tb=short                   # Short traceback format
uv run pytest tests/ -v --tb=long                    # Long traceback format
```

#### Using Python scripts

```bash
python scripts/test.py                        # Basic tests
python scripts/test.py --coverage             # With coverage report
python scripts/test.py --lint                 # Include linting
python scripts/test.py --fast                 # Fast tests only
```

#### Using shell script

```bash
bash scripts/run_tests.sh                     # Basic tests
bash scripts/run_tests.sh --coverage          # With coverage
bash scripts/run_tests.sh --lint              # Include linting
```

---

## GitHub Actions CI-CD

### Workflow: `.github/workflows/test.yml`

Runs automatically on:
- **Push** to `main` or `develop` branches
- **Pull requests** targeting `main` or `develop`

Tests on:
- Python 3.10, 3.11, 3.12
- Ubuntu Linux

Steps:
1. **Linting**: flake8 checks
2. **Testing**: pytest with coverage
3. **Coverage Upload**: to Codecov

### View Results

1. Go to your GitHub repository
2. Click **Actions** tab
3. Click the workflow run to see detailed logs

### Local CI Simulation

To test the same workflow locally:

```bash
# Run linting + tests + coverage (simulates CI)
python scripts/test.py --lint --coverage
```

---

## Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit.

### Setup

```bash
# Install pre-commit hooks
make pre-commit-install
# or
uv run pre-commit install
```

### What Runs

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Unit tests

### Bypass Hooks (if needed)

```bash
git commit --no-verify
```

---

## Writing New Tests

### Structure

```python
import pytest
import torch

class TestMyFeature:
    """Test suite for my feature."""
    
    @pytest.fixture
    def setup_data(self):
        """Fixture for test setup."""
        return torch.randn(10, 80, 200)
    
    def test_my_feature_basic(self, setup_data):
        """Test basic functionality."""
        assert setup_data.shape == (10, 80, 200)
    
    @pytest.mark.parametrize("param", [1, 2, 3])
    def test_my_feature_parametrized(self, param):
        """Parametrized test."""
        assert param > 0
```

### Fixtures

Common fixtures (add to `conftest.py` if needed):

```python
@pytest.fixture
def lfbe_transform():
    """LFBETransform instance."""
    from data.datasets import LFBETransform
    return LFBETransform()

@pytest.fixture
def dummy_audio():
    """Dummy 2-second audio."""
    return torch.randn(1, 32000)
```

### Markers

```python
@pytest.mark.slow
def test_slow_operation():
    """Skip with: pytest -m 'not slow'"""
    pass

@pytest.mark.gpu
def test_on_gpu():
    """Run only on GPU machines."""
    pass
```

---

## Coverage Reports

### Generate Coverage Report

```bash
make test-cov
```

### View HTML Report

```bash
# Generate
pytest tests/ --cov=data --cov=models --cov=training --cov-report=html

# View in browser
open htmlcov/index.html          # macOS
xdg-open htmlcov/index.html      # Linux
start htmlcov/index.html         # Windows
```

---

## Troubleshooting

### Tests Fail with "ModuleNotFoundError"

```bash
# Install the package in editable mode
pip install -e .
```

### "torch" not found

```bash
# Install PyTorch
pip install torch torchaudio
```

### Pre-commit hooks fail

```bash
# Format and fix issues
make format

# Then commit again
git add .
git commit -m "Auto-formatted"
```

### Coverage report is low

```bash
# Check which files are missing coverage
pytest tests/ --cov=data --cov-report=term-missing
```

---

## Continuous Integration Best Practices

1. **Run locally before pushing**: `make test`
2. **Add tests for new features**: Coverage > 80%
3. **Use descriptive test names**: `test_feature_with_specific_condition`
4. **Test edge cases**: Empty inputs, very large inputs, etc.
5. **Use fixtures for setup**: Keep tests DRY
6. **Mark slow tests**: Use `@pytest.mark.slow`

---

## Next Steps

- Add model tests (`tests/test_models.py`)
- Add integration tests (`tests/test_integration.py`)
- Add performance benchmarks in `tests/test_benchmarks.py`

---

For questions, see `pytest --help` or visit [pytest docs](https://docs.pytest.org/).
