.PHONY: help test test-cov lint format install pre-commit-install clean

help:
	@echo "DISENT-KWS Development Commands"
	@echo "================================"
	@echo "make test              - Run all tests"
	@echo "make test-cov          - Run tests with coverage report"
	@echo "make test-v            - Run tests with verbose output"
	@echo "make test-specific     - Run specific test (e.g., make test-specific TEST=test_lfbe_transform)"
	@echo "make lint              - Run flake8 linting"
	@echo "make format            - Auto-format code with black"
	@echo "make install           - Install dependencies"
	@echo "make pre-commit-install - Install pre-commit hooks"
	@echo "make clean             - Remove build artifacts"
	@echo "make clean-test        - Remove test artifacts"

test:
	uv run pytest tests/ -v --tb=short

test-cov:
	uv run pytest tests/ -v --tb=short --cov=data --cov=models --cov=training --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

test-v:
	uv run pytest tests/ -vv --tb=long

test-specific:
	uv run pytest tests/ -k $(TEST) -v --tb=short

test-dataloaders:
	uv run pytest tests/test_dataloaders.py -v --tb=short

lint:
	uv run flake8 data/ models/ training/ --max-line-length=120 --count

format:
	uv run black data/ models/ training/ tests/
	uv run isort data/ models/ training/ tests/

install:
	uv sync --all-extras

pre-commit-install:
	uv run pre-commit install
	@echo "Pre-commit hooks installed!"

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf dist/ build/

clean-test:
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
