#!/bin/bash
# Local test runner for DISENT-KWS
# Usage: ./scripts/run_tests.sh [--coverage] [--lint]

set -e

echo "🧪 DISENT-KWS Test Suite"
echo "========================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv not found. Please install uv first.${NC}"
    exit 1
fi

# Run linting if requested
if [[ "$*" == *"--lint"* ]]; then
    echo -e "\n${YELLOW}Running flake8...${NC}"
    uv run flake8 data/ models/ training/ --max-line-length=120 --count || true
fi

# Run tests
echo -e "\n${YELLOW}Running pytest...${NC}"
if [[ "$*" == *"--coverage"* ]]; then
    uv run pytest tests/ -v --tb=short --cov=data --cov=models --cov=training --cov-report=html --cov-report=term
    echo -e "\n${GREEN}✅ Coverage report generated: htmlcov/index.html${NC}"
else
    uv run pytest tests/ -v --tb=short
fi

echo -e "\n${GREEN}✅ All tests passed!${NC}"
