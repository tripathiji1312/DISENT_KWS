#!/usr/bin/env python3
"""
Local test runner with detailed output.
Usage: python scripts/test.py [--coverage] [--lint]
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, name, silent=False):
    """Run command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔨 {name}")
    print(f"{'='*60}")
    
    try:
        if silent:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(result.stdout)
                print(result.stderr)
        else:
            result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print(f"✅ {name} passed")
            return True
        else:
            print(f"❌ {name} failed")
            return False
    except FileNotFoundError:
        print(f"⚠️  {name} skipped (command not found)")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run DISENT-KWS tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))

    # Check if uv is available
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Error: uv not found. Please install uv first.")
        sys.exit(1)

    results = {}

    # Linting
    if args.lint:
        lint_cmd = ["uv", "run", "flake8", "data/", "models/", "training/", "--max-line-length=120", "--count"]
        results["flake8"] = run_command(lint_cmd, "Linting (flake8)", silent=True)

    # Tests
    test_cmd = ["uv", "run", "pytest", "tests/", "-v", "--tb=short"]
    if args.coverage:
        test_cmd.extend(["--cov=data", "--cov=models", "--cov=training", "--cov-report=html"])
    if args.fast:
        test_cmd.extend(["-m", "not slow"])
    
    results["pytest"] = run_command(test_cmd, "Running tests")

    # Summary
    print(f"\n{'='*60}")
    print("📊 Summary")
    print(f"{'='*60}")
    for name, status in results.items():
        if status is None:
            print(f"⚠️  {name}: SKIPPED")
        elif status:
            print(f"✅ {name}: PASSED")
        else:
            print(f"❌ {name}: FAILED")

    if args.coverage:
        print("\n📈 Coverage report: htmlcov/index.html")

    # Exit with failure if any test failed
    if any(v is False for v in results.values()):
        sys.exit(1)

    print("\n✨ All tests passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
