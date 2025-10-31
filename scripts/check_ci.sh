#!/bin/bash
# Run all CI checks locally before pushing
# This script mimics the CI workflow to catch issues early

set -e  # Exit on first error

echo "=================================================="
echo "Running CI Checks Locally"
echo "=================================================="
echo ""

echo "→ [1/5] Type checking with mypy..."
mypy src/arc_prometheus tests/
echo "✅ Type checking passed"
echo ""

echo "→ [2/5] Linting with ruff..."
ruff check src/ tests/ scripts/
echo "✅ Linting passed"
echo ""

echo "→ [3/5] Checking code formatting..."
ruff format --check src/ tests/ scripts/
echo "✅ Formatting passed"
echo ""

echo "→ [4/5] Security checks with bandit..."
bandit -r src/ -ll
echo "✅ Security checks passed"
echo ""

echo "→ [5/5] Running tests..."
pytest tests/ -v --cov=src/arc_prometheus --cov-report=term-missing
echo "✅ All tests passed"
echo ""

echo "=================================================="
echo "✅ All CI checks passed! Safe to push."
echo "=================================================="
