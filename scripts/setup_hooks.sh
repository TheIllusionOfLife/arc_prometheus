#!/bin/bash
# Setup git hooks for development
# This script installs both pre-commit and pre-push hooks

set -e

echo "🔧 Setting up git hooks..."
echo ""

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "❌ Error: Not in a git repository root"
    echo "   Run this script from the project root directory"
    exit 1
fi

# Install pre-commit hooks (ruff, mypy, bandit, etc.)
echo "→ Installing pre-commit hooks..."
if command -v uv &> /dev/null; then
    uv run pre-commit install
    echo "✅ Pre-commit hooks installed"
elif command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  pre-commit not found. Install with: pip install pre-commit"
    exit 1
fi

# Install pre-push hooks (runs full CI suite)
echo ""
echo "→ Installing pre-push hooks..."
if [ -f .github/hooks/pre-push ]; then
    # Create hooks directory if it doesn't exist
    mkdir -p .git/hooks

    # Copy pre-push hook
    cp .github/hooks/pre-push .git/hooks/pre-push
    chmod +x .git/hooks/pre-push
    echo "✅ Pre-push hooks installed"
else
    echo "❌ Error: .github/hooks/pre-push not found"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ Git hooks setup complete!"
echo "=================================================="
echo ""
echo "Hooks installed:"
echo "  • Pre-commit: Fast checks (ruff, mypy, bandit)"
echo "  • Pre-push: Full CI suite (all checks + tests)"
echo ""
echo "To test the hooks:"
echo "  make ci          # Run all checks manually"
echo "  git commit       # Will run pre-commit hooks"
echo "  git push         # Will run pre-push hooks"
echo ""
echo "To skip hooks (not recommended):"
echo "  git commit --no-verify"
echo "  git push --no-verify"
echo ""
