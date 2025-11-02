.PHONY: test test-integration test-all test-cov lint typecheck format format-check ci clean help install check-uv

# Check if uv is installed
check-uv:
	@command -v uv > /dev/null 2>&1 || { echo "❌ Error: uv not found. Install with: pip install uv"; exit 1; }

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install project dependencies
	pip install -e ".[dev]"

test: check-uv  ## Run tests (excluding integration tests)
	uv run pytest tests/ -v -m "not integration"

test-integration: check-uv  ## Run integration tests only (requires GEMINI_API_KEY)
	uv run pytest tests/ -v -m "integration"

test-all: check-uv  ## Run all tests including integration tests
	uv run pytest tests/ -v

test-cov: check-uv  ## Run tests with coverage report (excluding integration tests)
	uv run pytest tests/ -m "not integration" --cov=src/arc_prometheus --cov-report=term-missing --cov-report=html

lint: check-uv  ## Run linter (ruff check)
	uv run ruff check src/ tests/ scripts/

lint-fix: check-uv  ## Run linter with auto-fix
	uv run ruff check --fix src/ tests/ scripts/

format: check-uv  ## Format code with ruff
	uv run ruff format src/ tests/ scripts/

format-check: check-uv  ## Check code formatting without changes
	uv run ruff format --check src/ tests/ scripts/

typecheck: check-uv  ## Run type checker (mypy)
	uv run mypy src/arc_prometheus

security: check-uv  ## Run security checks (bandit)
	uv run bandit -r src/ -ll

ci:  ## Run all CI checks (typecheck, lint, format-check, security, test)
	@echo "=== Running Type Checks ==="
	@$(MAKE) typecheck
	@echo "\n=== Running Linter ==="
	@$(MAKE) lint
	@echo "\n=== Checking Code Format ==="
	@$(MAKE) format-check
	@echo "\n=== Running Security Checks ==="
	@$(MAKE) security
	@echo "\n=== Running Tests ==="
	@$(MAKE) test
	@echo "\n✓ All CI checks passed!"

clean:  ## Clean up cache files
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
