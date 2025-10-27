.PHONY: test lint typecheck format format-check ci clean help install

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install project dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage report
	pytest tests/ --cov=src/arc_prometheus --cov-report=term-missing --cov-report=html

lint:  ## Run linter (ruff check)
	ruff check src/ tests/ scripts/

lint-fix:  ## Run linter with auto-fix
	ruff check --fix src/ tests/ scripts/

format:  ## Format code with ruff
	ruff format src/ tests/ scripts/

format-check:  ## Check code formatting without changes
	ruff format --check src/ tests/ scripts/

typecheck:  ## Run type checker (mypy)
	mypy src/arc_prometheus tests/

security:  ## Run security checks (bandit)
	bandit -r src/ -ll

ci:  ## Run all CI checks (typecheck, lint, format-check, test)
	@echo "=== Running Type Checks ==="
	@$(MAKE) typecheck
	@echo "\n=== Running Linter ==="
	@$(MAKE) lint
	@echo "\n=== Checking Code Format ==="
	@$(MAKE) format-check
	@echo "\n=== Running Tests ==="
	@$(MAKE) test
	@echo "\n✓ All CI checks passed!"

clean:  ## Clean up cache files
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
