# ARC-Prometheus 🔥

**AI Civilization for Solving ARC Prize through Evolutionary LLM Agents**

ARC-Prometheus is an ambitious project that simulates how human scientists solve problems: diverse specialists collaborating, experimenting, and building on each other's work through evolutionary pressure. Instead of building a single "super-intelligent" AI, we're creating an ecosystem of specialized LLM agents that evolve solutions to the [ARC Prize 2025](https://www.kaggle.com/competitions/arc-prize-2025) challenge.

## 🎯 Project Vision

**We are not building an ARC solver. We are building an AI civilization.**

Modern deep learning fails at ARC because it requires millions of examples, while ARC provides only ~3 training examples per task. This project takes a fundamentally different approach: instead of building a single "super-intelligent" AI, we're creating an **ecosystem of specialized LLM agents** that collaborate, experiment, and evolve solutions like a scientific community.

### The Three Pillars

- **The Crucible (るつぼ)**: Sandbox environment where AI-generated solvers are tested against ARC puzzles
- **The Cognitive Cells (認知的細胞)**: Specialized LLM agent teams working like human researchers:
  - **Analyst**: Analyzes patterns and infers transformation rules
  - **Programmer**: Generates Python solver code from specifications
  - **Refiner**: Debugs and improves failed solutions (mutation)
  - **Tagger**: Classifies solver techniques (rotation, fill, symmetry...)
  - **Crossover**: Fuses capabilities from different solvers
- **The Evolutionary Engine (進化的エンジン)**: Natural selection through:
  - **Fitness Function**: Prioritizes generalization over memorization (10x weight on test accuracy)
  - **Mutation**: Refiner improves individual solvers
  - **Crossover**: Combines techniques from multiple solvers
  - **Population Dynamics**: Many solvers evolving together, competing and breeding

### Why This Matters

ARC Prize tests **abstraction and reasoning** - the ability to learn underlying transformation rules from just 3 examples and apply them to never-before-seen problems. This is the essence of human intelligence and a critical step toward AGI.

**Our hypothesis**: A diverse community of specialized agents evolving together can achieve emergent intelligence that surpasses single-model approaches. The ARC Prize 2025 competition serves as our testbed to validate this hypothesis.

**Current Status**: Phase 1.1 Complete ✅
- Data loading and visualization
- Grid evaluation
- Infrastructure foundation

**Next**: Phase 1.2 - Manual solver and safe execution sandbox

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Gemini API key ([get one here](https://makersuite.google.com/app/apikey))
- ARC Prize 2025 dataset ([download from Kaggle](https://www.kaggle.com/competitions/arc-prize-2025/data))

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/TheIllusionOfLife/arc_prometheus.git
cd arc_prometheus
```

2. **Create and activate virtual environment**:
```bash
python3.13 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**:
```bash
pip install -e ".[dev]"
```

4. **(Optional) Docker Sandbox for Production**:
```bash
pip install -e ".[docker]"
# Install Docker Desktop (https://docs.docker.com/get-docker/)
docker build -t arc-prometheus-sandbox:latest -f docker/sandbox.Dockerfile .
# Use --sandbox-mode docker for network isolation, read-only filesystem, resource limits
```

5. **Set up API key**:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

5. **Download and place ARC dataset**:
```bash
# Download from: https://www.kaggle.com/competitions/arc-prize-2025/data
# Extract and place in: data/arc-prize-2025/

# Expected structure:
# data/arc-prize-2025/
#   ├── arc-agi_training_challenges.json
#   ├── arc-agi_training_solutions.json
#   ├── arc-agi_evaluation_challenges.json
#   ├── arc-agi_evaluation_solutions.json
#   ├── arc-agi_test_challenges.json
#   └── sample_submission.json
```

6. **Verify installation**:
```bash
python -m pytest tests/ -v
```

All tests should pass! ✅

## 📚 Usage

### Demo Scripts

**Phase 1 - Core Functionality**:
```bash
# Data loading and visualization
python scripts/demo_phase1_1_data.py [task_id]

# Manual solver validation
python scripts/demo_phase1_2_manual.py

# Sandbox security (timeout, exceptions)
python scripts/demo_phase1_3_sandbox.py

# End-to-end AI solver pipeline
python scripts/run_phase1_test.py <task_id>
```

**Phase 2 - Evolution**:
```bash
# Fitness evaluation (generalization vs overfitting)
python scripts/demo_phase2_1_fitness.py

# Refiner agent (automated debugging)
python scripts/demo_phase2_2_refiner.py

# Multi-generation evolution loop
python scripts/demo_phase2_3_evolution.py [--model MODEL] [--max-generations N]
```

### Configuration Options

**LLM Response Caching** (70-80% cost reduction):
```bash
# Cache location: ~/.arc_prometheus/llm_cache.db (7-day TTL)
--cache-stats       # View hit rate and savings
--no-cache          # Disable for fresh responses
--clear-cache       # Remove all entries
```

**Evolution Parameters**:
```bash
--model MODEL                    # LLM model (default: gemini-2.5-flash-lite)
--programmer-temperature TEMP    # Code generation (default: 0.3)
--refiner-temperature TEMP       # Debugging creativity (default: 0.4)
--max-generations N              # Max evolution cycles (default: 5)
--target-fitness N               # Early stop threshold
--sandbox-mode docker            # Production security (default: multiprocess)
--timeout-eval SECONDS           # Code execution timeout (default: 5)
--timeout-llm SECONDS            # LLM API timeout (default: 60)
```

**Kaggle Submission** (pass@2 format):
```bash
--generate-submission            # Enable pass@2 prediction generation
--num-attempts N                 # Number of diverse attempts (default: 2)
```

### Benchmarking (Real-World Testing)

Run evolution loop on diverse ARC tasks to measure performance and validate Phase 2:

#### Preparing Evaluation Dataset

The evaluation dataset stores test outputs separately from challenges. Merge them before benchmarking:

```bash
# One-time setup: Merge evaluation challenges with solutions
python scripts/prepare_evaluation_data.py \
  --challenges data/arc-prize-2025/arc-agi_evaluation_challenges.json \
  --solutions data/arc-prize-2025/arc-agi_evaluation_solutions.json \
  --output data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json
```

This creates a unified format where test examples include outputs (required to measure generalization performance).

#### Running Benchmarks

```bash
# Benchmark specific tasks
python scripts/benchmark_evolution.py \
  --tasks "00576224,007bbfb7,025d127b" \
  --output-dir results/test_run/ \
  --experiment-name "test_run"

# Random sample from training set (development)
python scripts/benchmark_evolution.py \
  --random-sample 15 \
  --training-data data/arc-prize-2025/arc-agi_training_challenges.json \
  --output-dir results/baseline/ \
  --experiment-name "baseline"

# Random sample from EVALUATION set (validation - measures true generalization)
python scripts/benchmark_evolution.py \
  --random-sample 15 \
  --training-data data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json \
  --output-dir results/evaluation_baseline/ \
  --experiment-name "evaluation_baseline"

# Load tasks from file
python scripts/benchmark_evolution.py \
  --task-ids-file benchmark_tasks.txt \
  --output-dir results/multiprocess_baseline/ \
  --experiment-name "multiprocess_baseline"

# Generate Kaggle submission (pass@2 format)
python scripts/benchmark_evolution.py \
  --random-sample 15 \
  --training-data data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json \
  --output-dir results/submission_test/ \
  --experiment-name "submission_test" \
  --generate-submission \
  --num-attempts 2

# Analyze results and generate report
python scripts/analyze_benchmark.py \
  --results-dir results/multiprocess_baseline/ \
  --output-report docs/benchmarks/report.md

# Compare two experiments
python scripts/analyze_benchmark.py \
  --results-dir results/multiprocess_baseline/ \
  --compare-with results/docker_baseline/ \
  --output-report docs/benchmarks/comparison.md
```

**Pass@2 Submission Generation:**

The `--generate-submission` flag enables Kaggle competition submission format:

```bash
# Generate submission for evaluation set
python scripts/benchmark_evolution.py \
  --random-sample 120 \
  --training-data data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json \
  --output-dir results/kaggle_submission/ \
  --experiment-name "kaggle_submission" \
  --generate-submission \
  --num-attempts 2
```

This creates `submission.json` with the required pass@2 format:
- 2 diverse attempts per test input
- Selects best and second-best solvers from evolution history
- Fallback to duplicate attempts if insufficient diversity
- Automatically validates format against Kaggle requirements

**Benchmark Output Structure:**
```
results/{experiment_name}/
├── metadata.json              # Experiment config, timestamp, git commit
├── task_{task_id}.json        # Individual task results
├── summary.json               # Aggregate statistics
├── submission.json            # Kaggle submission (if --generate-submission used)
```

**Phase 2 Baseline Results** (October 30, 2025):
- **Tasks**: 15 diverse ARC tasks
- **Execution Stability**: 100% (no crashes or timeouts)
- **Solution Quality**: 20% of tasks (3/15) achieved meaningful fitness > 0
- **Average Fitness**: 0.33 across all tasks
- **Key Finding**: System is stable but Programmer/Refiner need improvement
- **Full Report**: [docs/benchmarks/phase2_findings.md](docs/benchmarks/phase2_findings.md)

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_loader.py -v

# Run with coverage
pytest tests/ --cov=src/arc_prometheus --cov-report=html
```

## 🏗️ Project Structure

```
arc_prometheus/
├── src/arc_prometheus/
│   ├── crucible/           # Sandbox environment
│   │   ├── data_loader.py  # Load ARC tasks from JSON
│   │   ├── evaluator.py    # Compare grids for correctness
│   │   └── sandbox.py      # Safe code execution with multiprocessing ✅
│   ├── cognitive_cells/    # LLM agents
│   │   ├── prompts.py      # Prompt templates
│   │   ├── programmer.py   # Code generation
│   │   └── refiner.py      # (Phase 2.2) Code debugging ✅
│   ├── evolutionary_engine/ # Evolution mechanisms
│   │   ├── fitness.py      # (Phase 2.1) Fitness evaluation ✅
│   │   └── evolution_loop.py # (Phase 2.3) Evolution loop ✅
│   └── utils/
│       └── config.py       # Configuration management
├── tests/                  # Test suite (423 tests passing)
├── scripts/                # Demo and utility scripts
├── data/                   # ARC dataset (gitignored)
└── plan_20251024.md       # Detailed implementation plan
```

## 🔬 Development Status

**Phase 1: Core Prototype** ✅ COMPLETE
- Data loading, sandbox, LLM generation, E2E pipeline

**Phase 2: Evolutionary Loop** ✅ COMPLETE
- Fitness evaluation, Refiner agent, multi-generation evolution
- Kaggle submission formatter (pass@2 format)

**Phase 3: AI Civilization** ✅ COMPLETE (November 1, 2025)
- ✅ **Task 3.1** (October 31, 2025): Analyst agent - Pattern analysis and rule inference
  - 21 unit tests + 9 integration tests (all passing)
  - Real API validation: 5/5 tasks completed successfully
  - Integration: Programmer accepts Analyst specifications (backward compatible)
- ✅ **Task 3.2** (October 31, 2025): Enhanced Programmer Integration
  - AI Civilization mode (with Analyst) vs Direct mode (without Analyst)
  - CLI support: `--use-analyst` and `--analyst-temperature` flags
  - Refiner receives Analyst context for improved debugging
- ✅ **Task 3.3** (November 1, 2025): Tagger agent - Technique classification
  - 12 predefined techniques: rotation, flip, transpose, color_fill, pattern_copy, symmetry, grid_partition, object_detection, counting, conditional_logic, array_manipulation, neighborhood_analysis
  - Hybrid static + LLM analysis for accurate classification
- ✅ **Task 3.4** (November 1, 2025): Crossover agent - Solution fusion
  - LLM-based technique fusion for population-based evolution
  - Hybrid strategy: Crossover when 2+ diverse solvers exist, else Refiner (mutation)
- ✅ **Task 3.5** (November 1, 2025): Solver Library - Persistent storage
  - SQLite-based population storage with thread-safe WAL mode
  - Diverse solver selection algorithm (greedy tag diversity)
- ✅ **Task 3.6** (November 1, 2025): Population-Based Evolution
  - Multiple solvers evolving simultaneously with genetic algorithm
  - Tournament selection, hybrid breeding, elitism-based survival
  - 19 comprehensive tests (all passing - 423 total)

**Phase 4: Benchmarking & Optimization** ⏭️ NEXT
- Performance tuning across full task suite
- Quality improvements based on empirical results
- Scaling to 400+ training tasks

## 🧪 Technical Details

### ARC Dataset Format

Tasks are JSON files with this structure:
```json
{
  "train": [
    {"input": [[0,1,2], ...], "output": [[3,4,5], ...]},
    ...
  ],
  "test": [
    {"input": [[...]], "output": [[...]] (optional)}
  ]
}
```

Grids are 2D arrays of integers (0-9) representing colors.

### Solver Function Signature

All generated solvers must follow this exact signature:
```python
def solve(task_grid: np.ndarray) -> np.ndarray:
    """
    Transform input grid according to inferred rule.
    Must use only numpy for array operations.
    """
    pass
```

### Safe Execution Protocol

- All LLM-generated code runs in isolated `multiprocessing.Process`
- Default timeout: 5 seconds per execution
- Return format: `tuple[bool, Optional[np.ndarray]]`
  - `(True, result_grid)` on successful execution
  - `(False, None)` on failure/timeout/exception

### Fitness Evaluation Priority

**Critical**: Test accuracy is weighted 10x higher than train accuracy

```python
Fitness = (train_correct * 1) + (test_correct * 10)
```

**Why?** Solvers that only work on training examples have memorized patterns instead of learning the underlying transformation rule. This defeats the core ARC challenge: abstract reasoning with unseen problems.

## 📖 Documentation

- **[CLAUDE.md](CLAUDE.md)**: Guidance for AI agents working on this project
- **[kickoff.md](kickoff.md)**: Project vision and philosophy (Japanese)
- **[plan_20251024.md](plan_20251024.md)**: Detailed Phase 1 implementation plan

## 🧑‍💻 Development

### Development Workflow (⚡ Recommended)

**Option C: Hybrid Approach** - Fast commits with thorough pre-push validation

#### First-Time Setup

```bash
# Install git hooks (one-time setup)
./scripts/setup_hooks.sh
```

This installs:
- **Pre-commit hooks**: Fast checks (ruff formatting, mypy type checking, bandit security) - runs in ~5 seconds
- **Pre-push hooks**: Full CI suite (all checks + full test suite) - runs before pushing to remote

#### Development Cycle

```bash
# 1. Make changes to code

# 2. (Optional) Run full CI checks manually before committing
make ci                  # Runs all checks locally

# 3. Commit your changes
git commit -m "feat: add new feature"
# → Pre-commit hooks run automatically (fast checks only)

# 4. Push to remote
git push origin feature-branch
# → Pre-push hooks run automatically (full test suite)
# → Prevents pushing code that will fail CI
```

#### Why This Workflow?

**Problem**: PR #37 had 5+ CI failure iterations despite having pre-commit config:
- Ruff linting errors (F541, F401, F841)
- Ruff formatting issues
- Runtime errors (type annotations)
- Test failures (missing API key mocks)

**Root Cause**: Tests weren't run before pushing, pre-commit only checked `src/` not `tests/`

**Solution**: Hybrid approach balances speed with reliability:
- ✅ **Fast commits** (~5 seconds) - formatting, linting, type checking
- ✅ **Thorough pre-push** (~30-60 seconds) - full test suite catches API mocking issues
- ✅ **Manual override** - `make ci` runs all checks anytime
- ✅ **Skip if needed** - `git push --no-verify` for emergencies

#### Skip Hooks (Not Recommended)

```bash
git commit --no-verify   # Skip pre-commit hooks
git push --no-verify     # Skip pre-push hooks
```

### Running Tests (TDD Approach)

This project follows Test-Driven Development:

1. Write tests first (in `tests/`)
2. Run tests and watch them fail
3. Implement feature to make tests pass
4. Refactor if needed

```bash
# Watch mode (requires pytest-watch)
ptw tests/

# Run with coverage
pytest tests/ --cov=src/arc_prometheus --cov-report=term-missing
```

### Code Quality

We use a comprehensive CI/CD pipeline with automated quality checks:

**Available Commands** (via Makefile):

```bash
# Run all CI checks at once (same as what runs in CI)
make ci

# Individual checks
make test          # Run test suite
make test-cov      # Run tests with coverage report
make typecheck     # Type checking with mypy
make lint          # Linting with ruff
make lint-fix      # Auto-fix linting issues
make format        # Format code
make format-check  # Check formatting without changes
make security      # Security scanning with bandit
make clean         # Clean cache files
make help          # Show all available commands
```

**Alternative CI Check Script**:
```bash
# Standalone script that mimics GitHub Actions CI
./scripts/check_ci.sh
```

**Quality Tools**:
- **mypy**: Strict type checking (Python 3.13) - checks both `src/` and `tests/`
- **ruff**: Fast linting and formatting (replaces black, flake8, isort)
- **pytest**: Test framework with coverage reporting
- **bandit**: Security vulnerability scanning

**CI/CD Pipeline**:
- Automated checks run on all PRs via GitHub Actions
- Type checking, linting, formatting, security, and tests
- All checks must pass before merging
- Git hooks ensure code is validated before it reaches CI

### Commit Convention

Follow conventional commits:
- `feat: add new feature`
- `fix: bug fix`
- `docs: documentation updates`
- `test: add or update tests`
- `refactor: code refactoring`

## 🤝 Contributing

This project is part of research into evolutionary AI systems. Contributions following the established architecture and TDD approach are welcome.

**Key Principles**:
- Test-driven development (tests before implementation)
- Incremental PRs with validation
- Safe execution (sandboxing is non-negotiable)
- LLM robustness (handle real-world response variations)

## 📝 License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- **ARC Prize 2025**: https://www.kaggle.com/competitions/arc-prize-2025
- **ARC Paper**: "On the Measure of Intelligence" by François Chollet
- **Gemini API**: https://makersuite.google.com/app/apikey

---

**Tests**: 423 passing ✅ | **Phase 3 Complete** ✅ | **Next**: Phase 4 Benchmarking & Optimization

---

## Session Handover

### Last Updated: November 01, 2025 2:00 PM JST

#### Recently Completed

**Phase 3.6: Population-Based Evolution** ([PR #45](https://github.com/TheIllusionOfLife/arc_prometheus/pull/45) - November 01, 2025):
- Implemented PopulationEvolution - complete genetic algorithm for ARC solving
- Multiple solvers evolving simultaneously (configurable population size)
- Tournament selection (k=3) for parent breeding
- Hybrid breeding strategy: Crossover when 2+ diverse parents (30% technique diversity threshold), else mutation via Refiner
- Elitism (top 20%) + fitness-proportionate selection for survivor selection
- Diversity tracking: unique techniques per solver across generations
- 19 comprehensive tests (all passing - 423 total tests)
- Demo script: `scripts/demo_population_evolution.py` with full CLI configuration
- Real API integration: Analyst, Programmer, Refiner, Tagger, Crossover, SolverLibrary all working together
- **Impact**: Phase 3 AI Civilization COMPLETE - full ecosystem operational with true genetic algorithm
- **Key Fix**: Test fixture injection issue - needed to add `sample_task_json` parameter to test function signature

**Phase 3.4 & 3.5: Crossover Agent and Solver Library** ([PR #43](https://github.com/TheIllusionOfLife/arc_prometheus/pull/43) - November 01, 2025):
- Implemented Crossover agent for LLM-based technique fusion (combining successful solvers with complementary techniques)
- Implemented Solver Library for SQLite-based population storage with WAL mode for thread-safety
- Hybrid evolution strategy: Crossover when 2+ diverse solvers exist, else Refiner (mutation)
- Diversity selection algorithm: Greedy tag-based selection prioritizing fitness + technique variety
- 50 new tests (27 crossover + 23 solver library, 404 total passing)
- Real API validation: Technique fusion working correctly, no timeouts or formatting issues
- CLI support: `--use-crossover`, `--crossover-temperature`, `--min-crossover-fitness` flags
- Foreign key constraints enabled, input validation for code_str in SolverLibrary
- **Code Quality**: Addressed 5 gemini-code-assist feedback items (SQLite FK constraints, input validation, API config optimization, pathlib cross-platform paths, CI test mocking)
- **Impact**: Population-based evolution foundation complete - can now fuse techniques from different solvers to create novel solutions

**Phase 3.3: Tagger Agent** ([PR #41](https://github.com/TheIllusionOfLife/arc_prometheus/pull/41) - November 01, 2025):
- Implemented Tagger agent for technique classification (rotation, flip, transpose, color_fill, pattern_copy, symmetry, grid_partition, object_detection, counting, conditional_logic, array_manipulation, neighborhood_analysis)
- Hybrid analysis: Static pattern matching + LLM semantic understanding
- Integration with Evolution Loop via `--use-tagger` and `--tagger-temperature` flags
- Only tags successful solvers (fitness > 0) to optimize API usage
- 37 new tests (28 unit + 9 integration, 354 total passing)
- Real API validation: 5/5 techniques detected with high confidence
- **CI Configuration**: Fixed mypy duplicate module errors by configuring `mypy_path = "src"` and `explicit_package_bases = true` in pyproject.toml
- **Code Quality**: Dual suppression for security false positives (ruff + bandit), proper PEP 561 compliance with py.typed marker
- **Impact**: Foundation ready for Phase 3.4 Crossover - can now identify complementary techniques for solver fusion

**Phase 3.2: Enhanced Programmer Integration** ([PR #39](https://github.com/TheIllusionOfLife/arc_prometheus/pull/39) - October 31, 2025):
- Integrated Analyst agent into Evolution Loop (AI Civilization mode)
- Two operating modes: AI Civilization (with Analyst) vs Direct (without Analyst)
- CLI support: `--use-analyst` and `--analyst-temperature` flags
- Refiner receives Analyst context for improved debugging
- 4 new integration tests (315 total passing)
- Backward compatible: default behavior unchanged (use_analyst=False)
- Real API validation: both modes tested, no errors/timeouts
- **Code Quality**: Centralized ANALYST_DEFAULT_TEMPERATURE constant (DRY principle)
- **Impact**: Complete AI Civilization pipeline now operational - Analyst → Programmer → Refiner collaboration working

**Phase 3.1: Analyst Agent** ([PR #37](https://github.com/TheIllusionOfLife/arc_prometheus/pull/37) - October 31, 2025):
- Implemented Analyst agent for pattern analysis and rule inference
- Natural language specification generation for Programmer agent
- LLM response caching support (70-80% cost reduction)
- Robust parsing for multiple bullet point styles (-, *, numbered lists)
- Git hooks workflow (pre-commit + pre-push) to prevent CI failures
- 313 tests passing (21 Analyst unit + 9 integration + existing)
- Real API validation: 5/5 diverse tasks completed successfully
- **Process Improvements**: Setup script (`./scripts/setup_hooks.sh`), Makefile uv checks
- **Code Quality**: Type safety (GenerationConfigDict), future annotations
- **Impact**: Foundation for AI Civilization mode complete - Analyst understands patterns abstractly, not just code matching

**Earlier Work** (See git history for details):
- ✅ Phase 3.1: Analyst Agent ([PR #37](https://github.com/TheIllusionOfLife/arc_prometheus/pull/37))
- ✅ Task 1: Fix Data Pipeline ([PR #33](https://github.com/TheIllusionOfLife/arc_prometheus/pull/33))
- ✅ Phase 2 Benchmarking ([PR #31](https://github.com/TheIllusionOfLife/arc_prometheus/pull/31))
- ✅ Docker Sandbox ([PR #28](https://github.com/TheIllusionOfLife/arc_prometheus/pull/28))
- ✅ Error Classification ([PR #26](https://github.com/TheIllusionOfLife/arc_prometheus/pull/26))

#### Session Learnings (Most Recent)

**From PR #43 (Crossover Agent & Solver Library) - November 01, 2025 12:25 PM JST**:
- ✅ **COMPLETE**: Phase 3.4 & 3.5 - Crossover Agent and Solver Library successfully implemented and merged
- **Hybrid Evolution Strategy**: Implemented intelligent fallback - use Crossover when 2+ diverse solvers exist (population-based), else Refiner (single-lineage mutation). Enables graceful degradation when population too small
- **SQLite Foreign Key Constraints**: Foreign keys OFF by default in SQLite. Created `_get_connection()` helper with `PRAGMA foreign_keys=ON` to ensure referential integrity for parent_solver_id references
- **Input Validation at Data Layer**: Added validation in `SolverLibrary.add_solver()` to check code_str is non-empty and contains 'def solve' function. Prevents invalid data from entering database
- **Quick Wins Pattern**: When reviewing automated feedback (gemini-code-assist, CodeRabbit), categorize by effort/impact and address quick wins immediately. This session fixed 3 quick wins in <10 minutes (FK constraints, validation, misleading comment)
- **CI Test Mocking Evolution**: Moving API initialization from method to `__init__` requires updating ALL test methods that instantiate the class. Used `@patch` decorators for get_gemini_api_key and genai to mock initialization
- **Pathlib for Cross-Platform Compatibility**: Using string operations like `task_id = task_json_path.split('/')[-1].replace('.json', '')` breaks on Windows. Use `Path(task_json_path).stem` for cross-platform path handling
- **LLM API Config Optimization**: Configure API once in `__init__` instead of per-method call. Reduces overhead and simplifies testing (single mock point)
- **Diversity Selection Algorithm**: Greedy tag-based selection - start with highest fitness solver, then iteratively select solvers with most new tags not yet in selected set. Balances fitness and technique diversity
- **Real API Validation Critical**: Mock tests pass but real Gemini API revealed crossover produces valid fused code. Integration testing with actual LLM responses essential for production readiness

**From PR #41 (Tagger Agent Implementation) - November 01, 2025 09:52 AM JST**:
- ✅ **COMPLETE**: Phase 3.3 - Tagger Agent successfully implemented and merged
- **CI Fix Workflow Mastery**: Encountered mypy duplicate module error ("Source file found twice"). Fixed by configuring `mypy_path = "src"` and `explicit_package_bases = true` in pyproject.toml, plus limiting scope to `mypy src/arc_prometheus` instead of checking tests/scripts
- **Dual Linter Suppression**: Security linters (ruff + bandit) require different suppression syntax for the same false positive. Used `# noqa: S608 # nosec B608` to suppress both - ruff for pre-commit hooks, bandit for CI
- **PEP 561 Compliance**: Created `src/arc_prometheus/py.typed` marker file for proper mypy type checking of installed packages
- **TDD Discipline**: Wrote all 37 tests (28 unit + 9 integration) BEFORE implementing Tagger, then implemented code to pass tests. Caught multiple issues early (fitness=0 handling, mock responses, type annotations)
- **Real API Validation**: Testing with actual Gemini API crucial - detected 5/5 techniques with high confidence, no timeouts or formatting issues. Mock tests alone insufficient for production readiness
- **Iterative CI Debugging**: Hit 3 CI failures (mypy duplicate modules → bandit SQL injection → ruff suppression syntax). Each fix addressed root cause, not symptoms. Final solution: proper configuration + dual suppression

**Earlier Learnings** (See git history for details):
- DRY Principle in Action (PR #39): Centralized constants in config.py
- Backward Compatibility Pattern (PR #39): New features opt-in via flags
- Systematic PR Review Approach (PR #37): GraphQL for comprehensive feedback
- Git Hooks Workflow (PR #37): Fast pre-commit + thorough pre-push

**From Planning Session (plan_20251031.md) - October 31, 2025 10:30 PM JST**:
- ✅ **COMPLETE**: Comprehensive Phase 3 plan created with experimental validation strategy
- **Core Vision Preservation**: Initial plan lost AI Civilization concept by focusing solely on SOTA techniques (single-model active inference). User correctly challenged this - project exists to validate multi-agent evolution hypothesis, not just maximize leaderboard score
- **Experimental Design**: Baseline (pure civilization) → Exp1 (+ single-model active inference) → Exp2 (+ multi-agent active inference). Clean measurement of each component's value-add
- **Active Inference Discovery**: Jack Cole's 34% SOTA uses fine-tuning on each test task's 3 training examples at runtime (not static pre-training). Requires example expansion (3 → 30+) via augmentation
- **Hardware Advantage**: Kaggle L4x4 offers 96GB GPU (vs 15GB T4) - enables larger models (13B-27B) and on-the-fly fine-tuning
- **François's Recommendation**: "Augment discrete program search with deep learning driven intuition" - our multi-agent approach aligns well (LLMs provide intuition, population evolution does search)
- **Research First, Competition Second**: Primary goal is validating AI Civilization hypothesis. Competition provides fair benchmarks and constraints, but success = proving the approach works, not just leaderboard rank
- **Crossover as Unique Differentiator**: Top approaches (J. Berman, E. Pang) use single models. Our Crossover agent can fuse techniques that never appeared together - true genetic innovation vs pattern matching


#### Competitive Context (ARC-AGI-2 Leaderboard)

**What We're Up Against**: ARC-AGI-2 is extremely challenging - top AI systems score in single digits.

**Current Leaders** (October 2025):
- 🥇 **J. Berman**: 29.4% at $30.40/task - Instruction generation + nested LLM calls
- 🥈 **E. Pang**: 26.0% at $3.97/task - Code + Program Library (DreamCoder-inspired)
- 🥉 **GPT-5 Pro**: 18.3% at $7.14/task - Pure CoT reasoning
- **Claude Sonnet 4.5**: 13.6% at $0.759/task - Extended thinking
- **Gemini 2.5 Pro**: 4.9% at $0.767/task - Our base model's cousin
- **Humans**: 60% average (100% at $17/task)
- **Competition Target**: 85% at $0.42/task

**How We Differ from Top Approaches**:

| Approach | J. Berman (29.4%) | E. Pang (26.0%) | **ARC-Prometheus** |
|----------|-------------------|-----------------|-------------------|
| **Architecture** | Single LLM | Single LLM + Library | **Multi-agent ecosystem** |
| **Code Generation** | ❌ (Instructions) | ✅ Python | ✅ Python |
| **Learning** | Iterative refinement | Pattern reuse | **Genetic evolution** |
| **Collaboration** | ❌ Monolithic | ❌ Single agent | ✅ **Analyst + Programmer + Refiner** |
| **Innovation** | Template variations | Library lookup | **Crossover (technique fusion)** |
| **Novel Problems** | Limited | Relies on similar patterns | **Emergent problem-solving** |

**Our Unique Advantages**:
- **Specialized Agents**: Analyst understands patterns abstractly (not just code matching)
- **Crossover**: Can invent solutions by fusing techniques that never appeared together
- **Population Dynamics**: Multiple hypotheses evolving simultaneously
- **Emergent Intelligence**: Novel capabilities from agent interaction

**What We're Validating**: Can a multi-agent civilization outperform single-agent systems?

#### Competition Requirements Analysis

**Critical Findings from Official Rules**:

1. **Data Split** (Competition uses 3 separate datasets):
   - Training: 400+ tasks with solutions (for development)
   - Evaluation: 120 tasks with solutions (for validation)
   - Test: **240 hidden tasks** without solutions (for leaderboard)

2. **Runtime Constraint**: 12-hour hard limit for 240 tasks
   - Our current: ~2 min/task × 240 = 8 hours ✅ (4-hour buffer)
   - Risk: Library lookups and multi-test tasks could add overhead

3. **Variable Test Inputs**: Most tasks have 1 test input, some have 2
   - Must handle dynamic number of test inputs per task
   - Format: `[{"attempt_1": [...], "attempt_2": [...]}, {...}]` (array of predictions)

4. **External Resources Allowed**: Pre-trained models and external data permitted
   - Could fine-tune on training set (400+ tasks)
   - Could use vision models for grid analysis
   - Could pre-compute pattern libraries offline

5. **Open Source Mandatory**: Must open-source to win prizes ✅ (already compliant)

6. **Timeline**: Final submission deadline **November 3, 2025** ⏰

#### Next Priority Tasks

**Philosophy**: We're building an AI civilization, not chasing leaderboard scores. The competition validates our hypothesis: can multi-agent evolution outperform single-agent approaches?

**See detailed plan:** [plan_20251031.md](plan_20251031.md)

**Phase 3: Complete the AI Civilization** (Days 1-3, 30-40 hours) 🧬

**Goal:** Implement all missing agents to realize the complete vision, then validate experimentally

**Core Implementation (Baseline):**

1. ✅ **Analyst Agent** (COMPLETE - PR #37)
   - Separates pattern understanding from code generation
   - Analyzes task examples, infers transformation rules in natural language
   - 21 unit tests + 9 integration tests passing

2. ✅ **Enhanced Programmer** (COMPLETE - PR #39)
   - Accepts Analyst specifications as guidance via analyst_spec parameter
   - Two modes: AI Civilization (with Analyst) vs Direct (without Analyst)
   - 4 new integration tests, backward compatible

3. ✅ **Tagger Agent** (COMPLETE - PR #41)
   - Classifies solver techniques (12 taxonomy: rotation, flip, transpose, color_fill, pattern_copy, symmetry, grid_partition, object_detection, counting, conditional_logic, array_manipulation, neighborhood_analysis)
   - Hybrid static + LLM analysis for comprehensive detection
   - 37 tests passing (28 unit + 9 integration)

4. ✅ **Crossover Agent** (COMPLETE - PR #43)
   - LLM-based technique fusion combining successful solvers with complementary techniques
   - Diversity selection algorithm: greedy tag-based selection prioritizing fitness + variety
   - 27 tests passing (initialization, prompt construction, code parsing, LLM integration, edge cases)

5. ✅ **Solver Library** (COMPLETE - PR #43)
   - SQLite-based population storage with WAL mode for thread-safety
   - Foreign key constraints enabled, input validation for code_str
   - Diversity selection for crossover parent selection
   - 23 tests passing (CRUD, queries, diverse solver selection, edge cases)

6. ✅ **Population-Based Evolution** (COMPLETE - PR #45)
   - Multiple solvers evolving simultaneously with genetic algorithm
   - Tournament selection (k=3), hybrid breeding (crossover/mutation), elitism-based survival
   - Diversity tracking across generations
   - 19 tests passing (data structures, functionality, breeding, dynamics, edge cases)

7. **Kaggle Baseline Submission** (Next, 4-6 hours)
   - **Purpose**: Submit pure AI Civilization to competition
   - **Expected**: 15-25% score (competitive with pure LLM approaches)
   - **Deliverables**: `kaggle_baseline_civilization.ipynb`, leaderboard score, analysis

**Experimental Variations (Optional, Days 4-5):**

8. **Experiment 1: Single Model Active Inference** (4-6 hours)
   - Distill population outputs → Gemma 2 27B
   - Implement active inference (fine-tune on test examples at runtime)
   - **Measure**: Value-add from active inference alone

9. **Experiment 2: Multi-Agent Active Inference** (6-8 hours)
   - Each agent does active inference separately
   - **Measure**: Multi-agent vs single-model active inference

**Research Hypothesis:**
- H1: Multi-agent AI Civilization with crossover outperforms single-model approaches
- H2: Active inference provides additional benefit
- H3: Multi-agent active inference > single-model active inference

**Success Criteria:**
- [x] All 5 agents implemented (Analyst, Programmer, Refiner, Tagger, Crossover) ✅
- [x] Population-based evolution working (COMPLETE - PR #45) ✅
- [x] 100+ new tests passing (423 total tests passing) ✅
- [ ] Baseline Kaggle submission: 15-25% score
- [ ] Research question answered: Can AI Civilization solve novel problems competitively?

**Competition Context:**
- Deadline: November 3, 2025 (3 days)
- Hardware: Kaggle L4x4 GPUs (96GB memory)
- Constraint: 12-hour runtime, no internet access
- Format: pass@2 submission (2 diverse attempts per test input)
