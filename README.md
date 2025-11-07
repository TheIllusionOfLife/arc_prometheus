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

**Tests**: 562 total (all passing) ✅ | **Phase 3 Complete** ✅ | **Kaggle Competition Concluded** ✅ | **Next**: Active Inference User Validation

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

6. **(Optional) Download Code Gemma for Local Testing**:
```bash
# For Kaggle offline notebook development (15.93 GB)
uv run python scripts/download_codegemma.py

# Models are stored in: models/codegemma-7b/
# Note: models/ is gitignored (too large for version control)
```

7. **Download and place ARC dataset**:
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

8. **Verify installation**:
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

**Active Inference** (training example augmentation):
```bash
--use-active-inference           # Enable data augmentation
--augmentation-factor N          # Variations per example (default: 10)
                                 # Trade-off: +5% completion rate, 3.5x slower
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

#### Phase 1: Baseline Validation (Validation-First Strategy)

```bash
# Quick test (10 tasks, ~5-10 minutes)
python scripts/benchmark_evolution.py --random-sample 10 --training-data data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json --output-dir results/quick/ --experiment-name "quick_test"

# Analyze results and get decision recommendation
python scripts/analyze_baseline.py results/quick/ --sample-size

# Full validation (40 tasks, ~1 hour) + comparison
python scripts/benchmark_evolution.py --random-sample 40 --training-data data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json --output-dir results/baseline/ --experiment-name "baseline"
python scripts/analyze_baseline.py results/baseline/ results/experiment/ --compare
```

See `plan_phase4_validation_first.md` for decision gates and complete strategy.

**Pass@2 Submission:**
```bash
python scripts/benchmark_evolution.py --random-sample 120 --training-data data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json --output-dir results/submission/ --generate-submission --num-attempts 2
```
Creates `submission.json` with 2 diverse attempts per test input, auto-validated against Kaggle format.

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

- ✅ **Phase 1**: Core Prototype (data loading, sandbox, LLM generation, E2E pipeline)
- ✅ **Phase 2**: Evolutionary Loop (fitness, refiner, evolution, pass@2 submission)
- ✅ **Phase 3**: AI Civilization (Analyst, Programmer, Refiner, Tagger, Crossover, Solver Library, Population Evolution) - 518 tests passing
- ⏭️ **Phase 4**: Benchmarking & Optimization (performance tuning, scaling to 400+ tasks)

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

### Development Workflow

**Hybrid Approach**: Fast commits (~5s) + thorough pre-push (~30-60s)

```bash
# One-time setup
./scripts/setup_hooks.sh

# Development cycle
make ci                          # (Optional) Run all checks manually
git commit -m "feat: feature"    # → Pre-commit: fast checks (ruff, mypy, bandit)
git push origin branch           # → Pre-push: full test suite

# Skip hooks (emergencies only)
git commit --no-verify
git push --no-verify
```

**Why?** Prevents CI failures from untested code (PR #37 had 5+ iterations). Pre-commit checks formatting/types, pre-push runs tests.

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

**Tests**: 518 total (all passing) ✅ | **Phase 3 Complete** ✅ | **Kaggle Competition Concluded** ✅ | **Next**: Active Inference Implementation

---

## Session Handover

### Last Updated: November 08, 2025 03:01 AM JST

#### Current Status: Kaggle Competition Concluded - Active Inference Development

**Competition Status:**
- Kaggle ARC Prize 2025 competition concluded
- Notebook documentation complete (PR #59)
- Focus shifted from Kaggle constraints to score improvement research

**Pydantic Schema Migration with Cache Error Handling:** MERGED
- Status: PR #55 successfully merged to main with all CI passing
- Architectural improvement: Hybrid schema (dict for API, Pydantic for validation)
- Added 7 missing length constraints to Pydantic models
- Graceful cache invalidation handling for schema migrations
- Cache validation error recovery prevents crashes on stale entries
- Test coverage: 518 tests (all passing)
- **Impact**: Robust schema validation + automatic cache recovery on model changes

#### Recently Completed

**PR #62: Address PR #60 MEDIUM Priority Feedback** ([PR #62](https://github.com/TheIllusionOfLife/arc_prometheus/pull/62) - November 08, 2025):
- Addressed all MEDIUM priority code quality feedback from PR #60 review (gemini-code-assist + claude)
- **Stratified Sampling**: Implemented diverse transformation mix for small augmentation counts - ensures rotations, flips, and color permutations are represented even with a small number of variations (e.g., `num_variations=3`)
- **PEP 8 Compliance**: Moved augmentation imports from local if blocks to top-level try/except pattern (consistent with project conventions)
- **Real CLI Tests**: Replaced placeholder tests with actual argument parser validation for `benchmark_evolution.py` (validates default values, custom inputs, flag combinations)
- **Named Constants**: Replaced magic numbers (3, 5) with lowercase named constants (num_rotations, num_flips) per PEP 8 function-scope requirements
- **Edge Case Test Coverage**: Added 6 unit tests for stratified sampling (count=1, count=2, category representation, reproducibility with seed)
- **Test Coverage**: 568 tests passing (562 existing + 6 new stratified sampling tests)
- **All Review Feedback Addressed**: gemini-code-assist magic numbers fix + claude edge case test recommendations
- **Impact**: Improved code maintainability and test coverage for Active Inference augmentation

**PR #61: Session Handover - PR #60 Active Inference Complete** ([PR #61](https://github.com/TheIllusionOfLife/arc_prometheus/pull/61) - November 07, 2025):
- Documentation update capturing PR #60 learnings and outcomes
- Documented systematic multi-reviewer feedback handling approach
- Captured 5-commit fix strategy and technical patterns (duplicate detection, seed reproducibility, performance optimization)
- Updated session handover timestamp with accurate date formatting

**PR #55: Pydantic Schema Validation Migration** ([PR #55](https://github.com/TheIllusionOfLife/arc_prometheus/pull/55) - November 03, 2025):
- Migrated from dict-based to Pydantic v2 schema validation (PR #53 bug fix)
- Hybrid architecture: dict schemas for Gemini API compatibility + Pydantic models for runtime validation
- Added 7 missing length constraints: `pattern` (150), `approach` (100), `approach_summary` (100), `synthesis_strategy` (150), `diversity_justification` (100), `observations` (1-3), `successful_patterns/failed_patterns` (max 3)
- Enhanced field validators to check per-item lengths for list fields
- Graceful cache error handling: try-except ValidationError with automatic regeneration on stale cache
- Removed redundant validation checks (Pydantic handles constraint validation)
- Fixed Makefile test targets: separate `test`, `test-integration`, `test-all` targets
- Post-merge cache clearing instructions in PR description
- **Code Quality**: Addressed all gemini-code-assist and claude[bot] review feedback (9 items total)
- **Cache Recovery**: ValidationError handling prevents crashes when Pydantic models change after schema migrations
- **Impact**: Production-ready schema validation with robust error recovery for evolving schemas

**PR #53: Test-Time Multi-Persona Ensemble** ([PR #53](https://github.com/TheIllusionOfLife/arc_prometheus/pull/53) - November 03, 2025):
- Implemented test-time ensemble architecture (Phase 1 Tasks 1.1-1.4)
- Multi-Persona Analyst: 5 specialized personas for diverse interpretations (temperature=1.0)
- Multi-Solution Programmer: Generates 5 solutions from interpretations (temperature=0.7)
- Synthesis Agent: Meta-learning from solution performance, generates final hybrid solution
- Gemini structured output: JSON schemas reduce tokens by ~70%, eliminate parsing errors
- Repository cleanup: Removed 11,703 lines of experiment results, added comprehensive .gitignore
- Enhanced conftest.py: Preserves/restores original GEMINI_API_KEY for developer workflows
- Quick wins: Error message debugging details, placeholder array documentation
- All critical reviewer feedback addressed across 3 commits
- **Impact**: Architectural pivot complete - test-time ensemble ready for validation (Tasks 1.5-1.6)


#### Session Learnings (Most Recent)

**From PR #62 (Code Quality Improvements) - November 08, 2025 03:01 AM JST**:
- ✅ **COMPLETE**: PR #62 merged - All MEDIUM priority feedback from PR #60 addressed with enhanced test coverage
- **Stratified Sampling for Diversity**: Implemented algorithm ensuring diverse transformation mix (rotations, flips, colors) for ANY augmentation count, preventing small counts from only getting geometric transforms. Uses proportional sampling with at least 1 from each category.
- **PEP 8 Named Constants in Functions**: Learned ruff N806 requires LOWERCASE for function-scope variables even when conceptually constants (e.g., `num_rotations = 3` not `NUM_ROTATIONS = 3`). Module-level constants use UPPERCASE, function-level use lowercase.
- **Non-Blocking Review Recommendations**: Claude marked stratified sampling test recommendations as "non-blocking action items" - still implemented them for completeness. Distinguishing blocking vs non-blocking feedback prevents delays while improving quality.
- **Edge Case Test Coverage**: Added 6 unit tests for stratified sampling edge cases (e.g., `count=1`, `count=2`), category representation, and reproducibility with seed. These tests validate algorithm branches that standard tests missed.
- **Systematic Review Response**: Addressed gemini-code-assist feedback (magic numbers → named constants) immediately, then claude feedback (edge case tests). All reviewers approved with no additional requests.

**From PR #55 (Pydantic Schema Migration + Cache Error Handling) - November 03, 2025 10:05 AM JST**:
- ✅ **COMPLETE**: PR #55 merged - Pydantic schema validation with graceful cache error recovery
- **Cache Validation for Schema Migrations (Pattern #34)**: When Pydantic models change (e.g., adding max_length constraints), cached responses from old schemas cause ValidationError. Solution: Wrap model_validate_json() in try-except, log warning, fall through to regenerate fresh response. Invalid cache entries automatically overwritten.
- **Hybrid Schema Architecture**: Dict schemas define Gemini API output format (must be simple for API compatibility), Pydantic models enforce runtime validation (can have complex constraints). Separation of concerns: API contract vs runtime safety.
- **Missing Validation Constraints**: During migration, removed max_length from dict schemas for API compatibility but forgot to add to Pydantic Field() definitions. Lesson: Schema migration requires TWO updates - remove from dict, ADD to Pydantic model.
- **Redundant Validation Anti-Pattern**: If Pydantic model has `min_length=1, max_length=5` on list field, don't add explicit `if len(data) != 5: raise ValueError()` check in parsing code. Pydantic raises ValidationError BEFORE parsing method is called. Trust the framework.
- **Field Validator Enhancement**: For list fields with per-item constraints (e.g., observations: list[str] each ≤80 chars), use @field_validator to check each item individually, not just list length. Example: `for obs in v: if len(obs) > 80: raise ValueError()`
- **Post-Merge Cache Clearing**: After schema migrations, users with existing caches need instructions. Added to PR description: "rm -rf ~/.arc_prometheus/llm_cache.db" prevents validation errors on first run with new models.
- **GraphQL PR Review Efficiency**: Continued /fix_pr_graphql pattern - single query fetches all feedback sources. This session: 2 reviewers (gemini-code-assist, claude[bot]), addressed all cache error handling suggestions immediately.
- **Test Coverage for Constraints**: Added length constraints didn't break any tests because existing test data already complied. Good test data should include BOTH valid and boundary cases to catch constraint violations.

**From PR #60 (Active Inference) - November 07, 2025 10:00 PM JST**:
- ✅ **COMPLETE**: PR #60 merged - Active Inference training example augmentation with +5% success rate improvement
- **Multi-Reviewer Feedback Handling**: Systematic extraction and prioritization of feedback from 3 AI reviewers (coderabbitai, gemini-code-assist, claude) across different comment threads
- **5-Commit Fix Strategy**: Addressed feedback in priority order - (1) CRITICAL: seed control (2) MEDIUM: CLI validation, console display, documentation (3) MEDIUM: performance optimization, diversity enforcement (4) CRITICAL: API tracking, variation warnings (5) MUST FIX: duplicate detection
- **Duplicate Detection Pattern**: Added deduplication logic to prevent token waste when augmentation_factor > 13 (max unique variations). Logs info when duplicates removed, prevents wasting tokens on identical examples
- **API Cost Tracking Accuracy**: Fixed metric showing incorrect "30 API calls" with augmentation_factor=10. Corrected to show accurate "3 API calls" + "10x token multiplier" - important for user cost awareness
- **Seed Reproducibility**: Implemented full seed control for color permutations with min_swaps diversity enforcement. Documented limitation that seed only affects color perms, not geometric transforms
- **Performance Optimization**: Replaced O(10×n) color mapping loop with O(n) lookup table using numpy indexing - 5-10x speedup for large grids
- **Test Coverage**: 565 tests passing (26 augmentation unit + 9 integration + 530 existing), all reviewers approved after systematic fixes
- **Production Quality Metrics**: +5% success rate validated on 25 tasks, 3.5x slower execution (acceptable trade-off for 10x prompt increase), comprehensive error handling and warnings

**From PR #53 (Test-Time Ensemble + PR Review) - November 03, 2025 07:26 AM JST**:
- ✅ **COMPLETE**: PR #53 merged - test-time multi-persona ensemble operational
- **Comprehensive PR Review Workflow**: Used GraphQL to extract ALL feedback sources (PR comments, reviews, line comments, CI annotations) from 4 reviewers (claude, coderabbitai, gemini-code-assist, chatgpt-codex-connector)
- **Critical Issue Prioritization**: Addressed CRITICAL issues first (results bloat 11,660 lines, conftest.py env var preservation), then quick wins (error messages, documentation), deferred optional refactoring
- **Post-Commit Review Vigilance**: Always check timestamps - new reviews can arrive AFTER pushing fixes. CodeRabbit reviewed at different cadence than other bots
- **.gitignore Pattern Best Practices**: Use consistent recursive patterns (`results/**/*.json`) over mixed patterns. Removed redundant `results/*.log` (covered by `results/**/*.log`)
- **Test Fixture Environment Safety**: Save original env vars before modification: `original_key = os.environ.get("KEY")`, restore on cleanup to prevent disrupting developer workflows
- **Quick Wins from Approval Reviews**: Even "APPROVE" reviews contain actionable quick wins. Addressed 2/5 suggestions in <10 minutes (error detail messages, placeholder rationale comments)
- **Results Directory Hygiene**: Use .gitignore patterns + `git rm -r --cached results/` to remove bloat, add results/.gitkeep to maintain structure. Prevents future repository bloat
- **Approval-with-Suggestions Pattern**: Not all feedback requires fixes before merge. Distinguish CRITICAL (blocking) vs NICE-TO-HAVE (future PRs) based on test status and reviewer priority labels
- **CI Type Check Priority**: mypy errors most common; run `uv run mypy src/` locally before push to catch issues early

#### Next Priority Tasks

Currently in maintenance mode - all Active Inference development complete (PR #60, #61, #62 merged). No active tasks.

**Future Research Directions** (when/if resumed):
1. **Active Inference Evaluation**: Benchmark augmentation_factor impact on diverse ARC task types
   - Source: PR #60 - showed a +5% success rate on a sample of 25 tasks, which needs broader validation
   - Context: Understand which task patterns benefit from augmentation vs. those where it hurts
   - Approach: Systematic evaluation across task categories (rotation, fill, pattern, logic)
2. **Stratified Sampling Optimization**: Tune category proportions for maximum diversity
   - Source: PR #62 - current implementation uses equal weighting across categories
   - Context: Some categories (color permutations) may provide more diversity than others
   - Approach: Experiment with weighted sampling based on transformation diversity metrics

---

**For detailed project history**: See git log and [plan_20251101.md](plan_20251101.md) for Phase 4 priorities.
