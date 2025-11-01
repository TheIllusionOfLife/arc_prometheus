# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ARC-Prometheus** is an AI civilization project designed to solve the ARC Prize (Abstraction and Reasoning Corpus) challenge through an evolutionary ecosystem of LLM agents.

### Core Architecture

The system consists of three main components:

1. **The Crucible (るつぼ)** - Sandbox environment for ARC puzzle execution and validation
2. **The Cognitive Cells (認知的細胞)** - Specialized LLM agent teams:
   - **Analyst**: Analyzes ARC input/output examples to infer transformation rules
   - **Programmer**: Generates Python solver code based on Analyst specifications
   - **Refiner**: Debugs and improves failed solver code
   - **Tagger** (Phase 3): Identifies techniques used in successful solvers
3. **The Evolutionary Engine (進化的エンジン)** - Evolution mechanisms:
   - **Mutation**: Code refinement through error analysis
   - **Crossover**: Fusion of successful solvers with different capabilities
   - **Fitness Function**: Evaluates solver generalization (prioritizes test accuracy over train)

### Development Philosophy

This project follows a strict Test-Driven Development approach and evolutionary principles:

- **Few-shot learning**: ARC tasks provide only ~3 training examples on average
- **Generalization over memorization**: Test accuracy is weighted 10x higher than train accuracy
- **Emergent intelligence**: Multiple specialized agents collaborate like a scientific community
- **Safe execution**: All generated code runs in sandboxed environments (multiprocessing or Docker)

## Development Phases

### Phase 1: Core Prototype ✅ COMPLETE
Build the minimal ecosystem to run a single ARC task end-to-end:
- ✅ **Phase 1.1-1.5**: Foundation through End-to-End Pipeline
  - Data loading, visualization, grid evaluation
  - Manual solver validation (task 05269061)
  - Safe execution sandbox with multiprocessing
  - LLM code generation (Gemini API)
  - Complete E2E orchestration
  - 93 tests passing
- **Success Criteria**: ✅ AI-generated code solves ≥1 train pair

### Phase 2: Evolutionary Loop ✅ COMPLETE
Introduce selection pressure and mutation:
- ✅ **Phase 2.1 Complete**: Fitness function evaluation
  - Formula: `fitness = (train_correct * 1) + (test_correct * 10)`
  - 10x weight prioritizes generalization over memorization
  - Handles ARC evaluation format (missing test outputs)
  - 11 comprehensive tests (104 total passing)
  - Demo with 3 scenarios (perfect, overfitting, timeout)
- ✅ **Phase 2.2 Complete**: Refiner agent (code debugging/mutation)
  - LLM-based debugging with error analysis
  - 12 comprehensive tests (116 total passing)
  - Real API: 100% success rate (syntax, logic, timeout fixes)
- ✅ **Phase 2.3 Complete**: Multi-generation evolution loop
  - Complete cycle: Generate → Evaluate → Refine → Repeat
  - Generation tracking with fitness improvement metrics
  - 13 new tests (129 total passing)
  - Early termination when target fitness reached
- ✅ **Task 1.1 Complete**: CLI configuration externalization
  - Comprehensive argparse-based CLI system
  - Separate --programmer-temperature and --refiner-temperature
  - Optional parameters threaded through Programmer/Refiner/Evolution Loop
  - 22 new tests (151 total passing)
  - Backward compatible with config.py defaults

### Phase 3: AI Civilization 🚧 IN PROGRESS
Expand to full multi-agent ecosystem with crossover:
- ✅ **Phase 3.1 Complete** (October 31, 2025): Analyst Agent
  - Pattern analysis and rule inference from ARC examples
  - Structured output: pattern_description, key_observations, suggested_approach, confidence
  - Integration with Programmer (AI Civilization mode vs Direct mode)
  - 21 unit tests + 9 integration tests (all passing - 311 total)
  - Real API validation: 5/5 diverse tasks completed successfully
  - Production ready: No timeouts, truncation, or API errors
- ✅ **Phase 3.2 Complete** (October 31, 2025 - Task 3.2): Enhanced Programmer Integration
  - Evolution loop integrated with Analyst agent
  - CLI support via `--use-analyst` and `--analyst-temperature` flags
  - Two modes: AI Civilization (with Analyst) vs Direct (without Analyst)
  - Refiner receives Analyst context for improved debugging
  - 21 tests (4 new Analyst integration + 17 existing all passing - 315 total)
  - Backward compatible: default behavior unchanged (use_analyst=False)
  - Real API validated: both modes work correctly, no errors/timeouts
- ✅ **Phase 3.3 Complete** (November 1, 2025): Tagger Agent
  - Technique classification via hybrid static + LLM analysis
  - 12 predefined techniques: rotation, flip, transpose, color_fill, pattern_copy, symmetry, grid_partition, object_detection, counting, conditional_logic, array_manipulation, neighborhood_analysis
  - CLI support via `--use-tagger` and `--tagger-temperature` flags
  - 37 tests (28 unit + 9 integration tests, all passing - 354 total)
  - Real API validated: 5/5 techniques detected with high confidence
  - Production ready: No timeouts, truncation, or formatting issues
  - Only tags successful solvers (fitness > 0) to optimize API usage
  - CI configuration: Fixed mypy duplicate module errors with proper package base configuration
- ✅ **Phase 3.4 Complete** (November 1, 2025): Crossover Agent
  - LLM-based technique fusion for population-based evolution
  - Hybrid strategy: Crossover when 2+ diverse solvers exist, else Refiner (mutation)
  - Compatibility assessment and confidence scoring
  - CLI support via `--use-crossover` and `--crossover-temperature` flags (default: 0.5)
  - 27 tests (all passing - 381 total)
  - Backward compatible: default behavior unchanged (use_crossover=False)
  - Integration with Analyst for context-aware fusion
- ✅ **Phase 3.5 Complete** (November 1, 2025): Solver Library
  - SQLite-based persistent storage for solver population
  - Thread-safe WAL mode for concurrent access
  - Diverse solver selection algorithm (greedy tag diversity)
  - UUID-based solver tracking with lineage (parent_solver_id)
  - 23 tests (all passing - 404 total)
  - Integration with evolution loop: automatic storage when crossover enabled
  - Database: ~/.arc_prometheus/solver_library.db

## Common Commands

### Environment Setup (Phase 1.1)
```bash
# Download ARC Prize 2025 dataset from Kaggle
# https://www.kaggle.com/competitions/arc-prize-2025/data

# Install core dependencies
pip install numpy google-generativeai

# Future phases will add:
# pip install multiprocess celery redis pytest

# Install git hooks (IMPORTANT - prevents CI failures)
./scripts/setup_hooks.sh
```

### Development Workflow (Prevent CI Failures) ⚠️

**MANDATORY FOR ALL DEVELOPMENT**: Install git hooks to catch issues before CI
```bash
# One-time setup
./scripts/setup_hooks.sh

# This installs:
# - Pre-commit hooks: Fast checks (ruff, mypy, bandit) - ~5 seconds
# - Pre-push hooks: Full CI suite (all checks + tests) - ~30-60 seconds
```

**Before Every Commit**:
```bash
# Option 1: Let git hooks handle it automatically
git commit -m "feat: your changes"
# → Pre-commit hooks run automatically

# Option 2: Run all checks manually first
make ci
git commit -m "feat: your changes"
```

**Before Every Push**:
```bash
git push origin your-branch
# → Pre-push hooks run automatically (includes full test suite)
# → Prevents pushing code that will fail CI
```

**Why This Matters** (Lessons from PR #37):
- PR #37 had 5+ CI failure iterations despite pre-commit config
- Issues: ruff linting, formatting, type annotations, missing test mocks
- Root cause: Tests not run before push, mypy only checked `src/` not `tests/`
- Solution: Hybrid approach - fast commits, thorough pre-push validation
- Result: Catches issues locally before wasting CI/reviewer time

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_evolution_loop.py -v

# Run with coverage
uv run pytest tests/ --cov=src/arc_prometheus
```

### Evolution Loop CLI (Phase 2.3 + Phase 3.2 + Phase 3.3)
```bash
# Direct mode (Phase 2 - without Analyst)
uv run python scripts/demo_phase2_3_evolution.py

# AI Civilization mode (Phase 3 - with Analyst)
uv run python scripts/demo_phase2_3_evolution.py --use-analyst

# With Tagger for technique classification (Phase 3.3)
uv run python scripts/demo_phase2_3_evolution.py --use-tagger

# Full AI Civilization: Analyst + Tagger
uv run python scripts/demo_phase2_3_evolution.py --use-analyst --use-tagger

# Customize temperatures
uv run python scripts/demo_phase2_3_evolution.py \
  --programmer-temperature 0.5 \
  --refiner-temperature 0.6 \
  --analyst-temperature 0.4 \
  --tagger-temperature 0.5  # Only used with --use-tagger

# Longer evolution runs
uv run python scripts/demo_phase2_3_evolution.py \
  --use-analyst \
  --use-tagger \
  --max-generations 10 \
  --model gemini-2.0-flash-thinking-exp

# Cache management
uv run python scripts/demo_phase2_3_evolution.py --cache-stats  # View stats
uv run python scripts/demo_phase2_3_evolution.py --clear-cache  # Clear all
uv run python scripts/demo_phase2_3_evolution.py --no-cache     # Disable
```

### Data Format
ARC tasks are JSON files with this structure:
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

**IMPORTANT - ARC Evaluation Format**:
- Training tasks (`arc-agi_training_challenges.json`) have outputs for both train AND test examples
- Evaluation tasks (`arc-agi_evaluation_challenges.json`) store test outputs SEPARATELY in `arc-agi_evaluation_solutions.json`
- **Use preprocessing script before benchmarking evaluation data**:
  ```bash
  python scripts/prepare_evaluation_data.py \
    --challenges data/arc-prize-2025/arc-agi_evaluation_challenges.json \
    --solutions data/arc-prize-2025/arc-agi_evaluation_solutions.json \
    --output data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json
  ```
- Code must handle missing "output" keys gracefully: `if "output" not in example: continue`
- Always use explicit dtype casting: `np.array(example["input"], dtype=np.int64)`

## Code Standards

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
**Note**: Use `np.ndarray` (the type) not `np.array` (the function) for type annotations.

### Safe Execution Protocol

**Sandbox Options**:
- **Multiprocessing** (default): Fast execution with basic isolation
  - Restricted builtins: eval, exec, compile, open removed
  - Timeout enforcement via process termination
  - **Limitation**: Cannot prevent filesystem/network access
- **Docker** (production-grade): Complete container isolation
  - Network disabled (no external communication)
  - Read-only filesystem (except /tmp tmpfs for numpy)
  - Memory limit: 512MB (configurable)
  - CPU limit: 50% of one core (configurable)
  - Process limit: 100 (prevents fork bombs)
  - Non-root user execution (UID 1000)

**Return Format**: `tuple[bool, Optional[np.ndarray], Optional[dict]]`
- `(True, result_grid, None)` on successful execution
- `(False, None, error_detail)` on failure (error_detail includes error_type, error_message, exception_class)

**Usage**:
- Default: `calculate_fitness(task_path, code)` uses multiprocessing
- Production: `calculate_fitness(task_path, code, sandbox_mode="docker")` uses Docker
- CLI: `--sandbox-mode docker` flag for evolution scripts

**Implementation**:
- Both sandboxes implement `ExecutionEnvironment` protocol
- Factory pattern in `fitness.py`: `_get_sandbox(sandbox_mode)` returns appropriate sandbox
- Docker requires: `pip install -e '.[docker]'` and image built via `docker build`

### LLM Integration (Gemini)
- Primary model: Google Gemini API
- Input format: ASCII art grids in prompts
- Expected output: Pure Python code blocks
- Parser must handle:
  - Code wrapped in \`\`\`python ... \`\`\` blocks
  - Raw code without delimiters
  - Multiple code blocks (extract the solve() function)
- Always include "use only numpy" constraint in prompts

**Note**: LLMs often include markdown formatting despite instructions. Implement robust parsing.

### LLM Response Caching
- **Cache location**: `~/.arc_prometheus/llm_cache.db` (SQLite)
- **Enabled by default** in all cognitive cells (Programmer, Refiner)
- **TTL**: 7 days (configurable)
- **Thread-safe**: Uses SQLite WAL mode for concurrent access
- **CLI control**:
  - `--no-cache`: Disable for specific run
  - `--cache-stats`: View hit rate and cost savings
  - `--clear-cache`: Remove all entries
- **Programmatic control**: Use `use_cache=False` parameter in `generate_solver()` and `refine_solver()`
- **Benefits**: 70-80% API cost reduction during development, instant responses for repeated prompts
- **Limitations**:
  - Cache NOT shared across processes
  - Statistics approximate (hit rate slightly inflated)
  - See docstring in `llm_cache.py` for details

### Fitness Evaluation: Prioritize Generalization Over Memorization
**Why Test Accuracy Matters 10x More**: Solvers that only work on training examples fail the core ARC challenge—abstract reasoning with unseen problems. Overfitting to training data means the AI has memorized patterns instead of learning the underlying transformation rule.

**Consequences of Prioritizing Train Accuracy**:
- Solver becomes a "lookup table" for known examples
- Fails completely on novel test cases
- Defeats the purpose of AGI research (generalization is the goal)
- Wastes computational resources on non-generalizable solutions

**Formula**: `Fitness = (train_correct * 1) + (test_correct * 10)`
- Example: 3/3 train + 1/1 test = 13 points (good generalization)
- Example: 3/3 train + 0/1 test = 3 points (pure overfitting)
- Solvers that timeout or crash receive fitness = 0

### Data Persistence (Phase 3)
Solver schema:
```python
{
    "solver_id": str,
    "task_id": str,
    "generation": int,
    "code_str": str,
    "fitness_score": float,
    "train_correct": int,
    "test_correct": int,
    "parent_solver_id": str | None,
    "tags": List[str]  # e.g., ["rotation", "fill", "symmetry"]
}
```

## Important Context

### Why This Approach?
Modern deep learning fails at ARC because it requires millions of examples, while ARC provides only 3. This project simulates how human scientists solve problems: diverse specialists collaborating, experimenting, and building on each other's work through evolutionary pressure.

### Technical Constraints
- **ARC dataset**: ~400 training tasks, ~100 evaluation tasks
- **Grid size**: Typically 3x3 to 30x30
- **Color values**: 0-9 (10 colors total)
- **No external libraries** in generated solvers except numpy
- **Execution safety**: Untrusted LLM code must not access filesystem/network

### Task-Specific vs General-Purpose Solvers
**Important Distinction**: This project has two types of solvers with different purposes:

1. **Task-Specific Validation Solvers** (Phases 1.2-1.3):
   - Manually written for ONE specific ARC task
   - Purpose: Validate infrastructure before LLM integration
   - May use hardcoded constants specific to that task (e.g., 7x7 grids for task 05269061)
   - Example: `scripts/demo_phase1_2_manual.py` - intentionally task-specific
   - **Not a bug**: Hardcoded values are BY DESIGN for validation purposes

2. **General-Purpose LLM-Generated Solvers** (Phase 1.4+):
   - Generated by LLM from task examples
   - Must handle variable grid sizes (3x3 to 30x30)
   - Should generalize to unseen test examples
   - Example: Future output of Programmer agent

**Code Review Guidance**: When reviewing Phase 1.2-1.3 code, task-specific implementations are correct and expected. Only Phase 1.4+ requires general-purpose solvers.

### Success Metrics
- **Phase 1**: Solve ≥1 train pair with generated code
- **Phase 2**: Improve test accuracy across generations
- **Phase 3**: Cross-task solver generalization (solver trained on task A solves task B)

## Kaggle Submission (pass@2 Format)

### Overview
The pass@2 submission format is required for the ARC Prize 2025 competition on Kaggle. It submits 2 diverse attempts per test input, and the score is 1 if at least one matches the ground truth.

### Implementation (Phase 2.5 - October 31, 2025)

**Module**: `src/arc_prometheus/evolutionary_engine/submission_formatter.py`

**Core Functions**:
1. `select_diverse_solvers(generations, num_attempts=2, diversity_metric="fitness")`:
   - Selects diverse solvers from evolution generation history
   - Removes duplicate code before selection
   - Default: fitness-based (best + second-best solver)
   - Alternative: generation_gap (early + late generation)
   - Raises `ValueError` if insufficient unique solvers

2. `generate_task_predictions(task_json_path, solver_codes, timeout=5, sandbox_mode="multiprocess")`:
   - Applies each solver to all test inputs in the task
   - Handles variable test input counts (0-3 per task)
   - Returns `[[0, 0], [0, 0]]` placeholder for failed solvers
   - Converts numpy arrays to Python lists for JSON serialization

3. `format_submission_json(task_predictions)`:
   - Validates structure matches Kaggle requirements
   - Returns submission dict ready for json.dump()

**Usage via Benchmark Script**:
```bash
python scripts/benchmark_evolution.py \
  --random-sample 120 \
  --training-data data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json \
  --output-dir results/kaggle_submission/ \
  --experiment-name "kaggle_submission" \
  --generate-submission \
  --num-attempts 2
```

**Output Format** (matches `sample_submission.json`):
```json
{
  "task_id": [
    {"attempt_1": [[grid]], "attempt_2": [[grid]]},  // Test input 0
    {"attempt_1": [[grid]], "attempt_2": [[grid]]},  // Test input 1 (if exists)
    ...
  ]
}
```

**Diversity Strategy**:
- Primary: Select best fitness and second-best fitness solvers
- Fallback: If <2 unique solvers exist, duplicate the best solver (with warning)
- This handles cases where evolution produces nearly identical code

**Validation**:
- Automatic format validation in benchmark script
- Checks: dict structure, list per task, attempt_1/attempt_2 keys, JSON serializable
- Test coverage: 18 unit tests in `tests/test_submission_formatter.py`

**Real API Testing** (October 31, 2025):
- Tested with 3 tasks using Gemini API
- Results: 3/3 successful, submission.json generated, format validated ✅
- No timeouts, truncation, or invalid values ✅
- Diversity: 2/3 tasks had different attempts, 1/3 identical (expected)

## GitHub Actions

This repository has Claude Code integrated via GitHub Actions:
- **@claude mentions**: Claude responds to `@claude` mentions in issues/PRs/comments (`.github/workflows/claude.yml`)
  - Triggered by: issue comments, PR comments, PR reviews, new issues
  - Can read CI results and perform code operations
  - Responds with code changes, analysis, or suggestions
- **Automated PR review**: Claude reviews all PRs for code quality, bugs, security, and best practices (`.github/workflows/claude-code-review.yml`)
  - Triggered on: PR open, PR synchronize (new commits)
  - Checks: code quality, potential bugs, performance, security, test coverage
  - Posts review comments with actionable feedback

## Reference Documents

- **kickoff.md**: Detailed project vision and complete task roadmap (in Japanese)
- **ARC Prize 2025**: https://www.kaggle.com/competitions/arc-prize-2025
- **ARC Paper**: "On the Measure of Intelligence" by François Chollet
- **plan_20251024.md**: Detailed Phase 1 implementation plan with incremental PRs
