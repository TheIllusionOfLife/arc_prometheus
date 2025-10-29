# ARC-Prometheus 🔥

**AI Civilization for Solving ARC Prize through Evolutionary LLM Agents**

ARC-Prometheus is an ambitious project that simulates how human scientists solve problems: diverse specialists collaborating, experimenting, and building on each other's work through evolutionary pressure. Instead of building a single "super-intelligent" AI, we're creating an ecosystem of specialized LLM agents that evolve solutions to the [ARC Prize 2025](https://www.kaggle.com/competitions/arc-prize-2025) challenge.

## 🎯 Project Vision

Modern deep learning fails at ARC because it requires millions of examples, while ARC provides only ~3 training examples per task. This project takes a fundamentally different approach:

- **The Crucible (るつぼ)**: Sandbox environment for safe code execution and validation
- **The Cognitive Cells (認知的細胞)**: Specialized LLM agent teams (Analyst, Programmer, Refiner, Tagger)
- **The Evolutionary Engine (進化的エンジン)**: Evolution mechanisms (Mutation, Crossover, Fitness Function)

### Why This Matters

ARC Prize tests **abstraction and reasoning** - the ability to learn underlying transformation rules from just 3 examples and apply them to never-before-seen problems. This is the essence of human intelligence and a critical step toward AGI.

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

4. **Set up API key**:
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

### Phase 1.1: Data Loading Demo

Run the demo script to see data loading and grid visualization:

```bash
# Use default task (first in dataset)
python scripts/demo_phase1_1_data.py

# Use specific task ID
python scripts/demo_phase1_1_data.py 007bbfb7
```

**Expected output**:
- Colored grid visualization for train examples
- Input/output comparisons
- Test examples (puzzles to solve)
- Evaluation demo showing grid comparison

### Phase 1.2: Manual Solver Demo

Run the manual solver demo to see validation with task 05269061:

```bash
python scripts/demo_phase1_2_manual.py
```

**Expected output**:
- Manual solver solves all 3 train examples (100% success rate)
- Predicted vs expected output comparison with colored grids
- Success message: "🎉 SUCCESS: Manual solver works perfectly!"

**What it demonstrates**:
- Manually-written solver using numpy operations
- Diagonal pattern extraction and grid filling
- Infrastructure validation before LLM integration

### Phase 1.3: Safe Execution Sandbox Demo

Run the sandbox demo to see safe execution of untrusted code:

```bash
python scripts/demo_phase1_3_sandbox.py
```

**Expected output**:
- Demo 1: Successful execution - Phase 1.2 solver in sandbox (3/3 correct, 100%)
- Demo 2: Timeout enforcement - Infinite loop terminated after 2 seconds
- Demo 3: Exception handling - ZeroDivisionError caught gracefully

**What it demonstrates**:
- Multiprocessing isolation for untrusted code execution
- Timeout enforcement (configurable, default 5 seconds)
- Exception handling (syntax errors, runtime errors)
- Return type validation (must be np.ndarray)
- Integration with Phase 1.2 manual solver

**Security notes**:
- Restricted builtins: eval, exec, compile, open removed
- **Limitation**: Cannot prevent filesystem/network access (multiprocessing limitation)
- **Production**: Use Docker with read-only filesystem and network disabled

### Phase 1.5: End-to-End Pipeline

Run complete AI solver pipeline on any ARC task:

```bash
python scripts/run_phase1_test.py <task_id>

# Example with known tasks
python scripts/run_phase1_test.py 00576224
python scripts/run_phase1_test.py 007bbfb7
python scripts/run_phase1_test.py 025d127b
```

**Expected output**:
- Task loading confirmation with train/test example counts
- LLM-generated solver code display (with line numbers)
- Execution results for each train example with colored grid visualization
- Success rate and performance metrics
- Saved solver file location (if any examples solved correctly)
- Clear error diagnostics with diffs when predictions don't match

**What it demonstrates**:
- Complete orchestration of all Phase 1 components
- Data loading → LLM generation → Sandbox execution → Evaluation
- End-to-end AI civilization capability
- Robust error handling (timeouts, exceptions, mismatches)
- User-friendly output with visual progress tracking

**Note**: LLM-generated code quality varies (inherent randomness). The pipeline correctly handles all scenarios:
- ✅ Successful solver execution and evaluation
- ✅ Code generation failures (missing returns, syntax errors)
- ✅ Runtime errors (division by zero, module imports)
- ✅ Timeout enforcement for infinite loops
- ✅ Grid shape/value mismatches with detailed diffs

### Phase 2.1: Fitness Function Evaluation

Evaluate solver quality with emphasis on generalization over memorization:

```bash
python scripts/demo_phase2_1_fitness.py
```

**Expected output**:
- Demo 1: Perfect solver achieving 100% accuracy (fitness = 13)
- Demo 2: Overfitting solver failing on test data (fitness = 3)
- Demo 3: Timeout enforcement with infinite loop (fitness = 0)
- Clear breakdown of train vs test performance
- Analysis of generalization capability

**What it demonstrates**:
- Fitness calculation: `(train_correct × 1) + (test_correct × 10)`
- 10x weight on test accuracy encourages generalization
- Timeout enforcement prevents infinite loops
- Clear error reporting for failure modes
- Foundation for evolutionary loop (Phase 2.2+)

**Key insight**: A solver with 100% train accuracy but 0% test accuracy (pure overfitting) receives fitness = 3, while a solver with lower train but some test accuracy can score higher. This prioritizes solvers that learn underlying rules rather than memorizing examples.

### Phase 2.2: Refiner Agent (Code Debugging)

Automatically debug and improve failed solver code through LLM-based refinement:

```bash
python scripts/demo_phase2_2_refiner.py
```

**Expected output**:
- Demo 1: Syntax error fixed (fitness 0 → 13, +13 improvement)
- Demo 2: Logic error corrected (fitness 0 → 13, +13 improvement)
- Demo 3: Infinite loop optimized (fitness 0 → 13, +13 improvement)
- Clear before/after fitness comparison with line-numbered code
- Progress indicators during Gemini API calls
- Summary showing total fitness gain across all scenarios

**What it demonstrates**:
- LLM-based code debugging with error analysis
- First evolutionary mechanism (Mutation) - automated code refinement
- Fitness improvement through iterative debugging
- Handles syntax errors, logic errors, and performance issues
- Foundation for evolution loop (Phase 2.3)

**Key features**:
- Uses Gemini API with temperature 0.4 (debugging creativity)
- Analyzes fitness results to identify failure patterns
- Generates corrected code with proper numpy-only constraints
- Re-evaluates refined code to verify improvement
- Tested with real API: 3/3 scenarios successful (100%)

### Phase 2.3: Evolution Loop

Run multi-generation solver evolution combining all Phase 2 components:

```bash
# Run with default configuration
python scripts/demo_phase2_3_evolution.py

# Use custom model (e.g., thinking model for complex tasks)
python scripts/demo_phase2_3_evolution.py --model gemini-2.0-flash-thinking-exp

# Adjust creativity with custom temperatures
python scripts/demo_phase2_3_evolution.py --programmer-temperature 0.5 --refiner-temperature 0.6

# Run more generations for difficult tasks
python scripts/demo_phase2_3_evolution.py --max-generations 10

# Combine multiple options
python scripts/demo_phase2_3_evolution.py \
  --model gemini-2.0-flash-thinking-exp \
  --max-generations 10 \
  --programmer-temperature 0.5 \
  --refiner-temperature 0.6

# See all available options
python scripts/demo_phase2_3_evolution.py --help
```

**Configuration Options**:
- `--model MODEL`: LLM model name (default: gemini-2.5-flash-lite)
- `--programmer-temperature TEMP`: Code generation creativity, 0.0-2.0 (default: 0.3)
- `--refiner-temperature TEMP`: Debugging creativity, 0.0-2.0 (default: 0.4)
- `--max-generations N`: Maximum evolution generations (default: 5)
- `--target-fitness N`: Stop when fitness reaches this value (optional)
- `--timeout-llm SECONDS`: LLM API call timeout (default: 60)
- `--timeout-eval SECONDS`: Code execution timeout (default: 5)
- `--no-verbose`: Disable verbose output

**Expected output**:
- Configuration display showing all parameters
- Demo 1: Simple task evolution (2-3 generations typical)
- Demo 2: Early convergence (perfect solver from start)
- Demo 3: Gradual improvement (up to 5 generations)
- Generation-by-generation fitness tracking
- Total improvement and timing statistics
- Analysis of evolution patterns (convergence, plateau, etc.)

**What it demonstrates**:
- Complete evolutionary cycle: Generate → Evaluate → Refine → Repeat
- Multi-generation tracking with improvement metrics
- Early termination when target fitness reached (efficiency)
- Graceful handling of plateau scenarios (no improvement)
- Foundation for Phase 3 (population-based evolution)

**Key features**:
- Combines Programmer (generation) + Fitness (evaluation) + Refiner (mutation)
- Configurable max generations and target fitness
- Per-generation timing for performance analysis
- Verbose/silent modes for different use cases
- Returns complete history for post-analysis

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
├── tests/                  # Test suite (127 tests passing)
├── scripts/                # Demo and utility scripts
├── data/                   # ARC dataset (gitignored)
└── plan_20251024.md       # Detailed implementation plan
```

## 🔬 Development Phases

### Phase 1: Core Prototype (Current)
**Goal**: Build minimal ecosystem for end-to-end ARC task solving

- [x] **1.1**: Environment setup + data loading ✅
- [x] **1.2**: Manual solver validation ✅
- [x] **1.3**: Safe execution sandbox ✅
- [x] **1.4**: LLM-based code generation (gemini-2.5-flash-lite) ✅
- [x] **1.5**: Complete end-to-end pipeline ✅

**Success Criteria**: AI-generated code solves ≥1 train pair

### Phase 2: Evolutionary Loop ✅ COMPLETE
**Goal**: Implement mutation and selection pressure

- [x] **2.1**: Fitness function evaluation ✅
- [x] **2.2**: Refiner agent for debugging ✅
- [x] **2.3**: Multi-generation evolution loop ✅

### Phase 3: Scaling and Crossover (Planned)
**Goal**: Full ARC dataset with genetic operations

- Solver library (SQLite → Cloud DB)
- Tagger for technique classification
- Crossover agent for capability fusion
- Distributed task queue (Celery/RabbitMQ)

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
# Run all CI checks at once
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

**Quality Tools**:
- **mypy**: Strict type checking (Python 3.13)
- **ruff**: Fast linting and formatting (replaces black, flake8, isort)
- **pytest**: Test framework with coverage reporting
- **bandit**: Security vulnerability scanning

**Pre-commit Hooks** (Optional):

```bash
# Install pre-commit hooks to run checks before each commit
pre-commit install

# Run manually
pre-commit run --all-files
```

**CI/CD Pipeline**:
- Automated checks run on all PRs via GitHub Actions
- Type checking, linting, formatting, security, and tests
- All checks must pass before merging

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

## 📊 Current Progress

```
Phase 1: Core Prototype  ✅ COMPLETE
├── [✅] 1.1: Environment Setup & Data Loading
├── [✅] 1.2: Manual Solver Validation
├── [✅] 1.3: Safe Execution Sandbox
├── [✅] 1.4: LLM Code Generation (gemini-2.5-flash-lite)
└── [✅] 1.5: End-to-End Pipeline

Phase 2: Evolutionary Loop  ✅ COMPLETE
├── [✅] 2.1: Fitness Function Evaluation
├── [✅] 2.2: Refiner Agent (Mutation)
└── [✅] 2.3: Evolution Loop

Tests: 127/127 passing ✅
Demo Phase 1.1: Working with real dataset ✅
Demo Phase 1.2: Manual solver (100% accuracy) ✅
Demo Phase 1.3: Sandbox execution (all demos passed) ✅
Demo Phase 1.4: LLM generation (First Victory achieved!) ✅
Demo Phase 1.5: Complete E2E pipeline (orchestration working!) ✅
Demo Phase 2.1: Fitness evaluation (generalization vs overfitting) ✅
Demo Phase 2.2: Refiner agent (3/3 scenarios improved, +39 total fitness) ✅
Demo Phase 2.3: Evolution loop (multi-generation tracking working!) ✅
```

---

**Next Steps**: Phase 3 - Scaling & Crossover. Build solver database, implement Crossover, scale to full ARC dataset. See [plan_20251024.md](plan_20251024.md) for details.

*"私たちがこれから目の当たりにするのは、AIが「思考」を学ぶ瞬間です。この歴史的な挑戦を、一緒に楽しみましょう！"*

---

## Session Handover

### Last Updated: October 29, 2025 06:23 PM JST

#### Recently Completed
- ✅ **Phase 2.3**: Evolution Loop - Multi-generation Evolution (PR #19 - MERGED!)
  - Implemented complete evolutionary cycle: Generate → Evaluate → Refine → Repeat
  - Created run_evolution_loop() with GenerationResult tracking
  - 13 new tests added (12 comprehensive, 1 edge case), for 129 total passing
  - Demo script with 3 scenarios: simple evolution, early convergence, gradual improvement
  - **Real API Testing**: All 3 demos successful with perfect output quality
    - No timeouts, no truncation, no broken formatting
    - Early termination working (stops when target fitness reached)
    - Per-generation timing and improvement tracking
  - **PR Review Fixes**: Addressed gemini-code-assist and claude feedback
    - Fixed IndexError when max_generations=0 (high priority)
    - Extracted MAX_CODE_DISPLAY_LINES constant (medium priority)
    - Added edge case test for max_generations=0
  - **Code Quality**: All checks passing (mypy, ruff, bandit, pytest)
  - **Phase 2 COMPLETE!** All evolutionary mechanisms working
  - **Time**: ~5 hours total (4 hours implementation + 1 hour review fixes and merge)

- ✅ **Phase 2.2**: Refiner Agent - Code Debugging (PR #17 - MERGED!)
  - Implemented LLM-based code debugging with Gemini API (temperature 0.4)
  - Created refiner prompt template with failure analysis context
  - 12 comprehensive tests (116 total passing: 104 existing + 12 new)
  - Demo script with 3 scenarios: syntax, logic, timeout fixes
  - **Real API Testing**: 3/3 scenarios successful (100% success rate)
    - Syntax error fix: 0 → 13 fitness (+13, missing colon corrected)
    - Logic error fix: 0 → 13 fitness (+13, add→multiply algorithm fixed)
    - Timeout fix: 0 → 13 fitness (+13, infinite loop removed)
  - Total fitness gain: +39 points across all scenarios
  - **Code Quality Improvements**: Addressed all claude-review feedback
    - Added FitnessResult TypedDict for type safety
    - Improved temp file cleanup with fallback logic
    - Made max_examples configurable in prompts
    - Improved error preview with char count (1000 chars with truncation details)
    - Extracted magic numbers to named constants (MAX_ERRORS_TO_SHOW)
    - Clarified prompt output format for code blocks
  - **CI Fixes**: Resolved type annotation differences (local vs CI environment)
    - Used typing.Any for generation_config to avoid environment-dependent type errors
  - All quality checks passing (mypy, ruff, bandit)
  - **First Evolutionary Mechanism (Mutation)**: Automated solver improvement working!
  - **Time**: ~6 hours from TDD to complete implementation, review fixes, and merge
  - **Key Achievement**: Foundation for evolution loop (Phase 2.3) established!

- ✅ **Phase 2.1**: Fitness Function Evaluation (PR #15 - MERGED!)
  - Implemented fitness calculation with 10x weight on test accuracy
  - Formula: `fitness = (train_correct * 1) + (test_correct * 10)`
  - Created new evolutionary_engine package
  - 11 comprehensive tests covering perfect solvers, overfitting, timeouts, errors, missing outputs
  - Demo script demonstrates generalization vs memorization with 3 scenarios
  - All 104 tests passing (93 existing + 11 new)
  - All quality checks passing (mypy, ruff, bandit)
  - **Review Process**: 2 rounds (gemini-code-assist + CodeRabbit), all issues addressed
  - **Key Fixes**: Missing output key handling for ARC evaluation tasks, explicit dtype casting
  - **Time**: ~4 hours from TDD to merge (implementation + testing + review fixes)
  - **Key Achievement**: Foundation for evolutionary loop established!

- ✅ **Phase 1.5**: End-to-End Pipeline (PR #13 - MERGED!)
  - Implemented complete E2E orchestration script `run_phase1_test.py`
  - Command-line interface for testing any ARC task
  - Comprehensive error handling and user-friendly output
  - Automatic solver saving for successful generations
  - 13 new integration tests (93/93 total, 100% pass rate)
  - **Phase 1 Milestone Complete!** 🎉
  - Manual testing: Tested with tasks 00576224, 007bbfb7, 025d127b
  - Robust handling of all failure modes (timeouts, exceptions, mismatches)
  - **Time**: ~4 hours from TDD to completion, testing, review fixes, and merge
  - **PR Review**: Addressed all 5 code quality issues (encoding, constants, safety checks, test duplication)

- ✅ **Phase 1.4**: LLM Code Generation (PR #11 - MERGED!)
  - Implemented Gemini API integration with gemini-2.5-flash-lite (latest, fastest model)
  - Created robust multi-strategy code parser (markdown → raw code → fallback)
  - Temperature optimization: 0.3 for consistent code generation (research-backed)
  - Unified Analyst+Programmer prompt design
  - Security warnings added to demo script (sandbox limitations)
  - 14 parser tests + 6 integration tests = 20 new tests (80/80 total, 100% pass rate)
  - **First Victory**: AI-generated code solves ARC train pairs!
  - Testing results: 100% on task 00576224 (2/2), graceful failure handling on task 007bbfb7
  - **Review Process**: 3 rounds of improvements, all critical/high/medium priority issues addressed
  - **Time**: ~6 hours from Oct 28 (implementation + comprehensive review resolution)

- ✅ **Phase 1.3**: Safe Execution Sandbox (PR #9 merged)
  - Implemented multiprocessing-based sandbox with timeout enforcement (5s default)
  - Critical security fixes: Builtins bypass prevention via sys.modules replacement
  - Queue reliability: Replaced unreliable empty() with get_nowait()
  - Error visibility: Added stderr logging for debugging
  - Code quality: Extracted DANGEROUS_BUILTINS as module constant
  - 23 new tests added (60/60 tests passing, 100% success rate)
  - Demo script validates 3 scenarios: successful execution, timeout, exception handling
  - Comprehensive PR review addressed: 5 reviewers, all critical/high/medium issues fixed
  - **Time**: 1 day from Oct 27-28 with systematic security hardening

- ✅ **CI/CD Pipeline**: Comprehensive quality tooling (PR #7 merged)
  - Implemented mypy, ruff, pytest, bandit with strict configurations
  - Created Makefile with 11 commands (ci, test, typecheck, lint, format, security, etc.)
  - Set up GitHub Actions workflow for automated PR checks
  - All 37 tests passing with zero regressions

- ✅ **Phase 1.2**: Manual Solver Validation (PR #5 merged)
  - Implemented solver for ARC task 05269061 (diagonal pattern extraction)
  - 100% success rate: all 3 train examples solved correctly
  - 14 new tests added, TDD approach

#### Next Priority Tasks
1. **Phase 3.1: Solver Library (SQLite)** ⭐ NEXT
   - Source: plan_20251024.md Phase 3.1, README line 295-299
   - Context: Persist successful solvers for reuse and analysis
   - Approach: Create SQLite database schema → CRUD operations → Query by fitness/tags
   - Schema: solver_id, task_id, generation, code, fitness, train_correct, test_correct, parent_id
   - Goal: Enable cross-task solver reuse and historical analysis

2. **Phase 3.2: Tagger Agent** (after 3.1)
   - Source: plan_20251024.md Phase 3.2, README line 296
   - Context: Classify solver techniques for crossover selection
   - Approach: LLM analyzes code → Identifies techniques → Tags (rotation, fill, symmetry, etc.)
   - Integration: Store tags in solver library
   - Goal: Enable intelligent crossover by technique combination

#### Known Issues / Blockers
- None - Phase 2 complete! Ready for Phase 3 (Solver Library + Crossover)

#### Session Learnings
- **Code Quality Constants Extraction** (Phase 1.5 PR Review): Extract magic numbers to named constants at module level (`SANDBOX_TIMEOUT = 5`, `PREVIEW_START_LINES = 20`). Improves maintainability and makes configuration explicit. Applied to timeout values and display parameters.
- **Empty Grid Safety Checks** (Phase 1.5 PR Review): Always validate `result_grid.size > 0` before array operations like `np.argwhere()`. Empty grids can cause IndexError. Add explicit `len(diff_positions) > 0` checks and use `int()` conversions for safety.
- **Pytest Fixture for Path Deduplication** (Phase 1.5 PR Review): When multiple tests construct same path, extract to `@pytest.fixture`. Eliminates 6-line duplication across tests and provides single source of truth.
- **File Encoding Specification** (Phase 1.5 PR Review): Always specify `encoding="utf-8"` in `open()` for cross-platform compatibility. Platform default encodings vary (Windows cp1252, Linux/Mac UTF-8).
- **E2E Pipeline Orchestration** (Phase 1.5): Chain components with comprehensive error handling at each stage. Show progressive feedback (load → generate → execute → evaluate). Never let early failures prevent reporting partial results.
- **User-Friendly CLI Output** (Phase 1.5): Use consistent visual formatting (═ ─ ✅ ❌ ⚠️) and progressive disclosure. Show what's happening at each step with timing info for transparency.
- **Solver Persistence Strategy** (Phase 1.5): Save generated code even with partial success (≥1 correct) - valuable for Phase 2 mutation and debugging. Include metadata (task_id, success_rate, timestamp) in saved files.
- **LLM Variability Acceptance** (Phase 1.5): LLM code generation is inherently variable. Build robust pipelines that handle all scenarios: success, syntax errors, runtime exceptions, incomplete code, constraint violations (e.g., scipy import). The pipeline's robustness matters more than individual LLM success.
- **Mock Patch at Import Level** (Phase 1.4): When mocking functions in tests, patch where they're imported, not where defined. `@patch("programmer.get_api_key")` not `@patch("utils.config.get_api_key")`. Mock must be active before function execution.
- **Temperature for Code Generation** (Phase 1.4): Research shows lower temps (0.2-0.3) produce more consistent, deterministic code than default 0.7. LLMs with high temperature generate creative but unreliable code.
- **Multi-Strategy Code Parsing** (Phase 1.4): Chain extraction strategies with fallbacks: Strategy 1 (markdown ``` blocks) → Strategy 2 (indentation-based raw code) → Strategy 3 (error). Indentation-based detection more robust than keyword matching.
- **Security Warnings in User Tools** (Phase 1.4): Always warn users about limitations in user-facing tools. Demo script now includes clear warning that multiprocessing sandbox doesn't prevent filesystem/network access (Docker needed for production).
- **Enhanced Error Messages** (Phase 1.4): Show first 300 + last 200 chars of response (not just first 500) for better debugging when parsing fails. Helps identify if issue is at start or end of response.
- **Decorator Detection in Code Parser** (Phase 1.4): Add "@" to Python statement detection to prevent premature termination when decorators present before function definitions.
- **Builtins Bypass Security** (Phase 1.3): Restricting `__builtins__` dict insufficient - code can bypass via `import builtins; builtins.eval(...)`. Must replace `sys.modules["builtins"]` with restricted module. Added test to verify bypass prevention.
- **Queue.empty() Unreliability** (Phase 1.3): `multiprocessing.Queue.empty()` has race conditions - feeder thread may not have transferred data when checked. Always use `get_nowait()` with try/except instead.
- **Security as Module Constants** (Phase 1.3): Extract security restrictions (dangerous functions, blocked keywords) as module-level frozenset constants for maintainability and clarity.
- **Systematic PR Review** (Phase 1.3): /fix_pr_graphql command successfully addressed all 5 reviewers' feedback systematically - critical security issues fixed, all CI checks passing.
- **AI Code Reviewer Verification** (PR #7): gemini-code-assist claimed all pre-commit hook versions were invalid (v0.14.2, v1.18.2, 1.8.0, v5.0.0), but GitHub API verification showed all versions exist. Always verify factual claims before accepting reviewer feedback. Correctness > Compliance.
- **Type Checking Environment Differences** (PR #7): CI has numpy type stubs (np.array_equal returns bool), local doesn't (returns Any). Using `cast(bool, ...)` triggered "redundant cast" error in CI. Solution: `bool(...)` works in both environments.
- **Dependency Verification** (PR #7): Always verify package existence before adding to dependencies. `types-python-dotenv` doesn't exist - python-dotenv doesn't provide type stubs. Use `ignore_missing_imports = true` in mypy instead.
- **CI/CD Failure Iteration** (PR #7): Systematic approach - fetch logs → identify exact error → verify locally → fix → test with `make ci` → push → monitor. GraphQL fetches ALL PR feedback in single query.
- **Critical Thinking in Code Review** (Phase 1.2): When user instructed "don't blindly trust the review", successfully identified that reviewer's "Critical" label on hardcoded 7x7 grid size was incorrect - task 05269061 always uses 7x7 grids, and Phase 1.2's goal is task-specific validation (not general-purpose solver). Added documentation clarification instead of unnecessary refactoring.
- **False Test Coverage Detection** (Phase 1.2): CodeRabbitAI identified critical issue where tests claimed 36/36 passing but most were placeholder `pass` statements that didn't actually validate anything. Complete rewrite added real assertions and integration tests.
- **Diagonal Pattern Recognition** (Phase 1.2): ARC task 05269061 requires grouping values by diagonal index (`row + col`), then applying rotation rules based on diagonal position (consecutive vs non-consecutive, top-left vs bottom-right). Iterative pattern analysis with real data was essential.
- **TDD Iteration Speed** (Phase 1.2): Writing tests first accelerated debugging - pattern mismatches were immediately visible, leading to 5 refinement cycles before achieving 100% accuracy.
- **Manual Solver as Validation** (Phase 1.2): Implementing a working solver before LLM integration validates that:
  1. The task is solvable with pure numpy
  2. `evaluate_grids()` correctly identifies matches
  3. The solver signature (`def solve(np.ndarray) -> np.ndarray`) works as designed
- **Lazy Validation Pattern** (Phase 1.1): API key validation converted from eager to lazy - enables Phase 1.1-1.2 features to work without Gemini API setup.
- **Enhanced load_task()** (Phase 1.1): Added optional `task_id` parameter to load from collection files directly.
- **ARC Evaluation Format Handling** (Phase 2.1): Test examples in ARC evaluation tasks often lack "output" keys. Always check `if "output" not in example:` before accessing. Skip invalid examples gracefully and only count evaluable examples in train_total/test_total.
- **Explicit NumPy dtype Casting** (Phase 2.1): Always specify `dtype=np.int64` in `np.array()` calls to ensure type consistency across grid operations. Prevents subtle type mismatches in grid comparisons.
- **Fitness Function 10x Weight Rationale** (Phase 2.1): Test accuracy weighted 10x higher than train (`fitness = train_correct * 1 + test_correct * 10`) ensures evolutionary pressure favors generalization over memorization. A solver with 100% train but 0% test (pure overfitting) scores only 3 points vs solver with partial generalization scoring higher.
- **Test Count Validation in CI** (Phase 2.1): Update test count assertions in README after adding new tests. Changed from `==` to `<=` for test_total when handling variable test example counts (due to missing outputs).
- **TypedDict for Type Safety** (Phase 2.2): Define structured dict types with TypedDict instead of generic `dict[str, Any]`. Provides better IDE autocomplete and type checking. Example: `FitnessResult` TypedDict for fitness evaluation results.
- **Type Annotation Environment Differences** (Phase 2.2): CI and local environments may have different type stub availability. When type checking passes locally but fails in CI, use `typing.Any` for parameters with environment-dependent type information. Avoids `type: ignore` comments that appear "unused" locally but are needed in CI.
- **Prompt Output Format Clarity** (Phase 2.2): LLM prompts should acknowledge parser flexibility. Instead of "Do NOT include code blocks", use "You may optionally wrap code in ```python blocks, but raw code is preferred". Aligns prompts with actual parser behavior that handles both formats.
- **Resource Cleanup Fallback Pattern** (Phase 2.2): When cleaning up temp files, provide fallback logic in finally blocks. Check both the intended variable and the original file handle to ensure cleanup even if early exceptions occur.
- **Systematic Code Review Addressing** (Phase 2.2): Prioritize review feedback by severity (Critical → Medium → Low). Address medium and critical issues immediately. Document low-priority issues for follow-up work. Used TodoWrite tool to track 7 review issues systematically.
- **CI Type Error Resolution** (Phase 2.2): When mypy passes locally but fails in CI due to type stubs, check if the library has different stub availability. Use `typing.Any` for cross-environment compatibility instead of platform-specific type annotations.
