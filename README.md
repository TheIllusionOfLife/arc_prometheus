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
│   │   ├── prompts.py      # (Phase 1.4) Prompt templates
│   │   └── programmer.py   # (Phase 1.4) Code generation
│   └── utils/
│       └── config.py       # Configuration management
├── tests/                  # Test suite (59 tests passing)
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

### Phase 2: Evolutionary Loop (Planned)
**Goal**: Implement mutation and selection pressure

- Fitness function: `(train_correct * 1) + (test_correct * 10)`
- Refiner agent for debugging
- Multi-generation evolution
- Test accuracy tracking

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

Tests: 93/93 passing ✅
Demo Phase 1.1: Working with real dataset ✅
Demo Phase 1.2: Manual solver (100% accuracy) ✅
Demo Phase 1.3: Sandbox execution (all demos passed) ✅
Demo Phase 1.4: LLM generation (First Victory achieved!) ✅
Phase 1.5: Complete E2E pipeline (orchestration working!) ✅
```

---

**Next Steps**: Phase 2 - Evolutionary Loop (fitness function, refiner agent, mutation). See [plan_20251024.md](plan_20251024.md) for details.

*"私たちがこれから目の当たりにするのは、AIが「思考」を学ぶ瞬間です。この歴史的な挑戦を、一緒に楽しみましょう！"*

---

## Session Handover

### Last Updated: October 28, 2025 07:45 PM JST

#### Recently Completed
- ✅ **Phase 1.5**: End-to-End Pipeline (PR #13 - READY FOR REVIEW!)
  - Implemented complete E2E orchestration script `run_phase1_test.py`
  - Command-line interface for testing any ARC task
  - Comprehensive error handling and user-friendly output
  - Automatic solver saving for successful generations
  - 13 new integration tests (93/93 total, 100% pass rate)
  - **Phase 1 Milestone Complete!** 🎉
  - Manual testing: Tested with tasks 00576224, 007bbfb7, 025d127b
  - Robust handling of all failure modes (timeouts, exceptions, mismatches)
  - **Time**: ~3 hours from TDD to completion with real user testing

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
1. **Phase 2.1: Fitness Function Implementation** ⭐ NEXT
   - Source: README Phase 2 section, plan_20251024.md Phase 2.1
   - Context: Begin evolutionary loop with fitness evaluation
   - Approach: Implement `calculate_fitness(solver_code, task_json_path) -> dict`
   - Formula: `fitness = (train_correct * 1) + (test_correct * 10)`
   - Goal: Prioritize generalization over memorization

2. **Phase 2.2: Refiner Agent (Mutation)**
   - Source: kickoff.md (line 46), plan_20251024.md Phase 2.2
   - Context: Debug and improve failed solver code
   - Approach: Create Refiner prompt, integrate error context, test with buggy solvers
   - Goal: Automated solver improvement through mutation

#### Known Issues / Blockers
- None - Phase 1 complete! All infrastructure ready for Phase 2 evolutionary loop

#### Session Learnings
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
