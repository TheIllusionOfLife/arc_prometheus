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

### Benchmarking (Real-World Testing)

Run evolution loop on diverse ARC tasks to measure performance and validate Phase 2:

```bash
# Benchmark specific tasks
python scripts/benchmark_evolution.py \
  --tasks "00576224,007bbfb7,025d127b" \
  --output-dir results/test_run/ \
  --experiment-name "test_run"

# Random sample from training set
python scripts/benchmark_evolution.py \
  --random-sample 15 \
  --training-data data/arc-prize-2025/arc-agi_training_challenges.json \
  --output-dir results/baseline/ \
  --experiment-name "baseline"

# Load tasks from file
python scripts/benchmark_evolution.py \
  --task-ids-file benchmark_tasks.txt \
  --output-dir results/multiprocess_baseline/ \
  --experiment-name "multiprocess_baseline"

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

**Benchmark Output Structure:**
```
results/{experiment_name}/
├── metadata.json              # Experiment config, timestamp, git commit
├── task_{task_id}.json        # Individual task results
├── summary.json               # Aggregate statistics
```

**Phase 2 Baseline Results** (October 30, 2025):
- **Tasks**: 15 diverse ARC tasks
- **Success Rate**: 100% (no crashes)
- **Average Fitness**: 0.33 (only 20% of tasks achieved fitness > 0)
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
├── tests/                  # Test suite (127 tests passing)
├── scripts/                # Demo and utility scripts
├── data/                   # ARC dataset (gitignored)
└── plan_20251024.md       # Detailed implementation plan
```

## 🔬 Development Status

**Phase 1: Core Prototype** ✅ COMPLETE
- Data loading, sandbox, LLM generation, E2E pipeline

**Phase 2: Evolutionary Loop** ✅ COMPLETE
- Fitness evaluation, Refiner agent, multi-generation evolution
- **Current Issue**: 20% success rate → Must fix Programmer/Refiner

**Phase 3: Scaling** (Planned)
- Solver library, Tagger, Crossover, distributed processing

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

---

**Tests**: 267 passing ✅ | **Next**: Fix Programmer/Refiner prompts (PR #31 findings) → Phase 3 Scaling

---

## Session Handover

### Last Updated: October 31, 2025 12:29 AM JST

#### Recently Completed

**Phase 2 Benchmarking** (PR #31 - October 31, 2025):
- 17 new tests (267 total passing), production-ready infrastructure
- **Critical Discovery**: 20% success rate (3/15 tasks), 82% logic errors
- **Impact**: Must fix Programmer/Refiner before Phase 3 (saved 6+ weeks)
- See: [docs/benchmarks/phase2_findings.md](docs/benchmarks/phase2_findings.md)

**Docker Sandbox** (PR #28 - October 30, 2025):
- Production-grade security with ExecutionEnvironment protocol
- Network disabled, read-only filesystem, resource limits
- CLI: `--sandbox-mode docker`

**Error Classification** (PR #26 - October 30, 2025):
- ErrorType enum (SYNTAX, RUNTIME, TIMEOUT, LOGIC, VALIDATION)
- 3-tuple return: `(success, result, error_detail)`
- Enables targeted Refiner debugging

#### Competitive Context (ARC-AGI-2 Leaderboard)

**What We're Up Against**: ARC-AGI-2 is extremely challenging - top AI systems score in single digits.

**Current Leaders** (October 2025):
- 🥇 **J. Berman**: 29.4% at $30.40/task - Instruction generation + nested LLM calls
- 🥈 **E. Pang**: 26.0% at $3.97/task - **Code + Program Library** (DreamCoder-inspired)
- 🥉 **GPT-5 Pro**: 18.3% at $7.14/task - Pure CoT reasoning
- **Claude Sonnet 4.5**: 13.6% at $0.759/task - Extended thinking
- **Gemini 2.5 Pro**: 4.9% at $0.767/task - Our base model's cousin
- **Humans**: 60% average (100% at $17/task)
- **Competition Target**: 85% at $0.42/task

**Critical Insight**: E. Pang's 26% approach uses:
1. ✅ Python code generation (like us!)
2. ✅ **Program library** with cross-task learning
3. ✅ Score-weighted sampling for 2 diverse attempts (pass@2)
4. ✅ Shows output differences in prompts (like our error details)

**Our Gap**: Missing pass@2 support and program library for cross-task learning.

#### Next Priority Tasks

**Strategy Pivot**: Leaderboard analysis reveals E. Pang's winning approach perfectly aligns with our Phase 3 plan! But we have critical evaluation gaps first.

**Phase 2.5: Critical Kaggle Requirements** (Week 1-2) ⚡ BLOCKS SUBMISSION:

1. **Implement pass@2 Output** ⭐ CRITICAL (2-3 days)
   - **Why**: Kaggle requires 2 attempts per test input (pass@2 metric)
   - **Currently**: We generate 1 output, cannot submit to competition
   - **Approach**:
     - Generate 2 diverse attempts with temperature/prompt variation
     - Output `submission.json` format: `{"task_id": [{"attempt_1": [[...]], "attempt_2": [[...]]}]}`
     - Implement diversity penalty or sample from generation history
   - **Success**: Can submit to Kaggle public leaderboard

2. **Fix Benchmark Evaluation** ⭐ CRITICAL (1 day)
   - **Why**: PR #31 benchmarked training data WITHOUT test outputs
   - **Currently**: Our "20%" measured train memorization, NOT generalization
   - **Approach**:
     - Merge `training_challenges.json` + `training_solutions.json`
     - Properly evaluate train→test generalization
     - Measure actual pass@2 score on test examples
   - **Success**: Know if we're at 5% or 25% on actual test performance

3. **Baseline Kaggle Submission** ⭐ HIGH (1 day)
   - **Why**: Ground truth on competitiveness vs SOTA
   - **Approach**:
     - Submit current system to public leaderboard
     - Compare against Claude (13.6%), Gemini (4.9%)
     - Adjust Phase 3 strategy based on actual score
   - **Decision Point**: If >10% → proceed Phase 3, if <5% → debug Programmer

**Phase 3: E. Pang's Library Strategy** (Week 3-5) 📚:

4. **Program Library + Cross-Task Learning** (3-4 days)
   - **Why**: E. Pang's 26% proves library >> prompt engineering
   - **Approach**:
     - SQLite storage for successful solvers (originally planned!)
     - Score-weighted retrieval (softmax sampling like E. Pang)
     - Library-augmented prompts: "Here are solvers from similar tasks..."
   - **Impact**: Cross-task knowledge transfer (E. Pang's key advantage)

5. **Tagger for Similarity Retrieval** (2-3 days)
   - **Why**: Query library for relevant patterns
   - **Approach**:
     - LLM tags techniques (rotation, fill, symmetry, pattern_matching)
     - Retrieve 2-3 most relevant programs per task
     - Include in Programmer/Refiner prompts
   - **Dependencies**: Program library (Task 4)

#### Known Issues / Blockers
- ✅ **RESOLVED - Security**: Docker Sandbox now available for production-grade security
  - **Status**: Task 2.1 complete (PR #28 merged October 30, 2025)
  - **Usage**: Use `--sandbox-mode docker` flag for production deployments
  - **Note**: Multiprocessing sandbox remains default for fast local development
  - **Security**: Docker provides network isolation, read-only filesystem, and resource limits

#### Session Learnings (Most Recent)

**From Competitive Analysis - October 31, 2025**:
- **Validate Against Competition Metric FIRST**: PR #31 benchmarked training data without test outputs → measured memorization not generalization. Always check: What metric? What dataset? What's the submission format?
- **Leaderboard Analysis Before Strategy**: E. Pang's 26% uses program library + pass@2 (exactly our Phase 3 plan!). Top approaches guide architecture choices better than intuition.
- **pass@2 Requirement**: Kaggle requires 2 diverse attempts per test. Score = 1 if either matches ground truth. Must implement before any submission.

**From PR #31 (Benchmarking) - October 31, 2025**:
- **Iterative Multi-Review PR Workflow**: Address Critical → High → Medium → Low priority systematically. Quick wins (5-10min) build reviewer trust and prevent follow-up reviews
- **CLI Flag Wiring Bug Pattern**: Thread parameters through ALL execution layers. Validate with targeted tests. Example: `--sandbox-mode docker` accepted but ignored until wired through benchmark → evolution_loop → calculate_fitness

**From Task 2.1 (Docker Sandbox) - October 30, 2025**:
- **ExecutionEnvironment Protocol**: Use Protocol for pluggable backends (multiprocess, docker) with zero coupling
- **Docker Security Layered Defense**: Network disabled + read-only filesystem + tmpfs /tmp + resource limits + non-root user
- **Safe Serialization for Untrusted Code**: NEVER pickle from containers running LLM-generated code. Use JSON. Fixed RCE vulnerability (PR #30)

**From Task 1.3 (Error Classification) - October 30, 2025**:
- **Error Classification Architecture**: ErrorType enum (SYNTAX, RUNTIME, TIMEOUT, LOGIC, VALIDATION) enables targeted Refiner debugging
- **3-Tuple Error Propagation**: Sandbox returns `(success, result, error_detail)` with structured error info
- **GraphQL for Complete PR Feedback**: Single query fetches ALL feedback (comments + reviews + line comments + CI)
- **Review State vs Content**: Never rely on APPROVED state - always read actual comment content

**Critical Patterns**:
- **Type Checking Environment Differences**: Use `bool(...)` not `cast(bool, ...)` for CI/local compatibility
- **Mock Patch at Import Level**: Patch where imported (`@patch("programmer.get_api_key")`) not where defined
- **Temperature for Code Generation**: Lower temps (0.2-0.3) produce more consistent code than default 0.7
- **Fitness 10x Weight Rationale**: `fitness = train_correct * 1 + test_correct * 10` prioritizes generalization over memorization
