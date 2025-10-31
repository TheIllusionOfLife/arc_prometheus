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
├── tests/                  # Test suite (267 tests passing)
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

**Phase 3: AI Civilization** 🚧 IN PROGRESS
- ✅ **Task 3.1** (October 31, 2025): Analyst agent - Pattern analysis and rule inference
  - 21 unit tests + 9 integration tests (all passing)
  - Real API validation: 5/5 tasks completed successfully
  - Integration: Programmer accepts Analyst specifications (backward compatible)
- ⏭️ **Task 3.2**: Enhanced Programmer (prompt optimization)
- ⏭️ **Task 3.3**: Refiner with Analyst context
- ⏭️ **Task 3.4**: Tagger agent (technique classification)
- ⏭️ **Task 3.5**: Crossover agent (solution fusion)
- ⏭️ **Task 3.6**: Population-based evolution

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

### Last Updated: October 31, 2025 10:30 PM JST

#### Recently Completed

**Task 1: Fix Data Pipeline** ([PR #33](https://github.com/TheIllusionOfLife/arc_prometheus/pull/33) - October 31, 2025):
- Created preprocessing script to merge evaluation challenges + solutions
- 5 comprehensive tests (261 total passing), validated with 120 real tasks
- **Impact**: Unblocked competition submission workflow - can now benchmark on evaluation dataset
- See: `scripts/prepare_evaluation_data.py`

**Phase 2 Benchmarking** ([PR #31](https://github.com/TheIllusionOfLife/arc_prometheus/pull/31) - October 31, 2025):
- 17 new tests (267 total passing), production-ready infrastructure
- **Critical Discovery**: 20% success rate (3/15 tasks), 82% logic errors
- **Impact**: Must fix Programmer/Refiner before Phase 3 (saved 6+ weeks)
- See: [docs/benchmarks/phase2_findings.md](docs/benchmarks/phase2_findings.md)

**Docker Sandbox** ([PR #28](https://github.com/TheIllusionOfLife/arc_prometheus/pull/28) - October 30, 2025):
- Production-grade security with ExecutionEnvironment protocol
- Network disabled, read-only filesystem, resource limits
- CLI: `--sandbox-mode docker`

**Error Classification** ([PR #26](https://github.com/TheIllusionOfLife/arc_prometheus/pull/26) - October 30, 2025):
- ErrorType enum (SYNTAX, RUNTIME, TIMEOUT, LOGIC, VALIDATION)
- 3-tuple return: `(success, result, error_detail)`
- Enables targeted Refiner debugging

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

1. **Analyst Agent** (Day 1, 6-8 hours)
   - **Why**: Separates pattern understanding from code generation
   - **Approach**: Analyzes task examples, infers transformation rules in natural language, generates specification for Programmer
   - **Impact**: Abstracts reasoning from implementation
   - **Deliverables**: `cognitive_cells/analyst.py`, 15+ tests

2. **Enhanced Programmer** (Day 1, 2-3 hours)
   - **Why**: Accept Analyst specifications as guidance
   - **Approach**: Extend to take analyst_spec parameter, generate code from specifications
   - **Deliverables**: Updated `cognitive_cells/programmer.py`

3. **Tagger Agent** (Day 2, 4-5 hours)
   - **Why**: Classify techniques for crossover selection
   - **Approach**: Static + LLM analysis to identify techniques (rotation, fill, symmetry, etc.)
   - **Deliverables**: `cognitive_cells/tagger.py`, technique taxonomy, 10+ tests

4. **Crossover Agent** (Day 2-3, 8-10 hours) ⭐ CRITICAL
   - **Why**: The unique differentiator - fuse techniques from different solvers
   - **Approach**: Select complementary parents, LLM-guided technique fusion, create novel solvers
   - **Impact**: Creates solutions that didn't exist in training data
   - **Deliverables**: `cognitive_cells/crossover.py`, parent selection strategies, 12+ tests

5. **Population-Based Evolution** (Day 3, 6-8 hours)
   - **Why**: Move from single-lineage to true genetic algorithm
   - **Approach**: Tournament selection, crossover breeding, mutation via Refiner, fitness-based survival
   - **Impact**: Parallel exploration of solution space
   - **Deliverables**: `evolutionary_engine/population_evolution.py`, 15+ tests

6. **Kaggle Baseline Submission** (Day 3, 4-6 hours)
   - **Purpose**: Submit pure AI Civilization to competition
   - **Expected**: 15-25% score (competitive with pure LLM approaches)
   - **Deliverables**: `kaggle_baseline_civilization.ipynb`, leaderboard score, analysis

**Experimental Variations (Optional, Days 4-5):**

7. **Experiment 1: Single Model Active Inference** (4-6 hours)
   - Distill population outputs → Gemma 2 27B
   - Implement active inference (fine-tune on test examples at runtime)
   - **Measure**: Value-add from active inference alone

8. **Experiment 2: Multi-Agent Active Inference** (6-8 hours)
   - Each agent does active inference separately
   - **Measure**: Multi-agent vs single-model active inference

**Research Hypothesis:**
- H1: Multi-agent AI Civilization with crossover outperforms single-model approaches
- H2: Active inference provides additional benefit
- H3: Multi-agent active inference > single-model active inference

**Success Criteria:**
- [ ] All 5 agents implemented (Analyst, Programmer, Refiner, Tagger, Crossover)
- [ ] Population-based evolution working
- [ ] 60+ new tests passing
- [ ] Baseline Kaggle submission: 15-25% score
- [ ] Research question answered: Can AI Civilization solve novel problems competitively?

**Competition Context:**
- Deadline: November 3, 2025 (3 days)
- Hardware: Kaggle L4x4 GPUs (96GB memory)
- Constraint: 12-hour runtime, no internet access
- Format: pass@2 submission (2 diverse attempts per test input)

#### Known Issues / Blockers
- ✅ **RESOLVED - Security**: Docker Sandbox now available for production-grade security
  - **Status**: Task 2.1 complete ([PR #28](https://github.com/TheIllusionOfLife/arc_prometheus/pull/28) merged October 30, 2025)
  - **Usage**: Use `--sandbox-mode docker` flag for production deployments
  - **Note**: Multiprocessing sandbox remains default for fast local development
  - **Security**: Docker provides network isolation, read-only filesystem, and resource limits

#### Session Learnings (Most Recent)

**From Planning Session (plan_20251031.md) - October 31, 2025 10:30 PM JST**:
- ✅ **COMPLETE**: Comprehensive Phase 3 plan created with experimental validation strategy
- **Core Vision Preservation**: Initial plan lost AI Civilization concept by focusing solely on SOTA techniques (single-model active inference). User correctly challenged this - project exists to validate multi-agent evolution hypothesis, not just maximize leaderboard score
- **Experimental Design**: Baseline (pure civilization) → Exp1 (+ single-model active inference) → Exp2 (+ multi-agent active inference). Clean measurement of each component's value-add
- **Active Inference Discovery**: Jack Cole's 34% SOTA uses fine-tuning on each test task's 3 training examples at runtime (not static pre-training). Requires example expansion (3 → 30+) via augmentation
- **Hardware Advantage**: Kaggle L4x4 offers 96GB GPU (vs 15GB T4) - enables larger models (13B-27B) and on-the-fly fine-tuning
- **François's Recommendation**: "Augment discrete program search with deep learning driven intuition" - our multi-agent approach aligns well (LLMs provide intuition, population evolution does search)
- **Research First, Competition Second**: Primary goal is validating AI Civilization hypothesis. Competition provides fair benchmarks and constraints, but success = proving the approach works, not just leaderboard rank
- **Crossover as Unique Differentiator**: Top approaches (J. Berman, E. Pang) use single models. Our Crossover agent can fuse techniques that never appeared together - true genetic innovation vs pattern matching

**From Task 2 (pass@2 Submission Format) - October 31, 2025 03:28 PM JST**:
- ✅ **COMPLETE**: Implemented pass@2 submission format for Kaggle (PR #35)
- **Diversity Selection**: 3 strategies (fitness, generation_gap, edit_distance placeholder)
- **Prediction Generation**: Handles variable test inputs per task (0-3), sandbox factory pattern
- **Format Validation**: Automated validation against Kaggle structure, JSON serialization checks
- **Fallback Behavior**: Duplicates best solver with console warnings when insufficient unique solvers
- **Testing**: 22 comprehensive tests (18 unit + 4 integration), real API validation (3 tasks, 100% success)
- **Quality Metrics**: 283 total tests passing, mypy clean, ruff clean, full CI passing
- **Review Iterations**: Multiple rounds of AI-assisted review - all feedback addressed
  - Fixed dynamic num_attempts parameter (3 HIGH bugs)
  - Fixed sandbox_mode wiring (MEDIUM bug)
  - Fixed hardcoded 2x2 placeholder with dynamic sizing
  - Added console warnings for fallback
  - Validated empty solver codes
  - Fixed misleading docstring

**Critical Learnings from PR #35**:
- **Systematic PR Review**: Use GraphQL single query for ALL feedback sources (comments + reviews + line comments)
- **Mandatory Verification Checklist**: 5-item checklist prevents missing feedback (count match, timestamps, author comments, review content, CI passing)
- **Post-Fix Verification**: Always run type check + targeted test before declaring "fix complete" (30s prevents hours of CI debugging)
- **Priority-Based Fixing**: Critical → High → Medium → Low, commit after each group, single push to save bot costs
- **Review State ≠ Content**: APPROVED can still contain improvement suggestions - always read the actual comment body
- **Fallback Architecture**: Raise ValueError in library function, catch and handle in caller with warnings (cleaner separation of concerns)

**From Task 1: Fix Data Pipeline (PR #33) - October 31, 2025**:
- **TDD with Real-World Validation**: Unit tests catch logic bugs, real data catches format assumptions. Always validate with production data (120 evaluation tasks) before merge
- **Set Operations for Efficient Validation**: Use `set(challenges) - set(solutions)` instead of nested loops - cleaner code, better performance (O(N) vs O(N²))
- **Deep Copy vs Shallow Copy**: When modifying nested structures (dicts with lists), use `copy.deepcopy()` to prevent side effects on original data
- **Single-Push Multi-Commit Strategy**: Commit locally after each fix, push once at end to save CI/bot costs. Example: 3 commits addressing different feedback, 1 push triggers bots once
- **Competition Data Format Preprocessing**: Split datasets (challenges + solutions) require preprocessing script before benchmarking. Validate data integrity with set operations first

**From Competitive Analysis & Philosophy Clarification - October 31, 2025**:
- **Competition as Testbed, Not Goal**: We're building an AI civilization to validate multi-agent evolution. Competition provides benchmarks and constraints, but doesn't dictate architecture. Chasing leaderboard rankings = cart before the horse.
- **Multi-Agent vs Single-Agent**: Top performers (J. Berman, E. Pang) are single-agent systems with clever tricks. Our hypothesis: specialized agents (Analyst, Programmer, Refiner, Tagger, Crossover) + genetic evolution = emergent intelligence that surpasses single models.
- **Crossover as Key Differentiator**: E. Pang uses library lookup (pattern matching). We use Crossover (technique fusion) - can invent solutions by combining capabilities that never appeared together. This is true genetic innovation.
- **Validate Against Competition Metric**: PR #31 benchmarked training data without test outputs → measured memorization not generalization. Always check: What metric? What dataset? What's the submission format?
- **pass@2 Requirement**: Kaggle requires 2 diverse attempts per test. Score = 1 if either matches ground truth. Must implement for submission compatibility.

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
