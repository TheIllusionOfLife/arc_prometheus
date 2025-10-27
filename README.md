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
│   │   └── sandbox.py      # (Phase 1.5) Safe code execution
│   ├── cognitive_cells/    # LLM agents
│   │   ├── prompts.py      # (Phase 1.4) Prompt templates
│   │   └── programmer.py   # (Phase 1.4) Code generation
│   └── utils/
│       └── config.py       # Configuration management
├── tests/                  # Test suite
├── scripts/                # Demo and utility scripts
├── data/                   # ARC dataset (gitignored)
└── plan_20251024.md       # Detailed implementation plan
```

## 🔬 Development Phases

### Phase 1: Core Prototype (Current)
**Goal**: Build minimal ecosystem for end-to-end ARC task solving

- [x] **1.1**: Environment setup + data loading ✅
- [ ] **1.2**: Manual solver validation
- [ ] **1.3**: Safe execution sandbox
- [ ] **1.4**: LLM-based code generation
- [ ] **1.5**: Complete end-to-end pipeline

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

```bash
# Type checking (future)
mypy src/

# Linting (future)
ruff check src/
```

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
Phase 1: Core Prototype
├── [✅] 1.1: Environment Setup & Data Loading
├── [ ] 1.2: Manual Solver Validation
├── [ ] 1.3: Safe Execution Sandbox
├── [ ] 1.4: LLM Code Generation
└── [ ] 1.5: End-to-End Pipeline

Tests: 23/23 passing ✅
Demo: Working with real dataset ✅
```

---

**Next Steps**: See [plan_20251024.md](plan_20251024.md) for detailed Phase 1.2-1.5 tasks.

*"私たちがこれから目の当たりにするのは、AIが「思考」を学ぶ瞬間です。この歴史的な挑戦を、一緒に楽しみましょう！"*

---

## Session Handover

### Last Updated: October 27, 2025 01:15 AM JST

#### Recently Completed
- ✅ **#3**: Phase 1.1 - Foundation and Data Infrastructure
  - Project structure with modern Python packaging (pyproject.toml)
  - ARC dataset loading with task_id parameter support
  - Grid evaluation and colorized terminal visualization
  - 23 comprehensive tests (TDD approach)
  - Configuration management with lazy API key validation
  - Demo script working with real ARC Prize 2025 dataset
- ✅ **#2**: CLAUDE.md for AI development guidance
- ✅ **Review feedback addressed**: All bot review comments from PR #3 systematically resolved

#### Next Priority Tasks
1. **Phase 1.2: Manual Solver Validation**
   - Source: plan_20251024.md (lines 205-249)
   - Context: Verify that manually-written solvers can solve train pairs before building LLM pipeline
   - Approach: Create example solvers for simple ARC tasks, test with crucible/evaluator.py

2. **Phase 1.3: Safe Execution Sandbox**
   - Source: plan_20251024.md (lines 250-286)
   - Context: Essential for running untrusted LLM-generated code safely
   - Approach: Implement multiprocessing-based sandbox with 5-second timeout, test with malicious code

3. **Phase 1.4: LLM Code Generation**
   - Source: plan_20251024.md (lines 287-360)
   - Context: Core intelligence - Gemini API for solver generation
   - Approach: Implement Analyst (rule inference) and Programmer (code generation) agents

#### Known Issues / Blockers
- None currently - Phase 1.1 complete and merged to main

#### Session Learnings
- **Lazy Validation Pattern**: API key validation converted from eager (blocks all imports) to lazy (call `get_gemini_api_key()` only when needed). Enables Phase 1.1 features to work without API setup.
- **Enhanced load_task()**: Added optional `task_id` parameter to load from collection files directly, eliminating temp file workaround in demo script.
- **Project Root Discovery**: Implemented robust search for `pyproject.toml` (up to 5 levels) with fallback, making imports work across different execution contexts.
- **Test Structure**: Used `pythonpath = ["src"]` in pyproject.toml to enable proper imports without sys.path manipulation.
- **Dependency Consolidation**: Removed requirements.txt, using only pyproject.toml with `[project.optional-dependencies]` for dev tools.
