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
- **Safe execution**: All generated code runs in sandboxed multiprocessing environments

## Development Phases

### Phase 1: Core Prototype (Current)
Build the minimal ecosystem to run a single ARC task end-to-end:
- Load ARC dataset (JSON format with train/test pairs)
- Implement grid evaluation and visualization
- Create LLM-based Analyst + Programmer pipeline (Gemini API)
- Build safe code execution sandbox (multiprocessing with timeout)
- Verify solver can solve train pairs

### Phase 2: Evolutionary Loop
Introduce selection pressure and mutation:
- Implement fitness function: `(train_correct * 1) + (test_correct * 10)`
- Add Refiner agent for code debugging
- Generate multiple solver candidates per task (10 initial, 5 generations)
- Track fitness improvement across generations

### Phase 3: Scaling and Crossover
Expand to full ARC dataset with genetic operations:
- Build solver library with SQLite (later cloud database)
- Implement Tagger for technique classification (rotation, fill, symmetry, etc.)
- Add Crossover agent to fuse different solver capabilities
- Deploy distributed task queue (Celery/RabbitMQ) for parallel processing
- Prepare cloud infrastructure (Docker/Kubernetes)

## Common Commands

### Environment Setup (Phase 1.1)
```bash
# Download ARC Prize 2025 dataset from Kaggle
# https://www.kaggle.com/competitions/arc-prize-2025/data

# Install core dependencies
pip install numpy google-generativeai

# Future phases will add:
# pip install multiprocess celery redis pytest
```

### Testing
```bash
# Run Phase 1 end-to-end test (when implemented)
python run_phase1_test.py <path_to_arc_task.json>

# Run unit tests (future)
pytest tests/
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
    {"input": [[...]], "output": [[...]]}
  ]
}
```
Grids are 2D arrays of integers (0-9) representing colors.

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
- All LLM-generated code runs in isolated `multiprocessing.Process`
- Default timeout: 5 seconds per execution
- Return format: `tuple[bool, Optional[np.ndarray]]`
  - `(True, result_grid)` on successful execution
  - `(False, None)` on failure/timeout/exception
- Handle: timeouts, runtime exceptions, invalid return types

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

### Success Metrics
- **Phase 1**: Solve ≥1 train pair with generated code
- **Phase 2**: Improve test accuracy across generations
- **Phase 3**: Cross-task solver generalization (solver trained on task A solves task B)

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
