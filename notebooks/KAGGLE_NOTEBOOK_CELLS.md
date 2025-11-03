# Kaggle Submission Notebook Cells

This document contains all cells for the Kaggle submission notebook implementing exact Gemini ensemble workflow with Code Gemma + Outlines.

**How to use:**
1. Create new Jupyter notebook on Kaggle
2. Copy each cell's content below into corresponding notebook cells
3. Mark Cell 0 as Markdown, all others as Code cells
4. Upload Code Gemma 7B model to Kaggle dataset
5. Update `MODEL_DIR` path in Cell 3 to match your dataset path
6. Run TEST_MODE = True (5 samples) first to validate timing
7. If timing OK, set TEST_MODE = False and run full 240 tasks

---

## Cell 0 (Markdown): Header

```markdown
# ARC Prize 2025 Submission: Test-Time Ensemble (Offline Inference)

**Strategy:** Exact replication of Gemini ensemble workflow with local Code Gemma 7B

**Architecture (from PR #58):**
1. **Multi-Persona Analyst** (temp=1.0): 4 diverse expert interpretations
2. **Multi-Solution Programmer** (temp=0.0): 4 solver implementations
3. **Synthesis Agent** (temp=0.0): 5th solution via meta-learning
4. **Pass@2 Output**: Best solution + Synthesis solution

**Key Features:**
- Structured JSON output via Outlines library (replaces Gemini API structured output)
- Exact prompt templates from `multi_persona_analyst.py`, `multi_solution_programmer.py`, `synthesis_agent.py`
- Exact temperatures: Analyst=1.0, Programmer=0.0, Synthesis=0.0
- Exact Pydantic schemas for validation
- Safe execution with multiprocess sandbox (5-second timeout)

**Constraints:**
- No internet access (offline inference only)
- 12-hour runtime for 240 tasks (target: ≤90 sec/task)
- Kaggle GPU: L4x4 (96GB VRAM) recommended

**Requirements:**
- Python 3.9+
- Code Gemma 7B (uploaded as Kaggle dataset)
- Outlines library for structured output
```

---

## Cell 1 (Code): Environment Setup + Dependencies

```python
# Cell 1: Environment Setup + Dependencies

import json
import multiprocessing
import os
import re
import time
from typing import Literal

import numpy as np

# Silence tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# NOTE: Kaggle notebooks run OFFLINE (no internet access during execution).
# You MUST pre-install dependencies by uploading them as a Kaggle dataset:
#
# Option 1: Upload pre-built wheels as a dataset
#   1. Download wheels locally: pip download outlines accelerate -d wheels/
#   2. Upload wheels/ directory as Kaggle dataset
#   3. In notebook: pip install /kaggle/input/your-wheels-dataset/*.whl
#
# Option 2: Use Kaggle's pre-installed packages
#   - Check available packages: !pip list
#   - Pydantic and transformers are pre-installed
#   - Only Outlines needs to be bundled
#
# For testing purposes, this cell attempts pip install (will fail offline)
print("Attempting to install Outlines library...")
import subprocess
import sys

try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "outlines", "accelerate"]
    )
    print("✅ Outlines installed successfully")
except Exception as e:
    print(f"⚠️  Pip install failed (expected in offline mode): {e}")
    print("Make sure dependencies are pre-bundled as a Kaggle dataset")

# Import required libraries
try:
    import outlines
    import torch
    from pydantic import BaseModel, Field
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_DEPENDENCIES = True
    print("✅ All dependencies available")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    HAS_DEPENDENCIES = False
    raise

print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
```

---

## Cell 2 (Code): Pydantic Schemas

```python
# Cell 2: Pydantic Schemas (Exact copies from src/arc_prometheus/utils/schemas.py)

# These schemas are identical to our Gemini workflow
# Outlines will enforce these schemas during generation (just like Gemini's structured output)

# Multi-Persona Analyst Schema
class Interpretation(BaseModel):
    """Single expert interpretation of an ARC task."""

    persona: str = Field(
        ...,
        description="Expert perspective name (e.g., 'Geometric Transformation Specialist')",
    )
    pattern: str = Field(
        ...,
        description="One-sentence transformation rule description",
    )
    observations: list[str] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="Key insights (1-3 items, each ≤85 chars)",
    )
    approach: str = Field(..., description="High-level implementation strategy (≤200 chars)")
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence in this interpretation"
    )


class MultiPersonaResponse(BaseModel):
    """Response containing 4 diverse expert interpretations."""

    interpretations: list[Interpretation] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Exactly 4 diverse expert interpretations",
    )


# Multi-Solution Programmer Schema
class Solution(BaseModel):
    """Single solver implementation linked to an interpretation."""

    interpretation_id: int = Field(
        ..., ge=1, le=4, description="Which interpretation this implements (1-4)"
    )
    code: str = Field(..., description="Complete solve() function implementation")
    approach_summary: str = Field(
        ..., description="Brief description of implementation approach (≤200 chars)"
    )


class MultiSolutionResponse(BaseModel):
    """Response containing solver implementations.

    Normally contains exactly 4 solutions, but can accept 1-4 when MAX_TOKENS
    truncates the response. The ensemble pipeline pads <4 solutions to 4 by
    duplicating the best solution.
    """

    solutions: list[Solution] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="1-4 solver implementations (target: 4)",
    )


# Synthesis Agent Schema
class SynthesisAnalysis(BaseModel):
    """Analysis of 4 solutions to inform synthesis."""

    successful_patterns: list[str] = Field(
        ...,
        max_length=3,
        description="Patterns from successful solutions (max 3 items, each ≤85 chars)",
    )
    failed_patterns: list[str] = Field(
        ...,
        max_length=5,
        description="Patterns from failed solutions (max 5 items, each ≤85 chars)",
    )
    synthesis_strategy: str = Field(
        ..., description="How to create diverse 5th solution (≤250 chars)"
    )


class SynthesisResponse(BaseModel):
    """Response containing synthesis of 4 solutions into a 5th diverse solution."""

    analysis: SynthesisAnalysis = Field(
        ..., description="Analysis of existing solutions"
    )
    code: str = Field(
        ..., description="Complete solve() function for synthesis solution"
    )
    diversity_justification: str = Field(
        ...,
        description="Why this solution is different from all 4 previous (≤200 chars)",
    )


print("✅ Pydantic schemas defined (identical to Gemini workflow)")
```

---

## Cell 3 (Code): Load Code Gemma with Outlines Wrapper

```python
# Cell 3: Load Code Gemma with Outlines Wrapper

import outlines
import torch

# Path to Code Gemma model (uploaded as Kaggle dataset)
# IMPORTANT: Point to the directory containing BOTH model and tokenizer files
MODEL_PATH = "/kaggle/input/codegemma-7b-instruct/codegemma-7b"

print("Loading Code Gemma 7B model with Outlines...")
try:
    # Let Outlines handle model loading directly from path
    # IMPORTANT: Pass MODEL_PATH string, NOT a pre-loaded model object
    # (outlines.models.transformers() expects a string path/name)
    outlines_model = outlines.models.transformers(
        MODEL_PATH,  # ✅ Pass directory path containing model + tokenizer
        device="auto",
        model_kwargs={"torch_dtype": torch.float16},  # Memory optimization
    )
    print(f"✅ Model loaded successfully with Outlines wrapper!")
    print(f"   Device: {outlines_model.device}")

except Exception as e:
    print(f"❌ ERROR: Model loading failed: {e}")
    print("This will cause the notebook to fail. Please check:")
    print("1. Model files exist in Kaggle dataset")
    print("2. Sufficient disk space (~16GB)")
    print(f"   MODEL_PATH: {MODEL_PATH}")
    raise


def generate_structured_json(
    prompt: str,
    schema: type[BaseModel],
    temperature: float = 0.0,
    max_tokens: int = 65536,
) -> BaseModel:
    """Generate structured JSON matching Pydantic schema.

    This replicates Gemini's structured output API behavior:
    - Input: prompt + Pydantic schema
    - Output: validated Pydantic model instance
    - Guaranteed valid JSON (no parsing errors)

    Args:
        prompt: The prompt to send to the model
        schema: Pydantic model class defining output structure
        temperature: 0.0 for deterministic, >0.0 for sampling
        max_tokens: Maximum tokens to generate

    Returns:
        Validated Pydantic model instance
    """
    # Select sampler based on temperature
    if temperature == 0.0:
        sampler = outlines.samplers.greedy()
    else:
        sampler = outlines.samplers.multinomial(temperature=temperature)

    # Create JSON generator constrained by Pydantic schema
    generator = outlines.generate.json(
        outlines_model,
        schema,
        sampler=sampler,
        max_tokens=max_tokens,
    )

    # Generate and return validated result
    result = generator(prompt)
    return result


print("✅ Structured JSON generation function ready")
```

**⚠️ IMPORTANT: Kaggle Testing Required**

This fix addresses the `HFValidationError` by passing the model path string to `outlines.models.transformers()` instead of a pre-loaded model object. The change has **not been tested locally** due to Code Gemma's 16GB VRAM requirement.

**Recommended Kaggle validation test:**
```python
# Test structured generation (add to Cell 4 for quick validation)
from pydantic import BaseModel, Field

class TestSchema(BaseModel):
    message: str = Field(..., description="A test message")

test_result = generate_structured_json(
    "Say hello in JSON format",
    TestSchema,
    temperature=0.0,
    max_tokens=50,
)
print(f"✅ Outlines test successful: {test_result.message}")
```

---

**NOTE: Due to length constraints, the remaining cells (4-7) contain the agent implementations and main loop. These are very long and detailed.**

**Instead of pasting them here, I recommend:**

1. I can create a separate `.py` file with all the agent code that you can review
2. You can manually type the cells in Jupyter on Kaggle using the code structure from our earlier conversation summary
3. Or we can create a minimal working version first to test the Outlines integration

**Would you prefer:**
- A) Full Python file with all agent code for review (not a notebook)?
- B) Create minimal test version first (just 1 analyst call, no full pipeline)?
- C) Continue with markdown documentation but split into multiple files?

Let me know and I'll proceed accordingly!
## Cell 4 (Code): Core Helper Functions

```python
# Cell 4: Core Helper Functions

def format_grid(grid: list[list[int]]) -> str:
    """Format grid as readable text (identical to Gemini agents)."""
    rows = []
    for row in grid:
        rows.append(" ".join(str(cell) for cell in row))
    return "\n".join(rows)


def execute_solver_safe(
    code: str, input_grid: np.ndarray, timeout: int = 5
) -> tuple[bool, np.ndarray | None, dict | None]:
    """
    Execute solver code with timeout (multiprocess sandbox).

    SECURITY NOTE: Provides process isolation but cannot prevent
    filesystem/network access (acceptable for Kaggle's isolated environment).

    Args:
        code: Solver code string
        input_grid: Input grid to transform
        timeout: Timeout in seconds

    Returns:
        (success, result_grid, error_detail)
    """

    def _run_solver(code_str, task_grid, result_queue):
        """Worker function for multiprocess execution"""
        try:
            # Create restricted namespace (remove dangerous builtins)
            safe_builtins = {
                k: v
                for k, v in __builtins__.items()
                if k not in ["eval", "exec", "compile", "open", "__import__"]
            }

            namespace = {
                "__builtins__": safe_builtins,
                "np": np,
                "task_grid": task_grid,
            }

            # Execute code
            exec(code_str, namespace)  # noqa: S102

            if "solve" not in namespace:
                result_queue.put(
                    (
                        False,
                        None,
                        {
                            "error_type": "missing_function",
                            "error_message": "No solve() function found",
                        },
                    )
                )
                return

            result = namespace["solve"](task_grid)

            # Validate result type
            if not isinstance(result, np.ndarray):
                result_queue.put(
                    (
                        False,
                        None,
                        {
                            "error_type": "invalid_return",
                            "error_message": f"Expected np.ndarray, got {type(result)}",
                        },
                    )
                )
                return

            result_queue.put((True, result, None))

        except Exception as e:
            result_queue.put(
                (False, None, {"error_type": type(e).__name__, "error_message": str(e)})
            )

    # Run in separate process
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_solver, args=(code, input_grid, result_queue)
    )

    try:
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return (
                False,
                None,
                {
                    "error_type": "timeout",
                    "error_message": f"Execution exceeded {timeout}s",
                },
            )

        if result_queue.empty():
            return (
                False,
                None,
                {
                    "error_type": "unknown",
                    "error_message": "Process terminated without result",
                },
            )

        return result_queue.get()

    finally:
        result_queue.close()
        result_queue.join_thread()


print("✅ Helper functions ready")
```

---

## Cell 5 (Code): Test-Time Ensemble Agents

**NOTE:** This cell is very long (~500 lines). It contains all 3 agent classes with EXACT prompts copied from Gemini agents.

Due to length, I'll provide it in a separate file: `notebooks/KAGGLE_CELL_5_AGENTS.md`

For now, here's the structure:

```python
# Cell 5: Test-Time Ensemble Agents (Exact Gemini Workflow Replication)

# IMPORTANT: These classes use EXACT prompts copied from:
# - src/arc_prometheus/cognitive_cells/multi_persona_analyst.py
# - src/arc_prometheus/cognitive_cells/multi_solution_programmer.py
# - src/arc_prometheus/cognitive_cells/synthesis_agent.py
#
# Temperatures match exactly: Analyst=1.0, Programmer=0.0, Synthesis=0.0

class OfflineMultiPersonaAnalyst:
    """Multi-Persona Analyst for offline inference with Code Gemma."""
    # ... (see separate file for full implementation)

class OfflineMultiSolutionProgrammer:
    """Multi-Solution Programmer for offline inference with Code Gemma."""
    # ... (see separate file for full implementation)

class OfflineSynthesisAgent:
    """Synthesis Agent for offline inference with Code Gemma."""
    # ... (see separate file for full implementation)

print("✅ Test-Time Ensemble agents ready (exact Gemini workflow)")
```

---

## Cell 6 (Code): Main Inference Loop

```python
# Cell 6: Main Inference Loop

# Configuration
TEST_DATA_PATH = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
TEST_MODE = True  # Set False for submission, True for 5-sample validation

print(f"Loading test tasks from: {TEST_DATA_PATH}")

try:
    with open(TEST_DATA_PATH) as f:
        test_tasks = json.load(f)
    print(f"Loaded {len(test_tasks)} test tasks")
except FileNotFoundError:
    print("Warning: Test data not found (using mock for local testing)")
    test_tasks = {}

if TEST_MODE:
    test_tasks = dict(list(test_tasks.items())[:5])
    print(f"⚠️  TEST MODE: Running on {len(test_tasks)} tasks for validation")
    print("Set TEST_MODE = False before submission!")

# Initialize agents (exact Gemini temperatures)
analyst = OfflineMultiPersonaAnalyst(temperature=1.0)
programmer = OfflineMultiSolutionProgrammer(temperature=0.0)
synthesis_agent = OfflineSynthesisAgent(temperature=0.0, timeout=5)

# Track progress and timing
submission = {}
start_time = time.time()
timing_stats = []

print(f"\nStarting test-time ensemble on {len(test_tasks)} tasks...")
print(
    f"Target: ≤90 sec/task ({len(test_tasks) * 90 / 60:.1f} min total, "
    f"{len(test_tasks) * 90 / 3600:.2f} hours)"
)

for idx, (task_id, task) in enumerate(test_tasks.items()):
    task_start = time.time()
    print(f"\n{'=' * 60}")
    print(f"Processing {idx + 1}/{len(test_tasks)}: {task_id}")

    try:
        # Step 1: Multi-Persona Analysis (temp=1.0, 4 interpretations)
        step_start = time.time()
        interpretations = analyst.analyze_task(task)
        analyst_time = time.time() - step_start
        print(f"  ✓ Analyst: {len(interpretations)} interpretations ({analyst_time:.1f}s)")

        # Step 2: Multi-Solution Generation (temp=0.0, 4 solutions)
        step_start = time.time()
        solutions = programmer.generate_multi_solutions(task, interpretations)
        programmer_time = time.time() - step_start
        print(f"  ✓ Programmer: {len(solutions)} solutions ({programmer_time:.1f}s)")

        # Step 3: Synthesis (temp=0.0, 5th solution via meta-learning)
        step_start = time.time()
        synthesis_result = synthesis_agent.synthesize_solution(
            task, solutions, interpretations
        )
        synthesis_time = time.time() - step_start
        print(f"  ✓ Synthesis: 5th solution generated ({synthesis_time:.1f}s)")

        # Step 4: Select best solution from 4 solutions
        best_solution = solutions[0]  # Default to first solution
        best_fitness = 0.0

        for solution in solutions:
            train_correct = 0
            for example in task.get("train", []):
                input_grid = np.array(example["input"], dtype=np.int64)
                expected = np.array(example["output"], dtype=np.int64)
                success, result, _ = execute_solver_safe(solution["code"], input_grid)
                if success and np.array_equal(result, expected):
                    train_correct += 1

            fitness = train_correct
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution

        # Step 5: Generate pass@2 predictions (best + synthesis)
        predictions = []
        for test_input in task.get("test", []):
            input_grid = np.array(test_input["input"], dtype=np.int64)

            # Attempt 1: Best solution
            success1, pred1, _ = execute_solver_safe(
                best_solution["code"], input_grid, timeout=5
            )
            attempt_1 = pred1.tolist() if success1 else input_grid.tolist()

            # Attempt 2: Synthesis solution
            success2, pred2, _ = execute_solver_safe(
                synthesis_result["code"], input_grid, timeout=5
            )
            attempt_2 = pred2.tolist() if success2 else input_grid.tolist()

            predictions.append({"attempt_1": attempt_1, "attempt_2": attempt_2})

        submission[task_id] = predictions

        # Record timing
        task_time = time.time() - task_start
        timing_stats.append(
            {
                "task_id": task_id,
                "total_time": task_time,
                "analyst_time": analyst_time,
                "programmer_time": programmer_time,
                "synthesis_time": synthesis_time,
            }
        )

        print(f"  ✅ Task complete in {task_time:.1f}s")

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        # Fallback: use input grids
        predictions = [
            {"attempt_1": test_input["input"], "attempt_2": test_input["input"]}
            for test_input in task.get("test", [])
        ]
        submission[task_id] = predictions

    # Progress update every 5 tasks
    if (idx + 1) % 5 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (len(test_tasks) - idx - 1)
        print(f"\n📊 Progress: {idx + 1}/{len(test_tasks)} tasks")
        print(f"   Elapsed: {elapsed / 60:.1f} min | Avg: {avg_time:.1f}s/task")
        print(f"   ETA: {remaining / 60:.1f} min")

# Final timing summary
total_time = time.time() - start_time
avg_time = total_time / len(test_tasks)

print(f"\n{'=' * 60}")
print("INFERENCE COMPLETE!")
print(f"Total time: {total_time / 60:.1f} min ({total_time / 3600:.2f} hours)")
print(f"Average: {avg_time:.1f} sec/task")
print(f"Target: 90 sec/task → {'✅ PASS' if avg_time <= 90 else '❌ FAIL'}")

# Detailed timing breakdown
if timing_stats:
    avg_analyst = sum(s["analyst_time"] for s in timing_stats) / len(timing_stats)
    avg_programmer = sum(s["programmer_time"] for s in timing_stats) / len(timing_stats)
    avg_synthesis = sum(s["synthesis_time"] for s in timing_stats) / len(timing_stats)

    print(f"\n⏱️  Timing Breakdown (averages):")
    print(f"   Analyst: {avg_analyst:.1f}s")
    print(f"   Programmer: {avg_programmer:.1f}s")
    print(f"   Synthesis: {avg_synthesis:.1f}s")
    print(f"   Total: {avg_analyst + avg_programmer + avg_synthesis:.1f}s")
```

---

## Cell 7 (Code): Save Submission + Validation

```python
# Cell 7: Save Submission + Validation

OUTPUT_PATH = "submission.json"

print(f"Saving submission to: {OUTPUT_PATH}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(submission, f, indent=2)

print("✅ Submission saved successfully!")
print(f"   Tasks: {len(submission)}")
print("   Format: pass@2 (2 attempts per test input)")

# Validate submission format
print("\nValidating submission format...")
valid = True
for task_id, predictions in submission.items():
    if not isinstance(predictions, list):
        print(f"❌ ERROR: {task_id} has invalid predictions type")
        valid = False
        continue

    for pred_idx, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            print(f"❌ ERROR: {task_id} prediction {pred_idx} is not a dict")
            valid = False
            break

        if "attempt_1" not in pred or "attempt_2" not in pred:
            print(f"❌ ERROR: {task_id} prediction {pred_idx} missing attempts")
            valid = False
            break

        # Check that attempts are lists (grids)
        if not isinstance(pred["attempt_1"], list) or not isinstance(
            pred["attempt_2"], list
        ):
            print(f"❌ ERROR: {task_id} prediction {pred_idx} has non-list attempts")
            valid = False
            break

if valid:
    print("\n✅ Submission format validated successfully!")
    print("\n📦 Ready for Kaggle submission!")
else:
    print("\n❌ Submission format validation FAILED!")
    print("Please fix errors before submitting.")

# If TEST_MODE, remind to switch to full dataset
if TEST_MODE:
    print("\n" + "=" * 60)
    print("⚠️  REMINDER: TEST MODE is enabled!")
    print("   Before final submission:")
    print("   1. Set TEST_MODE = False in Cell 6")
    print("   2. Verify you have 240 tasks loaded")
    print("   3. Re-run all cells")
    print("=" * 60)
```

---

## Summary

**Complete Notebook Structure:**
- ✅ Cell 0: Markdown header (architecture overview)
- ✅ Cell 1: Environment setup + dependencies
- ✅ Cell 2: Pydantic schemas (exact copies)
- ✅ Cell 3: Code Gemma + Outlines wrapper
- ✅ Cell 4: Helper functions (sandbox, grid formatting)
- 📝 Cell 5: Agent implementations (see separate file for full code)
- ✅ Cell 6: Main inference loop with timing
- ✅ Cell 7: Submission save + validation

**Next Step:**
Cell 5 is very long (~500 lines with all 3 agent classes). I'll create it in a separate file for easier review.
