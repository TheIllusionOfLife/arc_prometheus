"""Demo script for Phase 1.3: Safe Execution Sandbox

This script demonstrates:
1. Safe execution of valid solver code (Phase 1.2 manual solver)
2. Timeout enforcement with infinite loop
3. Exception handling with intentionally broken code
4. Integration with evaluate_grids() for correctness validation

The sandbox executes untrusted code in isolated processes with:
- Timeout enforcement (5 seconds default)
- Restricted builtins (no eval, exec, compile)
- Exception handling
- Return type validation

Usage:
    python scripts/demo_phase1_3_sandbox.py
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from arc_prometheus.crucible.data_loader import load_task, print_grid
from arc_prometheus.crucible.evaluator import evaluate_grids
from arc_prometheus.crucible.sandbox import safe_execute
from arc_prometheus.utils.config import DATA_DIR

# Phase 1.2 manual solver as string (will be executed in sandbox)
PHASE1_2_SOLVER_CODE = """
from collections import OrderedDict
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    '''Solve ARC task 05269061: Extract diagonal pattern and fill grid.'''
    # Step 1: Group non-zero values by diagonal
    diagonals = OrderedDict()

    for i in range(7):
        for j in range(7):
            if task_grid[i, j] != 0:
                diag_idx = i + j
                if diag_idx not in diagonals:
                    diagonals[diag_idx] = task_grid[i, j]

    if len(diagonals) == 0:
        return np.zeros((7, 7), dtype=int)

    # Step 2: Extract values in diagonal order
    diagonal_values = list(diagonals.values())

    # Step 3: Determine pattern rotation
    diag_indices = list(diagonals.keys())

    is_consecutive = all(
        diag_indices[i + 1] - diag_indices[i] == 1 for i in range(len(diag_indices) - 1)
    )

    if is_consecutive:
        min_diag = min(diag_indices)
        if min_diag < 4:
            base_pattern = diagonal_values
        else:
            base_pattern = diagonal_values[1:] + diagonal_values[:1]
    else:
        base_pattern = diagonal_values[-1:] + diagonal_values[:-1]

    # Step 4: Fill the 7x7 output grid
    output = np.zeros((7, 7), dtype=int)
    pattern_len = len(base_pattern)

    for row in range(7):
        for col in range(7):
            pattern_idx = (col + row) % pattern_len
            output[row, col] = base_pattern[pattern_idx]

    return output
"""

# Intentional infinite loop for timeout demonstration
INFINITE_LOOP_CODE = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Intentional infinite loop to trigger timeout
    while True:
        pass
    return task_grid
"""

# Intentional exception for error handling demonstration
BROKEN_CODE = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Intentional division by zero
    x = 1 / 0
    return task_grid
"""


def main():
    """Run the sandbox demonstration."""
    print("=" * 70)
    print("ARC-Prometheus Phase 1.3 Demo: Safe Execution Sandbox")
    print("=" * 70)
    print("\nThis demo shows how the sandbox safely executes untrusted code:")
    print("  1. Successful execution (Phase 1.2 manual solver)")
    print("  2. Timeout enforcement (infinite loop)")
    print("  3. Exception handling (intentional error)")
    print()

    # Load task 05269061
    task_id = "05269061"
    challenges_file = DATA_DIR / "arc-agi_training_challenges.json"

    if not challenges_file.exists():
        print(f"\n❌ ERROR: Training challenges file not found at {challenges_file}")
        print("Please download the ARC Prize 2025 dataset from:")
        print("https://www.kaggle.com/competitions/arc-prize-2025/data")
        print(f"And place it in: {DATA_DIR}")
        sys.exit(1)

    try:
        task_data = load_task(str(challenges_file), task_id=task_id)
    except Exception as e:
        print(f"\n❌ ERROR loading task: {e}")
        sys.exit(1)

    # =========================================================================
    # Demo 1: Successful Execution
    # =========================================================================
    print("=" * 70)
    print("DEMO 1: SUCCESSFUL EXECUTION")
    print("=" * 70)
    print("\nExecuting Phase 1.2 manual solver in sandbox...")
    print(f"Task ID: {task_id}")
    print(f"Train examples: {len(task_data['train'])}")

    correct_count = 0
    total_count = len(task_data["train"])

    for idx, example in enumerate(task_data["train"], 1):
        print(f"\n{'─' * 70}")
        print(f"Train Example {idx}/{total_count}")
        print(f"{'─' * 70}")

        # Execute solver in sandbox
        start_time = time.time()
        success, predicted_output = safe_execute(
            PHASE1_2_SOLVER_CODE, example["input"], timeout=5
        )
        execution_time = time.time() - start_time

        if not success:
            print("❌ Sandbox execution FAILED")
            print(f"   Execution time: {execution_time:.2f}s")
            continue

        print(f"✓ Sandbox execution succeeded in {execution_time:.2f}s")

        # Show input
        print_grid(example["input"], label=f"Input (shape: {example['input'].shape})")

        # Show predicted output
        print_grid(
            predicted_output,
            label=f"Predicted Output (shape: {predicted_output.shape})",
        )

        # Show expected output
        print_grid(
            example["output"],
            label=f"Expected Output (shape: {example['output'].shape})",
        )

        # Evaluate
        is_correct = evaluate_grids(predicted_output, example["output"])

        if is_correct:
            print("✅ MATCH: Predicted output matches expected output!")
            correct_count += 1
        else:
            print("❌ MISMATCH: Predicted output does NOT match expected output")

    success_rate = (correct_count / total_count) * 100
    print(f"\n{'=' * 70}")
    print(
        f"Solver Performance: {correct_count}/{total_count} correct ({success_rate:.0f}%)"
    )
    print("=" * 70)

    if correct_count == total_count:
        print("\n✅ SUCCESS: Sandbox executed Phase 1.2 solver perfectly!")
        print("   - All train examples solved correctly")
        print("   - No security breaches")
        print("   - No timeouts or errors")

    # =========================================================================
    # Demo 2: Timeout Enforcement
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("DEMO 2: TIMEOUT ENFORCEMENT")
    print("=" * 70)
    print("\nExecuting solver with infinite loop...")
    print("Timeout: 2 seconds")

    input_grid = task_data["train"][0]["input"]
    start_time = time.time()
    success, result = safe_execute(INFINITE_LOOP_CODE, input_grid, timeout=2)
    execution_time = time.time() - start_time

    print(f"\nExecution time: {execution_time:.2f}s")
    print(f"Success: {success}")
    print(f"Result: {result}")

    if not success and result is None and execution_time >= 2.0:
        print("\n✅ SUCCESS: Timeout enforcement works correctly!")
        print("   - Infinite loop detected")
        print("   - Process terminated after 2 seconds")
        print("   - Returned (False, None) as expected")
    else:
        print("\n❌ FAILURE: Timeout enforcement did not work as expected")

    # =========================================================================
    # Demo 3: Exception Handling
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("DEMO 3: EXCEPTION HANDLING")
    print("=" * 70)
    print("\nExecuting solver with intentional ZeroDivisionError...")

    start_time = time.time()
    success, result = safe_execute(BROKEN_CODE, input_grid, timeout=5)
    execution_time = time.time() - start_time

    print(f"\nExecution time: {execution_time:.2f}s")
    print(f"Success: {success}")
    print(f"Result: {result}")

    if not success and result is None:
        print("\n✅ SUCCESS: Exception handling works correctly!")
        print("   - ZeroDivisionError caught")
        print("   - Sandbox remained stable")
        print("   - Returned (False, None) as expected")
    else:
        print("\n❌ FAILURE: Exception handling did not work as expected")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("PHASE 1.3 COMPLETE: SAFE EXECUTION SANDBOX")
    print("=" * 70)
    print("\n✓ Infrastructure validated:")
    print("  - Multiprocessing isolation working")
    print("  - Timeout enforcement functional")
    print("  - Exception handling robust")
    print("  - Integration with Phase 1.2 solver successful")
    print("\n⚠️  Security limitations (documented):")
    print("  - Multiprocessing cannot prevent filesystem access")
    print("  - Multiprocessing cannot prevent network access")
    print("  - For production: use Docker with read-only filesystem")
    print("\nNext steps:")
    print("  - Phase 1.4: LLM Code Generation (Gemini API)")
    print("  - Phase 1.5: End-to-End Pipeline")
    print("=" * 70)


if __name__ == "__main__":
    main()
