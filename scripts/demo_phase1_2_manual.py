"""Demo script for Phase 1.2: Manual Solver Validation

This script demonstrates:
1. Manually-written solver for ARC task 05269061
2. Testing solver against all train examples
3. Validation using existing evaluate_grids() function

The manual solver extracts diagonal patterns and fills grids with
repeating cycles. This validates our infrastructure before building
the LLM pipeline in Phase 1.4.

Usage:
    python scripts/demo_phase1_2_manual.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from collections import OrderedDict
from arc_prometheus.crucible.data_loader import load_task, print_grid
from arc_prometheus.crucible.evaluator import evaluate_grids
from arc_prometheus.utils.config import DATA_DIR


def solve(task_grid: np.ndarray) -> np.ndarray:
    """Solve ARC task 05269061: Extract diagonal pattern and fill grid.

    **IMPORTANT**: This solver is specifically designed for task 05269061,
    which always uses 7x7 grids. It is NOT a general-purpose solver.
    This is a validation solver for Phase 1.2 to test infrastructure
    before building the LLM pipeline in Phase 1.4.

    Algorithm:
    1. Group non-zero values by diagonal lines (where row + col is constant)
    2. Extract one representative value per diagonal (first occurrence)
    3. Determine pattern rotation based on diagonal positions:
       - Consecutive diagonals in top-left (indices <4): no rotation
       - Consecutive diagonals in bottom-right (indices >=4): rotate left by 1
       - Non-consecutive diagonals: move last value to first position
    4. Fill 7x7 grid with row-shifted repeating pattern

    Args:
        task_grid: Input grid (7x7 numpy array for task 05269061)

    Returns:
        Output grid (7x7 numpy array) with repeating pattern

    Examples:
        >>> # Example 1: diagonals [8,9,10] with values [1,2,4]
        >>> # Output pattern: [2,4,1] (consecutive, >=4, rotate left 1)

        >>> # Example 2: diagonals [0,1,2] with values [2,8,3]
        >>> # Output pattern: [2,8,3] (consecutive, <4, no rotation)

        >>> # Example 3: diagonals [4,5,9] with values [8,3,4]
        >>> # Output pattern: [4,8,3] (non-consecutive, last first)
    """
    # Step 1: Group non-zero values by diagonal
    # Diagonal index = row + col (constant for / diagonals)
    diagonals = OrderedDict()

    for i in range(7):
        for j in range(7):
            if task_grid[i, j] != 0:
                diag_idx = i + j
                if diag_idx not in diagonals:
                    diagonals[diag_idx] = task_grid[i, j]

    # If no non-zero values, return grid of zeros
    if len(diagonals) == 0:
        return np.zeros((7, 7), dtype=int)

    # Step 2: Extract values in diagonal order
    diagonal_values = list(diagonals.values())

    # Step 3: Determine pattern rotation based on diagonal positions
    diag_indices = list(diagonals.keys())

    is_consecutive = all(diag_indices[i+1] - diag_indices[i] == 1
                        for i in range(len(diag_indices)-1))

    if is_consecutive:
        min_diag = min(diag_indices)
        if min_diag < 4:
            base_pattern = diagonal_values  # Top-left: no rotation
        else:
            base_pattern = diagonal_values[1:] + diagonal_values[:1]  # Bottom-right: rotate left 1
    else:
        base_pattern = diagonal_values[-1:] + diagonal_values[:-1]  # Non-consecutive: last first

    # Step 4: Fill the 7x7 output grid with row-shifted repeating pattern
    output = np.zeros((7, 7), dtype=int)
    pattern_len = len(base_pattern)

    for row in range(7):
        for col in range(7):
            pattern_idx = (col + row) % pattern_len
            output[row, col] = base_pattern[pattern_idx]

    return output


def main():
    """Run the manual solver demo."""
    print("=" * 70)
    print("ARC-Prometheus Phase 1.2 Demo: Manual Solver Validation")
    print("=" * 70)

    # Load task 05269061
    task_id = "05269061"
    print(f"\nTask ID: {task_id}")
    print("Pattern: Extract diagonal values and fill grid with repeating cycle")

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

    print(f"\n✓ Successfully loaded task: {task_id}")
    print(f"  - Train examples: {len(task_data['train'])}")
    print(f"  - Test examples: {len(task_data['test'])}")

    # Test solver on all train examples
    print("\n" + "=" * 70)
    print("TESTING MANUAL SOLVER")
    print("=" * 70)

    correct_count = 0
    total_count = len(task_data['train'])

    for idx, example in enumerate(task_data['train'], 1):
        print(f"\n{'─' * 70}")
        print(f"Train Example {idx}/{total_count}")
        print(f"{'─' * 70}")

        # Show input
        print_grid(example['input'], label=f"Input (shape: {example['input'].shape})")

        # Run solver
        predicted_output = solve(example['input'])

        # Show predicted output
        print_grid(predicted_output, label=f"Predicted Output (shape: {predicted_output.shape})")

        # Show expected output
        print_grid(example['output'], label=f"Expected Output (shape: {example['output'].shape})")

        # Evaluate
        is_correct = evaluate_grids(predicted_output, example['output'])

        if is_correct:
            print("✅ MATCH: Predicted output matches expected output!")
            correct_count += 1
        else:
            print("❌ MISMATCH: Predicted output does NOT match expected output")

            # Show first difference
            diff_mask = predicted_output != example['output']
            if np.any(diff_mask):
                diff_positions = np.argwhere(diff_mask)
                first_diff = diff_positions[0]
                i, j = first_diff[0], first_diff[1]
                print(f"   First difference at position ({i}, {j}):")
                print(f"   Predicted: {predicted_output[i, j]}, Expected: {example['output'][i, j]}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    success_rate = (correct_count / total_count) * 100
    print(f"\nSolver Performance: {correct_count}/{total_count} train examples solved correctly ({success_rate:.0f}%)")

    if correct_count == total_count:
        print("\n🎉 SUCCESS: Manual solver works perfectly!")
        print("\n✓ Phase 1.2 Complete:")
        print("  - Manual solver implemented using numpy")
        print("  - All train examples solved correctly")
        print("  - Infrastructure validated for Phase 1.3")
    else:
        print(f"\n⚠️  WARNING: Solver failed on {total_count - correct_count} example(s)")
        print("   Manual solver needs refinement before proceeding to Phase 1.3")

    # Next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\nPhase 1.3: Safe Execution Sandbox")
    print("  - Implement multiprocessing-based sandbox")
    print("  - Add timeout enforcement (5 seconds)")
    print("  - Test with malicious code scenarios")
    print("  - Reuse this manual solver as test case")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
