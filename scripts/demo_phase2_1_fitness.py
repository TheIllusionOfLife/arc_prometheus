#!/usr/bin/env python3
"""
Phase 2.1 Demo: Fitness Function Evaluation

Demonstrates fitness calculation with emphasis on generalization over memorization.
Tests with various solvers to show the 10x weight on test accuracy.
"""

import json
import os
import tempfile
import time

from arc_prometheus.evolutionary_engine.fitness import calculate_fitness


def print_section_header(text: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 60}")
    print(f" {text}")
    print(f"{'=' * 60}\n")


def print_fitness_report(result: dict, task_id: str) -> None:
    """Print detailed fitness evaluation report."""
    print(f"Task ID: {task_id}")
    print(f"\n{'─' * 60}")
    print("📊 Train Performance:")
    print(
        f"├─ Correct: {result['train_correct']}/{result['train_total']} "
        f"({result['train_accuracy'] * 100:.1f}%)"
    )
    print(f"├─ Score contribution: {result['train_correct']} points")

    print(f"\n{'─' * 60}")
    print("🎯 Test Performance:")
    print(
        f"├─ Correct: {result['test_correct']}/{result['test_total']} "
        f"({result['test_accuracy'] * 100:.1f}%)"
    )
    print(f"├─ Score contribution: {result['test_correct'] * 10} points")

    print(f"\n{'─' * 60}")
    print(f"🏆 Overall Fitness: {result['fitness']}")

    # Interpretation
    print(f"\n{'─' * 60}")
    print("💡 Analysis:")

    if result["test_accuracy"] == 0 and result["train_accuracy"] > 0:
        print("⚠️  This solver OVERFITS to training data.")
        print("   It memorizes train examples but fails to generalize.")
    elif result["test_accuracy"] == 1.0 and result["train_accuracy"] == 1.0:
        print("✅ Perfect solver! Achieves 100% on both train and test.")
        print("   Demonstrates strong generalization capability.")
    elif result["test_accuracy"] > result["train_accuracy"]:
        print("🌟 Excellent generalization! Test accuracy exceeds train.")
        print("   This solver learned the underlying rule well.")
    elif result["test_accuracy"] > 0:
        print("✓  Partial generalization. Solver works on some unseen examples.")
        print("   Could be improved with refinement.")
    else:
        print("❌ No generalization. Solver fails on all test examples.")
        print("   Needs significant improvement or redesign.")

    print("\n   Test accuracy is weighted 10x to encourage generalization.")
    print("   Formula: fitness = (train_correct × 1) + (test_correct × 10)")

    if result["execution_errors"]:
        print(f"\n{'─' * 60}")
        print("⚠️  Execution Errors:")
        for error in result["execution_errors"][:3]:  # Show first 3 errors
            print(f"   • {error}")
        if len(result["execution_errors"]) > 3:
            print(f"   ... and {len(result['execution_errors']) - 3} more errors")


def demo_with_perfect_solver(task_path: str, task_id: str) -> None:
    """Demo 1: Perfect solver (doubles input values)."""
    print_section_header("Demo 1: Perfect Solver (Correct Rule)")

    print("Testing a solver that correctly learns the rule: Output = Input × 2")
    print("This should achieve 100% accuracy on both train and test.\n")

    solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Perfect solver: doubles all input values
    return task_grid * 2
"""

    print("Solver code:")
    print(solver_code)

    start_time = time.time()
    result = calculate_fitness(task_path, solver_code)
    elapsed = time.time() - start_time

    print_fitness_report(result, task_id)
    print(f"\n⏱️  Evaluation time: {elapsed:.3f} seconds")


def demo_with_overfitting_solver(task_path: str, task_id: str) -> None:
    """Demo 2: Solver that overfits to training data."""
    print_section_header("Demo 2: Overfitting Solver (Memorization)")

    print("Testing a solver that memorizes specific train examples.")
    print("This works on train data but fails to generalize to test data.\n")

    # Hardcoded solver that only knows train examples
    solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Hardcoded rules for specific train inputs (overfitting)
    if task_grid[0, 0] == 1:
        return np.array([[2, 4], [6, 8]])
    elif task_grid[0, 0] == 5:
        return np.array([[10, 12], [14, 16]])
    elif task_grid[0, 0] == 0:
        return np.array([[0, 2], [4, 6]])

    # Fails on unseen test input (9, 10, ...)
    return np.zeros_like(task_grid)
"""

    print("Solver code:")
    print(solver_code)
    print("\nNote: This memorizes train examples but returns zeros for test input.")
    print("      Demonstrates overfitting - no generalization!\n")

    start_time = time.time()
    result = calculate_fitness(task_path, solver_code)
    elapsed = time.time() - start_time

    print_fitness_report(result, task_id)
    print(f"\n⏱️  Evaluation time: {elapsed:.3f} seconds")


def demo_with_timeout_solver(task_path: str, task_id: str) -> None:
    """Demo 3: Solver with infinite loop (timeout)."""
    print_section_header("Demo 3: Faulty Solver (Infinite Loop)")

    print("Testing a solver with an infinite loop.")
    print("Demonstrates timeout enforcement and error handling.\n")

    solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Infinite loop (will timeout)
    while True:
        pass
    return task_grid
"""

    print("Solver code:")
    print(solver_code)

    print("\n⏳ Running with 2-second timeout...\n")

    start_time = time.time()
    result = calculate_fitness(task_path, solver_code, timeout=2)
    elapsed = time.time() - start_time

    print_fitness_report(result, task_id)
    print(f"\n⏱️  Evaluation time: {elapsed:.3f} seconds")
    print("   (Timeout enforced - solver terminated)")


def main():
    """Run Phase 2.1 fitness function demo."""
    print("\n" + "=" * 60)
    print(" Phase 2.1: Fitness Function Demo")
    print(" Evaluating Solver Quality with Generalization Focus")
    print("=" * 60)

    # Create synthetic task for demo (double the input values)
    task_id = "demo_double"
    task_data = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 4], [6, 8]]},
            {"input": [[5, 6], [7, 8]], "output": [[10, 12], [14, 16]]},
            {"input": [[0, 1], [2, 3]], "output": [[0, 2], [4, 6]]},
        ],
        "test": [{"input": [[9, 10], [11, 12]], "output": [[18, 20], [22, 24]]}],
    }

    # Create temporary task file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as task_file:
        json.dump(task_data, task_file)
        task_path = task_file.name

    try:
        print(f"\nUsing task: {task_id} (synthetic demo task)")
        print("Rule: Output = Input × 2")
        print(f"Train examples: {len(task_data['train'])}")
        print(f"Test examples: {len(task_data['test'])}")

        # Run demos
        demo_with_perfect_solver(task_path, task_id)
        demo_with_overfitting_solver(task_path, task_id)
        demo_with_timeout_solver(task_path, task_id)

        # Final summary
        print_section_header("Demo Complete")
        print("✅ Fitness function successfully demonstrated!")
        print("\nKey Takeaways:")
        print("1. Test accuracy is weighted 10x higher than train accuracy")
        print("2. This encourages solvers that generalize, not memorize")
        print("3. Timeout enforcement prevents infinite loops")
        print("4. Clear error reporting helps identify failure modes")
        print("\n🎯 Next: Phase 2.2 - Refiner Agent for solver debugging")
        print("=" * 60 + "\n")
    finally:
        # Clean up temporary file
        os.remove(task_path)


if __name__ == "__main__":
    main()
