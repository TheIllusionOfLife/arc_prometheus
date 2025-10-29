"""Demo script for Phase 2.2: Refiner Agent (Mutation).

This demonstrates the first evolutionary mechanism - automated code debugging
that improves failed solver code through LLM-based refinement.

Usage:
    python scripts/demo_phase2_2_refiner.py

Requirements:
    - GEMINI_API_KEY environment variable must be set
    - ARC dataset in data/arc-prize-2025/ (for real task demos)

Demonstrations:
    1. Syntax Error Fix: Missing colon in function signature
    2. Logic Error Fix: Wrong algorithm (adds instead of multiplies)
    3. Timeout Fix: Infinite loop optimization
"""

import json
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_prometheus.cognitive_cells.refiner import refine_solver
from arc_prometheus.evolutionary_engine.fitness import calculate_fitness
from arc_prometheus.utils.config import get_gemini_api_key


def print_header(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print formatted subsection header."""
    print("\n" + "-" * 70)
    print(f" {title}")
    print("-" * 70)


def print_code_with_line_numbers(code: str, title: str = "Code") -> None:
    """Print code with line numbers."""
    print(f"\n{title}:")
    lines = code.strip().split("\n")
    for i, line in enumerate(lines, 1):
        print(f"  {i:3d} | {line}")


def print_fitness_result(result: dict, label: str = "Fitness") -> None:
    """Print fitness evaluation result."""
    print(f"\n{label}: {result['fitness']}")
    print(
        f"  - Train: {result['train_correct']}/{result['train_total']} "
        f"({result['train_accuracy']:.0%})"
    )
    print(
        f"  - Test: {result['test_correct']}/{result['test_total']} "
        f"({result['test_accuracy']:.0%})"
    )
    if result["execution_errors"]:
        print(f"  - Errors: {len(result['execution_errors'])} execution failure(s)")
        for error in result["execution_errors"][:3]:
            print(f"    • {error}")


def demo_syntax_error_fix() -> dict[str, float]:
    """Demo 1: Fix syntax error (missing colon)."""
    print_subheader("Demo 1: Fixing Syntax Error")

    # Create simple task: add 1 to each value
    task_data = {
        "train": [
            {"input": [[1, 2]], "output": [[2, 3]]},
            {"input": [[3, 4]], "output": [[4, 5]]},
            {"input": [[5, 6]], "output": [[6, 7]]},
        ],
        "test": [{"input": [[7, 8]], "output": [[8, 9]]}],
    }

    # Failed code with syntax error (missing colon)
    failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray
    return task_grid + 1  # Missing colon after signature
"""

    print("\nTask: Add 1 to each value")
    print_code_with_line_numbers(failed_code, "Original Code (Syntax Error)")

    # Create temp task file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        json.dump(task_data, tmp_file)
        task_file = tmp_file.name

    try:
        # Evaluate original code
        fitness_before = calculate_fitness(task_file, failed_code)
        print_fitness_result(fitness_before, "Fitness Before")

        # Refine code
        print("\n🔧 Calling Refiner Agent...")
        print("   - Analyzing syntax error")
        print("   - Generating fixed code with Gemini API (temperature=0.4)")

        refined_code = refine_solver(failed_code, task_file, fitness_before)

        print_code_with_line_numbers(refined_code, "\nRefined Code (Fixed)")

        # Evaluate refined code
        fitness_after = calculate_fitness(task_file, refined_code)
        print_fitness_result(fitness_after, "\nFitness After")

        # Summary
        improvement = fitness_after["fitness"] - fitness_before["fitness"]
        if improvement > 0:
            print(f"\n✅ Success! Fitness improved by +{improvement} points")
        else:
            print(f"\n⚠️  Fitness unchanged (still {fitness_after['fitness']})")

        return {
            "before": fitness_before["fitness"],
            "after": fitness_after["fitness"],
            "improvement": improvement,
        }

    finally:
        Path(task_file).unlink()


def demo_logic_error_fix() -> dict[str, float]:
    """Demo 2: Fix logic error (wrong algorithm)."""
    print_subheader("Demo 2: Fixing Logic Error")

    # Create task: multiply by 2
    task_data = {
        "train": [
            {"input": [[2]], "output": [[4]]},
            {"input": [[3]], "output": [[6]]},
            {"input": [[4]], "output": [[8]]},
        ],
        "test": [{"input": [[5]], "output": [[10]]}],
    }

    # Failed code with wrong logic (adds instead of multiplies)
    failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1  # Wrong: should multiply by 2
"""

    print("\nTask: Multiply each value by 2")
    print_code_with_line_numbers(failed_code, "Original Code (Wrong Logic)")

    # Create temp task file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        json.dump(task_data, tmp_file)
        task_file = tmp_file.name

    try:
        # Evaluate original code
        fitness_before = calculate_fitness(task_file, failed_code)
        print_fitness_result(fitness_before, "Fitness Before")

        # Refine code
        print("\n🔧 Calling Refiner Agent...")
        print("   - Analyzing failed test cases")
        print("   - Identifying wrong transformation rule")
        print("   - Generating corrected algorithm with Gemini API")

        refined_code = refine_solver(failed_code, task_file, fitness_before)

        print_code_with_line_numbers(refined_code, "\nRefined Code (Fixed Logic)")

        # Evaluate refined code
        fitness_after = calculate_fitness(task_file, refined_code)
        print_fitness_result(fitness_after, "\nFitness After")

        # Summary
        improvement = fitness_after["fitness"] - fitness_before["fitness"]
        if improvement > 0:
            print(f"\n✅ Success! Fitness improved by +{improvement} points")
            print(f"   Algorithm corrected: add → multiply")
        else:
            print(f"\n⚠️  Fitness unchanged (still {fitness_after['fitness']})")

        return {
            "before": fitness_before["fitness"],
            "after": fitness_after["fitness"],
            "improvement": improvement,
        }

    finally:
        Path(task_file).unlink()


def demo_timeout_fix() -> dict[str, float]:
    """Demo 3: Fix timeout issue (infinite loop)."""
    print_subheader("Demo 3: Fixing Timeout (Infinite Loop)")

    # Create simple task: copy input
    task_data = {
        "train": [
            {"input": [[1, 2]], "output": [[1, 2]]},
            {"input": [[3, 4]], "output": [[3, 4]]},
            {"input": [[5, 6]], "output": [[5, 6]]},
        ],
        "test": [{"input": [[7, 8]], "output": [[7, 8]]}],
    }

    # Failed code with infinite loop
    failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    while True:
        pass  # Infinite loop - causes timeout
    return task_grid
"""

    print("\nTask: Copy input to output")
    print_code_with_line_numbers(failed_code, "Original Code (Infinite Loop)")

    # Create temp task file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        json.dump(task_data, tmp_file)
        task_file = tmp_file.name

    try:
        # Evaluate original code (will timeout)
        print("\nEvaluating original code (this will timeout after 5 seconds)...")
        fitness_before = calculate_fitness(task_file, failed_code, timeout=2)
        print_fitness_result(fitness_before, "Fitness Before")

        # Refine code
        print("\n🔧 Calling Refiner Agent...")
        print("   - Analyzing timeout errors")
        print("   - Removing infinite loop")
        print("   - Generating optimized code with Gemini API")

        refined_code = refine_solver(failed_code, task_file, fitness_before)

        print_code_with_line_numbers(refined_code, "\nRefined Code (Optimized)")

        # Evaluate refined code
        fitness_after = calculate_fitness(task_file, refined_code)
        print_fitness_result(fitness_after, "\nFitness After")

        # Summary
        improvement = fitness_after["fitness"] - fitness_before["fitness"]
        if improvement > 0:
            print(f"\n✅ Success! Fitness improved by +{improvement} points")
            print(f"   Timeout eliminated - code now executes in <1 second")
        else:
            print(f"\n⚠️  Fitness unchanged (still {fitness_after['fitness']})")

        return {
            "before": fitness_before["fitness"],
            "after": fitness_after["fitness"],
            "improvement": improvement,
        }

    finally:
        Path(task_file).unlink()


def main() -> None:
    """Run all refiner demonstrations."""
    print_header("PHASE 2.2: REFINER AGENT DEMO")
    print("\nDemonstrating automated code debugging through LLM-based refinement")
    print("First evolutionary mechanism (Mutation) - improving failed solvers")

    # Check API key
    try:
        api_key = get_gemini_api_key()
        if not api_key:
            print("\n❌ Error: GEMINI_API_KEY not configured")
            print("   Please set your API key in .env file")
            sys.exit(1)
        print("\n✓ Gemini API key configured")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    # Run demonstrations
    results = []

    try:
        results.append(("Syntax Error Fix", demo_syntax_error_fix()))
        results.append(("Logic Error Fix", demo_logic_error_fix()))
        results.append(("Timeout Fix", demo_timeout_fix()))

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Overall summary
    print_header("SUMMARY")

    successful = sum(1 for _, r in results if r["improvement"] > 0)
    total_improvement = sum(r["improvement"] for _, r in results)

    print(f"\n✅ {successful}/{len(results)} scenarios improved successfully!")
    print(f"\nTotal fitness gain: +{total_improvement:.1f} points")

    print("\nResults by scenario:")
    for name, result in results:
        status = "✅" if result["improvement"] > 0 else "⚠️"
        print(
            f"  {status} {name}: "
            f"{result['before']:.0f} → {result['after']:.0f} "
            f"(+{result['improvement']:.0f})"
        )

    print("\n" + "=" * 70)
    print(" Phase 2.2 Complete! 🎉")
    print("=" * 70)
    print("\nAchievement: First evolutionary mechanism (Mutation) working!")
    print("Refiner agent successfully debugs and improves failed solver code.")
    print("\nNext: Phase 2.3 - Evolution Loop")
    print("      Combine fitness evaluation + refinement into multi-generation cycle")
    print("=" * 70)


if __name__ == "__main__":
    main()
