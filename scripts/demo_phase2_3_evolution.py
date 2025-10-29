"""Demo script for Phase 2.3: Evolution Loop - Multi-generation solver evolution.

This demonstrates the complete evolutionary cycle:
1. Generate initial solver (Programmer)
2. Evaluate fitness
3. Refine if below target (Refiner - Mutation)
4. Track improvement across generations
5. Terminate when target reached or max generations hit

Usage:
    python scripts/demo_phase2_3_evolution.py

Requirements:
    - GEMINI_API_KEY environment variable must be set

Demonstrations:
    1. Simple Task Evolution: Overfitting → Generalization (2-3 generations)
    2. Early Convergence: Perfect solver from start (1 generation)
    3. Gradual Improvement: Progressive refinement (5 generations)
"""

import json
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop
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
    max_lines = 40  # Limit display for readability
    for i, line in enumerate(lines[:max_lines], 1):
        print(f"  {i:3d} | {line}")
    if len(lines) > max_lines:
        print(f"  ... ({len(lines) - max_lines} more lines)")


def print_generation_summary(results: list) -> None:
    """Print evolution summary across all generations."""
    print_subheader("Evolution Summary")
    print(f"\nGenerations completed: {len(results)}")
    print(f"Initial fitness: {results[0]['fitness_result']['fitness']:.1f}")
    print(f"Final fitness: {results[-1]['fitness_result']['fitness']:.1f}")

    total_improvement = (
        results[-1]["fitness_result"]["fitness"]
        - results[0]["fitness_result"]["fitness"]
    )
    print(f"Total improvement: {total_improvement:+.1f} points")

    total_time = sum(r["total_time"] for r in results)
    print(f"Total time: {total_time:.1f}s")

    # Generation-by-generation breakdown
    print("\nGeneration Details:")
    for result in results:
        gen = result["generation"]
        fitness = result["fitness_result"]["fitness"]
        improvement = result["improvement"]
        train = result["fitness_result"]["train_correct"]
        train_total = result["fitness_result"]["train_total"]
        test = result["fitness_result"]["test_correct"]
        test_total = result["fitness_result"]["test_total"]
        time_taken = result["total_time"]

        print(
            f"  Gen {gen}: fitness={fitness:.1f} ({improvement:+.1f}) "
            f"[train: {train}/{train_total}, test: {test}/{test_total}] "
            f"({time_taken:.1f}s)"
        )


def demo_1_simple_evolution():
    """Demo 1: Simple task with clear evolution pattern.

    Task: Multiply all values by 2
    Expected: Initial solver may overfit, refinement fixes generalization
    """
    print_subheader("Demo 1: Simple Task Evolution")
    print("\nTask: Multiply all grid values by 2")
    print("Expected: 2-3 generations to reach perfect fitness")

    # Create simple task
    task_data = {
        "train": [
            {"input": [[1, 2]], "output": [[2, 4]]},
            {"input": [[3, 4]], "output": [[6, 8]]},
            {"input": [[5, 6]], "output": [[10, 12]]},
        ],
        "test": [{"input": [[7, 8]], "output": [[14, 16]]}],
    }

    # Create temp file
    tmp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w", suffix=".json", delete=False
    )

    try:
        json.dump(task_data, tmp_file)
        tmp_file.close()
        task_file = tmp_file.name

        print("\n🧬 Starting evolution...")

        # Run evolution with target fitness of 13 (3 train + 1 test * 10)
        results = run_evolution_loop(
            task_file,
            max_generations=5,
            target_fitness=13.0,
            verbose=False,  # Use custom output format
        )

        # Display results
        print("\n✅ Evolution complete!")
        print_generation_summary(results)

        # Show final solver
        if results[-1]["fitness_result"]["fitness"] >= 13:
            print("\n🏆 Perfect solver achieved!")
            print_code_with_line_numbers(
                results[-1]["solver_code"], "Final Solver Code"
            )
        else:
            print(
                f"\n⚠️  Evolution stopped but target not reached "
                f"(fitness: {results[-1]['fitness_result']['fitness']:.1f})"
            )

        return results

    finally:
        # Clean up temp file
        Path(tmp_file.name).unlink(missing_ok=True)


def demo_2_early_convergence():
    """Demo 2: Trivial task where initial solver is perfect.

    Task: Copy input to output
    Expected: 1 generation (initial is perfect, no refinement needed)
    """
    print_subheader("Demo 2: Early Convergence")
    print("\nTask: Copy input to output (identity function)")
    print("Expected: 1 generation (perfect from start)")

    # Create trivial task
    task_data = {
        "train": [
            {"input": [[1, 2, 3]], "output": [[1, 2, 3]]},
            {"input": [[4, 5, 6]], "output": [[4, 5, 6]]},
        ],
        "test": [{"input": [[7, 8, 9]], "output": [[7, 8, 9]]}],
    }

    tmp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w", suffix=".json", delete=False
    )

    try:
        json.dump(task_data, tmp_file)
        tmp_file.close()
        task_file = tmp_file.name

        print("\n🧬 Starting evolution...")

        # Run evolution with target fitness
        results = run_evolution_loop(
            task_file,
            max_generations=5,
            target_fitness=12.0,  # 2 train + 1 test * 10
            verbose=False,
        )

        print("\n✅ Evolution complete!")
        print_generation_summary(results)

        if len(results) == 1:
            print("\n💡 Analysis: Perfect solver generated immediately!")
            print("   No refinement needed - early termination saved API calls.")

        return results

    finally:
        Path(tmp_file.name).unlink(missing_ok=True)


def demo_3_gradual_improvement():
    """Demo 3: More complex task showing gradual improvement.

    Task: Add 5 to each value
    Expected: May take multiple generations to converge
    """
    print_subheader("Demo 3: Gradual Improvement")
    print("\nTask: Add 5 to each grid value")
    print("Expected: Gradual improvement over 3-5 generations")

    # Create task
    task_data = {
        "train": [
            {"input": [[1, 2]], "output": [[6, 7]]},
            {"input": [[3, 4]], "output": [[8, 9]]},
            {"input": [[10, 15]], "output": [[15, 20]]},
        ],
        "test": [
            {"input": [[20, 25]], "output": [[25, 30]]},
            {"input": [[0, 5]], "output": [[5, 10]]},
        ],
    }

    tmp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w", suffix=".json", delete=False
    )

    try:
        json.dump(task_data, tmp_file)
        tmp_file.close()
        task_file = tmp_file.name

        print("\n🧬 Starting evolution...")

        # Run evolution without target (run all generations)
        results = run_evolution_loop(
            task_file,
            max_generations=5,
            target_fitness=None,  # No early termination
            verbose=False,
        )

        print("\n✅ Evolution complete!")
        print_generation_summary(results)

        # Analyze improvement pattern
        improvements = [r["improvement"] for r in results]
        if any(imp > 0 for imp in improvements[1:]):
            print("\n💡 Analysis: Evolution showed positive improvements")
        elif all(imp <= 0 for imp in improvements[1:]):
            print("\n⚠️  Analysis: Evolution struggled to improve (plateau)")

        return results

    finally:
        Path(tmp_file.name).unlink(missing_ok=True)


def main():
    """Run all evolution loop demos."""
    print_header("PHASE 2.3: EVOLUTION LOOP DEMO")
    print("\nDemonstrating multi-generation solver evolution")
    print("Combines: Programmer → Fitness → Refiner → Repeat")

    # Check API key
    try:
        api_key = get_gemini_api_key()
        if not api_key:
            print("\n❌ ERROR: GEMINI_API_KEY not configured")
            print("Please set GEMINI_API_KEY environment variable")
            return 1
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return 1

    print("\n✓ Gemini API key configured")

    # Run demos
    demo_results = []

    try:
        # Demo 1: Simple evolution
        result1 = demo_1_simple_evolution()
        demo_results.append(("Simple Task Evolution", result1))

        # Demo 2: Early convergence
        result2 = demo_2_early_convergence()
        demo_results.append(("Early Convergence", result2))

        # Demo 3: Gradual improvement
        result3 = demo_3_gradual_improvement()
        demo_results.append(("Gradual Improvement", result3))

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Final summary
    print_header("OVERALL SUMMARY")

    successful = sum(
        1
        for _, results in demo_results
        if results[-1]["fitness_result"]["fitness"]
        > results[0]["fitness_result"]["fitness"]
    )

    print(f"\n✅ {successful}/{len(demo_results)} demos showed improvement!")

    print("\nEvolution Results:")
    for name, results in demo_results:
        initial = results[0]["fitness_result"]["fitness"]
        final = results[-1]["fitness_result"]["fitness"]
        improvement = final - initial
        generations = len(results)

        status = "✅" if improvement > 0 else "⚠️ "
        print(
            f"  {status} {name}: Gen 0: {initial:.1f} → Gen {generations - 1}: "
            f"{final:.1f} ({improvement:+.1f}) [{generations} generation(s)]"
        )

    print_header("Phase 2.3 Complete! 🎉")
    print("\nAchievement: Multi-generation evolution working!")
    print("Solvers improve iteratively through fitness-guided refinement.")
    print("\nNext: Phase 3 - Scaling & Crossover")
    print("      Build solver database, implement Crossover, scale to full dataset")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
