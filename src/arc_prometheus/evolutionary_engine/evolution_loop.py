"""Multi-generation evolution loop - iterative solver improvement (Phase 2.3).

This module implements the complete evolutionary cycle:
1. Generate initial solver (Programmer)
2. Evaluate fitness
3. Refine if below target (Refiner - Mutation)
4. Track improvement across generations
5. Terminate when target fitness reached or max generations hit
"""

import time
from typing import TypedDict

from ..cognitive_cells.programmer import generate_solver
from ..cognitive_cells.refiner import refine_solver
from ..crucible.data_loader import load_task
from .fitness import FitnessResult, calculate_fitness


class GenerationResult(TypedDict):
    """Result from a single evolution generation.

    Attributes:
        generation: Generation number (0-indexed)
        solver_code: Current solver code for this generation
        fitness_result: Complete fitness evaluation result
        refinement_count: Number of refinements applied this generation (0 or 1)
        total_time: Time taken for this generation in seconds
        improvement: Fitness improvement from previous generation
    """

    generation: int
    solver_code: str
    fitness_result: FitnessResult
    refinement_count: int
    total_time: float
    improvement: float


def run_evolution_loop(
    task_json_path: str,
    max_generations: int = 5,
    target_fitness: float | None = None,
    timeout_per_eval: int = 5,
    timeout_per_llm: int = 60,
    verbose: bool = True,
) -> list[GenerationResult]:
    """Run multi-generation evolution loop on ARC task.

    Process:
        1. Generate initial solver from train examples (Programmer)
        2. Evaluate fitness with calculate_fitness()
        3. If fitness < target_fitness, refine code (Refiner - Mutation)
        4. Repeat for max_generations or until target_fitness reached
        5. Track and return all generation results

    Args:
        task_json_path: Path to ARC task JSON file
        max_generations: Maximum evolution generations (default: 5)
        target_fitness: Stop when fitness >= this value (default: None = never stop early)
        timeout_per_eval: Timeout for sandbox execution per example in seconds (default: 5)
        timeout_per_llm: Timeout for LLM API calls in seconds (default: 60)
        verbose: Print progress information (default: True)

    Returns:
        List of GenerationResult dicts, one per generation

    Raises:
        FileNotFoundError: If task file not found
        ValueError: If API key not configured
        Exception: If LLM API call fails

    Example:
        >>> results = run_evolution_loop("task.json", max_generations=3, target_fitness=11)  # doctest: +SKIP
        >>> print(f"Final fitness: {results[-1]['fitness_result']['fitness']}")  # doctest: +SKIP
        Final fitness: 13
        >>> # Check improvement over generations
        >>> for r in results:  # doctest: +SKIP
        ...     print(f"Gen {r['generation']}: fitness = {r['fitness_result']['fitness']}")
        Gen 0: fitness = 3
        Gen 1: fitness = 13

    Notes:
        - Generation 0 is always initial generation (no refinement)
        - Subsequent generations refine if fitness < target_fitness
        - Early termination when target_fitness reached saves API calls
        - Each generation tracks its own timing for performance analysis
    """
    # Load task once (used for prompt creation)
    task_data = load_task(task_json_path)
    train_pairs = task_data.get("train", [])

    if not train_pairs:
        raise ValueError(f"Task {task_json_path} has no train examples")

    results: list[GenerationResult] = []
    current_code: str = ""
    previous_fitness: float = 0.0

    for generation in range(max_generations):
        gen_start_time = time.time()

        if verbose:
            print(f"\n{'=' * 70}")
            print(f" Generation {generation}")
            print(f"{'=' * 70}")

        # Generation 0: Generate initial solver
        if generation == 0:
            if verbose:
                print("\n📝 Generating initial solver from train examples...")

            current_code = generate_solver(train_pairs, timeout=timeout_per_llm)

            if verbose:
                print(f"✅ Initial solver generated ({len(current_code)} characters)")

            refinement_count = 0

        # Subsequent generations: Refine if needed
        else:
            # Check if refinement needed
            if target_fitness is not None and previous_fitness >= target_fitness:
                # Target already reached, stop evolution
                if verbose:
                    print(
                        f"\n🎯 Target fitness {target_fitness} reached in generation {generation - 1}"
                    )
                    print("Evolution complete!")
                break

            # Refine code
            if verbose:
                print(
                    f"\n🔧 Refining solver (fitness {previous_fitness:.1f} < target {target_fitness if target_fitness else 'N/A'})..."
                )

            # Get previous fitness result for refiner context
            prev_result = results[-1]["fitness_result"]
            current_code = refine_solver(
                current_code, task_json_path, prev_result, timeout=timeout_per_llm
            )

            if verbose:
                print(f"✅ Solver refined ({len(current_code)} characters)")

            refinement_count = 1

        # Evaluate fitness
        if verbose:
            print("\n📊 Evaluating fitness...")

        fitness_result = calculate_fitness(
            task_json_path, current_code, timeout=timeout_per_eval
        )

        current_fitness = fitness_result["fitness"]

        if verbose:
            print(f"Fitness: {current_fitness:.1f}")
            print(
                f"  Train: {fitness_result['train_correct']}/{fitness_result['train_total']} "
                f"({fitness_result['train_accuracy']:.0%})"
            )
            print(
                f"  Test: {fitness_result['test_correct']}/{fitness_result['test_total']} "
                f"({fitness_result['test_accuracy']:.0%})"
            )

            if generation > 0:
                improvement = current_fitness - previous_fitness
                print(f"  Improvement: {improvement:+.1f}")

        # Calculate metrics
        gen_total_time = time.time() - gen_start_time
        improvement = (
            float(current_fitness - previous_fitness) if generation > 0 else 0.0
        )

        # Record generation result
        generation_result: GenerationResult = {
            "generation": generation,
            "solver_code": current_code,
            "fitness_result": fitness_result,
            "refinement_count": refinement_count,
            "total_time": gen_total_time,
            "improvement": improvement,
        }

        results.append(generation_result)

        # Update for next iteration
        previous_fitness = current_fitness

        if verbose:
            print(f"⏱️  Generation time: {gen_total_time:.2f}s")

        # Check early termination
        if target_fitness is not None and current_fitness >= target_fitness:
            if verbose:
                print(f"\n🎯 Target fitness {target_fitness} reached!")
                print("Evolution complete!")
            break

    if verbose:
        print(f"\n{'=' * 70}")
        print(" Evolution Summary")
        print(f"{'=' * 70}")
        print(f"Generations completed: {len(results)}")
        print(f"Initial fitness: {results[0]['fitness_result']['fitness']:.1f}")
        print(f"Final fitness: {results[-1]['fitness_result']['fitness']:.1f}")
        total_improvement = (
            results[-1]["fitness_result"]["fitness"]
            - results[0]["fitness_result"]["fitness"]
        )
        print(f"Total improvement: {total_improvement:+.1f}")
        total_time = sum(r["total_time"] for r in results)
        print(f"Total time: {total_time:.2f}s")

    return results
