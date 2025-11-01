#!/usr/bin/env python3
"""Demo script for PopulationEvolution - genetic algorithm for ARC solving.

This demonstrates the full AI Civilization with population-based evolution:
- Multiple solvers evolving simultaneously
- Tournament selection for parents
- Crossover (technique fusion) + Mutation (refinement)
- Elitism-based survivor selection
- Diversity tracking across generations
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from arc_prometheus.evolutionary_engine.population_evolution import (  # noqa: E402
    PopulationEvolution,
)


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate population-based evolution for ARC solving"
    )

    # Task selection
    parser.add_argument(
        "task_path",
        nargs="?",
        default=str(
            project_root
            / "data"
            / "arc-prize-2025"
            / "arc-agi_training_challenges.json"
        ),
        help="Path to ARC task JSON file (default: training challenges)",
    )

    # Population parameters
    parser.add_argument(
        "--population-size",
        type=int,
        default=5,
        help="Number of solvers in population (default: 5)",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=3,
        help="Maximum evolution generations (default: 3)",
    )
    parser.add_argument(
        "--target-fitness",
        type=float,
        default=None,
        help="Early stopping fitness threshold (default: None)",
    )

    # Genetic algorithm parameters
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.5,
        help="Probability of crossover (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.2,
        help="Probability of mutation (0.0-1.0, default: 0.2)",
    )

    # LLM parameters
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="LLM model name (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response caching",
    )

    args = parser.parse_args()

    # Extract task ID from path
    task_path = Path(args.task_path)
    if not task_path.exists():
        print(f"❌ Task file not found: {task_path}")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("🧬 Population-Based Evolution Demo")
    print(f"{'=' * 80}\n")

    print("Configuration:")
    print(f"  Task: {task_path.stem}")
    print(f"  Population Size: {args.population_size}")
    print(f"  Max Generations: {args.max_generations}")
    print(f"  Crossover Rate: {args.crossover_rate}")
    print(f"  Mutation Rate: {args.mutation_rate}")
    print(f"  Model: {args.model}")
    print(f"  Cache: {'Disabled' if args.no_cache else 'Enabled'}")
    if args.target_fitness:
        print(f"  Target Fitness: {args.target_fitness}")
    print()

    # Initialize population evolution
    pop_evo = PopulationEvolution(
        population_size=args.population_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        model_name=args.model,
        use_cache=not args.no_cache,
        verbose=True,
    )

    # Run evolution
    try:
        result = pop_evo.evolve_population(
            task_json_path=str(task_path),
            max_generations=args.max_generations,
            target_fitness=args.target_fitness,
        )

        # Display results
        print(f"\n{'=' * 80}")
        print("📊 Final Results")
        print(f"{'=' * 80}\n")

        print("Best Solver:")
        print(f"  ID: {result.best_solver.solver_id}")
        print(f"  Fitness: {result.best_solver.fitness_score}")

        # Calculate train accuracy
        train_pct = (
            (result.best_solver.train_correct / result.best_solver.train_total * 100)
            if result.best_solver.train_total > 0
            else 0.0
        )
        print(
            f"  Train: {result.best_solver.train_correct}/{result.best_solver.train_total} ({train_pct:.1f}%)"
        )

        # Calculate test accuracy
        test_pct = (
            (result.best_solver.test_correct / result.best_solver.test_total * 100)
            if result.best_solver.test_total > 0
            else 0.0
        )
        print(
            f"  Test: {result.best_solver.test_correct}/{result.best_solver.test_total} ({test_pct:.1f}%)"
        )
        print(
            f"  Techniques: {', '.join(result.best_solver.tags) if result.best_solver.tags else 'None identified'}"
        )
        print(f"  Generation: {result.best_solver.generation}")
        print()

        print("Population Statistics:")
        print(f"  Final Population Size: {len(result.final_population)}")
        print(f"  Generations Completed: {len(result.generation_history)}")
        print(f"  Total Time: {result.total_time:.2f}s")
        print()

        print("Generation History:")
        for stats in result.generation_history:
            print(
                f"  Gen {stats.generation}: "
                f"best={stats.best_fitness:.1f}, "
                f"avg={stats.average_fitness:.2f}, "
                f"diversity={stats.diversity_score:.2f}, "
                f"crossover={stats.crossover_events}, "
                f"mutation={stats.mutation_events}"
            )

        print(f"\n{'=' * 80}")
        print("✅ Evolution Complete!")
        print(f"{'=' * 80}\n")

    except KeyboardInterrupt:
        print("\n\n❌ Evolution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during evolution: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
