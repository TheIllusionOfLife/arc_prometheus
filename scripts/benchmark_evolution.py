"""Benchmark script for real-world ARC task evaluation.

This script runs the evolution loop on diverse ARC tasks to measure performance,
identify bottlenecks, and validate the Phase 2 system before building Phase 3.

Usage:
    # Benchmark specific tasks
    python scripts/benchmark_evolution.py \\
        --tasks "00576224,007bbfb7,025d127b" \\
        --output-dir results/multiprocess_baseline/ \\
        --experiment-name "multiprocess_baseline"

    # Random sample from training set
    python scripts/benchmark_evolution.py \\
        --random-sample 15 \\
        --training-data data/arc-prize-2025/arc-agi_training_challenges.json \\
        --output-dir results/docker_baseline/ \\
        --experiment-name "docker_baseline" \\
        --sandbox-mode docker

    # Load task IDs from file
    python scripts/benchmark_evolution.py \\
        --task-ids-file benchmark_tasks.txt \\
        --output-dir results/test_run/ \\
        --experiment-name "test_run"

    # Resume interrupted run (skips completed tasks)
    python scripts/benchmark_evolution.py \\
        --task-ids-file benchmark_tasks.txt \\
        --output-dir results/test_run/ \\
        --experiment-name "test_run" \\
        --resume

Output:
    - Individual task results: {output_dir}/task_{task_id}.json
    - Aggregate summary: {output_dir}/summary.json
    - Experiment metadata: {output_dir}/metadata.json
"""

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

# Add src to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop
from arc_prometheus.evolutionary_engine.submission_formatter import (
    format_submission_json,
    generate_task_predictions,
    select_diverse_solvers,
)
from arc_prometheus.utils.config import MODEL_NAME as DEFAULT_MODEL_NAME


def validate_task_id(task_id: str) -> bool:
    """Validate ARC task ID format (8-character hexadecimal string).

    Args:
        task_id: Task ID to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_task_id("007bbfb7")
        True
        >>> validate_task_id("invalid")
        False
        >>> validate_task_id("007bbfb7extra")
        False
    """
    return bool(re.match(r"^[0-9a-f]{8}$", task_id.lower()))


def load_task_ids_from_file(filepath: str) -> list[str]:
    """Load task IDs from text file (one per line).

    Args:
        filepath: Path to text file containing task IDs

    Returns:
        List of task ID strings (8-char hex)

    Example file format:
        # Comment lines start with #
        00576224
        007bbfb7
        025d127b  # Inline comments also supported
    """
    task_ids = []

    with open(filepath) as f:
        for line in f:
            # Remove inline comments
            line = line.split("#")[0].strip()

            # Skip empty lines
            if not line:
                continue

            task_ids.append(line)

    # Validate all task IDs
    invalid = [tid for tid in task_ids if not validate_task_id(tid)]
    if invalid:
        raise ValueError(
            f"Invalid task ID format in {filepath}: {invalid}. "
            f"Expected 8-character hexadecimal strings."
        )

    return task_ids


def parse_task_ids(task_ids_str: str) -> list[str]:
    """Parse comma-separated task IDs from CLI argument.

    Args:
        task_ids_str: Comma-separated task IDs

    Returns:
        List of task ID strings

    Example:
        >>> parse_task_ids("00576224, 007bbfb7, 025d127b")
        ['00576224', '007bbfb7', '025d127b']
    """
    task_ids = [tid.strip() for tid in task_ids_str.split(",")]

    # Validate all task IDs
    invalid = [tid for tid in task_ids if not validate_task_id(tid)]
    if invalid:
        raise ValueError(
            f"Invalid task ID format: {invalid}. "
            f"Expected 8-character hexadecimal strings (e.g., '007bbfb7')."
        )

    return task_ids


def random_sample_tasks(
    training_challenges_path: str, n: int, seed: int | None = None
) -> list[str]:
    """Randomly sample N tasks from training challenges file.

    Args:
        training_challenges_path: Path to arc-agi_training_challenges.json
        n: Number of tasks to sample
        seed: Random seed for reproducibility (default: None)

    Returns:
        List of N random task IDs
    """
    with open(training_challenges_path) as f:
        training_data = json.load(f)

    all_task_ids = list(training_data.keys())

    if seed is not None:
        random.seed(seed)

    return random.sample(all_task_ids, min(n, len(all_task_ids)))


def get_completed_task_ids(output_dir: str) -> set[str]:
    """Get task IDs that already have result files in output directory.

    Args:
        output_dir: Directory containing result files

    Returns:
        Set of task IDs that have been completed
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        return set()

    completed = set()

    for result_file in output_path.glob("task_*.json"):
        # Extract task ID from filename: task_00576224.json -> 00576224
        task_id = result_file.stem.replace("task_", "")
        completed.add(task_id)

    return completed


def filter_remaining_tasks(
    all_task_ids: list[str], completed_task_ids: set[str]
) -> list[str]:
    """Filter out completed tasks to get remaining tasks.

    Args:
        all_task_ids: All task IDs to process
        completed_task_ids: Task IDs that are already completed

    Returns:
        List of remaining task IDs to process
    """
    return [tid for tid in all_task_ids if tid not in completed_task_ids]


def run_single_task_benchmark(
    task_id: str,
    training_challenges_path: str,
    max_generations: int = 5,
    target_fitness: float | None = None,
    sandbox_mode: str = "multiprocess",
    model_name: str | None = None,
    programmer_temperature: float | None = None,
    refiner_temperature: float | None = None,
    timeout_eval: int = 5,
    timeout_llm: int = 60,
    use_cache: bool = True,
    generate_submission: bool = False,
    num_attempts: int = 2,
) -> dict:
    """Run evolution loop benchmark on a single ARC task.

    Args:
        task_id: ARC task ID (8-char hex string)
        training_challenges_path: Path to arc-agi_training_challenges.json
        max_generations: Maximum evolution generations
        target_fitness: Optional early stopping fitness threshold
        sandbox_mode: "multiprocess" or "docker"
        model_name: LLM model name
        programmer_temperature: Temperature for code generation
        refiner_temperature: Temperature for debugging
        timeout_eval: Sandbox execution timeout per example
        timeout_llm: LLM API call timeout
        use_cache: Whether to use LLM response cache
        generate_submission: If True, generate pass@2 predictions
        num_attempts: Number of diverse attempts for pass@2 (default: 2)

    Returns:
        Dictionary with benchmark results:
        {
            "task_id": str,
            "success": bool,
            "generations": list[GenerationResult],
            "final_fitness": float,
            "total_generations": int,
            "total_time": float,
            "config": dict,
            "timestamp": str,
            "predictions": list[dict] | None,  # Only if generate_submission=True
            "error": str | None  # Only present if success=False
        }
    """
    start_time = datetime.now(UTC)

    result = {
        "task_id": task_id,
        "success": False,
        "timestamp": start_time.isoformat(),
        "config": {
            "max_generations": max_generations,
            "target_fitness": target_fitness,
            "sandbox_mode": sandbox_mode,
            "model_name": model_name,
            "programmer_temperature": programmer_temperature,
            "refiner_temperature": refiner_temperature,
            "timeout_eval": timeout_eval,
            "timeout_llm": timeout_llm,
            "use_cache": use_cache,
        },
    }

    try:
        # Load task data from collection
        # Note: load_task returns numpy arrays, but we need JSON
        # We'll load the raw JSON and write directly to temp file
        with open(training_challenges_path) as f:
            collection = json.load(f)

        if task_id not in collection:
            available_ids = list(collection.keys())[:5]
            raise ValueError(
                f"Task ID '{task_id}' not found in {training_challenges_path}. "
                f"Available IDs (showing first 5): {available_ids}"
            )

        task_data = collection[task_id]

        # Write to temp file for evolution loop
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(task_data, tmp_file)
            tmp_task_file = tmp_file.name

        try:
            # Run evolution loop
            generations = run_evolution_loop(
                task_json_path=tmp_task_file,
                max_generations=max_generations,
                target_fitness=target_fitness,
                timeout_per_eval=timeout_eval,
                timeout_per_llm=timeout_llm,
                verbose=False,  # Suppress console output for batch processing
                sandbox_mode=sandbox_mode,
                model_name=model_name,
                programmer_temperature=programmer_temperature,
                refiner_temperature=refiner_temperature,
                use_cache=use_cache,
            )

            # Calculate metrics
            total_time = sum(gen["total_time"] for gen in generations)
            final_fitness = generations[-1]["fitness_result"]["fitness"]

            result.update(
                {
                    "success": True,
                    "generations": generations,
                    "final_fitness": final_fitness,
                    "total_generations": len(generations),
                    "total_time": total_time,
                }
            )

            # Generate pass@2 predictions if requested
            if generate_submission:
                try:
                    # Select diverse solvers from generation history
                    solver_codes = select_diverse_solvers(
                        generations,
                        num_attempts=num_attempts,
                        diversity_metric="fitness",
                    )

                    # Generate predictions for all test inputs
                    predictions = generate_task_predictions(
                        task_json_path=tmp_task_file,
                        solver_codes=solver_codes,
                        timeout=timeout_eval,
                        sandbox_mode=sandbox_mode,
                    )

                    result["predictions"] = predictions

                except ValueError as e:
                    # Not enough unique solvers - use fallback
                    # Generate predictions with single solver (duplicate attempts)
                    print(f"⚠️  Warning for task {task_id}: {e}")
                    print("   Falling back to duplicated best solver")

                    if generations:
                        best_solver = max(
                            generations, key=lambda g: g["fitness_result"]["fitness"]
                        )
                        # Duplicate best solver for all attempts
                        solver_codes = [best_solver["solver_code"]] * num_attempts

                        predictions = generate_task_predictions(
                            task_json_path=tmp_task_file,
                            solver_codes=solver_codes,
                            timeout=timeout_eval,
                            sandbox_mode=sandbox_mode,
                        )

                        result["predictions"] = predictions
                        result["prediction_warning"] = str(e)

        finally:
            # Clean up temp file (check exists to avoid NameError)
            if "tmp_task_file" in locals():
                Path(tmp_task_file).unlink(missing_ok=True)

    except Exception as e:
        # Capture exception details
        result.update(
            {"success": False, "error": str(e), "error_type": type(e).__name__}
        )

    return result


def save_task_result(result: dict, output_dir: str) -> None:
    """Save individual task result to JSON file.

    Args:
        result: Task benchmark result dictionary
        output_dir: Directory to save result file

    Creates:
        {output_dir}/task_{task_id}.json
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    task_id = result["task_id"]
    result_file = output_path / f"task_{task_id}.json"

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)


def generate_experiment_metadata(
    experiment_name: str, task_ids: list[str], config: dict
) -> dict:
    """Generate metadata for the benchmark experiment.

    Args:
        experiment_name: Name/label for this experiment
        task_ids: List of task IDs being benchmarked
        config: Experiment configuration dictionary

    Returns:
        Metadata dictionary with timestamp, git commit, etc.
    """
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "num_tasks": len(task_ids),
        "task_ids": task_ids,
        "config": config,
    }

    # Try to get git commit hash
    try:
        # S607: Safe to suppress - using shell=False with hardcoded command list
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            shell=False,  # Explicit is better than implicit
        )
        metadata["git_commit"] = result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        metadata["git_commit"] = "unknown"

    return metadata


def calculate_aggregate_statistics(task_results: list[dict]) -> dict[str, Any]:
    """Calculate aggregate statistics from all task results.

    Args:
        task_results: List of task benchmark result dictionaries

    Returns:
        Dictionary with aggregate statistics:
        {
            "total_tasks": int,
            "successful_tasks": int,
            "failed_tasks": int,
            "success_rate": float,
            "avg_final_fitness": float,
            "median_final_fitness": float,
            "avg_generations": float,
            "avg_time_per_task": float,
            "total_time": float,
            "error_distribution": dict[str, int]
        }
    """
    total_tasks = len(task_results)
    successful = [r for r in task_results if r.get("success", False)]
    failed = [r for r in task_results if not r.get("success", False)]

    stats: dict[str, Any] = {
        "total_tasks": total_tasks,
        "successful_tasks": len(successful),
        "failed_tasks": len(failed),
        "success_rate": len(successful) / total_tasks if total_tasks > 0 else 0.0,
    }

    if successful:
        fitnesses = [r["final_fitness"] for r in successful]
        stats["avg_final_fitness"] = sum(fitnesses) / len(fitnesses)
        stats["median_final_fitness"] = median(fitnesses)
        stats["avg_generations"] = sum(
            r["total_generations"] for r in successful
        ) / len(successful)
        stats["avg_time_per_task"] = sum(r["total_time"] for r in successful) / len(
            successful
        )
    else:
        stats.update(
            {
                "avg_final_fitness": 0.0,
                "median_final_fitness": 0.0,
                "avg_generations": 0.0,
                "avg_time_per_task": 0.0,
            }
        )

    # Calculate total time across all tasks
    total_time = sum(
        r.get("total_time", 0.0) for r in task_results if "total_time" in r
    )
    stats["total_time"] = total_time

    # Aggregate error distribution from all generations
    error_distribution: dict[str, int] = {}

    for result in successful:
        for generation in result.get("generations", []):
            error_summary = generation["fitness_result"].get("error_summary", {})
            for error_type, count in error_summary.items():
                error_distribution[error_type] = (
                    error_distribution.get(error_type, 0) + count
                )

    stats["error_distribution"] = error_distribution

    return stats


def parse_benchmark_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for benchmark script.

    Args:
        args: Optional argument list (for testing). If None, uses sys.argv

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Benchmark evolution loop on diverse ARC tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Task selection (mutually exclusive)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--tasks", type=str, help='Comma-separated task IDs (e.g., "00576224,007bbfb7")'
    )
    task_group.add_argument(
        "--task-ids-file", type=str, help="Path to file with task IDs (one per line)"
    )
    task_group.add_argument(
        "--random-sample",
        type=int,
        metavar="N",
        help="Randomly sample N tasks from training dataset",
    )

    # Required arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--experiment-name", type=str, required=True, help="Name for this experiment"
    )

    # Training data (required for random sampling)
    parser.add_argument(
        "--training-data",
        type=str,
        default="data/arc-prize-2025/arc-agi_training_challenges.json",
        help="Path to training challenges JSON file (default: %(default)s)",
    )

    # Evolution configuration
    parser.add_argument(
        "--max-generations",
        type=int,
        default=5,
        help="Maximum evolution generations (default: %(default)s)",
    )
    parser.add_argument(
        "--target-fitness",
        type=float,
        default=None,
        help="Early stopping fitness threshold (default: None)",
    )
    parser.add_argument(
        "--sandbox-mode",
        type=str,
        choices=["multiprocess", "docker"],
        default="multiprocess",
        help="Sandbox execution mode (default: %(default)s)",
    )

    # LLM configuration
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="LLM model name (default: %(default)s)",
    )
    parser.add_argument(
        "--programmer-temperature",
        type=float,
        default=0.3,
        help="Temperature for code generation (default: %(default)s)",
    )
    parser.add_argument(
        "--refiner-temperature",
        type=float,
        default=0.4,
        help="Temperature for debugging (default: %(default)s)",
    )

    # Timeout configuration
    parser.add_argument(
        "--timeout-eval",
        type=int,
        default=5,
        help="Sandbox execution timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout-llm",
        type=int,
        default=60,
        help="LLM API call timeout in seconds (default: %(default)s)",
    )

    # Cache configuration
    parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Disable LLM response caching",
    )
    parser.set_defaults(use_cache=True)

    # Resume capability
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks that already have result files (resume interrupted run)",
    )

    # Random seed
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    # Submission generation (pass@2)
    parser.add_argument(
        "--generate-submission",
        action="store_true",
        help="Generate pass@2 predictions for Kaggle submission format",
    )
    parser.add_argument(
        "--num-attempts",
        type=int,
        default=2,
        help="Number of diverse attempts per test input (default: %(default)s for pass@2)",
    )

    return parser.parse_args(args)


def main() -> int:
    """Main benchmark execution function."""
    args = parse_benchmark_args()

    # Print header
    print("=" * 70)
    print(" ARC-Prometheus Evolution Loop Benchmark")
    print("=" * 70)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Output Directory: {args.output_dir}")

    # Determine task IDs to benchmark
    if args.tasks:
        task_ids = parse_task_ids(args.tasks)
        print(f"Task Selection: Explicit ({len(task_ids)} tasks)")
    elif args.task_ids_file:
        task_ids = load_task_ids_from_file(args.task_ids_file)
        print(f"Task Selection: From file {args.task_ids_file} ({len(task_ids)} tasks)")
    elif args.random_sample:
        task_ids = random_sample_tasks(
            args.training_data, n=args.random_sample, seed=args.seed
        )
        print(
            f"Task Selection: Random sample ({len(task_ids)} tasks, seed={args.seed})"
        )
    else:
        print("ERROR: Must specify --tasks, --task-ids-file, or --random-sample")
        return 1

    # Resume: filter out completed tasks
    if args.resume:
        completed = get_completed_task_ids(args.output_dir)
        if completed:
            task_ids = filter_remaining_tasks(task_ids, completed)
            print(f"Resume Mode: Skipping {len(completed)} completed tasks")
            print(f"Remaining: {len(task_ids)} tasks")

    if not task_ids:
        print("\n✅ All tasks already completed!")
        return 0

    # Display configuration
    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Sandbox Mode: {args.sandbox_mode}")
    print(f"  Max Generations: {args.max_generations}")
    print(f"  Target Fitness: {args.target_fitness or 'None'}")
    print(f"  Programmer Temperature: {args.programmer_temperature}")
    print(f"  Refiner Temperature: {args.refiner_temperature}")
    print(f"  Eval Timeout: {args.timeout_eval}s")
    print(f"  LLM Timeout: {args.timeout_llm}s")
    print(f"  Cache Enabled: {args.use_cache}")

    # Generate and save experiment metadata
    config = {
        "model": args.model,
        "sandbox_mode": args.sandbox_mode,
        "max_generations": args.max_generations,
        "target_fitness": args.target_fitness,
        "programmer_temperature": args.programmer_temperature,
        "refiner_temperature": args.refiner_temperature,
        "timeout_eval": args.timeout_eval,
        "timeout_llm": args.timeout_llm,
        "use_cache": args.use_cache,
        "seed": args.seed,
    }

    metadata = generate_experiment_metadata(args.experiment_name, task_ids, config)

    # Save metadata
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved to {output_path / 'metadata.json'}")

    # Run benchmarks
    print("\n" + "=" * 70)
    print(" Running Benchmarks")
    print("=" * 70)

    task_results = []

    for i, task_id in enumerate(task_ids, 1):
        print(f"\n[{i}/{len(task_ids)}] Task: {task_id}")
        print("-" * 70)

        # Run benchmark for this task
        result = run_single_task_benchmark(
            task_id=task_id,
            training_challenges_path=args.training_data,
            max_generations=args.max_generations,
            target_fitness=args.target_fitness,
            sandbox_mode=args.sandbox_mode,
            model_name=args.model,
            programmer_temperature=args.programmer_temperature,
            refiner_temperature=args.refiner_temperature,
            timeout_eval=args.timeout_eval,
            timeout_llm=args.timeout_llm,
            use_cache=args.use_cache,
            generate_submission=args.generate_submission,
            num_attempts=args.num_attempts,
        )

        # Display result summary
        if result["success"]:
            print(
                f"✅ Success: Fitness {result['final_fitness']:.1f} "
                f"({result['total_generations']} gens, {result['total_time']:.1f}s)"
            )
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")

        # Save individual result
        save_task_result(result, args.output_dir)
        task_results.append(result)

    # Calculate and save aggregate statistics
    print("\n" + "=" * 70)
    print(" Aggregate Statistics")
    print("=" * 70)

    stats = calculate_aggregate_statistics(task_results)

    print(f"\nTotal Tasks: {stats['total_tasks']}")
    print(f"Successful: {stats['successful_tasks']}")
    print(f"Failed: {stats['failed_tasks']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")

    if stats["successful_tasks"] > 0:
        print("\nFitness Metrics:")
        print(f"  Average: {stats['avg_final_fitness']:.2f}")
        print(f"  Median: {stats['median_final_fitness']:.2f}")
        print("\nEvolution Metrics:")
        print(f"  Avg Generations: {stats['avg_generations']:.2f}")
        print(f"  Avg Time per Task: {stats['avg_time_per_task']:.1f}s")
        print(f"  Total Time: {stats['total_time']:.1f}s")

        if stats["error_distribution"]:
            print("\nError Distribution:")
            for error_type, count in sorted(stats["error_distribution"].items()):
                print(f"  {error_type}: {count}")

    # Save summary
    with open(output_path / "summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Summary saved to {output_path / 'summary.json'}")

    # Generate Kaggle submission if requested
    if args.generate_submission:
        print("\n" + "=" * 70)
        print(" Generating Kaggle Submission (pass@2)")
        print("=" * 70)

        # Collect predictions from successful tasks
        task_predictions = {}
        tasks_with_predictions = 0
        tasks_with_warnings = 0

        for result in task_results:
            if result.get("success", False) and "predictions" in result:
                task_predictions[result["task_id"]] = result["predictions"]
                tasks_with_predictions += 1

                if "prediction_warning" in result:
                    tasks_with_warnings += 1

        if task_predictions:
            # Format and save submission
            submission = format_submission_json(
                task_predictions, num_attempts=args.num_attempts
            )
            submission_path = output_path / "submission.json"

            with open(submission_path, "w") as f:
                json.dump(submission, f, indent=2)

            print(f"\n✓ Submission generated: {submission_path}")
            print(
                f"  Tasks with predictions: {tasks_with_predictions}/{len(task_results)}"
            )

            if tasks_with_warnings:
                print(
                    f"  ⚠️  Warning: {tasks_with_warnings} tasks used duplicate solvers "
                    f"(insufficient diversity)"
                )

            # Validate structure
            print("\n  Validating submission format...")
            try:
                # Reload to ensure it's valid JSON
                with open(submission_path) as f:
                    loaded = json.load(f)

                # Basic checks
                assert isinstance(loaded, dict), "Submission must be a dict"

                for task_id, predictions in loaded.items():
                    assert isinstance(predictions, list), (
                        f"Task {task_id} predictions must be list"
                    )
                    for pred_idx, pred in enumerate(predictions):
                        assert isinstance(pred, dict), (
                            f"Task {task_id} pred {pred_idx} must be dict"
                        )
                        # Validate all required attempts dynamically
                        for attempt_num in range(1, args.num_attempts + 1):
                            attempt_key = f"attempt_{attempt_num}"
                            assert attempt_key in pred, (
                                f"Task {task_id} pred {pred_idx} missing {attempt_key}"
                            )

                print("  ✅ Submission format valid!")

            except (json.JSONDecodeError, AssertionError) as e:
                print(f"  ❌ Validation failed: {e}")

        else:
            print("\n⚠️  No predictions generated (all tasks failed)")

    print("\n" + "=" * 70)
    print(" Benchmark Complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
