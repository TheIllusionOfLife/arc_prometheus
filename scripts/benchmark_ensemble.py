"""Benchmark script for test-time ensemble evaluation on ARC tasks.

This script runs the test-time ensemble approach (multi-persona + synthesis)
on ARC tasks and compares performance against baseline evolution.

Usage:
    # Quick test (5 tasks)
    python scripts/benchmark_ensemble.py \\
        --random-sample 5 \\
        --seed 42 \\
        --training-data data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json \\
        --output-dir results/ensemble_test_5tasks/ \\
        --experiment-name "ensemble_quick_test"

    # Full comparison (20 tasks)
    python scripts/benchmark_ensemble.py \\
        --random-sample 20 \\
        --seed 42 \\
        --training-data data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json \\
        --output-dir results/ensemble_20tasks/ \\
        --experiment-name "ensemble_vs_baseline"

Output:
    - Individual task results: {output_dir}/task_{task_id}.json
    - Aggregate summary: {output_dir}/summary.json
    - Experiment metadata: {output_dir}/metadata.json
"""

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

# Add src to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_prometheus.inference.test_time_ensemble import solve_task_ensemble
from arc_prometheus.utils.config import MODEL_NAME as DEFAULT_MODEL_NAME


def load_task_ids(
    training_data_path: str,
    random_sample: int | None = None,
    seed: int | None = None,
) -> list[str]:
    """Load task IDs from training data file.

    Args:
        training_data_path: Path to ARC JSON file
        random_sample: Number of random tasks to sample (None = all tasks)
        seed: Random seed for reproducibility

    Returns:
        List of task IDs
    """
    with open(training_data_path) as f:
        data = json.load(f)

    task_ids = list(data.keys())

    if random_sample is not None:
        if seed is not None:
            random.seed(seed)
        task_ids = random.sample(task_ids, min(random_sample, len(task_ids)))

    return task_ids


def run_ensemble_single_task(
    task_id: str,
    training_data_path: str,
    model_name: str,
    analyst_temperature: float,
    programmer_temperature: float,
    synthesis_temperature: float,
    use_cache: bool,
    timeout: int,
    sandbox_mode: str,
    use_active_inference: bool = False,
    augmentation_factor: int = 10,
) -> dict[str, Any]:
    """Run test-time ensemble on a single task.

    Args:
        task_id: ARC task ID
        training_data_path: Path to ARC JSON file
        model_name: LLM model name
        analyst_temperature: Temperature for analyst
        programmer_temperature: Temperature for programmer
        synthesis_temperature: Temperature for synthesis
        use_cache: Enable LLM caching
        timeout: Execution timeout
        sandbox_mode: Sandbox type
        use_active_inference: Enable training augmentation
        augmentation_factor: Augmentation variations

    Returns:
        Dictionary with task results
    """
    start_time = time.time()

    # Load task
    with open(training_data_path) as f:
        data = json.load(f)

    if task_id not in data:
        return {
            "task_id": task_id,
            "success": False,
            "error": f"Task {task_id} not found in dataset",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    task = data[task_id]

    try:
        # Run ensemble
        predictions = solve_task_ensemble(
            task=task,
            model_name=model_name,
            analyst_temperature=analyst_temperature,
            programmer_temperature=programmer_temperature,
            synthesis_temperature=synthesis_temperature,
            use_cache=use_cache,
            timeout=timeout,
            sandbox_mode=sandbox_mode,
            use_active_inference=use_active_inference,
            augmentation_factor=augmentation_factor,
        )

        # Evaluate predictions against ground truth
        test_results = []
        for idx, (best_pred, synth_pred) in enumerate(predictions):
            if idx >= len(task.get("test", [])):
                break

            test_example = task["test"][idx]
            if "output" not in test_example:
                # No ground truth available
                test_results.append(
                    {
                        "test_input_index": idx,
                        "best_prediction_correct": None,
                        "synthesis_prediction_correct": None,
                        "pass_at_2": None,
                    }
                )
                continue

            expected = np.array(test_example["output"], dtype=np.int64)

            best_correct = best_pred.shape == expected.shape and np.array_equal(
                best_pred, expected
            )
            synth_correct = synth_pred.shape == expected.shape and np.array_equal(
                synth_pred, expected
            )
            pass_at_2 = best_correct or synth_correct

            test_results.append(
                {
                    "test_input_index": idx,
                    "best_prediction_correct": bool(best_correct),
                    "synthesis_prediction_correct": bool(synth_correct),
                    "pass_at_2": bool(pass_at_2),
                }
            )

        # Calculate accuracies
        if test_results and test_results[0]["pass_at_2"] is not None:
            pass_at_2_accuracy = mean(r["pass_at_2"] for r in test_results)
            best_only_accuracy = mean(
                r["best_prediction_correct"] for r in test_results
            )
            synthesis_only_accuracy = mean(
                r["synthesis_prediction_correct"] for r in test_results
            )
        else:
            pass_at_2_accuracy = None
            best_only_accuracy = None
            synthesis_only_accuracy = None

        total_time = time.time() - start_time

        return {
            "task_id": task_id,
            "success": True,
            "timestamp": datetime.now(UTC).isoformat(),
            "config": {
                "model_name": model_name,
                "analyst_temperature": analyst_temperature,
                "programmer_temperature": programmer_temperature,
                "synthesis_temperature": synthesis_temperature,
                "timeout": timeout,
                "sandbox_mode": sandbox_mode,
                "use_active_inference": use_active_inference,
                "augmentation_factor": augmentation_factor,
            },
            "test_results": test_results,
            "pass_at_2_accuracy": pass_at_2_accuracy,
            "best_only_accuracy": best_only_accuracy,
            "synthesis_only_accuracy": synthesis_only_accuracy,
            "total_time": total_time,
            "api_calls": 3,  # Analyst + Programmer + Synthesis
        }

    except Exception as e:
        return {
            "task_id": task_id,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
            "total_time": time.time() - start_time,
        }


def save_task_result(output_dir: Path, result: dict[str, Any]) -> None:
    """Save task result to JSON file.

    Args:
        output_dir: Output directory
        result: Task result dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = result["task_id"]
    output_file = output_dir / f"task_{task_id}.json"

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


def calculate_aggregate_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate aggregate statistics across all tasks.

    Args:
        results: List of task results

    Returns:
        Dictionary with aggregate statistics
    """
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]

    # Filter results with ground truth
    results_with_gt = [
        r for r in successful_results if r.get("pass_at_2_accuracy") is not None
    ]

    if results_with_gt:
        avg_pass_at_2 = mean(r["pass_at_2_accuracy"] for r in results_with_gt)
        avg_best_only = mean(r["best_only_accuracy"] for r in results_with_gt)
        avg_synthesis_only = mean(r["synthesis_only_accuracy"] for r in results_with_gt)
        diversity_benefit = avg_pass_at_2 - max(avg_best_only, avg_synthesis_only)
    else:
        avg_pass_at_2 = 0.0
        avg_best_only = 0.0
        avg_synthesis_only = 0.0
        diversity_benefit = 0.0

    times = [r.get("total_time", 0) for r in successful_results if "total_time" in r]
    avg_time = mean(times) if times else 0.0
    total_time = sum(times)

    total_api_calls = sum(r.get("api_calls", 0) for r in successful_results)

    return {
        "total_tasks": len(results),
        "successful_tasks": len(successful_results),
        "failed_tasks": len(failed_results),
        "success_rate": len(successful_results) / len(results) if results else 0.0,
        "pass_at_2_accuracy": avg_pass_at_2,
        "best_only_accuracy": avg_best_only,
        "synthesis_only_accuracy": avg_synthesis_only,
        "diversity_benefit": diversity_benefit,
        "avg_time_per_task": avg_time,
        "total_time": total_time,
        "total_api_calls": total_api_calls,
        "tasks_with_ground_truth": len(results_with_gt),
    }


def generate_metadata(
    experiment_name: str,
    task_ids: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generate experiment metadata.

    Args:
        experiment_name: Experiment name
        task_ids: List of task IDs
        args: Command-line arguments

    Returns:
        Metadata dictionary
    """
    # Get git info if available
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
        git_branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        git_commit = None
        git_branch = None

    return {
        "experiment_name": experiment_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "config": {
            "model_name": args.model_name,
            "analyst_temperature": args.analyst_temperature,
            "programmer_temperature": args.programmer_temperature,
            "synthesis_temperature": args.synthesis_temperature,
            "timeout": args.timeout,
            "sandbox_mode": args.sandbox_mode,
            "use_cache": args.use_cache,
            "training_data": args.training_data,
            "random_sample": args.random_sample,
            "seed": args.seed,
        },
        "git_commit": git_commit,
        "git_branch": git_branch,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark test-time ensemble on ARC tasks"
    )

    # Task selection
    parser.add_argument(
        "--training-data",
        type=str,
        default="data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json",
        help="Path to ARC training data JSON file",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        help="Number of random tasks to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for task sampling",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output files",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name of the experiment",
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"LLM model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--analyst-temperature",
        type=float,
        default=1.0,
        help="Temperature for analyst (default: 1.0)",
    )
    parser.add_argument(
        "--programmer-temperature",
        type=float,
        default=0.0,
        help="Temperature for programmer (default: 0.0)",
    )
    parser.add_argument(
        "--synthesis-temperature",
        type=float,
        default=0.0,
        help="Temperature for synthesis (default: 0.0)",
    )

    # Execution configuration
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout for sandbox execution (seconds)",
    )
    parser.add_argument(
        "--sandbox-mode",
        type=str,
        default="multiprocess",
        choices=["multiprocess", "docker"],
        help="Sandbox mode (default: multiprocess)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response caching",
    )

    # Active Inference (Phase 4)
    parser.add_argument(
        "--use-active-inference",
        action="store_true",
        help="Enable training example augmentation (Active Inference)",
    )
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=10,
        help="Number of variations per training example (default: 10)",
    )

    args = parser.parse_args()
    args.use_cache = not args.no_cache

    return args


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load task IDs
    print("=" * 70)
    print(" Test-Time Ensemble Benchmark")
    print("=" * 70)
    print()

    task_ids = load_task_ids(
        args.training_data,
        random_sample=args.random_sample,
        seed=args.seed,
    )

    print(f"Experiment: {args.experiment_name}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Tasks: {len(task_ids)}")
    print()
    print("Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Analyst Temperature: {args.analyst_temperature}")
    print(f"  Programmer Temperature: {args.programmer_temperature}")
    print(f"  Synthesis Temperature: {args.synthesis_temperature}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Sandbox: {args.sandbox_mode}")
    print(f"  Cache: {args.use_cache}")
    print()

    # Generate and save metadata
    output_dir = Path(args.output_dir)
    metadata = generate_metadata(args.experiment_name, task_ids, args)
    metadata_file = output_dir / "metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_file}")
    print()

    # Run benchmark
    print("=" * 70)
    print(" Running Benchmarks")
    print("=" * 70)
    print()

    results = []
    for idx, task_id in enumerate(task_ids, 1):
        print(f"[{idx}/{len(task_ids)}] Task: {task_id}")
        print("-" * 70)

        result = run_ensemble_single_task(
            task_id=task_id,
            training_data_path=args.training_data,
            model_name=args.model_name,
            analyst_temperature=args.analyst_temperature,
            programmer_temperature=args.programmer_temperature,
            synthesis_temperature=args.synthesis_temperature,
            use_cache=args.use_cache,
            timeout=args.timeout,
            sandbox_mode=args.sandbox_mode,
            use_active_inference=args.use_active_inference,
            augmentation_factor=args.augmentation_factor,
        )

        # Save individual result
        save_task_result(output_dir, result)

        # Print result
        if result["success"]:
            pass_at_2 = result.get("pass_at_2_accuracy")
            if pass_at_2 is not None:
                print(
                    f"✅ Success: pass@2={pass_at_2:.2f} ({result['total_time']:.1f}s)"
                )
            else:
                print(f"✅ Success: No ground truth ({result['total_time']:.1f}s)")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")

        print()
        results.append(result)

    # Calculate and save aggregate statistics
    print("=" * 70)
    print(" Aggregate Statistics")
    print("=" * 70)
    print()

    stats = calculate_aggregate_stats(results)

    print(f"Total Tasks: {stats['total_tasks']}")
    print(f"Successful: {stats['successful_tasks']}")
    print(f"Failed: {stats['failed_tasks']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print()
    print("Accuracy Metrics:")
    print(f"  pass@2: {stats['pass_at_2_accuracy']:.1%}")
    print(f"  Best-only: {stats['best_only_accuracy']:.1%}")
    print(f"  Synthesis-only: {stats['synthesis_only_accuracy']:.1%}")
    print(f"  Diversity benefit: {stats['diversity_benefit']:.1%}")
    print()
    print("Performance Metrics:")
    print(f"  Avg Time per Task: {stats['avg_time_per_task']:.1f}s")
    print(f"  Total Time: {stats['total_time']:.1f}s")
    print(f"  Total API Calls: {stats['total_api_calls']}")
    print()

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Summary saved to {summary_file}")
    print()

    print("=" * 70)
    print(" Benchmark Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
