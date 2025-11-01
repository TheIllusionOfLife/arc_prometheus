#!/usr/bin/env python3
"""
Analyze baseline validation results from benchmark_evolution.py.

This script supports two modes:
1. Single experiment analysis: Analyze one experiment directory
2. Comparison mode: Compare two experiment directories

Usage:
    # Analyze single experiment
    python scripts/analyze_baseline.py results/gemini_baseline_quick/

    # Compare two experiments
    python scripts/analyze_baseline.py \
        results/gemini_baseline_validation/ \
        results/gemini_active_inference/ \
        --compare

    # Calculate sample size from quick test
    python scripts/analyze_baseline.py results/gemini_baseline_quick/ --sample-size
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

# Decision gate thresholds for test accuracy
HIGH_SCORE_THRESHOLD = 8.0  # ≥8%: Proceed to Active Inference (Phase 2a)
MEDIUM_SCORE_THRESHOLD = 5.0  # 5-8%: Optimize hyperparameters (Phase 2b)
# <5%: Fundamental rethink needed


def load_summary(experiment_dir: str) -> dict[str, Any]:
    """
    Load summary.json from experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary with summary statistics

    Raises:
        FileNotFoundError: If summary.json doesn't exist
    """
    summary_path = Path(experiment_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"summary.json not found in {experiment_dir}. "
            "Run benchmark_evolution.py first."
        )

    with open(summary_path, encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
        return result


def calculate_sample_size(
    summary: dict[str, Any], target_duration: int = 3600, max_sample: int = 50
) -> tuple[int, float]:
    """
    Calculate optimal sample size based on time per task.

    Args:
        summary: Summary statistics from quick test
        target_duration: Target duration in seconds (default: 3600 = 1 hour)
        max_sample: Maximum sample size cap (default: 50)

    Returns:
        Tuple of (sample_size, estimated_time_seconds)
    """
    avg_time = summary.get("avg_time_per_task", 0.0)

    if avg_time <= 0:
        # Edge case: zero or negative time, return max sample
        return max_sample, 0.0

    # Calculate how many tasks fit in target duration
    optimal_sample = int(target_duration / avg_time)

    # Cap at max_sample for statistical validity
    sample_size = min(optimal_sample, max_sample)

    # Calculate estimated time
    estimated_time = sample_size * avg_time

    return sample_size, estimated_time


def get_decision(summary: dict[str, Any]) -> dict[str, Any]:
    """
    Determine next phase based on test accuracy score.

    Decision gates:
    - Score ≥8%: High → Phase 2a (Active Inference)
    - Score 5-8%: Medium → Phase 2b (Hyperparameter Tuning)
    - Score <5%: Low → Fundamental rethink needed

    Args:
        summary: Summary statistics with test_accuracy_pct

    Returns:
        Dictionary with decision category, next_phase, and reasoning
    """
    score = summary.get("test_accuracy_pct", 0.0)

    if score >= HIGH_SCORE_THRESHOLD:
        return {
            "category": "high",
            "next_phase": "Phase 2a (Active Inference)",
            "reasoning": "8% baseline + 5% Active Inference = 13% (competitive threshold)",
            "action": "Test Active Inference on same task sample",
        }
    elif score >= MEDIUM_SCORE_THRESHOLD:
        return {
            "category": "medium",
            "next_phase": "Phase 2b (Hyperparameter Tuning)",
            "reasoning": "Need stronger baseline before testing Active Inference",
            "action": "Optimize hyperparameters (generations, population, temperature, model)",
        }
    else:
        return {
            "category": "low",
            "next_phase": "Fundamental rethink needed",
            "reasoning": "Even with Active Inference (+5%), unlikely to reach competitive threshold",
            "action": "Analyze failure modes, improve prompts, try different model, increase generations",
        }


def format_statistics(summary: dict[str, Any]) -> str:
    """
    Format summary statistics for display.

    Args:
        summary: Summary statistics dictionary

    Returns:
        Formatted string for terminal output
    """
    lines = []
    lines.append("=" * 60)
    lines.append("BASELINE VALIDATION RESULTS")
    lines.append("=" * 60)

    # Core metrics
    test_acc = summary.get("test_accuracy_pct", 0.0)
    train_acc = summary.get("train_accuracy_pct", 0.0)
    lines.append(f"Test Accuracy:      {test_acc:>6.2f}%  ⭐ PRIMARY METRIC")
    lines.append(f"Train Accuracy:     {train_acc:>6.2f}%")
    lines.append("")

    # Task completion
    total_tasks = summary.get("total_tasks", 0)
    tasks_solved = summary.get("tasks_with_positive_fitness", 0)
    perfect_solvers = summary.get("perfect_solvers", 0)
    pct = (tasks_solved / total_tasks * 100) if total_tasks > 0 else 0.0
    lines.append(f"Tasks Solved:       {tasks_solved:>3}/{total_tasks:<3} ({pct:.1f}%)")
    lines.append(
        f"Perfect Solvers:    {perfect_solvers:>3}/{total_tasks:<3} (fitness=13)"
    )
    lines.append("")

    # Fitness metrics
    avg_fitness = summary.get("avg_final_fitness", 0.0)
    median_fitness = summary.get("median_final_fitness", 0.0)
    lines.append(f"Avg Fitness:        {avg_fitness:>6.2f}")
    lines.append(f"Median Fitness:     {median_fitness:>6.2f}")
    lines.append("")

    # Performance metrics
    avg_gens = summary.get("avg_generations", 0.0)
    avg_time = summary.get("avg_time_per_task", 0.0)
    total_time = summary.get("total_time", 0.0)
    lines.append(f"Avg Generations:    {avg_gens:>6.2f}")
    lines.append(f"Avg Time/Task:      {avg_time:>6.1f}s")
    lines.append(f"Total Time:         {total_time / 60:>6.1f} min")

    # Error distribution
    errors = summary.get("error_distribution", {})
    if errors:
        lines.append("")
        lines.append("Error Distribution:")
        for error_type, count in sorted(errors.items(), key=lambda x: -x[1]):
            lines.append(f"  {error_type:20} {count:>3}")

    lines.append("=" * 60)

    return "\n".join(lines)


def compare_experiments(
    baseline: dict[str, Any],
    comparison: dict[str, Any],
    baseline_name: str,
    comparison_name: str,
) -> dict[str, Any]:
    """
    Compare two experiments and calculate deltas.

    Args:
        baseline: Summary from baseline experiment
        comparison: Summary from comparison experiment
        baseline_name: Display name for baseline
        comparison_name: Display name for comparison

    Returns:
        Dictionary with comparison metrics and deltas
    """
    metrics: dict[str, Any] = {}

    # Calculate deltas for key metrics
    test_acc_baseline = cast(float, baseline.get("test_accuracy_pct", 0.0))
    test_acc_comparison = cast(float, comparison.get("test_accuracy_pct", 0.0))
    test_acc_delta = test_acc_comparison - test_acc_baseline
    test_acc_delta_pct = (
        (test_acc_delta / test_acc_baseline * 100) if test_acc_baseline > 0 else 0
    )

    metrics["baseline_name"] = baseline_name
    metrics["comparison_name"] = comparison_name
    metrics["test_accuracy_baseline"] = test_acc_baseline
    metrics["test_accuracy_comparison"] = test_acc_comparison
    metrics["test_accuracy_delta"] = test_acc_delta
    metrics["test_accuracy_delta_pct"] = test_acc_delta_pct
    metrics["improvement"] = test_acc_delta > 0

    # Additional metrics
    for key in [
        "train_accuracy_pct",
        "avg_final_fitness",
        "tasks_with_positive_fitness",
        "avg_time_per_task",
    ]:
        baseline_val = cast(float, baseline.get(key, 0.0))
        comparison_val = cast(float, comparison.get(key, 0.0))
        delta = comparison_val - baseline_val
        delta_pct = (delta / baseline_val * 100) if baseline_val > 0 else 0

        metrics[f"{key}_baseline"] = baseline_val
        metrics[f"{key}_comparison"] = comparison_val
        metrics[f"{key}_delta"] = delta
        metrics[f"{key}_delta_pct"] = delta_pct

    return metrics


def format_comparison(comparison: dict[str, Any]) -> str:
    """
    Format comparison results for display.

    Args:
        comparison: Comparison metrics from compare_experiments()

    Returns:
        Formatted string for terminal output
    """
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT COMPARISON")
    lines.append("=" * 80)
    lines.append(f"Baseline:    {comparison['baseline_name']}")
    lines.append(f"Comparison:  {comparison['comparison_name']}")
    lines.append("=" * 80)

    # Header
    lines.append(
        f"{'Metric':<30} {'Baseline':>12} {'Comparison':>12} {'Δ':>12} {'Δ%':>10}"
    )
    lines.append("-" * 80)

    # Test accuracy (primary metric)
    test_delta = comparison["test_accuracy_delta"]
    test_delta_pct = comparison["test_accuracy_delta_pct"]
    delta_sign = "+" if test_delta >= 0 else ""
    lines.append(
        f"{'⭐ Test Accuracy':<30} "
        f"{comparison['test_accuracy_baseline']:>11.2f}% "
        f"{comparison['test_accuracy_comparison']:>11.2f}% "
        f"{delta_sign}{test_delta:>11.2f}% "
        f"{delta_sign}{test_delta_pct:>9.1f}%"
    )

    # Other metrics
    metric_names = {
        "train_accuracy_pct": "Train Accuracy",
        "avg_final_fitness": "Avg Fitness",
        "tasks_with_positive_fitness": "Tasks Solved",
        "avg_time_per_task": "Avg Time/Task (s)",
    }

    for key, display_name in metric_names.items():
        baseline_val = comparison[f"{key}_baseline"]
        comp_val = comparison[f"{key}_comparison"]
        delta = comparison[f"{key}_delta"]
        delta_pct = comparison[f"{key}_delta_pct"]
        delta_sign = "+" if delta >= 0 else ""

        # Format based on metric type
        if "pct" in key:
            lines.append(
                f"{display_name:<30} "
                f"{baseline_val:>11.2f}% "
                f"{comp_val:>11.2f}% "
                f"{delta_sign}{delta:>11.2f}% "
                f"{delta_sign}{delta_pct:>9.1f}%"
            )
        else:
            lines.append(
                f"{display_name:<30} "
                f"{baseline_val:>12.2f} "
                f"{comp_val:>12.2f} "
                f"{delta_sign}{delta:>12.2f} "
                f"{delta_sign}{delta_pct:>9.1f}%"
            )

    lines.append("=" * 80)

    # Recommendation
    if comparison["improvement"]:
        improvement_val = comparison["test_accuracy_delta"]
        if improvement_val >= 2.0:
            lines.append(
                f"✅ STRONG IMPROVEMENT: +{improvement_val:.1f}% test accuracy"
            )
            lines.append("   Recommendation: Keep the improved configuration")
        else:
            lines.append(
                f"⚠️  MARGINAL IMPROVEMENT: +{improvement_val:.1f}% test accuracy"
            )
            lines.append("   Recommendation: Consider cost/benefit trade-off")
    else:
        regression_val = abs(comparison["test_accuracy_delta"])
        lines.append(f"❌ REGRESSION: -{regression_val:.1f}% test accuracy")
        lines.append("   Recommendation: Revert to baseline configuration")

    lines.append("=" * 80)

    return "\n".join(lines)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze baseline validation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single experiment
  python scripts/analyze_baseline.py results/gemini_baseline_quick/

  # Calculate sample size from quick test
  python scripts/analyze_baseline.py results/gemini_baseline_quick/ --sample-size

  # Compare two experiments
  python scripts/analyze_baseline.py \\
      results/gemini_baseline_validation/ \\
      results/gemini_active_inference/ \\
      --compare
        """,
    )

    parser.add_argument(
        "experiment_dir",
        nargs="+",
        help="Experiment directory (or two directories for comparison)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare two experiments (requires exactly 2 directories)",
    )
    parser.add_argument(
        "--sample-size",
        action="store_true",
        help="Calculate optimal sample size for full validation",
    )
    parser.add_argument(
        "--target-duration",
        type=int,
        default=3600,
        help="Target duration in seconds for sample size calculation (default: 3600 = 1 hour)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.compare and len(args.experiment_dir) != 2:
        print(
            "❌ Error: --compare requires exactly 2 experiment directories",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.sample_size and len(args.experiment_dir) != 1:
        print(
            "❌ Error: --sample-size requires exactly 1 experiment directory",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        if args.compare:
            # Comparison mode
            baseline_dir, comparison_dir = args.experiment_dir
            baseline = load_summary(baseline_dir)
            comparison_exp = load_summary(comparison_dir)

            # Extract experiment names from paths
            baseline_name = Path(baseline_dir).name
            comparison_name = Path(comparison_dir).name

            comparison_results = compare_experiments(
                baseline, comparison_exp, baseline_name, comparison_name
            )
            print(format_comparison(comparison_results))

        elif args.sample_size:
            # Sample size calculation mode
            experiment_dir = args.experiment_dir[0]
            summary = load_summary(experiment_dir)

            sample_size, estimated_time = calculate_sample_size(
                summary, target_duration=args.target_duration
            )

            print("=" * 60)
            print("SAMPLE SIZE CALCULATION")
            print("=" * 60)
            print("Quick Test Results:")
            print(f"  Tasks:            {summary.get('total_tasks', 0)}")
            print(f"  Avg Time/Task:    {summary.get('avg_time_per_task', 0):.1f}s")
            print(f"  Test Accuracy:    {summary.get('test_accuracy_pct', 0):.1f}%")
            print("")
            print("Recommendation:")
            print(f"  Sample Size:      {sample_size} tasks")
            print(
                f"  Estimated Time:   {estimated_time / 60:.1f} min ({estimated_time:.0f}s)"
            )
            print(f"  Target Duration:  {args.target_duration / 60:.0f} min")
            print("=" * 60)

        else:
            # Single experiment analysis mode
            experiment_dir = args.experiment_dir[0]
            summary = load_summary(experiment_dir)

            # Print statistics
            print(format_statistics(summary))
            print("")

            # Get and print decision
            decision = get_decision(summary)
            print("DECISION GATE")
            print("=" * 60)
            print(f"Score Category:     {decision['category'].upper()}")
            print(f"Next Phase:         {decision['next_phase']}")
            print(f"Reasoning:          {decision['reasoning']}")
            print(f"Action:             {decision['action']}")
            print("=" * 60)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in summary file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
