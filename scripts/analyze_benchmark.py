"""Simple analysis script for benchmark results.

Loads benchmark results from a directory and generates a markdown report.

Usage:
    python scripts/analyze_benchmark.py \\
        --results-dir results/multiprocess_baseline/ \\
        --output-report docs/benchmarks/multiprocess_baseline.md

    # Compare two experiments
    python scripts/analyze_benchmark.py \\
        --results-dir results/multiprocess_baseline/ \\
        --compare-with results/docker_baseline/ \\
        --output-report docs/benchmarks/comparison.md
"""

import argparse
import json
import sys
from pathlib import Path


def load_benchmark_results(results_dir: str) -> dict:
    """Load all benchmark results from directory.

    Args:
        results_dir: Directory containing result files

    Returns:
        Dictionary with metadata, summary, and task results
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Load metadata
    metadata_file = results_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Load summary
    summary_file = results_path / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
    else:
        summary = {}

    # Load individual task results
    task_results = []
    for task_file in sorted(results_path.glob("task_*.json")):
        with open(task_file) as f:
            task_results.append(json.load(f))

    return {
        "metadata": metadata,
        "summary": summary,
        "tasks": task_results,
    }


def generate_markdown_report(
    results: dict, compare_results: dict | None = None
) -> str:
    """Generate markdown report from benchmark results.

    Args:
        results: Primary benchmark results
        compare_results: Optional secondary results for comparison

    Returns:
        Markdown-formatted report string
    """
    md = []
    metadata = results["metadata"]
    summary = results["summary"]

    # Header
    md.append(f"# Benchmark Report: {metadata.get('experiment_name', 'Unknown')}")
    md.append("")
    md.append(f"**Date**: {metadata.get('timestamp', 'Unknown')}")
    md.append(f"**Git Commit**: `{metadata.get('git_commit', 'Unknown')}`")
    md.append(f"**Tasks**: {metadata.get('num_tasks', 0)}")
    md.append("")

    # Configuration
    md.append("## Configuration")
    md.append("")
    config = metadata.get("config", {})
    for key, value in sorted(config.items()):
        md.append(f"- **{key}**: {value}")
    md.append("")

    # Overall Results
    md.append("## Overall Results")
    md.append("")
    md.append(f"- **Total Tasks**: {summary.get('total_tasks', 0)}")
    md.append(f"- **Successful**: {summary.get('successful_tasks', 0)}")
    md.append(f"- **Failed**: {summary.get('failed_tasks', 0)}")
    md.append(
        f"- **Success Rate**: {summary.get('success_rate', 0) * 100:.1f}%"
    )
    md.append("")

    if summary.get("successful_tasks", 0) > 0:
        md.append("### Fitness Metrics")
        md.append("")
        md.append(f"- **Average**: {summary.get('avg_final_fitness', 0):.2f}")
        md.append(f"- **Median**: {summary.get('median_final_fitness', 0):.2f}")
        md.append("")

        md.append("### Evolution Metrics")
        md.append("")
        md.append(
            f"- **Avg Generations**: {summary.get('avg_generations', 0):.2f}"
        )
        md.append(
            f"- **Avg Time per Task**: {summary.get('avg_time_per_task', 0):.1f}s"
        )
        md.append(f"- **Total Time**: {summary.get('total_time', 0):.1f}s")
        md.append("")

        if summary.get("error_distribution"):
            md.append("### Error Distribution")
            md.append("")
            for error_type, count in sorted(
                summary["error_distribution"].items()
            ):
                md.append(f"- **{error_type}**: {count}")
            md.append("")

    # Task-by-Task Results
    md.append("## Task-by-Task Results")
    md.append("")
    md.append("| Task ID | Success | Final Fitness | Generations | Time (s) |")
    md.append("|---------|---------|---------------|-------------|----------|")

    for task in results["tasks"]:
        task_id = task["task_id"]
        success = "✅" if task.get("success") else "❌"
        fitness = task.get("final_fitness", 0.0)
        gens = task.get("total_generations", 0)
        time = task.get("total_time", 0.0)

        if task.get("success"):
            md.append(
                f"| {task_id} | {success} | {fitness:.1f} | {gens} | {time:.1f} |"
            )
        else:
            error = task.get("error", "Unknown")[:30]
            md.append(f"| {task_id} | {success} | N/A | N/A | {error}... |")

    md.append("")

    # Comparison section (if provided)
    if compare_results:
        comp_summary = compare_results["summary"]
        comp_metadata = compare_results["metadata"]

        md.append("## Comparison")
        md.append("")
        md.append(
            f"Comparing with: **{comp_metadata.get('experiment_name', 'Unknown')}**"
        )
        md.append("")

        md.append("| Metric | Primary | Comparison | Difference |")
        md.append("|--------|---------|------------|------------|")

        # Success rate
        sr1 = summary.get("success_rate", 0)
        sr2 = comp_summary.get("success_rate", 0)
        diff_sr = (sr1 - sr2) * 100
        md.append(
            f"| Success Rate | {sr1*100:.1f}% | {sr2*100:.1f}% | {diff_sr:+.1f}% |"
        )

        # Average fitness
        if summary.get("successful_tasks", 0) > 0:
            af1 = summary.get("avg_final_fitness", 0)
            af2 = comp_summary.get("avg_final_fitness", 0)
            diff_af = af1 - af2
            md.append(
                f"| Avg Fitness | {af1:.2f} | {af2:.2f} | {diff_af:+.2f} |"
            )

            # Avg time
            at1 = summary.get("avg_time_per_task", 0)
            at2 = comp_summary.get("avg_time_per_task", 0)
            diff_at = at1 - at2
            pct_diff = (diff_at / at2 * 100) if at2 > 0 else 0
            md.append(
                f"| Avg Time/Task | {at1:.1f}s | {at2:.1f}s | {diff_at:+.1f}s ({pct_diff:+.1f}%) |"
            )

        md.append("")

    return "\n".join(md)


def parse_args(args=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and generate report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        default=None,
        help="Optional second results directory for comparison",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Path to save markdown report (default: print to stdout)",
    )

    return parser.parse_args(args)


def main():
    """Main analysis function."""
    args = parse_args()

    # Load primary results
    print(f"Loading results from {args.results_dir}...")
    results = load_benchmark_results(args.results_dir)

    # Load comparison results if provided
    compare_results = None
    if args.compare_with:
        print(f"Loading comparison results from {args.compare_with}...")
        compare_results = load_benchmark_results(args.compare_with)

    # Generate report
    print("Generating report...")
    report = generate_markdown_report(results, compare_results)

    # Output report
    if args.output_report:
        output_path = Path(args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"✓ Report saved to {args.output_report}")
    else:
        print("\n" + "=" * 70)
        print(report)
        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
