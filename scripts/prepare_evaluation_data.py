#!/usr/bin/env python3
"""Prepare ARC evaluation dataset for benchmarking.

This script merges ARC evaluation challenges with their solutions to create
a unified format compatible with the training data structure. The evaluation
dataset stores test outputs separately from challenges, unlike training data
where outputs are embedded in the challenge JSON.

Usage:
    python scripts/prepare_evaluation_data.py \\
        --challenges data/arc-prize-2025/arc-agi_evaluation_challenges.json \\
        --solutions data/arc-prize-2025/arc-agi_evaluation_solutions.json \\
        --output data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


def merge_evaluation_data(
    challenges_path: str | Path,
    solutions_path: str | Path,
    output_path: str | Path,
) -> None:
    """Merge evaluation challenges with solutions into unified format.

    Args:
        challenges_path: Path to arc-agi_evaluation_challenges.json
        solutions_path: Path to arc-agi_evaluation_solutions.json
        output_path: Path to output merged JSON file

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If data validation fails (mismatched counts, missing tasks)
        json.JSONDecodeError: If input files are invalid JSON
    """
    # Load input files
    print(f"Loading challenges from: {challenges_path}")
    with open(challenges_path) as f:
        challenges: dict[str, Any] = json.load(f)

    print(f"Loading solutions from: {solutions_path}")
    with open(solutions_path) as f:
        solutions: dict[str, list[list[list[int]]]] = json.load(f)

    # Validate all challenge tasks have solutions (using set operations)
    print("\nValidating data integrity...")
    print(f"  Challenges: {len(challenges)} tasks")
    print(f"  Solutions: {len(solutions)} tasks")

    challenge_ids = set(challenges)
    solution_ids = set(solutions)
    missing_tasks = list(challenge_ids - solution_ids)

    if missing_tasks:
        raise ValueError(
            f"Tasks {missing_tasks} found in challenges but not found in solutions"
        )

    # Merge solutions into challenges
    print("\nMerging test outputs into challenges...")
    merged = {}
    mismatches = []

    for task_id, task_data in challenges.items():
        # Copy task structure (deepcopy to prevent side effects on original data)
        merged_task = {
            "train": task_data["train"],
            "test": copy.deepcopy(task_data["test"]),  # Will modify test examples
        }

        # Merge test outputs
        test_inputs = task_data["test"]
        test_outputs = solutions[task_id]

        # Validate counts match
        if len(test_inputs) != len(test_outputs):
            mismatches.append(
                f"  - {task_id}: {len(test_inputs)} test inputs, "
                f"{len(test_outputs)} solutions"
            )
            continue

        # Add outputs to test examples
        for i, output in enumerate(test_outputs):
            merged_task["test"][i]["output"] = output

        merged[task_id] = merged_task

    if mismatches:
        raise ValueError(
            "Test input/output count mismatches:\n" + "\n".join(mismatches)
        )

    # Write merged output
    print(f"\nWriting merged data to: {output_path}")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\n✅ Successfully merged {len(merged)} tasks")
    print(f"   Output: {output_path}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Merge ARC evaluation challenges with solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/prepare_evaluation_data.py \\
        --challenges data/arc-prize-2025/arc-agi_evaluation_challenges.json \\
        --solutions data/arc-prize-2025/arc-agi_evaluation_solutions.json \\
        --output data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json
        """,
    )

    parser.add_argument(
        "--challenges",
        required=True,
        help="Path to arc-agi_evaluation_challenges.json",
    )
    parser.add_argument(
        "--solutions",
        required=True,
        help="Path to arc-agi_evaluation_solutions.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output merged JSON file",
    )

    args = parser.parse_args()

    # Validate input files exist before processing
    challenges_file = Path(args.challenges)
    solutions_file = Path(args.solutions)

    if not challenges_file.exists():
        print(
            f"❌ Error: Challenges file not found: {args.challenges}", file=sys.stderr
        )
        return 1

    if not solutions_file.exists():
        print(
            f"❌ Error: Solutions file not found: {args.solutions}", file=sys.stderr
        )
        return 1

    try:
        merge_evaluation_data(
            challenges_path=args.challenges,
            solutions_path=args.solutions,
            output_path=args.output,
        )
        return 0
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"❌ Validation Error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
