"""Demo script for Phase 1.1: Data Loading and Evaluation

This script demonstrates:
1. Loading an ARC task from the dataset
2. Displaying train examples with print_grid()
3. Evaluating grids for correctness

Usage:
    python scripts/demo_phase1_1_data.py [task_id]

If no task_id is provided, uses the first task in the training set.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_prometheus.crucible.data_loader import load_task, print_grid
from arc_prometheus.crucible.evaluator import evaluate_grids
from arc_prometheus.utils.config import DATA_DIR
import numpy as np


def main():
    """Run the data loading demo."""
    print("=" * 70)
    print("ARC-Prometheus Phase 1.1 Demo: Data Loading and Evaluation")
    print("=" * 70)

    # Get task ID from command line or use first task
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
        print(f"\nLoading task ID: {task_id}")
    else:
        # Load the training challenges to get first task
        challenges_file = DATA_DIR / "arc-agi_training_challenges.json"
        if not challenges_file.exists():
            print(f"\n❌ ERROR: Training challenges file not found at {challenges_file}")
            print("Please download the ARC Prize 2025 dataset from:")
            print("https://www.kaggle.com/competitions/arc-prize-2025/data")
            print(f"And place it in: {DATA_DIR}")
            sys.exit(1)

        with open(challenges_file, 'r') as f:
            challenges = json.load(f)

        task_id = list(challenges.keys())[0]
        print(f"\nNo task ID provided, using first task: {task_id}")

    # Load the task
    task_file = DATA_DIR / "arc-agi_training_challenges.json"

    try:
        # Load all tasks and extract the specific one
        with open(task_file, 'r') as f:
            all_tasks = json.load(f)

        if task_id not in all_tasks:
            print(f"\n❌ ERROR: Task ID '{task_id}' not found in dataset")
            print(f"Available task IDs: {len(all_tasks)} total")
            print(f"First few: {list(all_tasks.keys())[:5]}")
            sys.exit(1)

        # Convert to the format expected by load_task (save as temp file)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(all_tasks[task_id], tmp)
            tmp_path = tmp.name

        task_data = load_task(tmp_path)

        # Clean up temp file
        Path(tmp_path).unlink()

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR loading task: {e}")
        sys.exit(1)

    # Display task information
    print(f"\n✓ Successfully loaded task: {task_id}")
    print(f"  - Train examples: {len(task_data['train'])}")
    print(f"  - Test examples: {len(task_data['test'])}")

    # Display all train examples
    print("\n" + "=" * 70)
    print("TRAIN EXAMPLES")
    print("=" * 70)

    for idx, example in enumerate(task_data['train'], 1):
        print(f"\n--- Train Example {idx}/{len(task_data['train'])} ---")

        print_grid(example['input'], label=f"Input (shape: {example['input'].shape})")
        print_grid(example['output'], label=f"Output (shape: {example['output'].shape})")

        # Test evaluation: input should NOT equal output (typically)
        is_same = evaluate_grids(example['input'], example['output'])
        if is_same:
            print("⚠️  Note: Input and output are identical (pass-through task)")
        else:
            print("✓ Input and output are different (transformation applied)")

    # Display test examples
    print("\n" + "=" * 70)
    print("TEST EXAMPLES")
    print("=" * 70)

    for idx, example in enumerate(task_data['test'], 1):
        print(f"\n--- Test Example {idx}/{len(task_data['test'])} ---")

        print_grid(example['input'], label=f"Input (shape: {example['input'].shape})")

        if 'output' in example:
            print_grid(example['output'], label=f"Output (shape: {example['output'].shape})")
            print("✓ Test has ground truth output available")
        else:
            print("(No ground truth output - this is the puzzle to solve!)")

    # Demonstrate evaluation functionality
    print("\n" + "=" * 70)
    print("EVALUATION DEMO")
    print("=" * 70)

    print("\nTesting evaluate_grids() function:")

    # Test 1: Identical grids
    grid1 = task_data['train'][0]['input']
    grid2 = task_data['train'][0]['input'].copy()
    result = evaluate_grids(grid1, grid2)
    print(f"  - Identical grids: {result} ✓" if result else f"  - Identical grids: {result} ❌")

    # Test 2: Different grids
    grid3 = task_data['train'][0]['output']
    result = evaluate_grids(grid1, grid3)
    print(f"  - Different grids: {result == False} ✓" if not result else f"  - Different grids: {result == False} ❌")

    # Summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\n✓ Successfully demonstrated:")
    print("  1. Loading ARC tasks from JSON files")
    print("  2. Displaying grids with visual formatting")
    print("  3. Evaluating grid equality")
    print("\nNext step: Phase 1.2 - Manual solver implementation")
    print("=" * 70)


if __name__ == "__main__":
    main()
