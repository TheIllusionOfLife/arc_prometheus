"""Fitness evaluation for solver quality assessment (Phase 2.1)."""

import json

import numpy as np

from ..crucible.data_loader import load_task
from ..crucible.evaluator import evaluate_grids
from ..crucible.sandbox import safe_execute


def calculate_fitness(
    task_json_path: str, solver_code: str, timeout: int = 5
) -> dict[str, int | float | list[str]]:
    """
    Evaluate solver performance on ARC task.

    This function calculates a fitness score that heavily weights test accuracy
    (10x) to encourage generalization over memorization of training examples.

    Args:
        task_json_path: Path to ARC task JSON file
        solver_code: Python code string containing solve() function
        timeout: Execution timeout per example in seconds (default: 5)

    Returns:
        Dictionary containing:
            - fitness (float): Overall score = (train_correct * 1) + (test_correct * 10)
            - train_correct (int): Number of train examples solved correctly
            - train_total (int): Total number of train examples
            - test_correct (int): Number of test examples solved correctly
            - test_total (int): Total number of test examples
            - train_accuracy (float): Train correctness ratio (0.0 to 1.0)
            - test_accuracy (float): Test correctness ratio (0.0 to 1.0)
            - execution_errors (list[str]): Error messages from failed executions

    Example:
        >>> result = calculate_fitness("task.json", solver_code)
        >>> print(f"Fitness: {result['fitness']}")
        Fitness: 13
        >>> print(f"Train: {result['train_correct']}/{result['train_total']}")
        Train: 3/3
        >>> print(f"Test: {result['test_correct']}/{result['test_total']}")
        Test: 1/1

    Note:
        The 10x weight on test accuracy ensures that solvers which only memorize
        training examples receive low fitness scores, encouraging true generalization.
    """
    # Initialize counters
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    execution_errors: list[str] = []

    try:
        # Load task data
        task_data = load_task(task_json_path)

        # Evaluate on train examples
        train_examples = task_data.get("train", [])
        train_total = len(train_examples)

        for idx, example in enumerate(train_examples):
            input_grid = np.array(example["input"])
            expected_output = np.array(example["output"])

            # Execute solver in sandbox
            success, result_grid = safe_execute(solver_code, input_grid, timeout)

            if not success:
                execution_errors.append(f"Train example {idx}: Execution failed")
                continue

            # Compare result with expected output
            if result_grid is not None and evaluate_grids(result_grid, expected_output):
                train_correct += 1

        # Evaluate on test examples
        test_examples = task_data.get("test", [])
        test_total = len(test_examples)

        for idx, example in enumerate(test_examples):
            input_grid = np.array(example["input"])
            expected_output = np.array(example["output"])

            # Execute solver in sandbox
            success, result_grid = safe_execute(solver_code, input_grid, timeout)

            if not success:
                execution_errors.append(f"Test example {idx}: Execution failed")
                continue

            # Compare result with expected output
            if result_grid is not None and evaluate_grids(result_grid, expected_output):
                test_correct += 1

    except FileNotFoundError:
        execution_errors.append(f"Task file not found: {task_json_path}")
    except json.JSONDecodeError:
        execution_errors.append(f"Invalid JSON in task file: {task_json_path}")
    except Exception as e:
        execution_errors.append(f"Unexpected error: {str(e)}")

    # Calculate fitness: train_correct * 1 + test_correct * 10
    fitness = (train_correct * 1) + (test_correct * 10)

    # Calculate accuracies (handle division by zero)
    train_accuracy = train_correct / train_total if train_total > 0 else 0.0
    test_accuracy = test_correct / test_total if test_total > 0 else 0.0

    return {
        "fitness": fitness,
        "train_correct": train_correct,
        "train_total": train_total,
        "test_correct": test_correct,
        "test_total": test_total,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "execution_errors": execution_errors,
    }
