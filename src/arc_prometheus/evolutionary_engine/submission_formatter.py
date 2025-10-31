"""Submission formatter for Kaggle pass@2 format (Phase 2.5).

This module handles:
1. Diversity selection from evolution generation history
2. Prediction generation for multiple test inputs
3. Submission JSON formatting matching Kaggle requirements
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np

from ..crucible.data_loader import load_task
from ..crucible.sandbox import MultiprocessSandbox
from ..crucible.sandbox_protocol import ExecutionEnvironment
from .evolution_loop import GenerationResult


def _get_sandbox(sandbox_mode: str) -> ExecutionEnvironment:
    """Factory function to create appropriate sandbox instance.

    Args:
        sandbox_mode: 'multiprocess' or 'docker'

    Returns:
        Sandbox instance implementing ExecutionEnvironment protocol
    """
    if sandbox_mode == "docker":
        try:
            from ..crucible.docker_sandbox import DockerSandbox

            return DockerSandbox()
        except ImportError as e:
            raise RuntimeError(
                "Docker sandbox requires docker package. "
                "Install with: pip install -e '.[docker]'"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Docker sandbox not available: {e}. "
                "Build image with: docker build -t arc-prometheus-sandbox:latest "
                "-f docker/sandbox.Dockerfile ."
            ) from e
    elif sandbox_mode == "multiprocess":
        return MultiprocessSandbox()
    else:
        raise ValueError(
            f"Invalid sandbox_mode: {sandbox_mode}. Must be 'multiprocess' or 'docker'"
        )


def select_diverse_solvers(
    generations: list[GenerationResult],
    num_attempts: int = 2,
    diversity_metric: Literal["fitness", "generation_gap", "edit_distance"] = "fitness",
) -> list[str]:
    """Select diverse solvers from generation history for pass@2.

    Selects num_attempts diverse solver codes from the generation history.
    Removes duplicate code before selection.

    Args:
        generations: List of GenerationResult from evolution loop
        num_attempts: Number of diverse solvers to select (default: 2 for pass@2)
        diversity_metric: Strategy for diversity selection:
            - "fitness": Select best fitness + next different solver
            - "generation_gap": Select from early and late generations
            - "edit_distance": Select by maximum code difference (future)

    Returns:
        List of solver code strings (length = num_attempts)

    Raises:
        ValueError: If fewer than num_attempts unique solvers available

    Example:
        >>> from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop
        >>> generations = run_evolution_loop("task.json", max_generations=5)  # doctest: +SKIP
        >>> solvers = select_diverse_solvers(generations, num_attempts=2)  # doctest: +SKIP
        >>> len(solvers)  # doctest: +SKIP
        2
    """
    if not generations:
        raise ValueError("No generations provided")

    # Remove duplicates by code
    unique_generations: list[GenerationResult] = []
    seen_codes: set[str] = set()

    for gen in generations:
        code = gen["solver_code"]
        if code not in seen_codes:
            unique_generations.append(gen)
            seen_codes.add(code)

    # Check if we have enough unique solvers
    if len(unique_generations) < num_attempts:
        raise ValueError(
            f"Not enough unique solvers: need {num_attempts}, "
            f"have {len(unique_generations)}"
        )

    # Select based on diversity metric
    if diversity_metric == "fitness":
        return _select_by_fitness(unique_generations, num_attempts)
    elif diversity_metric == "generation_gap":
        return _select_by_generation_gap(unique_generations, num_attempts)
    elif diversity_metric == "edit_distance":
        # For now, fallback to fitness (edit distance can be added later)
        return _select_by_fitness(unique_generations, num_attempts)
    else:
        raise ValueError(f"Unknown diversity metric: {diversity_metric}")


def _select_by_fitness(
    generations: list[GenerationResult], num_attempts: int
) -> list[str]:
    """Select solvers by fitness ranking.

    Strategy:
    1. Select solver with highest fitness
    2. Select next different solvers by fitness

    Args:
        generations: Unique generation results
        num_attempts: Number of solvers to select

    Returns:
        List of solver code strings

    Raises:
        ValueError: If not enough valid solvers after filtering empty codes
    """
    # Filter out empty or whitespace-only code
    valid_gens = [g for g in generations if g["solver_code"].strip()]

    if len(valid_gens) < num_attempts:
        raise ValueError(
            f"Not enough valid solvers: need {num_attempts}, "
            f"have {len(valid_gens)} (after filtering empty code)"
        )

    # Sort by fitness (descending)
    sorted_gens = sorted(
        valid_gens, key=lambda g: g["fitness_result"]["fitness"], reverse=True
    )

    # Select top num_attempts
    selected_codes = [gen["solver_code"] for gen in sorted_gens[:num_attempts]]

    return selected_codes


def _select_by_generation_gap(
    generations: list[GenerationResult], num_attempts: int
) -> list[str]:
    """Select solvers from early and late generations.

    Strategy:
    1. Select from earliest generation
    2. Select from latest generation
    3. If more needed, select from middle

    Args:
        generations: Unique generation results
        num_attempts: Number of solvers to select

    Returns:
        List of solver code strings

    Raises:
        ValueError: If not enough valid solvers after filtering empty codes
    """
    # Filter out empty or whitespace-only code
    valid_gens = [g for g in generations if g["solver_code"].strip()]

    if len(valid_gens) < num_attempts:
        raise ValueError(
            f"Not enough valid solvers: need {num_attempts}, "
            f"have {len(valid_gens)} (after filtering empty code)"
        )

    # Sort by generation number
    sorted_gens = sorted(valid_gens, key=lambda g: g["generation"])

    if num_attempts == 2:
        # Select first and last
        return [sorted_gens[0]["solver_code"], sorted_gens[-1]["solver_code"]]
    else:
        # Distribute evenly across generation range
        indices = [
            int(i * (len(sorted_gens) - 1) / (num_attempts - 1))
            for i in range(num_attempts)
        ]
        return [sorted_gens[i]["solver_code"] for i in indices]


def generate_task_predictions(
    task_json_path: str,
    solver_codes: list[str],
    timeout: int = 5,
    sandbox_mode: str = "multiprocess",
) -> list[dict[str, list[list[int]]]]:
    """Apply multiple solvers to all test inputs in a task.

    Generates pass@2 predictions by running each solver on each test input.
    Returns zero-filled placeholder matching input grid shape if solver fails or times out.

    Args:
        task_json_path: Path to ARC task JSON file
        solver_codes: List of solver code strings (length should match num_attempts)
        timeout: Execution timeout per test input in seconds (default: 5)
        sandbox_mode: "multiprocess" or "docker" (default: "multiprocess")

    Returns:
        List of predictions, one dict per test input:
        [
            {"attempt_1": [[...]], "attempt_2": [[...]]},  # Test input 0
            {"attempt_1": [[...]], "attempt_2": [[...]]},  # Test input 1
            ...
        ]

    Note:
        - Handles variable number of test inputs per task (0-3 typically)
        - If solver fails/times out, returns zero-filled placeholder matching input grid shape
        - Converts numpy arrays to Python lists for JSON serialization

    Example:
        >>> task_file = "path/to/task.json"  # doctest: +SKIP
        >>> solver_1 = "def solve(grid): return grid * 2"  # doctest: +SKIP
        >>> solver_2 = "def solve(grid): return grid * 3"  # doctest: +SKIP
        >>> preds = generate_task_predictions(task_file, [solver_1, solver_2])  # doctest: +SKIP
        >>> len(preds)  # doctest: +SKIP
        1  # One test input
        >>> preds[0]["attempt_1"]  # doctest: +SKIP
        [[2, 4], [6, 8]]  # Result from solver_1
    """
    # Load task
    task_data = load_task(task_json_path)
    test_examples = task_data.get("test", [])

    # If no test inputs, return empty list
    if not test_examples:
        return []

    # Create sandbox instance
    sandbox = _get_sandbox(sandbox_mode)

    predictions: list[dict[str, list[list[int]]]] = []

    # Process each test input
    for test_example in test_examples:
        test_input = test_example["input"]
        test_grid = np.array(test_input, dtype=np.int64)

        prediction: dict[str, list[list[int]]] = {}

        # Generate attempts with each solver
        for attempt_idx, solver_code in enumerate(solver_codes, start=1):
            attempt_key = f"attempt_{attempt_idx}"

            # Execute solver using sandbox
            success, result, error_detail = sandbox.execute(
                solver_code, test_grid, timeout=timeout
            )

            if success and result is not None:
                # Convert numpy array to Python list
                result_list = result.tolist()
                prediction[attempt_key] = result_list
            else:
                # Use placeholder matching input grid shape for failed execution
                placeholder = np.zeros_like(test_grid).tolist()
                prediction[attempt_key] = placeholder

        predictions.append(prediction)

    return predictions


def format_submission_json(
    task_predictions: dict[str, list[dict[str, list[list[int]]]]],
    num_attempts: int = 2,
) -> dict[str, list[dict[str, list[list[int]]]]]:
    """Format predictions into Kaggle submission JSON structure.

    This is essentially a pass-through function that validates the structure.
    The input should already match the expected Kaggle format.

    Args:
        task_predictions: Dict mapping task_id -> list of predictions
            where each prediction is {"attempt_1": [[...]], "attempt_2": [[...]], ...}
        num_attempts: Number of attempts per test input (default: 2 for pass@2)

    Returns:
        Submission dict ready for json.dump(), matching sample_submission.json structure

    Structure:
        {
            "task_id_1": [
                {"attempt_1": [[grid]], "attempt_2": [[grid]], ...},  # Test input 0
                {"attempt_1": [[grid]], "attempt_2": [[grid]], ...},  # Test input 1
                ...
            ],
            "task_id_2": [...],
            ...
        }

    Example:
        >>> predictions = {
        ...     "00576224": [{"attempt_1": [[0]], "attempt_2": [[1]]}],
        ...     "007bbfb7": [{"attempt_1": [[2]], "attempt_2": [[3]]}]
        ... }
        >>> submission = format_submission_json(predictions, num_attempts=2)
        >>> submission["00576224"][0]["attempt_1"]
        [[0]]
        >>> import json
        >>> json_str = json.dumps(submission)  # Should not raise
        >>> len(json_str) > 0
        True
    """
    # Validate structure
    for task_id, predictions in task_predictions.items():
        if not isinstance(predictions, list):
            raise ValueError(
                f"Predictions for task {task_id} must be a list, "
                f"got {type(predictions)}"
            )

        for pred_idx, pred in enumerate(predictions):
            if not isinstance(pred, dict):
                raise ValueError(
                    f"Prediction {pred_idx} for task {task_id} must be a dict, "
                    f"got {type(pred)}"
                )

            # Check for all required attempts (attempt_1, attempt_2, ..., attempt_N)
            for attempt_num in range(1, num_attempts + 1):
                attempt_key = f"attempt_{attempt_num}"
                if attempt_key not in pred:
                    raise ValueError(
                        f"Prediction {pred_idx} for task {task_id} missing '{attempt_key}'"
                    )

    # Return as-is (structure already correct)
    return task_predictions


def save_submission_json(
    submission: dict[str, list[dict[str, list[list[int]]]]], output_path: str | Path
) -> None:
    """Save submission dictionary to JSON file.

    Args:
        submission: Formatted submission dictionary
        output_path: Path to save submission.json

    Example:
        >>> submission = {"task1": [{"attempt_1": [[0]], "attempt_2": [[1]]}]}  # doctest: +SKIP
        >>> save_submission_json(submission, "submission.json")  # doctest: +SKIP
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)
