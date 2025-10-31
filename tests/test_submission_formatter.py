"""Tests for submission_formatter.py - pass@2 output format generation (TDD).

This module tests the pass@2 submission format generation for Kaggle:
- Diversity selection from generation history
- Prediction generation for variable test inputs
- Submission JSON formatting
"""

import json
from pathlib import Path
from typing import Any

import pytest

from arc_prometheus.evolutionary_engine.evolution_loop import GenerationResult
from arc_prometheus.evolutionary_engine.submission_formatter import (
    format_submission_json,
    generate_task_predictions,
    select_diverse_solvers,
)

# === Test Data Fixtures ===


@pytest.fixture
def sample_generations() -> list[GenerationResult]:
    """Create sample generation results for testing diversity selection."""
    return [
        {
            "generation": 0,
            "solver_code": "def solve(grid):\n    return grid",
            "fitness_result": {
                "fitness": 3.0,
                "train_correct": 3,
                "train_total": 3,
                "test_correct": 0,
                "test_total": 1,
                "train_accuracy": 1.0,
                "test_accuracy": 0.0,
                "execution_errors": [],
                "error_details": [],
                "error_summary": {},
            },
            "refinement_count": 0,
            "total_time": 10.0,
            "improvement": 0.0,
        },
        {
            "generation": 1,
            "solver_code": "def solve(grid):\n    return grid * 2",
            "fitness_result": {
                "fitness": 13.0,
                "train_correct": 3,
                "train_total": 3,
                "test_correct": 1,
                "test_total": 1,
                "train_accuracy": 1.0,
                "test_accuracy": 1.0,
                "execution_errors": [],
                "error_details": [],
                "error_summary": {},
            },
            "refinement_count": 1,
            "total_time": 12.0,
            "improvement": 10.0,
        },
        {
            "generation": 2,
            "solver_code": "def solve(grid):\n    return grid + 1",
            "fitness_result": {
                "fitness": 5.0,
                "train_correct": 2,
                "train_total": 3,
                "test_correct": 0,
                "test_total": 1,
                "train_accuracy": 0.667,
                "test_accuracy": 0.0,
                "execution_errors": ["example_0"],
                "error_details": [],
                "error_summary": {},
            },
            "refinement_count": 1,
            "total_time": 11.0,
            "improvement": -8.0,
        },
        {
            "generation": 3,
            "solver_code": "def solve(grid):\n    return np.rot90(grid)",
            "fitness_result": {
                "fitness": 7.0,
                "train_correct": 2,
                "train_total": 3,
                "test_correct": 0,
                "test_total": 1,
                "train_accuracy": 0.667,
                "test_accuracy": 0.0,
                "execution_errors": ["example_2"],
                "error_details": [],
                "error_summary": {},
            },
            "refinement_count": 1,
            "total_time": 13.0,
            "improvement": 2.0,
        },
        {
            "generation": 4,
            "solver_code": "def solve(grid):\n    return grid[::-1]",
            "fitness_result": {
                "fitness": 2.0,
                "train_correct": 2,
                "train_total": 3,
                "test_correct": 0,
                "test_total": 1,
                "train_accuracy": 0.667,
                "test_accuracy": 0.0,
                "execution_errors": ["example_0", "test_0"],
                "error_details": [],
                "error_summary": {},
            },
            "refinement_count": 1,
            "total_time": 10.5,
            "improvement": -5.0,
        },
    ]


@pytest.fixture
def sample_task_1_test() -> dict[str, Any]:
    """Create sample task with 1 test input."""
    return {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 4], [6, 8]]},
            {"input": [[5, 6], [7, 8]], "output": [[10, 12], [14, 16]]},
        ],
        "test": [{"input": [[9, 10], [11, 12]], "output": [[18, 20], [22, 24]]}],
    }


@pytest.fixture
def sample_task_2_tests() -> dict[str, Any]:
    """Create sample task with 2 test inputs."""
    return {
        "train": [
            {"input": [[1, 2]], "output": [[2, 4]]},
            {"input": [[3, 4]], "output": [[6, 8]]},
        ],
        "test": [
            {"input": [[5, 6]], "output": [[10, 12]]},
            {"input": [[7, 8]], "output": [[14, 16]]},
        ],
    }


@pytest.fixture
def sample_task_3_tests() -> dict[str, Any]:
    """Create sample task with 3 test inputs."""
    return {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [
            {"input": [[3]], "output": [[6]]},
            {"input": [[5]], "output": [[10]]},
            {"input": [[7]], "output": [[14]]},
        ],
    }


@pytest.fixture
def simple_solver_code() -> str:
    """Simple working solver for testing."""
    return """import numpy as np

def solve(grid: np.ndarray) -> np.ndarray:
    return grid * 2
"""


@pytest.fixture
def failing_solver_code() -> str:
    """Solver that will timeout for testing error handling."""
    return """import numpy as np

def solve(grid: np.ndarray) -> np.ndarray:
    while True:
        pass
    return grid
"""


# === Diversity Selection Tests ===


def test_select_diverse_solvers_basic(
    sample_generations: list[GenerationResult],
) -> None:
    """Test basic diversity selection returns 2 different solvers."""
    solvers = select_diverse_solvers(sample_generations, num_attempts=2)

    assert len(solvers) == 2
    assert isinstance(solvers[0], str)
    assert isinstance(solvers[1], str)
    assert solvers[0] != solvers[1], "Selected solvers should be different"


def test_select_diverse_solvers_fitness_based(
    sample_generations: list[GenerationResult],
) -> None:
    """Test that highest fitness solver is selected first."""
    solvers = select_diverse_solvers(
        sample_generations, num_attempts=2, diversity_metric="fitness"
    )

    # Best fitness is 13.0 from generation 1
    assert "grid * 2" in solvers[0], "Should select highest fitness solver first"

    # Second solver should be different
    assert solvers[1] != solvers[0]


def test_select_diverse_solvers_removes_duplicates() -> None:
    """Test that duplicate code is removed before selection."""
    generations_with_duplicates: list[GenerationResult] = [
        {
            "generation": 0,
            "solver_code": "def solve(grid):\n    return grid",
            "fitness_result": {
                "fitness": 5.0,
                "train_correct": 2,
                "train_total": 3,
                "test_correct": 0,
                "test_total": 1,
                "train_accuracy": 0.667,
                "test_accuracy": 0.0,
                "execution_errors": [],
                "error_details": [],
                "error_summary": {},
            },
            "refinement_count": 0,
            "total_time": 10.0,
            "improvement": 0.0,
        },
        {
            "generation": 1,
            "solver_code": "def solve(grid):\n    return grid",  # Duplicate
            "fitness_result": {
                "fitness": 5.0,
                "train_correct": 2,
                "train_total": 3,
                "test_correct": 0,
                "test_total": 1,
                "train_accuracy": 0.667,
                "test_accuracy": 0.0,
                "execution_errors": [],
                "error_details": [],
                "error_summary": {},
            },
            "refinement_count": 1,
            "total_time": 11.0,
            "improvement": 0.0,
        },
        {
            "generation": 2,
            "solver_code": "def solve(grid):\n    return grid * 2",
            "fitness_result": {
                "fitness": 7.0,
                "train_correct": 2,
                "train_total": 3,
                "test_correct": 0,
                "test_total": 1,
                "train_accuracy": 0.667,
                "test_accuracy": 0.0,
                "execution_errors": [],
                "error_details": [],
                "error_summary": {},
            },
            "refinement_count": 1,
            "total_time": 12.0,
            "improvement": 2.0,
        },
    ]

    solvers = select_diverse_solvers(generations_with_duplicates, num_attempts=2)

    assert len(solvers) == 2
    assert solvers[0] != solvers[1]


def test_select_diverse_solvers_insufficient_unique() -> None:
    """Test error when insufficient unique solvers available."""
    generations_only_one: list[GenerationResult] = [
        {
            "generation": 0,
            "solver_code": "def solve(grid):\n    return grid",
            "fitness_result": {
                "fitness": 3.0,
                "train_correct": 3,
                "train_total": 3,
                "test_correct": 0,
                "test_total": 1,
                "train_accuracy": 1.0,
                "test_accuracy": 0.0,
                "execution_errors": [],
                "error_details": [],
                "error_summary": {},
            },
            "refinement_count": 0,
            "total_time": 10.0,
            "improvement": 0.0,
        }
    ]

    with pytest.raises(ValueError, match="Not enough unique solvers"):
        select_diverse_solvers(generations_only_one, num_attempts=2)


def test_select_diverse_solvers_generation_gap() -> None:
    """Test generation gap diversity selects from early and late generations."""
    generations = [
        {
            "generation": 0,
            "solver_code": "def solve(grid):\n    return grid",
            "fitness_result": {
                "fitness": 3.0,
                "train_correct": 3,
                "train_total": 3,
                "test_correct": 0,
                "test_total": 1,
                "train_accuracy": 1.0,
                "test_accuracy": 0.0,
                "execution_errors": [],
                "error_details": [],
                "error_summary": {},
            },
            "refinement_count": 0,
            "total_time": 10.0,
            "improvement": 0.0,
        },
        {
            "generation": 4,
            "solver_code": "def solve(grid):\n    return grid * 2",
            "fitness_result": {
                "fitness": 13.0,
                "train_correct": 3,
                "train_total": 3,
                "test_correct": 1,
                "test_total": 1,
                "train_accuracy": 1.0,
                "test_accuracy": 1.0,
                "execution_errors": [],
                "error_details": [],
                "error_summary": {},
            },
            "refinement_count": 1,
            "total_time": 12.0,
            "improvement": 10.0,
        },
    ]

    solvers = select_diverse_solvers(
        generations,
        num_attempts=2,
        diversity_metric="generation_gap",  # type: ignore[arg-type]
    )

    assert len(solvers) == 2
    assert solvers[0] != solvers[1]


# === Prediction Generation Tests ===


def test_generate_task_predictions_1_test(
    sample_task_1_test: dict[str, Any], simple_solver_code: str, tmp_path: Path
) -> None:
    """Test prediction generation for task with 1 test input."""
    # Create temporary task file
    task_file = tmp_path / "task.json"
    with open(task_file, "w") as f:
        json.dump(sample_task_1_test, f)

    solver_codes = [
        simple_solver_code,
        simple_solver_code,
    ]  # Same solver for simplicity

    predictions = generate_task_predictions(
        task_json_path=str(task_file),
        solver_codes=solver_codes,
        timeout=5,
        sandbox_mode="multiprocess",
    )

    # Should have 1 prediction (1 test input)
    assert len(predictions) == 1

    # Each prediction should have attempt_1 and attempt_2
    assert "attempt_1" in predictions[0]
    assert "attempt_2" in predictions[0]

    # Results should be Python lists (not numpy arrays)
    assert isinstance(predictions[0]["attempt_1"], list)
    assert isinstance(predictions[0]["attempt_2"], list)

    # Check correct computation (grid * 2)
    expected = [[18, 20], [22, 24]]
    assert predictions[0]["attempt_1"] == expected
    assert predictions[0]["attempt_2"] == expected


def test_generate_task_predictions_2_tests(
    sample_task_2_tests: dict[str, Any], simple_solver_code: str, tmp_path: Path
) -> None:
    """Test prediction generation for task with 2 test inputs."""
    task_file = tmp_path / "task.json"
    with open(task_file, "w") as f:
        json.dump(sample_task_2_tests, f)

    solver_codes = [simple_solver_code, simple_solver_code]

    predictions = generate_task_predictions(
        task_json_path=str(task_file),
        solver_codes=solver_codes,
        timeout=5,
        sandbox_mode="multiprocess",
    )

    # Should have 2 predictions (2 test inputs)
    assert len(predictions) == 2

    # Each prediction should have both attempts
    for pred in predictions:
        assert "attempt_1" in pred
        assert "attempt_2" in pred
        assert isinstance(pred["attempt_1"], list)
        assert isinstance(pred["attempt_2"], list)

    # Check correct computations
    assert predictions[0]["attempt_1"] == [[10, 12]]
    assert predictions[1]["attempt_1"] == [[14, 16]]


def test_generate_task_predictions_3_tests(
    sample_task_3_tests: dict[str, Any], simple_solver_code: str, tmp_path: Path
) -> None:
    """Test prediction generation for task with 3 test inputs."""
    task_file = tmp_path / "task.json"
    with open(task_file, "w") as f:
        json.dump(sample_task_3_tests, f)

    solver_codes = [simple_solver_code, simple_solver_code]

    predictions = generate_task_predictions(
        task_json_path=str(task_file),
        solver_codes=solver_codes,
        timeout=5,
        sandbox_mode="multiprocess",
    )

    # Should have 3 predictions (3 test inputs)
    assert len(predictions) == 3

    for pred in predictions:
        assert "attempt_1" in pred
        assert "attempt_2" in pred


def test_generate_task_predictions_solver_timeout(
    sample_task_1_test: dict[str, Any], failing_solver_code: str, tmp_path: Path
) -> None:
    """Test that solver timeout results in placeholder grid."""
    task_file = tmp_path / "task.json"
    with open(task_file, "w") as f:
        json.dump(sample_task_1_test, f)

    solver_codes = [failing_solver_code, failing_solver_code]

    predictions = generate_task_predictions(
        task_json_path=str(task_file),
        solver_codes=solver_codes,
        timeout=1,  # Short timeout to fail quickly
        sandbox_mode="multiprocess",
    )

    # Should still return predictions with placeholder
    assert len(predictions) == 1
    assert "attempt_1" in predictions[0]
    assert "attempt_2" in predictions[0]

    # Placeholder should be [[0, 0], [0, 0]]
    assert predictions[0]["attempt_1"] == [[0, 0], [0, 0]]
    assert predictions[0]["attempt_2"] == [[0, 0], [0, 0]]


def test_generate_task_predictions_diverse_solvers(
    sample_task_1_test: dict[str, Any], tmp_path: Path
) -> None:
    """Test that different solvers produce different attempts."""
    task_file = tmp_path / "task.json"
    with open(task_file, "w") as f:
        json.dump(sample_task_1_test, f)

    solver_1 = """import numpy as np
def solve(grid: np.ndarray) -> np.ndarray:
    return grid * 2
"""

    solver_2 = """import numpy as np
def solve(grid: np.ndarray) -> np.ndarray:
    return grid * 3
"""

    solver_codes = [solver_1, solver_2]

    predictions = generate_task_predictions(
        task_json_path=str(task_file),
        solver_codes=solver_codes,
        timeout=5,
        sandbox_mode="multiprocess",
    )

    # Attempts should be different
    assert predictions[0]["attempt_1"] != predictions[0]["attempt_2"]

    # Check correct values
    assert predictions[0]["attempt_1"] == [[18, 20], [22, 24]]  # * 2
    assert predictions[0]["attempt_2"] == [[27, 30], [33, 36]]  # * 3


# === Submission Format Tests ===


def test_format_submission_json_single_task() -> None:
    """Test submission formatting for single task."""
    task_predictions = {"00576224": [{"attempt_1": [[1, 2]], "attempt_2": [[3, 4]]}]}

    submission = format_submission_json(task_predictions)

    assert "00576224" in submission
    assert len(submission["00576224"]) == 1
    assert submission["00576224"][0]["attempt_1"] == [[1, 2]]
    assert submission["00576224"][0]["attempt_2"] == [[3, 4]]


def test_format_submission_json_multiple_tasks() -> None:
    """Test submission formatting for multiple tasks."""
    task_predictions = {
        "00576224": [{"attempt_1": [[1, 2]], "attempt_2": [[3, 4]]}],
        "007bbfb7": [
            {"attempt_1": [[5, 6]], "attempt_2": [[7, 8]]},
            {"attempt_1": [[9, 10]], "attempt_2": [[11, 12]]},
        ],
    }

    submission = format_submission_json(task_predictions)

    assert len(submission) == 2
    assert "00576224" in submission
    assert "007bbfb7" in submission
    assert len(submission["007bbfb7"]) == 2


def test_format_submission_json_matches_sample_structure() -> None:
    """Test that output structure matches sample_submission.json."""
    task_predictions = {
        "test_task": [
            {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]},
            {"attempt_1": [[1, 1], [1, 1]], "attempt_2": [[2, 2], [2, 2]]},
        ]
    }

    submission = format_submission_json(task_predictions)

    # Check top-level structure
    assert isinstance(submission, dict)
    assert "test_task" in submission

    # Check per-task structure (list of predictions)
    assert isinstance(submission["test_task"], list)
    assert len(submission["test_task"]) == 2

    # Check per-prediction structure
    for pred in submission["test_task"]:
        assert isinstance(pred, dict)
        assert "attempt_1" in pred
        assert "attempt_2" in pred
        assert isinstance(pred["attempt_1"], list)
        assert isinstance(pred["attempt_2"], list)


def test_format_submission_json_serializable() -> None:
    """Test that submission is JSON serializable."""
    task_predictions = {"task1": [{"attempt_1": [[1, 2]], "attempt_2": [[3, 4]]}]}

    submission = format_submission_json(task_predictions)

    # Should not raise
    json_str = json.dumps(submission)

    # Should be able to load back
    loaded = json.loads(json_str)
    assert loaded == submission


# === Integration Tests ===


def test_end_to_end_submission_generation(
    sample_generations: list[GenerationResult],
    sample_task_1_test: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Test complete flow: select solvers → generate predictions → format."""
    # Create task file
    task_file = tmp_path / "task.json"
    with open(task_file, "w") as f:
        json.dump(sample_task_1_test, f)

    # Step 1: Select diverse solvers
    solver_codes = select_diverse_solvers(sample_generations, num_attempts=2)
    assert len(solver_codes) == 2

    # Step 2: Generate predictions
    predictions = generate_task_predictions(
        task_json_path=str(task_file),
        solver_codes=solver_codes,
        timeout=5,
        sandbox_mode="multiprocess",
    )
    assert len(predictions) == 1

    # Step 3: Format as submission
    task_predictions = {"test_task": predictions}
    submission = format_submission_json(task_predictions)

    # Validate final structure
    assert "test_task" in submission
    assert len(submission["test_task"]) == 1
    assert "attempt_1" in submission["test_task"][0]
    assert "attempt_2" in submission["test_task"][0]

    # Should be JSON serializable
    json_str = json.dumps(submission)
    assert len(json_str) > 0


# === Edge Cases ===


def test_generate_predictions_empty_test_list(
    tmp_path: Path, simple_solver_code: str
) -> None:
    """Test handling of task with no test inputs."""
    task_no_tests = {"train": [{"input": [[1]], "output": [[2]]}], "test": []}

    task_file = tmp_path / "task.json"
    with open(task_file, "w") as f:
        json.dump(task_no_tests, f)

    solver_codes = [simple_solver_code, simple_solver_code]

    predictions = generate_task_predictions(
        task_json_path=str(task_file),
        solver_codes=solver_codes,
        timeout=5,
        sandbox_mode="multiprocess",
    )

    # Should return empty list
    assert len(predictions) == 0


def test_generate_predictions_invalid_solver(
    sample_task_1_test: dict[str, Any], tmp_path: Path
) -> None:
    """Test handling of syntactically invalid solver."""
    task_file = tmp_path / "task.json"
    with open(task_file, "w") as f:
        json.dump(sample_task_1_test, f)

    invalid_solver = """def solve(grid):
    return grid +  # Syntax error
"""

    solver_codes = [invalid_solver, invalid_solver]

    predictions = generate_task_predictions(
        task_json_path=str(task_file),
        solver_codes=solver_codes,
        timeout=5,
        sandbox_mode="multiprocess",
    )

    # Should return placeholder for failed solvers
    assert len(predictions) == 1
    assert predictions[0]["attempt_1"] == [[0, 0], [0, 0]]
    assert predictions[0]["attempt_2"] == [[0, 0], [0, 0]]


def test_select_diverse_solvers_all_zero_fitness() -> None:
    """Test diversity selection when all solvers have zero fitness."""
    generations = [
        {
            "generation": i,
            "solver_code": f"def solve(grid):\n    return grid + {i}",
            "fitness_result": {
                "fitness": 0.0,
                "train_correct": 0,
                "train_total": 3,
                "test_correct": 0,
                "test_total": 1,
                "train_accuracy": 0.0,
                "test_accuracy": 0.0,
                "execution_errors": [],
                "error_details": [],
                "error_summary": {},
            },
            "refinement_count": 0 if i == 0 else 1,
            "total_time": 10.0,
            "improvement": 0.0,
        }
        for i in range(5)
    ]

    solvers = select_diverse_solvers(generations, num_attempts=2)  # type: ignore[arg-type]

    # Should still select 2 different solvers
    assert len(solvers) == 2
    assert solvers[0] != solvers[1]
