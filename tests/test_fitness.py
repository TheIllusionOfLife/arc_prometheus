"""Tests for fitness evaluation system (Phase 2.1)."""

import json

import pytest

from arc_prometheus.evolutionary_engine.fitness import calculate_fitness


class TestFitnessCalculation:
    """Test fitness evaluation with various solver scenarios."""

    def test_perfect_solver_all_correct(self, tmp_path):
        """Test solver that solves all train and test examples correctly."""
        # Create simple task: copy input to output
        task_data = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
                {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]},
                {"input": [[9, 0], [1, 2]], "output": [[9, 0], [1, 2]]},
            ],
            "test": [{"input": [[3, 4], [5, 6]], "output": [[3, 4], [5, 6]]}],
        }

        task_file = tmp_path / "task.json"

        task_file.write_text(json.dumps(task_data))

        # Perfect solver: just returns input as-is
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid.copy()
"""

        result = calculate_fitness(str(task_file), solver_code)

        assert result["fitness"] == 13  # 3*1 + 1*10
        assert result["train_correct"] == 3
        assert result["train_total"] == 3
        assert result["test_correct"] == 1
        assert result["test_total"] == 1
        assert result["train_accuracy"] == 1.0
        assert result["test_accuracy"] == 1.0
        assert len(result["execution_errors"]) == 0

    def test_train_only_solver_no_generalization(self, tmp_path):
        """Test solver that only works on train examples (overfitting)."""
        task_data = {
            "train": [
                {"input": [[1, 1]], "output": [[2, 2]]},
                {"input": [[2, 2]], "output": [[3, 3]]},
                {"input": [[3, 3]], "output": [[4, 4]]},
            ],
            "test": [{"input": [[5, 5]], "output": [[6, 6]]}],
        }

        task_file = tmp_path / "task.json"

        task_file.write_text(json.dumps(task_data))

        # Solver that memorizes train examples but fails on test
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Hardcoded lookup table (overfitting)
    lookup = {
        (1, 1): np.array([[2, 2]]),
        (2, 2): np.array([[3, 3]]),
        (3, 3): np.array([[4, 4]])
    }
    key = tuple(task_grid[0])
    if key in lookup:
        return lookup[key]
    # Returns wrong result for unseen inputs
    return np.array([[0, 0]])
"""

        result = calculate_fitness(str(task_file), solver_code)

        assert result["fitness"] == 3  # 3*1 + 0*10 (demonstrates overfitting)
        assert result["train_correct"] == 3
        assert result["test_correct"] == 0
        assert result["train_accuracy"] == 1.0
        assert result["test_accuracy"] == 0.0

    def test_partial_success(self, tmp_path):
        """Test solver with partial success on train and test."""
        task_data = {
            "train": [
                {"input": [[1]], "output": [[2]]},
                {"input": [[2]], "output": [[4]]},
                {"input": [[3]], "output": [[6]]},
            ],
            "test": [
                {"input": [[4]], "output": [[8]]},
                {"input": [[5]], "output": [[10]]},
            ],
        }

        task_file = tmp_path / "task.json"

        task_file.write_text(json.dumps(task_data))

        # Solver that works on first two train and first test
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    value = task_grid[0, 0]
    # Works for 1, 2, 4 but not 3, 5
    if value in [1, 2, 4]:
        return np.array([[value * 2]])
    return np.array([[0]])  # Wrong for 3, 5
"""

        result = calculate_fitness(str(task_file), solver_code)

        assert result["fitness"] == 12  # 2*1 + 1*10
        assert result["train_correct"] == 2
        assert result["train_total"] == 3
        assert result["test_correct"] == 1
        assert result["test_total"] == 2

    def test_solver_with_timeout(self, tmp_path):
        """Test solver that causes infinite loop (timeout)."""
        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }

        task_file = tmp_path / "task.json"

        task_file.write_text(json.dumps(task_data))

        # Infinite loop solver
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    while True:
        pass
    return task_grid
"""

        result = calculate_fitness(str(task_file), solver_code, timeout=1)

        assert result["fitness"] == 0
        assert result["train_correct"] == 0
        assert result["test_correct"] == 0
        assert (
            "timeout" in str(result["execution_errors"]).lower()
            or len(result["execution_errors"]) > 0
        )

    def test_solver_with_runtime_error(self, tmp_path):
        """Test solver that raises runtime exception."""
        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }

        task_file = tmp_path / "task.json"

        task_file.write_text(json.dumps(task_data))

        # Solver with division by zero
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    x = 1 / 0  # Runtime error
    return task_grid
"""

        result = calculate_fitness(str(task_file), solver_code)

        assert result["fitness"] == 0
        assert result["train_correct"] == 0
        assert len(result["execution_errors"]) > 0

    def test_solver_with_syntax_error(self, tmp_path):
        """Test solver with invalid Python syntax."""
        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }

        task_file = tmp_path / "task.json"

        task_file.write_text(json.dumps(task_data))

        # Invalid syntax
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + + +  # Syntax error
"""

        result = calculate_fitness(str(task_file), solver_code)

        assert result["fitness"] == 0
        assert result["train_correct"] == 0
        assert len(result["execution_errors"]) > 0

    def test_empty_task_data(self, tmp_path):
        """Test task with no train/test examples."""
        task_data = {"train": [], "test": []}

        task_file = tmp_path / "task.json"

        task_file.write_text(json.dumps(task_data))

        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
"""

        result = calculate_fitness(str(task_file), solver_code)

        assert result["fitness"] == 0
        assert result["train_correct"] == 0
        assert result["train_total"] == 0
        assert result["test_correct"] == 0
        assert result["test_total"] == 0
        assert result["train_accuracy"] == 0.0
        assert result["test_accuracy"] == 0.0

    def test_fitness_calculation_weights_test_10x(self, tmp_path):
        """Verify that test accuracy is weighted 10x higher than train."""
        task_data = {
            "train": [{"input": [[1]], "output": [[1]]}] * 10,  # 10 train examples
            "test": [{"input": [[1]], "output": [[1]]}] * 1,  # 1 test example
        }

        task_file = tmp_path / "task.json"

        task_file.write_text(json.dumps(task_data))

        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
"""

        result = calculate_fitness(str(task_file), solver_code)

        # 10 train correct (10 points) + 1 test correct (10 points) = 20
        assert result["fitness"] == 20
        assert result["train_correct"] == 10
        assert result["test_correct"] == 1

        # Verify: 1 test example is worth 10 train examples
        assert result["test_correct"] * 10 == result["train_correct"] * 1

    def test_return_format_completeness(self, tmp_path):
        """Test that result dict contains all required keys."""
        task_data = {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]], "output": [[2]]}],
        }

        task_file = tmp_path / "task.json"

        task_file.write_text(json.dumps(task_data))

        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
"""

        result = calculate_fitness(str(task_file), solver_code)

        required_keys = {
            "fitness",
            "train_correct",
            "train_total",
            "test_correct",
            "test_total",
            "train_accuracy",
            "test_accuracy",
            "execution_errors",
        }

        assert set(result.keys()) == required_keys

        # Verify types
        assert isinstance(result["fitness"], (int, float))
        assert isinstance(result["train_correct"], int)
        assert isinstance(result["train_total"], int)
        assert isinstance(result["test_correct"], int)
        assert isinstance(result["test_total"], int)
        assert isinstance(result["train_accuracy"], float)
        assert isinstance(result["test_accuracy"], float)
        assert isinstance(result["execution_errors"], list)

    def test_task_with_missing_output_keys(self, tmp_path):
        """Test handling of examples without 'output' key (ARC evaluation tasks)."""
        # Simulate ARC evaluation task format where test examples lack outputs
        task_data = {
            "train": [
                {"input": [[1]], "output": [[2]]},
                {"input": [[2]], "output": [[4]]},
            ],
            "test": [
                {"input": [[3]]},  # No output key
                {"input": [[4]]},  # No output key
            ],
        }

        task_file = tmp_path / "task_no_outputs.json"
        task_file.write_text(json.dumps(task_data))

        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid * 2
"""

        result = calculate_fitness(str(task_file), solver_code)

        # Should evaluate train examples normally
        assert result["train_total"] == 2
        assert result["train_correct"] == 2

        # Should skip test examples without output
        assert result["test_total"] == 0
        assert result["test_correct"] == 0

        # Fitness should only count train
        assert result["fitness"] == 2  # 2*1 + 0*10


class TestFitnessWithRealDataset:
    """Test fitness calculation with actual ARC dataset."""

    @pytest.fixture
    def arc_dataset_path(self):
        """Path to ARC training challenges."""
        return "data/arc-prize-2025/arc-agi_training_challenges.json"

    def test_fitness_with_real_task(self, arc_dataset_path, tmp_path):
        """Test fitness calculation with a real ARC task."""
        from pathlib import Path

        # Skip if dataset not available
        if not Path(arc_dataset_path).exists():
            pytest.skip("ARC dataset not available")

        # Load a known task (007bbfb7 used in Phase 1)
        with open(arc_dataset_path) as f:
            all_tasks = json.load(f)

        task_id = "007bbfb7"
        if task_id not in all_tasks:
            pytest.skip(f"Task {task_id} not found in dataset")

        # Create temporary file with single task
        task_file = tmp_path / f"{task_id}.json"
        task_file.write_text(json.dumps(all_tasks[task_id]))

        # Simple solver (likely to fail, but should execute)
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Simple solver: return input unchanged
    return task_grid.copy()
"""

        result = calculate_fitness(str(task_file), solver_code)

        # Verify structure (not exact values, as solver will likely fail)
        assert "fitness" in result
        assert "train_correct" in result
        assert "test_correct" in result
        assert result["train_total"] == len(all_tasks[task_id]["train"])
        # Test total might be 0 if test examples lack 'output' key (evaluation tasks)
        assert result["test_total"] <= len(all_tasks[task_id]["test"])
        assert (
            0
            <= result["fitness"]
            <= (result["train_total"] + result["test_total"] * 10)
        )
