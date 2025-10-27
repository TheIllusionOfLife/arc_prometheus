"""Tests for Phase 1.2: Manual solver implementation.

This test suite validates that a manually-written solver can correctly
solve ARC task 05269061 before we build the LLM pipeline.
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_prometheus.crucible.data_loader import load_task
from arc_prometheus.crucible.evaluator import evaluate_grids
from arc_prometheus.utils.config import DATA_DIR

# Import the solver to test
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from demo_phase1_2_manual import solve


class TestManualSolverSignature:
    """Test that solver function follows the required signature."""

    def test_solver_exists(self):
        """Test that solve function is importable and callable."""
        assert callable(solve), "solve function should be callable"

    def test_solver_signature(self):
        """Test that solve function has correct signature.

        Required: def solve(task_grid: np.ndarray) -> np.ndarray
        """
        sig = inspect.signature(solve)
        params = list(sig.parameters.keys())

        assert len(params) == 1, "solve should take exactly one parameter"
        assert params[0] == "task_grid", "Parameter should be named 'task_grid'"

        # Verify return type annotation exists (if present)
        if sig.return_annotation != inspect.Parameter.empty:
            assert "ndarray" in str(sig.return_annotation), "Should return np.ndarray"


class TestManualSolverTask05269061:
    """Test manual solver on actual ARC task 05269061."""

    @pytest.fixture
    def task_data(self):
        """Load task 05269061 from the dataset."""
        challenges_file = DATA_DIR / "arc-agi_training_challenges.json"
        if not challenges_file.exists():
            pytest.skip(f"Dataset not found at {challenges_file}")

        return load_task(str(challenges_file), task_id="05269061")

    def test_task_loaded_correctly(self, task_data):
        """Verify task data structure is correct."""
        assert "train" in task_data
        assert "test" in task_data
        assert len(task_data["train"]) == 3
        assert len(task_data["test"]) >= 1

    def test_task_grid_shapes(self, task_data):
        """Verify all grids are 7x7 as expected."""
        for example in task_data["train"]:
            assert example["input"].shape == (7, 7)
            assert example["output"].shape == (7, 7)

    def test_task_example1_pattern(self, task_data):
        """Test the pattern extraction for example 1.

        Input contains diagonal with values [1, 2, 4]
        Output should be filled grid with repeating pattern.
        """
        example = task_data["train"][0]
        input_grid = example["input"]
        expected_output = example["output"]

        # Verify input has diagonal pattern
        non_zero_values = input_grid[input_grid != 0]
        assert len(non_zero_values) > 0

        # Expected output first row should be [2, 4, 1, 2, 4, 1, 2]
        expected_first_row = np.array([2, 4, 1, 2, 4, 1, 2])
        np.testing.assert_array_equal(expected_output[0], expected_first_row)

        # Test that solver produces correct output
        predicted_output = solve(input_grid)
        assert predicted_output.shape == expected_output.shape, (
            "Solver output shape should match expected shape"
        )
        np.testing.assert_array_equal(
            predicted_output,
            expected_output,
            "Solver output should match expected output for example 1",
        )

    def test_task_example2_pattern(self, task_data):
        """Test the pattern extraction for example 2.

        Input contains diagonal with values [2, 8, 3]
        Output should be filled grid with repeating pattern.
        """
        example = task_data["train"][1]
        input_grid = example["input"]
        expected_output = example["output"]

        # Expected output first row should be [2, 8, 3, 2, 8, 3, 2]
        expected_first_row = np.array([2, 8, 3, 2, 8, 3, 2])
        np.testing.assert_array_equal(expected_output[0], expected_first_row)

        # Test that solver produces correct output
        predicted_output = solve(input_grid)
        assert predicted_output.shape == expected_output.shape, (
            "Solver output shape should match expected shape"
        )
        np.testing.assert_array_equal(
            predicted_output,
            expected_output,
            "Solver output should match expected output for example 2",
        )

    def test_task_example3_pattern(self, task_data):
        """Test the pattern extraction for example 3.

        Input contains diagonal with values [8, 3, 4]
        Output should be filled grid with repeating pattern.
        """
        example = task_data["train"][2]
        input_grid = example["input"]
        expected_output = example["output"]

        # Expected output first row should be [4, 8, 3, 4, 8, 3, 4]
        expected_first_row = np.array([4, 8, 3, 4, 8, 3, 4])
        np.testing.assert_array_equal(expected_output[0], expected_first_row)

        # Test that solver produces correct output
        predicted_output = solve(input_grid)
        assert predicted_output.shape == expected_output.shape, (
            "Solver output shape should match expected shape"
        )
        np.testing.assert_array_equal(
            predicted_output,
            expected_output,
            "Solver output should match expected output for example 3",
        )

    def test_solver_integration_all_train_examples(self, task_data):
        """Integration test: solver should solve all train examples correctly."""
        for idx, example in enumerate(task_data["train"], 1):
            predicted = solve(example["input"])
            is_correct = evaluate_grids(predicted, example["output"])
            assert is_correct, f"Solver failed on train example {idx}"


class TestSolverReturnType:
    """Test that solver returns correct type and shape."""

    def test_returns_numpy_array(self):
        """Test that solver returns np.ndarray type."""
        test_grid = np.zeros((7, 7), dtype=int)
        test_grid[0, 0] = 1
        result = solve(test_grid)
        assert isinstance(result, np.ndarray), "solve() should return np.ndarray"

    def test_returns_correct_shape(self):
        """Test that solver returns 7x7 grid for task 05269061."""
        test_grid = np.zeros((7, 7), dtype=int)
        test_grid[3, 3] = 5
        result = solve(test_grid)
        assert result.shape == (7, 7), "solve() should return 7x7 grid for this task"

    def test_returns_integer_values(self):
        """Test that solver returns integer values (0-9)."""
        test_grid = np.zeros((7, 7), dtype=int)
        test_grid[0, 0] = 1
        result = solve(test_grid)
        assert result.dtype in [np.int32, np.int64, int], (
            "solve() should return integer array"
        )
        assert np.all((result >= 0) & (result <= 9)), (
            "All values should be in range 0-9"
        )


class TestSolverEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_grid(self):
        """Test behavior with empty grid (all zeros)."""
        empty_grid = np.zeros((7, 7), dtype=int)
        result = solve(empty_grid)

        assert isinstance(result, np.ndarray), "Should return array for empty input"
        assert result.shape == (7, 7), "Should return 7x7 grid"
        # With no pattern, should return zeros
        assert np.all(result == 0), "Empty input should produce zero-filled output"

    def test_single_value_grid(self):
        """Test behavior with grid containing single non-zero value."""
        grid = np.zeros((7, 7), dtype=int)
        grid[3, 3] = 5
        result = solve(grid)

        assert isinstance(result, np.ndarray), (
            "Should return array for single-value input"
        )
        assert result.shape == (7, 7), "Should return 7x7 grid"
        # With single value, pattern length is 1, so entire grid should be that value
        assert np.all(result == 5), "Single-value pattern should fill entire grid"

    def test_full_grid(self):
        """Test behavior with grid that's already filled (no zeros)."""
        full_grid = np.ones((7, 7), dtype=int)
        result = solve(full_grid)

        assert isinstance(result, np.ndarray), "Should return array for full input"
        assert result.shape == (7, 7), "Should return 7x7 grid"
        # With all same values, should fill grid with that value
        assert np.all(result == 1), (
            "Full grid with single unique value should produce uniform output"
        )
