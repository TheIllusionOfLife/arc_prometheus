"""Tests for Phase 1.2: Manual solver implementation.

This test suite validates that a manually-written solver can correctly
solve ARC task 05269061 before we build the LLM pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_prometheus.crucible.data_loader import load_task
from arc_prometheus.crucible.evaluator import evaluate_grids
from arc_prometheus.utils.config import DATA_DIR


class TestManualSolverSignature:
    """Test that solver function follows the required signature."""

    def test_solver_exists(self):
        """Test that solve function is importable."""
        # Import will be from demo script - test after implementation
        # This test documents the requirement
        pass

    def test_solver_signature(self):
        """Test that solve function has correct signature.

        Required: def solve(task_grid: np.ndarray) -> np.ndarray
        """
        # This test documents the required function signature
        # Implementation will be validated by integration tests
        pass


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

    def test_task_example2_pattern(self, task_data):
        """Test the pattern extraction for example 2.

        Input contains diagonal with values [2, 8, 3]
        Output should be filled grid with repeating pattern.
        """
        example = task_data["train"][1]
        expected_output = example["output"]

        # Expected output first row should be [2, 8, 3, 2, 8, 3, 2]
        expected_first_row = np.array([2, 8, 3, 2, 8, 3, 2])
        np.testing.assert_array_equal(expected_output[0], expected_first_row)

    def test_task_example3_pattern(self, task_data):
        """Test the pattern extraction for example 3.

        Input contains diagonal with values [8, 3, 4]
        Output should be filled grid with repeating pattern.
        """
        example = task_data["train"][2]
        expected_output = example["output"]

        # Expected output first row should be [4, 8, 3, 4, 8, 3, 4]
        expected_first_row = np.array([4, 8, 3, 4, 8, 3, 4])
        np.testing.assert_array_equal(expected_output[0], expected_first_row)


class TestSolverReturnType:
    """Test that solver returns correct type and shape."""

    def test_returns_numpy_array(self):
        """Test that solver returns np.ndarray type."""
        # Will be validated by integration test
        # This documents the requirement
        pass

    def test_returns_correct_shape(self):
        """Test that solver returns 7x7 grid for task 05269061."""
        # Will be validated by integration test
        # This documents the requirement: must return same shape as expected output
        pass

    def test_returns_integer_values(self):
        """Test that solver returns integer values (0-9)."""
        # Will be validated by integration test
        # ARC grids must have integer values 0-9
        pass


class TestSolverEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_grid(self):
        """Test behavior with empty grid (all zeros)."""
        empty_grid = np.zeros((7, 7), dtype=int)
        # Should handle gracefully - exact behavior TBD
        # This documents that edge case should be considered
        pass

    def test_single_value_grid(self):
        """Test behavior with grid containing single non-zero value."""
        grid = np.zeros((7, 7), dtype=int)
        grid[3, 3] = 5
        # Should handle gracefully
        pass

    def test_full_grid(self):
        """Test behavior with grid that's already filled (no zeros)."""
        full_grid = np.ones((7, 7), dtype=int)
        # Should handle gracefully
        pass
