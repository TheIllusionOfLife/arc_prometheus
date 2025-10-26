"""Tests for ARC grid evaluation functionality."""

import pytest
import numpy as np

from arc_prometheus.crucible.evaluator import evaluate_grids


class TestEvaluateGrids:
    """Tests for evaluate_grids function."""

    def test_identical_grids(self):
        """Test that identical grids are evaluated as equal."""
        grid_a = np.array([[1, 2, 3], [4, 5, 6]])
        grid_b = np.array([[1, 2, 3], [4, 5, 6]])

        assert evaluate_grids(grid_a, grid_b) is True

    def test_different_values(self):
        """Test that grids with different values are not equal."""
        grid_a = np.array([[1, 2], [3, 4]])
        grid_b = np.array([[1, 2], [3, 5]])  # Different last element

        assert evaluate_grids(grid_a, grid_b) is False

    def test_different_shapes(self):
        """Test that grids with different shapes are not equal."""
        grid_a = np.array([[1, 2], [3, 4]])
        grid_b = np.array([[1, 2, 3], [4, 5, 6]])

        assert evaluate_grids(grid_a, grid_b) is False

    def test_single_element_grids(self):
        """Test evaluation of 1x1 grids."""
        grid_a = np.array([[5]])
        grid_b = np.array([[5]])

        assert evaluate_grids(grid_a, grid_b) is True

        grid_c = np.array([[6]])
        assert evaluate_grids(grid_a, grid_c) is False

    def test_large_grids(self):
        """Test evaluation of larger grids (30x30)."""
        grid_a = np.ones((30, 30), dtype=int)
        grid_b = np.ones((30, 30), dtype=int)

        assert evaluate_grids(grid_a, grid_b) is True

        grid_b[15, 15] = 2  # Change one element
        assert evaluate_grids(grid_a, grid_b) is False

    def test_different_row_count(self):
        """Test grids with different number of rows."""
        grid_a = np.array([[1, 2]])
        grid_b = np.array([[1, 2], [3, 4]])

        assert evaluate_grids(grid_a, grid_b) is False

    def test_different_column_count(self):
        """Test grids with different number of columns."""
        grid_a = np.array([[1], [2]])
        grid_b = np.array([[1, 2], [3, 4]])

        assert evaluate_grids(grid_a, grid_b) is False

    def test_all_zeros(self):
        """Test grids with all zero values."""
        grid_a = np.zeros((3, 3), dtype=int)
        grid_b = np.zeros((3, 3), dtype=int)

        assert evaluate_grids(grid_a, grid_b) is True

    def test_all_colors(self):
        """Test grids with all ARC colors (0-9)."""
        grid_a = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        grid_b = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

        assert evaluate_grids(grid_a, grid_b) is True

    def test_type_conversion_from_list(self):
        """Test that function handles conversion from Python lists."""
        grid_a = np.array([[1, 2], [3, 4]])
        grid_b = [[1, 2], [3, 4]]  # Python list

        # Should work with lists converted to arrays
        result = evaluate_grids(grid_a, np.array(grid_b))
        assert result is True

    def test_empty_grids(self):
        """Test evaluation with empty grids."""
        grid_a = np.array([[]]).reshape(0, 0)
        grid_b = np.array([[]]).reshape(0, 0)

        # Empty grids should be considered equal
        assert evaluate_grids(grid_a, grid_b) is True

    def test_dtype_consistency(self):
        """Test that grids with different dtypes but same values are equal."""
        grid_a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        grid_b = np.array([[1, 2], [3, 4]], dtype=np.int64)

        assert evaluate_grids(grid_a, grid_b) is True
