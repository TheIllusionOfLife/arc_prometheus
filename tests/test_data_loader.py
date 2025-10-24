"""Tests for ARC data loading functionality."""

import json
import pytest
import numpy as np
from pathlib import Path
import tempfile
import sys
from io import StringIO

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_prometheus.crucible.data_loader import load_task, print_grid


class TestLoadTask:
    """Tests for load_task function."""

    def test_load_valid_task(self, tmp_path):
        """Test loading a valid ARC task JSON file."""
        task_data = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},
                {"input": [[0, 1]], "output": [[2, 3]]}
            ],
            "test": [
                {"input": [[9, 9], [8, 8]]}
            ]
        }

        task_file = tmp_path / "test_task.json"
        with open(task_file, 'w') as f:
            json.dump(task_data, f)

        result = load_task(str(task_file))

        assert "train" in result
        assert "test" in result
        assert len(result["train"]) == 2
        assert len(result["test"]) == 1
        assert isinstance(result["train"][0]["input"], np.ndarray)
        assert isinstance(result["train"][0]["output"], np.ndarray)
        assert result["train"][0]["input"].shape == (2, 2)
        np.testing.assert_array_equal(result["train"][0]["input"], [[1, 2], [3, 4]])

    def test_load_task_file_not_found(self):
        """Test error handling when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="ARC task file not found"):
            load_task("/nonexistent/path/task.json")

    def test_load_task_invalid_json(self, tmp_path):
        """Test error handling for invalid JSON format."""
        task_file = tmp_path / "invalid.json"
        with open(task_file, 'w') as f:
            f.write("{ invalid json }")

        with pytest.raises(ValueError, match="Invalid JSON format"):
            load_task(str(task_file))

    def test_load_task_missing_keys(self, tmp_path):
        """Test error handling when required keys are missing."""
        task_data = {"train": []}  # Missing 'test' key

        task_file = tmp_path / "missing_keys.json"
        with open(task_file, 'w') as f:
            json.dump(task_data, f)

        with pytest.raises(ValueError, match="must contain 'train' and 'test' keys"):
            load_task(str(task_file))

    def test_load_task_empty_train(self, tmp_path):
        """Test handling of task with empty train examples."""
        task_data = {
            "train": [],
            "test": [{"input": [[1, 2]]}]
        }

        task_file = tmp_path / "empty_train.json"
        with open(task_file, 'w') as f:
            json.dump(task_data, f)

        with pytest.raises(ValueError, match="Task must have at least one train example"):
            load_task(str(task_file))

    def test_load_task_various_grid_sizes(self, tmp_path):
        """Test loading tasks with various grid sizes."""
        task_data = {
            "train": [
                {"input": [[1]], "output": [[2]]},  # 1x1
                {"input": [[1, 2, 3, 4, 5]], "output": [[0, 0, 0, 0, 0]]},  # 1x5
                {"input": [[1, 2], [3, 4], [5, 6]], "output": [[0, 0], [0, 0], [0, 0]]}  # 3x2
            ],
            "test": [
                {"input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}  # 3x3
            ]
        }

        task_file = tmp_path / "various_sizes.json"
        with open(task_file, 'w') as f:
            json.dump(task_data, f)

        result = load_task(str(task_file))

        assert result["train"][0]["input"].shape == (1, 1)
        assert result["train"][1]["input"].shape == (1, 5)
        assert result["train"][2]["input"].shape == (3, 2)
        assert result["test"][0]["input"].shape == (3, 3)


class TestPrintGrid:
    """Tests for print_grid function."""

    def test_print_grid_basic(self, capsys):
        """Test basic grid printing."""
        grid = np.array([[1, 2], [3, 4]])
        print_grid(grid)

        captured = capsys.readouterr()
        output = captured.out

        # Should contain the numbers
        assert "1" in output
        assert "2" in output
        assert "3" in output
        assert "4" in output

    def test_print_grid_with_label(self, capsys):
        """Test grid printing with label."""
        grid = np.array([[5, 6]])
        print_grid(grid, label="Test Grid")

        captured = capsys.readouterr()
        output = captured.out

        assert "Test Grid" in output
        assert "5" in output
        assert "6" in output

    def test_print_grid_single_element(self, capsys):
        """Test printing 1x1 grid."""
        grid = np.array([[9]])
        print_grid(grid)

        captured = capsys.readouterr()
        output = captured.out

        assert "9" in output

    def test_print_grid_all_colors(self, capsys):
        """Test printing grid with all ARC colors (0-9)."""
        grid = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        print_grid(grid, label="All Colors")

        captured = capsys.readouterr()
        output = captured.out

        for i in range(10):
            assert str(i) in output

    def test_print_grid_large_grid(self, capsys):
        """Test printing larger grid (30x30)."""
        grid = np.ones((30, 30), dtype=int)
        print_grid(grid, label="30x30 Grid")

        captured = capsys.readouterr()
        output = captured.out

        assert "30x30 Grid" in output
        # Should contain multiple rows
        lines = output.strip().split('\n')
        assert len(lines) > 30  # At least 30 rows plus label/formatting
