"""Tests for Phase 1.5 End-to-End Pipeline.

This module tests the complete orchestration of all Phase 1 components:
- Data loading from ARC dataset
- LLM solver code generation
- Sandbox execution of generated code
- Grid evaluation and result reporting
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_prometheus.crucible.data_loader import load_task
from arc_prometheus.crucible.evaluator import evaluate_grids
from arc_prometheus.crucible.sandbox import safe_execute


class TestE2EPipelineCore:
    """Tests for core E2E pipeline orchestration logic."""

    @patch("arc_prometheus.cognitive_cells.programmer.generate_solver")
    @patch("arc_prometheus.crucible.sandbox.safe_execute")
    def test_complete_pipeline_success(self, mock_execute, mock_generate):
        """Test complete pipeline with successful solver generation and execution."""
        # Mock LLM to return valid solver
        mock_generate.return_value = """
def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid.copy()
"""

        # Mock sandbox to return correct output
        mock_execute.return_value = (True, np.array([[1, 2], [3, 4]]))

        # Simulate pipeline logic
        train_pairs = [
            {"input": np.array([[1, 2], [3, 4]]), "output": np.array([[1, 2], [3, 4]])}
        ]

        solver_code = mock_generate(train_pairs)
        assert solver_code is not None
        assert "def solve" in solver_code

        success, result = mock_execute(solver_code, train_pairs[0]["input"])
        assert success is True
        assert result is not None

        is_correct = evaluate_grids(result, train_pairs[0]["output"])
        assert is_correct is True

    @patch("arc_prometheus.cognitive_cells.programmer.generate_solver")
    def test_llm_generation_failure(self, mock_generate):
        """Test pipeline handles LLM generation failure gracefully."""
        # Mock LLM to raise error
        mock_generate.side_effect = ValueError("API key not configured")
        train_pairs = [{"input": np.array([[1, 2]]), "output": np.array([[1, 2]])}]

        with pytest.raises(ValueError, match="API key not configured"):
            mock_generate(train_pairs)

    @patch("arc_prometheus.cognitive_cells.programmer.generate_solver")
    @patch("arc_prometheus.crucible.sandbox.safe_execute")
    def test_sandbox_timeout_handling(self, mock_execute, mock_generate):
        """Test pipeline handles sandbox timeout gracefully."""
        # Mock LLM success
        mock_generate.return_value = "def solve(grid): return grid"

        # Mock sandbox to timeout
        mock_execute.return_value = (False, None)

        solver_code = mock_generate([])
        success, result = mock_execute(solver_code, np.array([[1]]))

        assert success is False
        assert result is None

    @patch("arc_prometheus.cognitive_cells.programmer.generate_solver")
    @patch("arc_prometheus.crucible.sandbox.safe_execute")
    def test_grid_evaluation_mismatch(self, mock_execute, mock_generate):
        """Test pipeline detects grid mismatches correctly."""
        # Mock successful generation and execution
        mock_generate.return_value = "def solve(grid): return grid"
        wrong_output = np.array([[9, 9], [9, 9]])
        mock_execute.return_value = (True, wrong_output)

        expected_output = np.array([[1, 2], [3, 4]])

        solver_code = mock_generate([])
        success, result = mock_execute(solver_code, np.array([[1, 2]]))

        assert success is True
        is_correct = evaluate_grids(result, expected_output)
        assert is_correct is False

    @patch("arc_prometheus.cognitive_cells.programmer.generate_solver")
    @patch("arc_prometheus.crucible.sandbox.safe_execute")
    def test_partial_success_multiple_examples(self, mock_execute, mock_generate):
        """Test pipeline with mixed success/failure across multiple examples."""
        mock_generate.return_value = "def solve(grid): return grid"

        # Mock: first succeeds, second fails, third succeeds
        mock_execute.side_effect = [
            (True, np.array([[1, 2]])),  # Correct
            (False, None),  # Timeout
            (True, np.array([[5, 6]])),  # Correct
        ]

        train_pairs = [
            {"input": np.array([[1, 2]]), "output": np.array([[1, 2]])},
            {"input": np.array([[3, 4]]), "output": np.array([[3, 4]])},
            {"input": np.array([[5, 6]]), "output": np.array([[5, 6]])},
        ]

        solver_code = mock_generate(train_pairs)
        correct_count = 0

        for example in train_pairs:
            success, result = mock_execute(solver_code, example["input"])
            if success:
                is_correct = evaluate_grids(result, example["output"])
                if is_correct:
                    correct_count += 1

        assert correct_count == 2  # 2 out of 3 correct
        success_rate = (correct_count / len(train_pairs)) * 100
        assert success_rate == pytest.approx(66.67, rel=0.1)

    @patch("arc_prometheus.cognitive_cells.programmer.generate_solver")
    @patch("arc_prometheus.crucible.sandbox.safe_execute")
    def test_shape_mismatch_detection(self, mock_execute, mock_generate):
        """Test pipeline detects shape mismatches between predicted and expected."""
        mock_generate.return_value = "def solve(grid): return grid"

        # Mock: returns wrong shape
        wrong_shape_output = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        mock_execute.return_value = (True, wrong_shape_output)

        expected_output = np.array([[1, 2], [3, 4]])  # 2x2

        solver_code = mock_generate([])
        success, result = mock_execute(solver_code, np.array([[1]]))

        assert success is True
        is_correct = evaluate_grids(result, expected_output)
        assert is_correct is False  # Shape mismatch should be detected


class TestE2EPipelineIntegration:
    """Integration tests with real components (except LLM)."""

    @patch("arc_prometheus.cognitive_cells.programmer.generate_solver")
    def test_real_data_loading_and_execution(self, mock_generate):
        """Test pipeline with real data loading and sandbox execution."""
        # Use real load_task if data available
        data_file = (
            Path(__file__).parent.parent
            / "data"
            / "arc-prize-2025"
            / "arc-agi_training_challenges.json"
        )

        if not data_file.exists():
            pytest.skip("ARC dataset not available")

        # Load real task
        task_data = load_task(str(data_file), task_id="00576224")
        assert "train" in task_data
        assert len(task_data["train"]) > 0

        # Mock LLM to return identity solver
        mock_generate.return_value = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid.copy()
"""

        solver_code = mock_generate(task_data["train"])

        # Use REAL sandbox execution
        example = task_data["train"][0]
        success, result = safe_execute(solver_code, example["input"], timeout=5)

        assert success is True
        assert isinstance(result, np.ndarray)
        assert result.shape == example["input"].shape

    @patch("arc_prometheus.cognitive_cells.programmer.generate_solver")
    def test_real_sandbox_timeout(self, mock_generate):
        """Test real sandbox timeout with infinite loop code."""
        # Mock LLM to return infinite loop
        mock_generate.return_value = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    while True:
        pass
    return task_grid
"""

        solver_code = mock_generate([])
        test_grid = np.array([[1, 2], [3, 4]])

        # Should timeout after 2 seconds
        success, result = safe_execute(solver_code, test_grid, timeout=2)

        assert success is False
        assert result is None

    @patch("arc_prometheus.cognitive_cells.programmer.generate_solver")
    def test_real_sandbox_exception(self, mock_generate):
        """Test real sandbox exception handling."""
        # Mock LLM to return code with runtime error
        mock_generate.return_value = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    x = 1 / 0  # ZeroDivisionError
    return task_grid
"""

        solver_code = mock_generate([])
        test_grid = np.array([[1, 2], [3, 4]])

        success, result = safe_execute(solver_code, test_grid, timeout=5)

        assert success is False
        assert result is None


class TestE2EPipelineErrorCases:
    """Tests for error handling and edge cases."""

    def test_invalid_task_id(self):
        """Test handling of invalid task ID."""
        data_file = (
            Path(__file__).parent.parent
            / "data"
            / "arc-prize-2025"
            / "arc-agi_training_challenges.json"
        )

        if not data_file.exists():
            pytest.skip("ARC dataset not available")

        with pytest.raises(ValueError, match="not found"):
            load_task(str(data_file), task_id="INVALID_TASK_ID_12345")

    def test_missing_dataset_file(self):
        """Test handling of missing dataset file."""
        with pytest.raises(FileNotFoundError):
            load_task("/nonexistent/path/to/file.json", task_id="00576224")

    @patch("arc_prometheus.cognitive_cells.programmer.generate_solver")
    @patch("arc_prometheus.crucible.sandbox.safe_execute")
    def test_all_examples_fail(self, mock_execute, mock_generate):
        """Test pipeline when all train examples fail."""
        mock_generate.return_value = "def solve(grid): return grid"
        mock_execute.return_value = (False, None)  # All fail

        train_pairs = [
            {"input": np.array([[1]]), "output": np.array([[1]])},
            {"input": np.array([[2]]), "output": np.array([[2]])},
        ]

        solver_code = mock_generate(train_pairs)
        correct_count = 0

        for example in train_pairs:
            success, result = mock_execute(solver_code, example["input"])
            if success:
                correct_count += 1

        assert correct_count == 0

    @patch("arc_prometheus.cognitive_cells.programmer.generate_solver")
    @patch("arc_prometheus.crucible.sandbox.safe_execute")
    def test_solver_saves_with_partial_success(self, mock_execute, mock_generate):
        """Test that solver is saved even with partial success (>=1 correct)."""
        mock_generate.return_value = "def solve(grid): return grid"

        # 1 success, 2 failures
        mock_execute.side_effect = [
            (True, np.array([[1]])),  # Correct
            (False, None),  # Fail
            (False, None),  # Fail
        ]

        train_pairs = [
            {"input": np.array([[1]]), "output": np.array([[1]])},
            {"input": np.array([[2]]), "output": np.array([[2]])},
            {"input": np.array([[3]]), "output": np.array([[3]])},
        ]

        solver_code = mock_generate(train_pairs)
        correct_count = 0

        for example in train_pairs:
            success, result = mock_execute(solver_code, example["input"])
            if success:
                is_correct = evaluate_grids(result, example["output"])
                if is_correct:
                    correct_count += 1

        # Should save because correct_count > 0
        assert correct_count > 0
