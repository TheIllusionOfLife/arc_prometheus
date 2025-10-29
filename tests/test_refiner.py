"""Tests for Refiner agent - LLM-based code debugging (Phase 2.2)."""

import json
from unittest.mock import Mock, patch

import pytest

from arc_prometheus.cognitive_cells.prompts import create_refiner_prompt
from arc_prometheus.cognitive_cells.refiner import refine_solver
from arc_prometheus.evolutionary_engine.fitness import calculate_fitness


class TestRefinerPromptTemplate:
    """Test refiner prompt template generation."""

    def test_create_refiner_prompt_structure(self, tmp_path):
        """Test that refiner prompt contains all required sections."""
        # Create simple task
        task_data = {
            "train": [
                {"input": [[1, 2]], "output": [[2, 3]]},
                {"input": [[3, 4]], "output": [[4, 5]]},
            ],
            "test": [{"input": [[5, 6]], "output": [[6, 7]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Failed code with syntax error
        failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray
    return task_grid + 1  # Missing colon
"""

        # Create fitness result with errors
        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 2,
            "test_correct": 0,
            "test_total": 1,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "execution_errors": ["Train example 0: Execution failed"],
        }

        prompt = create_refiner_prompt(failed_code, task_data, fitness_result)

        # Verify structure
        assert "debug" in prompt.lower() or "fix" in prompt.lower()
        assert "import numpy as np" in prompt
        assert "def solve(" in prompt
        assert "error" in prompt.lower()
        assert "1 2" in prompt or "[[1, 2]]" in prompt  # Task examples present

    def test_create_refiner_prompt_with_syntax_error(self, tmp_path):
        """Test prompt includes syntax error messages."""
        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }

        failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + + +  # Syntax error
"""

        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 1,
            "test_correct": 0,
            "test_total": 1,
            "execution_errors": ["Train example 0: Execution failed"],
        }

        prompt = create_refiner_prompt(failed_code, task_data, fitness_result)

        # Verify error context is included
        assert "0/1" in prompt or "0 out of 1" in prompt or "failed" in prompt.lower()
        assert "Execution failed" in prompt or "error" in prompt.lower()

    def test_create_refiner_prompt_with_logic_error(self, tmp_path):
        """Test prompt includes failed examples with expected vs actual."""
        task_data = {
            "train": [
                {"input": [[1]], "output": [[2]]},
                {"input": [[2]], "output": [[4]]},
            ],
            "test": [{"input": [[3]], "output": [[6]]}],
        }

        # Wrong code: adds 1 instead of multiplying by 2
        failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
"""

        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 2,
            "test_correct": 0,
            "test_total": 1,
            "execution_errors": [],
        }

        prompt = create_refiner_prompt(failed_code, task_data, fitness_result)

        # Verify task examples are present
        assert "1" in prompt  # Input value
        assert "2" in prompt or "4" in prompt  # Expected output values
        assert "train" in prompt.lower() or "example" in prompt.lower()


class TestRefinerCodeExtraction:
    """Test extracting refined code from LLM responses."""

    def test_extract_refined_code_reuses_programmer_parser(self):
        """Test that refiner uses same code parser as programmer."""
        # This is an integration test - we're verifying the parser works
        from arc_prometheus.cognitive_cells.programmer import (
            extract_code_from_response,
        )

        response = """Here's the fixed code:

```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid * 2  # Fixed: multiply instead of add
```
"""
        code = extract_code_from_response(response)

        assert "import numpy as np" in code
        assert "def solve(" in code
        assert "return task_grid * 2" in code
        assert "Here's the fixed code" not in code

    def test_extract_refined_code_handles_explanation(self):
        """Test parsing when LLM includes debugging explanation."""
        from arc_prometheus.cognitive_cells.programmer import (
            extract_code_from_response,
        )

        response = """The bug was in line 4 - you were adding instead of multiplying.

import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid * 2

This should now work correctly."""

        code = extract_code_from_response(response)

        assert "def solve(" in code
        assert "return task_grid * 2" in code
        assert "The bug was" not in code


class TestRefinerIntegration:
    """Test refiner integration with mocked Gemini API."""

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.refiner.get_gemini_api_key")
    def test_refine_solver_fixes_syntax_error(
        self, mock_get_api_key, mock_model_class, tmp_path
    ):
        """Test refiner fixes syntax error in code."""
        # Mock API key
        mock_get_api_key.return_value = "test-api-key"

        # Mock Gemini response with fixed code
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        # Create task
        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Failed code (syntax error)
        failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray
    return task_grid + 1  # Missing colon
"""

        # Fitness result
        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 1,
            "test_correct": 0,
            "test_total": 1,
            "execution_errors": ["Train example 0: Execution failed"],
        }

        # Call refiner
        refined_code = refine_solver(failed_code, str(task_file), fitness_result)

        assert "def solve(" in refined_code
        assert "import numpy as np" in refined_code
        assert "return task_grid + 1" in refined_code
        # Should NOT have syntax error - check for proper function signature
        assert ") -> np.ndarray:" in refined_code  # Has colon in signature

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.refiner.get_gemini_api_key")
    def test_refine_solver_fixes_logic_error(
        self, mock_get_api_key, mock_model_class, tmp_path
    ):
        """Test refiner fixes logic error (wrong algorithm)."""
        mock_get_api_key.return_value = "test-api-key"

        # Mock response with corrected logic
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid * 2  # Fixed: multiply instead of add
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[6]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Wrong logic
        failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1  # Wrong operation
"""

        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 1,
            "test_correct": 0,
            "test_total": 1,
            "execution_errors": [],
        }

        refined_code = refine_solver(failed_code, str(task_file), fitness_result)

        assert "def solve(" in refined_code
        assert "task_grid * 2" in refined_code

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.refiner.get_gemini_api_key")
    def test_refine_solver_fixes_timeout(
        self, mock_get_api_key, mock_model_class, tmp_path
    ):
        """Test refiner optimizes code with infinite loop."""
        mock_get_api_key.return_value = "test-api-key"

        # Mock response without infinite loop
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid.copy()
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        task_data = {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]], "output": [[2]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Code with timeout issue
        failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    while True:
        pass
    return task_grid
"""

        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 1,
            "test_correct": 0,
            "test_total": 1,
            "execution_errors": ["Train example 0: Execution failed"],
        }

        refined_code = refine_solver(failed_code, str(task_file), fitness_result)

        assert "def solve(" in refined_code
        # Should not have infinite loop
        assert "while True:" not in refined_code

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.refiner.get_gemini_api_key")
    def test_refine_solver_with_api_timeout(
        self, mock_get_api_key, mock_model_class, tmp_path
    ):
        """Test handling of Gemini API timeout."""
        mock_get_api_key.return_value = "test-api-key"

        # Mock timeout
        mock_model = Mock()
        mock_model.generate_content.side_effect = TimeoutError("API timeout")
        mock_model_class.return_value = mock_model

        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
"""

        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 1,
            "execution_errors": ["Train example 0: Execution failed"],
        }

        with pytest.raises(Exception, match="Gemini API call failed"):
            refine_solver(failed_code, str(task_file), fitness_result)


class TestRefinerFitnessImprovement:
    """Test end-to-end fitness improvement through refinement."""

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.refiner.get_gemini_api_key")
    def test_refined_code_improves_fitness(
        self, mock_get_api_key, mock_model_class, tmp_path
    ):
        """Test that refined code has better fitness than original."""
        mock_get_api_key.return_value = "test-api-key"

        # Create task: multiply by 2
        task_data = {
            "train": [
                {"input": [[2]], "output": [[4]]},
                {"input": [[3]], "output": [[6]]},
                {"input": [[4]], "output": [[8]]},
            ],
            "test": [{"input": [[5]], "output": [[10]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Original failed code (wrong logic - adds instead of multiplies)
        failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1  # Wrong - should multiply by 2
"""

        # Calculate fitness before (should be 0 - no examples solved)
        fitness_before = calculate_fitness(str(task_file), failed_code)
        assert fitness_before["fitness"] == 0  # 0 train + 0 test

        # Mock refined code
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid * 2  # Fixed
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        # Refine code
        refined_code = refine_solver(failed_code, str(task_file), fitness_before)

        # Calculate fitness after (should be better)
        fitness_after = calculate_fitness(str(task_file), refined_code)

        # Verify improvement
        assert fitness_after["fitness"] > fitness_before["fitness"]
        assert fitness_after["fitness"] == 13  # 3 train + 1 test * 10

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.refiner.get_gemini_api_key")
    def test_refiner_with_empty_failure_context(
        self, mock_get_api_key, mock_model_class, tmp_path
    ):
        """Test refiner handles edge case with no specific errors."""
        mock_get_api_key.return_value = "test-api-key"

        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        task_data = {"train": [{"input": [[1]], "output": [[1]]}], "test": []}

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
"""

        # Empty fitness result
        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 1,
            "test_correct": 0,
            "test_total": 0,
            "execution_errors": [],  # No specific errors
        }

        refined_code = refine_solver(failed_code, str(task_file), fitness_result)

        assert "def solve(" in refined_code
        assert "import numpy as np" in refined_code


class TestRefinerRealAPI:
    """Integration tests with real Gemini API (optional, requires API key)."""

    @pytest.mark.integration
    def test_refiner_with_real_gemini_api(self, tmp_path):
        """Test refiner with actual Gemini API call.

        This test is marked as integration and will be skipped if:
        - GEMINI_API_KEY is not set in environment
        - Running in CI without --run-integration flag
        """
        from arc_prometheus.utils.config import get_gemini_api_key

        # Skip if no API key
        try:
            api_key = get_gemini_api_key()
            if not api_key:
                pytest.skip("GEMINI_API_KEY not configured")
        except ValueError:
            pytest.skip("GEMINI_API_KEY not configured")

        # Create simple task
        task_data = {
            "train": [
                {"input": [[1, 2]], "output": [[2, 4]]},
                {"input": [[3, 4]], "output": [[6, 8]]},
            ],
            "test": [{"input": [[5, 6]], "output": [[10, 12]]}],
        }

        task_file = tmp_path / "real_task.json"
        task_file.write_text(json.dumps(task_data))

        # Failed code (adds instead of multiplies)
        failed_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
"""

        # Get fitness
        fitness_result = calculate_fitness(str(task_file), failed_code)
        assert fitness_result["fitness"] == 0

        # Call real refiner
        refined_code = refine_solver(failed_code, str(task_file), fitness_result)

        # Verify basic structure
        assert "def solve(" in refined_code
        assert "import numpy as np" in refined_code

        # Refined code should be different from original
        assert refined_code.strip() != failed_code.strip()

        # Test refined code (might not be perfect, but should be valid Python)
        fitness_refined = calculate_fitness(str(task_file), refined_code)

        # At minimum, refined code should execute without errors
        # (it might not solve the task perfectly, but should improve)
        assert len(fitness_refined["execution_errors"]) <= len(
            fitness_result["execution_errors"]
        )
