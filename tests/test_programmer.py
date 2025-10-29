"""Tests for LLM code generation and parsing functionality.

This test suite validates:
1. Code extraction from various LLM response formats
2. Handling of markdown formatting
3. Multiple code blocks
4. Error cases (no solve function, invalid format)
5. Gemini API integration with mocked responses
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from arc_prometheus.cognitive_cells.programmer import (
    extract_code_from_response,
    generate_solver,
)


class TestCodeExtraction:
    """Tests for parsing LLM responses to extract Python code."""

    def test_extract_from_triple_backticks(self):
        """Test extracting code from ```python ... ``` blocks."""
        response = """Here is the solution:
```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
```
"""
        code = extract_code_from_response(response)
        assert "import numpy as np" in code
        assert "def solve(" in code
        assert "return task_grid + 1" in code
        assert "```" not in code  # Should strip delimiters

    def test_extract_from_backticks_without_python_keyword(self):
        """Test extracting code from ``` blocks without 'python' keyword."""
        response = """Here is the solution:
```
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
```
"""
        code = extract_code_from_response(response)
        assert "import numpy as np" in code
        assert "def solve(" in code
        assert "```" not in code

    def test_extract_raw_code_without_delimiters(self):
        """Test extracting raw Python code without markdown."""
        response = """import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid * 2
"""
        code = extract_code_from_response(response)
        assert "import numpy as np" in code
        assert "def solve(" in code
        assert "return task_grid * 2" in code

    def test_extract_from_multiple_code_blocks(self):
        """Test extracting when response has multiple code blocks."""
        response = """First, let me show you the wrong approach:
```python
# Wrong code
pass
```

Now here's the correct solution:
```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
```
"""
        code = extract_code_from_response(response)
        # Should extract the block containing solve()
        assert "def solve(" in code
        assert "Wrong code" not in code

    def test_extract_with_markdown_formatting(self):
        """Test handling markdown formatting in response."""
        response = """**Solution:**

```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid.T  # Transpose
```

This transposes the grid."""
        code = extract_code_from_response(response)
        assert "import numpy as np" in code
        assert "def solve(" in code
        assert "**Solution:**" not in code

    def test_extract_with_explanation_mixed_in(self):
        """Test extracting code when explanation is mixed with code."""
        response = """The transformation rule is to add 1 to each cell:

import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Add 1 to all values
    return task_grid + 1

This works because the pattern shows incrementing."""
        code = extract_code_from_response(response)
        assert "import numpy as np" in code
        assert "def solve(" in code
        # Should not include trailing explanation
        assert "This works because" not in code

    def test_extract_fails_when_no_solve_function(self):
        """Test that extraction raises error if no solve() found."""
        response = """import numpy as np

def wrong_name(grid):
    return grid
"""
        with pytest.raises(ValueError, match="solve.*function.*not found"):
            extract_code_from_response(response)

    def test_extract_handles_indentation_correctly(self):
        """Test that indentation is preserved correctly."""
        response = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    if task_grid.size == 0:
        return task_grid
    return task_grid + 1
```"""
        code = extract_code_from_response(response)
        # Should preserve indentation
        assert "    if task_grid.size == 0:" in code
        assert "    return task_grid" in code

    def test_extract_with_complex_multiline_code(self):
        """Test extraction of complex code with multiple functions/logic."""
        response = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Complex logic
    result = np.zeros_like(task_grid)
    for i in range(task_grid.shape[0]):
        for j in range(task_grid.shape[1]):
            if task_grid[i, j] > 0:
                result[i, j] = task_grid[i, j] * 2
    return result
```"""
        code = extract_code_from_response(response)
        assert "import numpy as np" in code
        assert "def solve(" in code
        assert "for i in range" in code
        assert "for j in range" in code

    def test_extract_with_helper_functions(self):
        """Test extraction when code includes helper functions."""
        response = """```python
import numpy as np

def helper(arr):
    return arr * 2

def solve(task_grid: np.ndarray) -> np.ndarray:
    return helper(task_grid)
```"""
        code = extract_code_from_response(response)
        assert "import numpy as np" in code
        assert "def helper(" in code
        assert "def solve(" in code

    def test_extract_with_comments_and_docstrings(self):
        """Test extraction preserves comments and docstrings."""
        response = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    \"\"\"Transform the grid by adding 1.\"\"\"
    # This is a comment
    return task_grid + 1
```"""
        code = extract_code_from_response(response)
        assert '"""Transform the grid by adding 1."""' in code
        assert "# This is a comment" in code

    def test_extract_empty_response(self):
        """Test that empty response raises appropriate error."""
        response = ""
        with pytest.raises(ValueError, match="solve.*function.*not found"):
            extract_code_from_response(response)

    def test_extract_only_explanation_no_code(self):
        """Test response with only text explanation, no code."""
        response = """This puzzle transforms the grid by rotating it.
The pattern shows a 90-degree rotation applied to the input."""
        with pytest.raises(ValueError, match="solve.*function.*not found"):
            extract_code_from_response(response)

    def test_extract_malformed_markdown_unclosed_block(self):
        """Test handling of unclosed code blocks (malformed markdown).

        Strategy 1 (regex for ``` blocks) will fail, but Strategy 2
        (raw code extraction) should still succeed.
        """
        response = """Here is the solution:
```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1

Note: Missing closing backticks"""
        # Strategy 2 should handle this gracefully
        code = extract_code_from_response(response)
        assert "import numpy as np" in code
        assert "def solve(" in code
        assert "return task_grid + 1" in code
        # Should not include the explanation text
        assert "Note: Missing closing backticks" not in code


class TestGenerateSolver:
    """Tests for Gemini API integration."""

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    def test_generate_solver_success(self, mock_get_api_key, mock_model_class):
        """Test successful code generation with mocked API."""
        # Mock API key
        mock_get_api_key.return_value = "test-api-key"

        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[2]])}]
        code = generate_solver(train_pairs)

        assert "def solve(" in code
        assert "import numpy as np" in code
        assert "return task_grid + 1" in code

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    def test_generate_solver_api_timeout(self, mock_get_api_key, mock_model_class):
        """Test handling of API timeout."""
        # Mock API key
        mock_get_api_key.return_value = "test-api-key"

        # Mock timeout error
        mock_model = Mock()
        mock_model.generate_content.side_effect = TimeoutError("API timeout")
        mock_model_class.return_value = mock_model

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[2]])}]

        with pytest.raises(Exception, match="Gemini API call failed"):
            generate_solver(train_pairs, use_cache=False)

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    def test_generate_solver_unparseable_response(
        self, mock_get_api_key, mock_model_class
    ):
        """Test handling of unparseable LLM response."""
        # Mock API key
        mock_get_api_key.return_value = "test-api-key"

        # Mock response with no code
        mock_response = Mock()
        mock_response.text = "This is just text with no code"

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[2]])}]

        with pytest.raises(ValueError, match="Failed to parse LLM response"):
            generate_solver(train_pairs, use_cache=False)

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    def test_generate_solver_with_markdown_code_block(
        self, mock_get_api_key, mock_model_class
    ):
        """Test parsing response with markdown code blocks."""
        # Mock API key
        mock_get_api_key.return_value = "test-api-key"

        # Mock response with markdown
        mock_response = Mock()
        mock_response.text = """Here's the solution:

```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid.T  # Transpose
```

This transposes the grid."""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        train_pairs = [{"input": np.array([[1, 2]]), "output": np.array([[1], [2]])}]
        code = generate_solver(train_pairs)

        assert "def solve(" in code
        assert "return task_grid.T" in code
        assert "Here's the solution" not in code

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    def test_generate_solver_with_raw_code_response(
        self, mock_get_api_key, mock_model_class
    ):
        """Test parsing raw code without markdown delimiters."""
        # Mock API key
        mock_get_api_key.return_value = "test-api-key"

        # Mock response with raw code (no backticks)
        mock_response = Mock()
        mock_response.text = """import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid * 2"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[2]])}]
        code = generate_solver(train_pairs, use_cache=False)

        assert "def solve(" in code
        assert "return task_grid * 2" in code

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    def test_generate_solver_timeout_parameter(
        self, mock_get_api_key, mock_model_class
    ):
        """Test that timeout parameter is passed to API call."""
        # Mock API key
        mock_get_api_key.return_value = "test-api-key"

        # Mock response
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[1]])}]
        generate_solver(train_pairs, timeout=30, use_cache=False)

        # Verify timeout was passed
        call_kwargs = mock_model.generate_content.call_args[1]
        assert call_kwargs["request_options"]["timeout"] == 30

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    def test_generate_solver_custom_model(self, mock_get_api_key, mock_model_class):
        """Test that custom model_name parameter is used."""
        # Mock API key
        mock_get_api_key.return_value = "test-api-key"

        # Mock response
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[1]])}]
        generate_solver(train_pairs, model_name="gemini-2.0-flash-thinking-exp")

        # Verify custom model was used
        mock_model_class.assert_called_once_with("gemini-2.0-flash-thinking-exp")

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    def test_generate_solver_default_model(self, mock_get_api_key, mock_model_class):
        """Test that default model is used when model_name not specified."""
        # Mock API key
        mock_get_api_key.return_value = "test-api-key"

        # Mock response
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[1]])}]
        generate_solver(train_pairs, use_cache=False)

        # Verify default model from config was used
        from arc_prometheus.utils.config import MODEL_NAME

        mock_model_class.assert_called_once_with(MODEL_NAME)

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    def test_generate_solver_custom_temperature(
        self, mock_get_api_key, mock_model_class
    ):
        """Test that custom temperature parameter is used."""
        # Mock API key
        mock_get_api_key.return_value = "test-api-key"

        # Mock response
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[1]])}]
        generate_solver(train_pairs, temperature=0.7, use_cache=False)

        # Verify temperature was passed in generation_config
        call_kwargs = mock_model.generate_content.call_args[1]
        assert call_kwargs["generation_config"]["temperature"] == 0.7

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    def test_generate_solver_default_temperature(
        self, mock_get_api_key, mock_model_class
    ):
        """Test that default temperature is used when not specified."""
        # Mock API key
        mock_get_api_key.return_value = "test-api-key"

        # Mock response
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[1]])}]
        generate_solver(train_pairs, use_cache=False)

        # Verify default temperature from config was used
        from arc_prometheus.utils.config import PROGRAMMER_GENERATION_CONFIG

        call_kwargs = mock_model.generate_content.call_args[1]
        assert (
            call_kwargs["generation_config"]["temperature"]
            == PROGRAMMER_GENERATION_CONFIG["temperature"]
        )

    @patch("google.generativeai.GenerativeModel")
    @patch("arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    def test_generate_solver_all_custom_params(
        self, mock_get_api_key, mock_model_class
    ):
        """Test using all custom parameters together."""
        # Mock API key
        mock_get_api_key.return_value = "test-api-key"

        # Mock response
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[1]])}]
        generate_solver(
            train_pairs,
            model_name="gemini-2.0-flash-thinking-exp",
            temperature=0.9,
            timeout=120,
            use_cache=False,
        )

        # Verify all custom params were used
        mock_model_class.assert_called_once_with("gemini-2.0-flash-thinking-exp")
        call_kwargs = mock_model.generate_content.call_args[1]
        assert call_kwargs["generation_config"]["temperature"] == 0.9
        assert call_kwargs["request_options"]["timeout"] == 120
