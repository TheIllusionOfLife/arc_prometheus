"""Tests for LLM code generation and parsing functionality.

This test suite validates:
1. Code extraction from various LLM response formats
2. Handling of markdown formatting
3. Multiple code blocks
4. Error cases (no solve function, invalid format)
"""

import pytest

from arc_prometheus.cognitive_cells.programmer import extract_code_from_response


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
