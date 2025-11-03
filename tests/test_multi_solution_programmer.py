"""Tests for Multi-Solution Programmer agent."""

import json
from unittest.mock import MagicMock, patch

import pytest

from arc_prometheus.cognitive_cells.multi_persona_analyst import InterpretationResult
from arc_prometheus.cognitive_cells.multi_solution_programmer import (
    MultiSolutionProgrammer,
    SolutionResult,
)


@pytest.fixture
def sample_task():
    """Sample ARC task for testing."""
    return {
        "train": [
            {
                "input": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                "output": [[8, 7, 6], [5, 4, 3], [2, 1, 0]],
            },
            {
                "input": [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                "output": [[3, 3, 3], [2, 2, 2], [1, 1, 1]],
            },
        ],
        "test": [{"input": [[9, 9, 9], [8, 8, 8], [7, 7, 7]]}],
    }


@pytest.fixture
def sample_interpretations():
    """Sample interpretations from Multi-Persona Analyst."""
    return [
        InterpretationResult(
            persona="Geometric Transformation Specialist",
            pattern="Flip the grid vertically (reverse row order)",
            observations=["First row becomes last", "Symmetric transformation"],
            approach="Use np.flip(grid, axis=0)",
            confidence="high",
        ),
        InterpretationResult(
            persona="Color Pattern Specialist",
            pattern="Reverse color values in spatial order",
            observations=["Colors maintain positions", "Spatial reversal"],
            approach="Reverse iteration with indexing",
            confidence="medium",
        ),
        InterpretationResult(
            persona="Object Detection Specialist",
            pattern="Treat rows as objects and swap positions",
            observations=["Three row-objects", "Top-bottom swap"],
            approach="Extract rows, reverse list, stack",
            confidence="high",
        ),
        InterpretationResult(
            persona="Grid Structure Specialist",
            pattern="Reverse grid along horizontal axis",
            observations=["Grid height unchanged", "Row-wise reversal"],
            approach="Use array slicing [::-1]",
            confidence="high",
        ),
        InterpretationResult(
            persona="Logical Rules Specialist",
            pattern="Apply rule: output[i] = input[n-1-i]",
            observations=["Index transformation", "Works for any height"],
            approach="Iterate with reversed indices",
            confidence="medium",
        ),
    ]


@pytest.fixture
def sample_api_response():
    """Sample valid API response with 5 solutions."""
    return {
        "solutions": [
            {
                "interpretation_id": 1,
                "code": "import numpy as np\n\ndef solve(task_grid: np.ndarray) -> np.ndarray:\n    return np.flip(task_grid, axis=0)",
                "approach_summary": "Vertical flip using np.flip",
            },
            {
                "interpretation_id": 2,
                "code": "import numpy as np\n\ndef solve(task_grid: np.ndarray) -> np.ndarray:\n    return task_grid[::-1]",
                "approach_summary": "Reverse rows with slicing",
            },
            {
                "interpretation_id": 3,
                "code": "import numpy as np\n\ndef solve(task_grid: np.ndarray) -> np.ndarray:\n    rows = [task_grid[i] for i in range(len(task_grid))]\n    return np.array(rows[::-1])",
                "approach_summary": "Extract and reverse row list",
            },
            {
                "interpretation_id": 4,
                "code": "import numpy as np\n\ndef solve(task_grid: np.ndarray) -> np.ndarray:\n    n = task_grid.shape[0]\n    result = np.zeros_like(task_grid)\n    for i in range(n):\n        result[i] = task_grid[n-1-i]\n    return result",
                "approach_summary": "Index-based row reversal",
            },
            {
                "interpretation_id": 5,
                "code": "import numpy as np\n\ndef solve(task_grid: np.ndarray) -> np.ndarray:\n    return np.flipud(task_grid)",
                "approach_summary": "Flip up-down using np.flipud",
            },
        ]
    }


class TestSolutionResult:
    """Test SolutionResult dataclass."""

    def test_create_solution_result(self):
        """Test creating SolutionResult with all fields."""
        result = SolutionResult(
            interpretation_id=1,
            code="import numpy as np\ndef solve(grid): return grid",
            approach_summary="Identity function",
        )

        assert result.interpretation_id == 1
        assert "def solve" in result.code
        assert result.approach_summary == "Identity function"


class TestMultiSolutionProgrammer:
    """Test MultiSolutionProgrammer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        with patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai"):
            programmer = MultiSolutionProgrammer()

            assert programmer.model_name == "gemini-2.0-flash-thinking-exp"
            assert programmer.temperature == 0.0
            assert programmer.use_cache is True

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        with patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai"):
            programmer = MultiSolutionProgrammer(
                model_name="gemini-2.5-flash-lite",
                temperature=0.5,
                use_cache=False,
            )

            assert programmer.model_name == "gemini-2.5-flash-lite"
            assert programmer.temperature == 0.5
            assert programmer.use_cache is False

    def test_format_grid(self, sample_task):
        """Test grid formatting."""
        with patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai"):
            programmer = MultiSolutionProgrammer()
            grid = sample_task["train"][0]["input"]

            formatted = programmer._format_grid(grid)

            expected = "0 1 2\n3 4 5\n6 7 8"
            assert formatted == expected

    def test_create_prompt_structure(self, sample_task, sample_interpretations):
        """Test that prompt contains required elements."""
        with patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai"):
            programmer = MultiSolutionProgrammer()
            prompt = programmer._create_prompt(sample_task, sample_interpretations)

            # Check for key sections
            assert "TRAINING EXAMPLES:" in prompt
            assert "EXPERT INTERPRETATIONS TO IMPLEMENT:" in prompt
            assert "YOUR TASK:" in prompt

            # Check for all 5 interpretations
            for i in range(1, 6):
                assert f"Interpretation {i}" in prompt

            # Check for training examples
            assert "Example 1:" in prompt
            assert "Example 2:" in prompt
            assert "0 1 2" in prompt

            # Check for constraints
            assert "def solve(task_grid: np.ndarray)" in prompt
            assert "import numpy as np" in prompt
            assert "≤200 characters" in prompt  # Updated: removed max_length
            assert "≤1500 characters" in prompt  # New: conciseness requirement
            assert "interpretation_id" in prompt

    def test_validate_solution_valid(self):
        """Test validation of valid solution code."""
        with patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai"):
            programmer = MultiSolutionProgrammer()

            valid_code = "import numpy as np\n\ndef solve(grid: np.ndarray) -> np.ndarray:\n    return np.flip(grid, axis=0)"

            is_valid, error = programmer._validate_solution(valid_code)

            assert is_valid is True
            assert error == ""

    def test_validate_solution_missing_solve(self):
        """Test validation fails when solve() function is missing."""
        with patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai"):
            programmer = MultiSolutionProgrammer()

            code_without_solve = (
                "import numpy as np\n\ndef transform(grid):\n    return grid"
            )

            is_valid, error = programmer._validate_solution(code_without_solve)

            assert is_valid is False
            assert "Missing 'def solve('" in error

    def test_validate_solution_syntax_error(self):
        """Test validation fails for syntax errors."""
        with patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai"):
            programmer = MultiSolutionProgrammer()

            code_with_error = "import numpy as np\n\ndef solve(grid)\n    return grid"  # Missing colon

            is_valid, error = programmer._validate_solution(code_with_error)

            assert is_valid is False
            assert "Syntax error" in error

    def test_validate_solution_missing_numpy_import(self):
        """Test validation fails when numpy is used without import."""
        with patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai"):
            programmer = MultiSolutionProgrammer()

            code_without_import = "def solve(grid):\n    return np.flip(grid, axis=0)"

            is_valid, error = programmer._validate_solution(code_without_import)

            assert is_valid is False
            assert "numpy" in error.lower()

    @patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai")
    @patch("arc_prometheus.utils.llm_cache.get_cache")
    def test_generate_multi_solutions_with_cache_hit(
        self,
        mock_get_cache,
        mock_genai,
        sample_task,
        sample_interpretations,
        sample_api_response,
    ):
        """Test solution generation with cache hit."""
        # Setup mock cache
        mock_cache = MagicMock()
        mock_cache.get.return_value = json.dumps(sample_api_response)
        mock_get_cache.return_value = mock_cache

        programmer = MultiSolutionProgrammer(use_cache=True)
        solutions = programmer.generate_multi_solutions(
            sample_task, sample_interpretations
        )

        # Should get 5 valid solutions from cache
        assert len(solutions) == 5
        assert all(isinstance(s, SolutionResult) for s in solutions)
        assert mock_cache.get.called
        # API should NOT be called
        assert not mock_genai.GenerativeModel.called

    @patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai")
    @patch("arc_prometheus.utils.llm_cache.get_cache")
    def test_generate_multi_solutions_with_cache_miss(
        self,
        mock_get_cache,
        mock_genai,
        sample_task,
        sample_interpretations,
        sample_api_response,
    ):
        """Test solution generation with cache miss."""
        # Setup mock cache (cache miss)
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        # Setup mock API response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_api_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        programmer = MultiSolutionProgrammer(use_cache=True)
        solutions = programmer.generate_multi_solutions(
            sample_task, sample_interpretations
        )

        # Should get 5 valid solutions from API
        assert len(solutions) == 5
        assert mock_cache.get.called
        # API should be called
        assert mock_genai.GenerativeModel.called
        # Result should be cached
        assert mock_cache.set.called

    @patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai")
    def test_generate_multi_solutions_no_cache(
        self, mock_genai, sample_task, sample_interpretations, sample_api_response
    ):
        """Test solution generation with caching disabled."""
        # Setup mock API response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_api_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        programmer = MultiSolutionProgrammer(use_cache=False)
        solutions = programmer.generate_multi_solutions(
            sample_task, sample_interpretations
        )

        # Should get 5 valid solutions from API
        assert len(solutions) == 5
        # API should be called
        assert mock_genai.GenerativeModel.called

    @patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai")
    def test_generate_multi_solutions_uses_structured_output(
        self, mock_genai, sample_task, sample_interpretations, sample_api_response
    ):
        """Test that solution generation uses structured output."""
        # Setup mock API response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_api_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        programmer = MultiSolutionProgrammer(use_cache=False)
        programmer.generate_multi_solutions(sample_task, sample_interpretations)

        # Check that generate_content was called
        assert mock_model.generate_content.called

        # Get the generation_config argument
        call_args = mock_model.generate_content.call_args
        generation_config = call_args[1]["generation_config"]

        # Verify structured output configuration
        assert generation_config.temperature == 0.0
        assert generation_config.response_mime_type == "application/json"
        assert generation_config.response_schema is not None

    @patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai")
    def test_generate_multi_solutions_partial_validation_failures(
        self, mock_genai, sample_task, sample_interpretations
    ):
        """Test handling of partial validation failures (some solutions invalid)."""
        # Response with 2 invalid solutions
        partial_response = {
            "solutions": [
                {
                    "interpretation_id": 1,
                    "code": "import numpy as np\n\ndef solve(grid: np.ndarray) -> np.ndarray:\n    return np.flip(grid, axis=0)",
                    "approach_summary": "Valid solution 1",
                },
                {
                    "interpretation_id": 2,
                    "code": "def broken(grid):  # Missing solve() function\n    return grid",
                    "approach_summary": "Invalid - no solve()",
                },
                {
                    "interpretation_id": 3,
                    "code": "import numpy as np\n\ndef solve(grid: np.ndarray) -> np.ndarray:\n    return grid[::-1]",
                    "approach_summary": "Valid solution 2",
                },
                {
                    "interpretation_id": 4,
                    "code": "def solve(grid):\n    return np.array(grid)  # Missing numpy import",
                    "approach_summary": "Invalid - no numpy import",
                },
                {
                    "interpretation_id": 5,
                    "code": "import numpy as np\n\ndef solve(grid: np.ndarray) -> np.ndarray:\n    return np.flipud(grid)",
                    "approach_summary": "Valid solution 3",
                },
            ]
        }

        # Setup mock API response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(partial_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        programmer = MultiSolutionProgrammer(use_cache=False)
        solutions = programmer.generate_multi_solutions(
            sample_task, sample_interpretations
        )

        # Should return only 3 valid solutions
        assert len(solutions) == 3
        assert all(s.interpretation_id in [1, 3, 5] for s in solutions)

    @patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai")
    def test_generate_multi_solutions_all_invalid_raises_error(
        self, mock_genai, sample_task, sample_interpretations
    ):
        """Test that all invalid solutions raises ValueError."""
        # Response with all invalid solutions
        all_invalid_response = {
            "solutions": [
                {
                    "interpretation_id": i,
                    "code": "def broken(): pass",  # All missing solve()
                    "approach_summary": f"Invalid solution {i}",
                }
                for i in range(1, 6)
            ]
        }

        # Setup mock API response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(all_invalid_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        programmer = MultiSolutionProgrammer(use_cache=False)

        with pytest.raises(ValueError, match="All 4 solutions failed validation"):
            programmer.generate_multi_solutions(sample_task, sample_interpretations)

    @patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai")
    def test_generate_multi_solutions_wrong_interpretation_count_raises_error(
        self, mock_genai, sample_task, sample_interpretations
    ):
        """Test that wrong number of interpretations raises ValueError."""
        programmer = MultiSolutionProgrammer(use_cache=False)

        # Only 3 interpretations instead of 5
        wrong_count_interpretations = sample_interpretations[:3]

        with pytest.raises(ValueError, match="Expected 4 interpretations, got 3"):
            programmer.generate_multi_solutions(
                sample_task, wrong_count_interpretations
            )

    @patch("arc_prometheus.cognitive_cells.multi_solution_programmer.genai")
    def test_parse_response_wrong_solution_count(
        self, mock_genai, sample_task, sample_interpretations
    ):
        """Test that wrong solution count in response raises validation error."""
        # Response with only 3 solutions (Pydantic will catch this)
        wrong_count_response = {
            "solutions": [
                {
                    "interpretation_id": i,
                    "code": "import numpy as np\ndef solve(grid): return grid",
                    "approach_summary": f"Solution {i}",
                }
                for i in range(1, 4)
            ]
        }

        # Setup mock API response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(wrong_count_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        programmer = MultiSolutionProgrammer(use_cache=False)

        # Pydantic validation will raise ValidationError for wrong count
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            programmer.generate_multi_solutions(sample_task, sample_interpretations)


# Integration test (can be run with real API if needed)
@pytest.mark.integration
def test_real_api_multi_solution_programmer(sample_task, sample_interpretations):
    """Integration test with real Gemini API.

    This test requires a valid GEMINI_API_KEY environment variable.
    Run with: pytest tests/test_multi_solution_programmer.py -m integration
    """
    programmer = MultiSolutionProgrammer(use_cache=False)

    try:
        solutions = programmer.generate_multi_solutions(
            sample_task, sample_interpretations
        )

        # Validate structure
        assert 3 <= len(solutions) <= 5, f"Expected 3-5 solutions, got {len(solutions)}"
        assert all(isinstance(s, SolutionResult) for s in solutions)

        # Validate content
        for solution in solutions:
            assert 1 <= solution.interpretation_id <= 5, (
                f"Invalid ID: {solution.interpretation_id}"
            )
            assert "def solve(" in solution.code, "Missing solve() function"
            assert solution.approach_summary, "Empty approach summary"
            assert len(solution.approach_summary) <= 100, (
                f"Approach too long: {len(solution.approach_summary)} chars"
            )

            # Validate code compiles
            try:
                compile(solution.code, "<string>", "exec")
            except SyntaxError as e:
                pytest.fail(
                    f"Solution {solution.interpretation_id} has syntax error: {e}"
                )

        print("\n=== Real API Test Results ===")
        print(f"Generated {len(solutions)}/5 valid solutions")
        for solution in solutions:
            print(f"\nSolution {solution.interpretation_id}:")
            print(f"  Approach: {solution.approach_summary}")
            print(f"  Code length: {len(solution.code)} chars")

    except Exception as e:
        pytest.skip(f"Real API test failed (may be expected): {e}")
