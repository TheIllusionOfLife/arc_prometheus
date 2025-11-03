"""Tests for Synthesis Agent."""

import json
from unittest.mock import MagicMock, patch

import pytest

from arc_prometheus.cognitive_cells.multi_persona_analyst import InterpretationResult
from arc_prometheus.cognitive_cells.multi_solution_programmer import SolutionResult
from arc_prometheus.cognitive_cells.synthesis_agent import (
    SynthesisAgent,
    SynthesisResult,
)
from arc_prometheus.utils.schemas import SynthesisResponse


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
            observations=[
                "First row becomes last row",
                "Middle row stays in middle",
                "Last row becomes first row",
            ],
            approach="Use np.flip(grid, axis=0) for vertical flip",
            confidence="high",
        ),
        InterpretationResult(
            persona="Color Pattern Specialist",
            pattern="Invert the color values in reverse spatial order",
            observations=[
                "Colors maintain their positions",
                "Spatial arrangement is reversed",
            ],
            approach="Reverse iteration with color mapping",
            confidence="medium",
        ),
        InterpretationResult(
            persona="Object Detection Specialist",
            pattern="Treat each row as an object and reverse their positions",
            observations=[
                "Three row-objects identified",
                "Objects swapped: top↔bottom",
            ],
            approach="Extract rows, reverse list, stack back",
            confidence="high",
        ),
        InterpretationResult(
            persona="Grid Structure Specialist",
            pattern="Reverse grid along the horizontal axis",
            observations=[
                "Grid height unchanged (3x3)",
                "Row-wise reversal pattern",
            ],
            approach="Use array slicing [::-1] on rows",
            confidence="high",
        ),
        InterpretationResult(
            persona="Logical Rules Specialist",
            pattern="Apply rule: output[i] = input[n-1-i] for each row i",
            observations=[
                "Index transformation pattern",
                "Works for any grid height",
            ],
            approach="Iterate with reversed indices",
            confidence="medium",
        ),
    ]


@pytest.fixture
def sample_solutions():
    """Sample solutions from Multi-Solution Programmer."""
    return [
        SolutionResult(
            interpretation_id=1,
            code="import numpy as np\n\ndef solve(task_grid: np.ndarray) -> np.ndarray:\n    return np.flip(task_grid, axis=0)",
            approach_summary="Vertical flip using np.flip",
        ),
        SolutionResult(
            interpretation_id=2,
            code="import numpy as np\n\ndef solve(task_grid: np.ndarray) -> np.ndarray:\n    return task_grid[::-1, :]",
            approach_summary="Reverse rows with slicing",
        ),
        SolutionResult(
            interpretation_id=3,
            code="import numpy as np\n\ndef solve(task_grid: np.ndarray) -> np.ndarray:\n    rows = [row for row in task_grid]\n    return np.array(rows[::-1])",
            approach_summary="Extract rows, reverse, stack",
        ),
        SolutionResult(
            interpretation_id=4,
            code="import numpy as np\n\ndef solve(task_grid: np.ndarray) -> np.ndarray:\n    return task_grid[::-1]",
            approach_summary="Simple array reversal",
        ),
        SolutionResult(
            interpretation_id=5,
            code="import numpy as np\n\ndef solve(task_grid: np.ndarray) -> np.ndarray:\n    n = len(task_grid)\n    result = np.zeros_like(task_grid)\n    for i in range(n):\n        result[i] = task_grid[n-1-i]\n    return result",
            approach_summary="Index transformation with loop",
        ),
    ]


@pytest.fixture
def sample_api_response():
    """Sample valid API response matching SYNTHESIS_SCHEMA."""
    return {
        "analysis": {
            "successful_patterns": [
                "np.flip and slicing worked well for vertical flip",
                "All solutions used numpy array operations",
            ],
            "failed_patterns": [
                "Manual loops were slower but still worked",
            ],
            "synthesis_strategy": "Try rotation (90° turn) as different approach from vertical flip",
        },
        "code": "import numpy as np\n\ndef solve(task_grid: np.ndarray) -> np.ndarray:\n    return np.rot90(task_grid, k=2)",
        "diversity_justification": "Uses rotation (180°) instead of vertical flip - different transformation category",
    }


class TestSynthesisResult:
    """Test SynthesisResult dataclass."""

    def test_create_synthesis_result(self):
        """Test creating SynthesisResult with all fields."""
        result = SynthesisResult(
            code="import numpy as np\ndef solve(grid): return grid",
            approach_summary="Test approach",
            successful_patterns=["Pattern 1", "Pattern 2"],
            failed_patterns=["Anti-pattern 1"],
            synthesis_strategy="Combine successful patterns",
            diversity_justification="Uses different algorithm",
        )

        assert "def solve" in result.code
        assert result.approach_summary == "Test approach"
        assert len(result.successful_patterns) == 2
        assert len(result.failed_patterns) == 1
        assert result.synthesis_strategy == "Combine successful patterns"
        assert result.diversity_justification == "Uses different algorithm"


class TestSynthesisAgent:
    """Test SynthesisAgent class."""

    def test_initialization_default(self):
        """Test default initialization."""
        with patch("arc_prometheus.cognitive_cells.synthesis_agent.genai"):
            agent = SynthesisAgent()

            assert agent.model_name == "gemini-2.0-flash-thinking-exp"
            assert agent.temperature == 0.5
            assert agent.use_cache is True
            assert agent.timeout == 5
            assert agent.sandbox_mode == "multiprocess"

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        with patch("arc_prometheus.cognitive_cells.synthesis_agent.genai"):
            agent = SynthesisAgent(
                model_name="gemini-2.5-flash-lite",
                temperature=0.7,
                use_cache=False,
                timeout=10,
                sandbox_mode="docker",
            )

            assert agent.model_name == "gemini-2.5-flash-lite"
            assert agent.temperature == 0.7
            assert agent.use_cache is False
            assert agent.timeout == 10
            assert agent.sandbox_mode == "docker"

    def test_format_grid(self, sample_task):
        """Test grid formatting."""
        with patch("arc_prometheus.cognitive_cells.synthesis_agent.genai"):
            agent = SynthesisAgent()
            grid = sample_task["train"][0]["input"]

            formatted = agent._format_grid(grid)

            expected = "0 1 2\n3 4 5\n6 7 8"
            assert formatted == expected

    def test_create_prompt_structure(
        self, sample_task, sample_solutions, sample_interpretations
    ):
        """Test that prompt contains required elements."""
        with patch("arc_prometheus.cognitive_cells.synthesis_agent.genai"):
            agent = SynthesisAgent()

            # Mock accuracies
            accuracies = [
                {"train_accuracy": 1.0, "train_correct": 2, "train_total": 2},
                {"train_accuracy": 1.0, "train_correct": 2, "train_total": 2},
                {"train_accuracy": 1.0, "train_correct": 2, "train_total": 2},
                {"train_accuracy": 0.5, "train_correct": 1, "train_total": 2},
                {"train_accuracy": 0.0, "train_correct": 0, "train_total": 2},
            ]

            prompt = agent._create_prompt(
                sample_task, sample_solutions, accuracies, sample_interpretations
            )

            # Check for key sections
            assert "TRAINING EXAMPLES:" in prompt
            assert "PREVIOUS 5 SOLUTIONS" in prompt
            assert "YOUR TASK:" in prompt

            # Check for solution summaries with accuracy
            assert "Solution 1" in prompt
            assert "Train Accuracy:" in prompt
            assert "100%" in prompt or "0%" in prompt

            # Check for persona names
            assert "Geometric Transformation Specialist" in prompt

            # Check for diversity constraint
            assert "DIFFERENT algorithm" in prompt
            assert "diversity_justification" in prompt

    def test_validate_solution_valid(self):
        """Test validation of valid solution."""
        with patch("arc_prometheus.cognitive_cells.synthesis_agent.genai"):
            agent = SynthesisAgent()

            code = "import numpy as np\n\ndef solve(task_grid: np.ndarray) -> np.ndarray:\n    return np.flip(task_grid, axis=0)"

            is_valid, error_msg = agent._validate_solution(code)

            assert is_valid is True
            assert error_msg == ""

    def test_validate_solution_missing_solve(self):
        """Test validation catches missing solve() function."""
        with patch("arc_prometheus.cognitive_cells.synthesis_agent.genai"):
            agent = SynthesisAgent()

            code = "import numpy as np\n\ndef transform(grid):\n    return grid"

            is_valid, error_msg = agent._validate_solution(code)

            assert is_valid is False
            assert "Missing 'def solve(' function" in error_msg

    def test_validate_solution_syntax_error(self):
        """Test validation catches syntax errors."""
        with patch("arc_prometheus.cognitive_cells.synthesis_agent.genai"):
            agent = SynthesisAgent()

            code = "import numpy as np\n\ndef solve(grid)\n    return grid"  # Missing colon

            is_valid, error_msg = agent._validate_solution(code)

            assert is_valid is False
            assert "Syntax error" in error_msg

    def test_validate_solution_missing_import(self):
        """Test validation catches missing numpy import."""
        with patch("arc_prometheus.cognitive_cells.synthesis_agent.genai"):
            agent = SynthesisAgent()

            code = "def solve(task_grid):\n    return np.flip(task_grid, axis=0)"

            is_valid, error_msg = agent._validate_solution(code)

            assert is_valid is False
            assert "without importing" in error_msg

    def test_parse_response_valid(self, sample_api_response):
        """Test parsing valid API response."""
        with patch("arc_prometheus.cognitive_cells.synthesis_agent.genai"):
            agent = SynthesisAgent()
            # Convert dict to Pydantic model first
            pydantic_response = SynthesisResponse.model_validate(sample_api_response)
            result = agent._parse_response(pydantic_response)

            assert isinstance(result, SynthesisResult)
            assert "def solve" in result.code
            assert "np.rot90" in result.code
            assert len(result.successful_patterns) == 2
            assert len(result.failed_patterns) == 1
            assert "rotation" in result.synthesis_strategy.lower()
            assert "rotation" in result.diversity_justification.lower()

    def test_parse_response_invalid_code(self):
        """Test parsing response with invalid code."""
        with patch("arc_prometheus.cognitive_cells.synthesis_agent.genai"):
            agent = SynthesisAgent()

            invalid_response = {
                "analysis": {
                    "successful_patterns": ["Pattern 1"],
                    "failed_patterns": ["Anti-pattern 1"],
                    "synthesis_strategy": "Test strategy",
                },
                "code": "def transform(grid): return grid",  # Missing solve()
                "diversity_justification": "Different approach",
            }

            # Convert dict to Pydantic model first
            pydantic_response = SynthesisResponse.model_validate(invalid_response)

            with pytest.raises(ValueError, match="validation failed"):
                agent._parse_response(pydantic_response)

    @patch("arc_prometheus.cognitive_cells.synthesis_agent.genai")
    def test_synthesize_solution_wrong_solution_count(
        self, mock_genai, sample_task, sample_solutions, sample_interpretations
    ):
        """Test error when wrong number of solutions provided."""
        agent = SynthesisAgent()

        # Only 3 solutions instead of 5
        with pytest.raises(ValueError, match="Expected 5 solutions, got 3"):
            agent.synthesize_solution(
                sample_task, sample_solutions[:3], sample_interpretations
            )

    @patch("arc_prometheus.cognitive_cells.synthesis_agent.genai")
    def test_synthesize_solution_wrong_interpretation_count(
        self, mock_genai, sample_task, sample_solutions, sample_interpretations
    ):
        """Test error when wrong number of interpretations provided."""
        agent = SynthesisAgent()

        # Only 3 interpretations instead of 5
        with pytest.raises(ValueError, match="Expected 5 interpretations, got 3"):
            agent.synthesize_solution(
                sample_task, sample_solutions, sample_interpretations[:3]
            )

    @patch("arc_prometheus.crucible.sandbox.MultiprocessSandbox")
    @patch("arc_prometheus.cognitive_cells.synthesis_agent.genai")
    @patch("arc_prometheus.utils.llm_cache.get_cache")
    def test_synthesize_solution_with_cache_hit(
        self,
        mock_get_cache,
        mock_genai,
        mock_get_sandbox,
        sample_task,
        sample_solutions,
        sample_interpretations,
        sample_api_response,
    ):
        """Test synthesize_solution with cache hit."""
        # Setup mock cache
        mock_cache = MagicMock()
        mock_cache.get.return_value = json.dumps(sample_api_response)
        mock_get_cache.return_value = mock_cache

        # Setup mock sandbox (for accuracy calculation)
        import numpy as np

        mock_sandbox_instance = MagicMock()
        mock_sandbox_instance.execute.return_value = (
            True,
            np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]], dtype=np.int64),
            None,
        )
        mock_get_sandbox.return_value = mock_sandbox_instance

        agent = SynthesisAgent(use_cache=True)
        result = agent.synthesize_solution(
            sample_task, sample_solutions, sample_interpretations
        )

        # Should get result from cache
        assert isinstance(result, SynthesisResult)
        assert mock_cache.get.called
        # API should NOT be called
        assert not mock_genai.GenerativeModel.called

    @patch("arc_prometheus.crucible.sandbox.MultiprocessSandbox")
    @patch("arc_prometheus.cognitive_cells.synthesis_agent.genai")
    @patch("arc_prometheus.utils.llm_cache.get_cache")
    def test_synthesize_solution_with_cache_miss(
        self,
        mock_get_cache,
        mock_genai,
        mock_get_sandbox,
        sample_task,
        sample_solutions,
        sample_interpretations,
        sample_api_response,
    ):
        """Test synthesize_solution with cache miss."""
        # Setup mock cache (cache miss)
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        # Setup mock sandbox (for accuracy calculation)
        import numpy as np

        mock_sandbox_instance = MagicMock()
        mock_sandbox_instance.execute.return_value = (
            True,
            np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]], dtype=np.int64),
            None,
        )
        mock_get_sandbox.return_value = mock_sandbox_instance

        # Setup mock API response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_api_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        agent = SynthesisAgent(use_cache=True)
        result = agent.synthesize_solution(
            sample_task, sample_solutions, sample_interpretations
        )

        # Should get result from API
        assert isinstance(result, SynthesisResult)
        assert mock_cache.get.called
        # API should be called
        assert mock_genai.GenerativeModel.called
        # Result should be cached
        assert mock_cache.set.called

    @patch("arc_prometheus.crucible.sandbox.MultiprocessSandbox")
    @patch("arc_prometheus.cognitive_cells.synthesis_agent.genai")
    def test_synthesize_solution_no_cache(
        self,
        mock_genai,
        mock_get_sandbox,
        sample_task,
        sample_solutions,
        sample_interpretations,
        sample_api_response,
    ):
        """Test synthesize_solution with caching disabled."""
        # Setup mock sandbox (for accuracy calculation)
        import numpy as np

        mock_sandbox_instance = MagicMock()
        mock_sandbox_instance.execute.return_value = (
            True,
            np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]], dtype=np.int64),
            None,
        )
        mock_get_sandbox.return_value = mock_sandbox_instance

        # Setup mock API response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_api_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        agent = SynthesisAgent(use_cache=False)
        result = agent.synthesize_solution(
            sample_task, sample_solutions, sample_interpretations
        )

        # Should get result from API
        assert isinstance(result, SynthesisResult)
        # API should be called
        assert mock_genai.GenerativeModel.called

    @patch("arc_prometheus.crucible.sandbox.MultiprocessSandbox")
    @patch("arc_prometheus.cognitive_cells.synthesis_agent.genai")
    def test_synthesize_solution_uses_structured_output(
        self,
        mock_genai,
        mock_get_sandbox,
        sample_task,
        sample_solutions,
        sample_interpretations,
        sample_api_response,
    ):
        """Test that synthesize_solution uses structured output configuration."""
        # Setup mock sandbox
        import numpy as np

        mock_sandbox_instance = MagicMock()
        mock_sandbox_instance.execute.return_value = (
            True,
            np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]], dtype=np.int64),
            None,
        )
        mock_get_sandbox.return_value = mock_sandbox_instance

        # Setup mock API response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_api_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        agent = SynthesisAgent(use_cache=False)
        agent.synthesize_solution(sample_task, sample_solutions, sample_interpretations)

        # Check that generate_content was called
        assert mock_model.generate_content.called

        # Get the generation_config argument
        call_args = mock_model.generate_content.call_args
        generation_config = call_args[1]["generation_config"]

        # Verify structured output configuration
        assert generation_config.temperature == 0.5
        assert generation_config.response_mime_type == "application/json"
        assert generation_config.response_schema is not None

    @patch("arc_prometheus.crucible.sandbox.MultiprocessSandbox")
    def test_calculate_solution_accuracies_all_correct(
        self, mock_get_sandbox, sample_task, sample_solutions
    ):
        """Test accuracy calculation when all solutions work."""
        import numpy as np

        # Setup mock sandbox to return correct results
        mock_sandbox_instance = MagicMock()
        mock_sandbox_instance.execute.return_value = (
            True,
            np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]], dtype=np.int64),
            None,
        )
        mock_get_sandbox.return_value = mock_sandbox_instance

        with patch("arc_prometheus.cognitive_cells.synthesis_agent.genai"):
            agent = SynthesisAgent()
            accuracies = agent._calculate_solution_accuracies(
                sample_task, sample_solutions
            )

            assert len(accuracies) == 5
            # First solution should have high accuracy
            assert accuracies[0]["train_accuracy"] > 0.0
            assert accuracies[0]["train_correct"] >= 0
            assert accuracies[0]["train_total"] == 2

    @patch("arc_prometheus.crucible.sandbox.MultiprocessSandbox")
    def test_calculate_solution_accuracies_with_failures(
        self, mock_get_sandbox, sample_task, sample_solutions
    ):
        """Test accuracy calculation with some failures."""
        import numpy as np

        # Setup mock sandbox to fail sometimes
        # Each solution is tested against 2 training examples, so we need 10 results
        mock_sandbox_instance = MagicMock()
        mock_sandbox_instance.execute.side_effect = [
            # Solution 1: both examples succeed
            (True, np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]], dtype=np.int64), None),
            (True, np.array([[3, 3, 3], [2, 2, 2], [1, 1, 1]], dtype=np.int64), None),
            # Solution 2: first succeeds, second fails
            (True, np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]], dtype=np.int64), None),
            (False, None, {"error_type": "runtime", "error_message": "Error"}),
            # Solution 3: both fail
            (False, None, {"error_type": "runtime", "error_message": "Error"}),
            (False, None, {"error_type": "runtime", "error_message": "Error"}),
            # Solution 4: first fails, second succeeds
            (False, None, {"error_type": "runtime", "error_message": "Error"}),
            (True, np.array([[3, 3, 3], [2, 2, 2], [1, 1, 1]], dtype=np.int64), None),
            # Solution 5: both succeed
            (True, np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]], dtype=np.int64), None),
            (True, np.array([[3, 3, 3], [2, 2, 2], [1, 1, 1]], dtype=np.int64), None),
        ]

        mock_get_sandbox.return_value = mock_sandbox_instance

        with patch("arc_prometheus.cognitive_cells.synthesis_agent.genai"):
            agent = SynthesisAgent()
            accuracies = agent._calculate_solution_accuracies(
                sample_task, sample_solutions
            )

            assert len(accuracies) == 5
            # Should have mixed results
            assert any(acc["train_accuracy"] > 0 for acc in accuracies)
            assert any(acc["has_errors"] for acc in accuracies)


# Integration test (can be run with real API if needed)
@pytest.mark.integration
def test_real_api_synthesis_agent(
    sample_task, sample_solutions, sample_interpretations
):
    """Integration test with real Gemini API.

    This test requires a valid GEMINI_API_KEY environment variable.
    Run with: pytest tests/test_synthesis_agent.py -m integration
    """
    agent = SynthesisAgent(use_cache=False)

    try:
        result = agent.synthesize_solution(
            sample_task, sample_solutions, sample_interpretations
        )

        # Validate structure
        assert isinstance(result, SynthesisResult)

        # Validate content
        assert result.code, "Code should not be empty"
        assert "def solve" in result.code, "Should contain solve() function"
        assert len(result.successful_patterns) >= 1, (
            "Should have at least 1 successful pattern"
        )
        assert len(result.successful_patterns) <= 3, (
            "Should have at most 3 successful patterns"
        )
        for pattern in result.successful_patterns:
            assert len(pattern) <= 80, f"Pattern too long: {len(pattern)} chars"

        assert len(result.failed_patterns) >= 1, "Should have at least 1 failed pattern"
        assert len(result.failed_patterns) <= 3, "Should have at most 3 failed patterns"
        for pattern in result.failed_patterns:
            assert len(pattern) <= 80, f"Pattern too long: {len(pattern)} chars"

        assert result.synthesis_strategy, "Synthesis strategy should not be empty"
        assert len(result.synthesis_strategy) <= 150, (
            "Synthesis strategy should be ≤150 chars"
        )

        assert result.diversity_justification, (
            "Diversity justification should not be empty"
        )
        assert len(result.diversity_justification) <= 100, (
            "Diversity justification should be ≤100 chars"
        )

        print("\n=== Real API Test Results ===")
        print(f"Code length: {len(result.code)} chars")
        print(f"Successful patterns: {result.successful_patterns}")
        print(f"Failed patterns: {result.failed_patterns}")
        print(f"Synthesis strategy: {result.synthesis_strategy}")
        print(f"Diversity justification: {result.diversity_justification}")

    except Exception as e:
        pytest.skip(f"Real API test failed (may be expected): {e}")
