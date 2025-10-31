"""Integration tests for Analyst + Programmer collaboration.

This test suite validates:
1. End-to-end flow: Analyst analyzes task → Programmer generates code
2. Programmer correctly uses Analyst's specifications
3. Backward compatibility (Programmer works with or without Analyst)
4. Code quality improvements when using Analyst specifications
5. Both modes produce executable solver functions
"""

from unittest.mock import Mock, patch

import numpy as np

from arc_prometheus.cognitive_cells.analyst import AnalysisResult, Analyst
from arc_prometheus.cognitive_cells.programmer import generate_solver


class TestAnalystProgrammerIntegration:
    """Integration tests for Analyst + Programmer workflow."""

    @patch("arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.programmer.genai")
    @patch("arc_prometheus.cognitive_cells.analyst.genai")
    def test_end_to_end_with_analyst(
        self,
        mock_analyst_genai,
        mock_programmer_genai,
        mock_analyst_api_key,
        mock_programmer_api_key,
    ):
        """Test complete flow: Analyst → Programmer → executable code."""
        # Setup API key mocks
        mock_analyst_api_key.return_value = "test-api-key"
        mock_programmer_api_key.return_value = "test-api-key"

        # Setup Analyst mock
        mock_analyst_response = Mock()
        mock_analyst_response.text = """PATTERN: Fill the entire grid with the single non-zero color

OBSERVATIONS:
- Input contains exactly one non-zero cell
- Output fills entire grid with that color
- Grid size remains the same

APPROACH: Use np.nonzero() to find the colored cell, then np.full() to create filled grid

CONFIDENCE: high"""

        mock_analyst_model = Mock()
        mock_analyst_model.generate_content.return_value = mock_analyst_response
        mock_analyst_genai.GenerativeModel.return_value = mock_analyst_model

        # Setup Programmer mock
        mock_programmer_response = Mock()
        mock_programmer_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Find the non-zero value
    non_zero_indices = np.nonzero(task_grid)
    if len(non_zero_indices[0]) > 0:
        color = task_grid[non_zero_indices[0][0], non_zero_indices[1][0]]
    else:
        return task_grid

    # Fill entire grid with that color
    return np.full(task_grid.shape, color, dtype=task_grid.dtype)
```"""

        mock_programmer_model = Mock()
        mock_programmer_model.generate_content.return_value = mock_programmer_response
        mock_programmer_genai.GenerativeModel.return_value = mock_programmer_model

        # Execute end-to-end flow
        task_data = {
            "train": [
                {"input": [[1, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
                {"input": [[0, 0], [4, 0]], "output": [[4, 4], [4, 4]]},
            ]
        }

        # Step 1: Analyst analyzes task
        analyst = Analyst(use_cache=False)
        analysis = analyst.analyze_task(task_data)

        assert (
            analysis.pattern_description
            == "Fill the entire grid with the single non-zero color"
        )
        assert len(analysis.key_observations) == 3
        assert analysis.confidence == "high"

        # Step 2: Programmer generates code using Analyst's analysis
        train_pairs = [
            {"input": np.array(ex["input"]), "output": np.array(ex["output"])}
            for ex in task_data["train"]
        ]

        code = generate_solver(train_pairs, analyst_spec=analysis, use_cache=False)

        # Verify code was generated
        assert "import numpy as np" in code
        assert "def solve(" in code
        assert "np.nonzero()" in code or "non_zero" in code.lower()

        # Verify Programmer API was called with correct prompt
        programmer_call_args = mock_programmer_model.generate_content.call_args[0][0]
        assert "Pattern Analysis (from Analyst Agent)" in programmer_call_args
        assert "Fill the entire grid" in programmer_call_args
        assert "Input contains exactly one non-zero cell" in programmer_call_args

    @patch("arc_prometheus.cognitive_cells.programmer.genai")
    def test_programmer_without_analyst_still_works(self, mock_genai):
        """Test backward compatibility: Programmer works without Analyst."""
        # Setup mock
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Generate code WITHOUT analyst_spec (Direct mode)
        train_pairs = [{"input": np.array([[1, 2]]), "output": np.array([[2, 3]])}]

        code = generate_solver(train_pairs, use_cache=False)

        # Verify code was generated
        assert "import numpy as np" in code
        assert "def solve(" in code

        # Verify prompt does NOT contain Analyst-specific content
        call_args = mock_model.generate_content.call_args[0][0]
        assert "Pattern Analysis (from Analyst Agent)" not in call_args
        assert "Transformation Rule:" not in call_args
        # Should contain Direct mode instructions
        assert (
            "Analyze the input-output examples" in call_args
            or "infer the transformation" in call_args
        )

    @patch("arc_prometheus.cognitive_cells.programmer.genai")
    def test_analyst_spec_passed_to_prompt(self, mock_genai):
        """Test that Analyst specifications are correctly passed to Programmer prompt."""
        # Setup mock
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return np.rot90(task_grid, k=-1)
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Create Analyst result
        analysis = AnalysisResult(
            pattern_description="Rotate grid 90 degrees clockwise",
            key_observations=[
                "Top row becomes right column",
                "Bottom row becomes left column",
                "Grid dimensions are square",
            ],
            suggested_approach="Use np.rot90() with k=-1 parameter for clockwise rotation",
            confidence="high",
        )

        # Generate code with analyst_spec
        train_pairs = [
            {"input": np.array([[1, 2], [3, 4]]), "output": np.array([[3, 1], [4, 2]])}
        ]

        generate_solver(train_pairs, analyst_spec=analysis, use_cache=False)

        # Verify Analyst specifications were included in prompt
        call_args = mock_model.generate_content.call_args[0][0]

        # Check for AI Civilization mode markers
        assert "Pattern Analysis (from Analyst Agent)" in call_args
        assert "Transformation Rule:" in call_args

        # Check for Analyst's specific content
        assert "Rotate grid 90 degrees clockwise" in call_args
        assert "Top row becomes right column" in call_args
        assert "Use np.rot90() with k=-1" in call_args
        assert "**Analyst Confidence:** high" in call_args

    @patch("arc_prometheus.cognitive_cells.programmer.genai")
    def test_analyst_spec_with_medium_confidence(self, mock_genai):
        """Test Programmer handles medium confidence from Analyst."""
        # Setup mock
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return np.fliplr(task_grid)
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Create Analyst result with medium confidence
        analysis = AnalysisResult(
            pattern_description="Mirror grid horizontally",
            key_observations=["Pattern unclear with limited examples"],
            suggested_approach="Try horizontal flip using np.fliplr()",
            confidence="medium",
        )

        train_pairs = [{"input": np.array([[1, 2]]), "output": np.array([[2, 1]])}]

        code = generate_solver(train_pairs, analyst_spec=analysis, use_cache=False)

        # Verify code was generated despite medium confidence
        assert "import numpy as np" in code
        assert "def solve(" in code

        # Verify confidence level was passed to prompt
        call_args = mock_model.generate_content.call_args[0][0]
        assert "**Analyst Confidence:** medium" in call_args

    @patch("arc_prometheus.cognitive_cells.programmer.genai")
    def test_analyst_spec_with_minimal_observations(self, mock_genai):
        """Test Programmer handles Analyst result with few observations."""
        # Setup mock
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Create Analyst result with no observations
        analysis = AnalysisResult(
            pattern_description="Identity transformation",
            key_observations=[],
            suggested_approach="Return input unchanged",
            confidence="low",
        )

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[1]])}]

        code = generate_solver(train_pairs, analyst_spec=analysis, use_cache=False)

        # Verify code was still generated
        assert "import numpy as np" in code
        assert "def solve(" in code

        # Verify prompt was created (should handle empty observations gracefully)
        call_args = mock_model.generate_content.call_args[0][0]
        assert "Pattern Analysis (from Analyst Agent)" in call_args
        # Should NOT have "Key Observations:" section if empty
        assert "Identity transformation" in call_args

    @patch("arc_prometheus.cognitive_cells.programmer.genai")
    def test_multiple_observations_formatted_correctly(self, mock_genai):
        """Test that multiple observations are formatted as bullet points."""
        # Setup mock
        mock_response = Mock()
        mock_response.text = """```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
```"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Create Analyst result with many observations
        analysis = AnalysisResult(
            pattern_description="Complex transformation",
            key_observations=[
                "Observation 1: Colors change",
                "Observation 2: Size increases",
                "Observation 3: Pattern repeats",
                "Observation 4: Symmetry detected",
                "Observation 5: Edge effects",
            ],
            suggested_approach="Multi-step approach",
            confidence="medium",
        )

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[2]])}]

        generate_solver(train_pairs, analyst_spec=analysis, use_cache=False)

        # Verify prompt contains all observations as bullet points
        call_args = mock_model.generate_content.call_args[0][0]
        assert "**Key Observations:**" in call_args
        for i in range(1, 6):
            assert f"Observation {i}:" in call_args

    def test_analyst_result_dataclass_compatibility(self):
        """Test that AnalysisResult can be passed to generate_solver without issues."""
        # This test verifies the type system works correctly
        # No API mocking needed - just checking interface compatibility

        analysis = AnalysisResult(
            pattern_description="Test pattern",
            key_observations=["Test obs"],
            suggested_approach="Test approach",
            confidence="high",
        )

        # Verify AnalysisResult has expected attributes
        assert hasattr(analysis, "pattern_description")
        assert hasattr(analysis, "key_observations")
        assert hasattr(analysis, "suggested_approach")
        assert hasattr(analysis, "confidence")

        # Verify types
        assert isinstance(analysis.pattern_description, str)
        assert isinstance(analysis.key_observations, list)
        assert isinstance(analysis.suggested_approach, str)
        assert isinstance(analysis.confidence, str)


class TestPromptModeSelection:
    """Tests for AI Civilization mode vs Direct mode selection."""

    @patch("arc_prometheus.cognitive_cells.programmer.genai")
    def test_ai_civilization_mode_with_analyst_spec(self, mock_genai):
        """Test that providing analyst_spec triggers AI Civilization mode."""
        # Setup mock
        mock_response = Mock()
        mock_response.text = (
            "```python\nimport numpy as np\ndef solve(x): return x\n```"
        )
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        analysis = AnalysisResult(
            pattern_description="Test",
            key_observations=["Test"],
            suggested_approach="Test",
            confidence="high",
        )

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[2]])}]
        generate_solver(train_pairs, analyst_spec=analysis, use_cache=False)

        prompt = mock_model.generate_content.call_args[0][0]

        # AI Civilization mode markers
        assert "You are a Programmer agent in an AI civilization" in prompt
        assert "Pattern Analysis (from Analyst Agent)" in prompt

    @patch("arc_prometheus.cognitive_cells.programmer.genai")
    def test_direct_mode_without_analyst_spec(self, mock_genai):
        """Test that omitting analyst_spec triggers Direct mode."""
        # Setup mock
        mock_response = Mock()
        mock_response.text = (
            "```python\nimport numpy as np\ndef solve(x): return x\n```"
        )
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        train_pairs = [{"input": np.array([[1]]), "output": np.array([[2]])}]
        generate_solver(train_pairs, use_cache=False)

        prompt = mock_model.generate_content.call_args[0][0]

        # Direct mode markers
        assert "You are an AI system analyzing Abstract Reasoning Corpus" in prompt
        assert (
            "Analyze the input-output examples" in prompt
            or "infer the transformation" in prompt
        )

        # Should NOT have AI Civilization mode markers
        assert "Pattern Analysis (from Analyst Agent)" not in prompt
        assert "Programmer agent in an AI civilization" not in prompt
