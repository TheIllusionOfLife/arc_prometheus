"""Tests for Analyst agent - pattern analysis and rule inference.

This test suite validates:
1. Pattern analysis from ARC task examples
2. Natural language rule description generation
3. Key observations extraction
4. Implementation approach suggestions
5. Confidence level assessment
6. Response parsing for structured output
7. Integration with Programmer agent
8. Edge cases (single example, large grids, complex patterns)
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from arc_prometheus.cognitive_cells.analyst import (
    AnalysisResult,
    Analyst,
    parse_analysis_response,
)


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_analysis_result_creation(self):
        """Test creating an AnalysisResult with all fields."""
        result = AnalysisResult(
            pattern_description="Fill the grid with the non-zero color",
            key_observations=[
                "Input has one non-zero cell",
                "Output fills entire grid with that color",
            ],
            suggested_approach="Find non-zero value, create grid filled with it",
            confidence="high",
        )

        assert result.pattern_description == "Fill the grid with the non-zero color"
        assert len(result.key_observations) == 2
        assert result.suggested_approach.startswith("Find non-zero value")
        assert result.confidence == "high"

    def test_analysis_result_with_medium_confidence(self):
        """Test AnalysisResult with medium confidence."""
        result = AnalysisResult(
            pattern_description="Rotate and fill pattern",
            key_observations=["Rotation detected", "Color change detected"],
            suggested_approach="Apply rotation then fill",
            confidence="medium",
        )

        assert result.confidence == "medium"

    def test_analysis_result_with_low_confidence(self):
        """Test AnalysisResult with low confidence for complex patterns."""
        result = AnalysisResult(
            pattern_description="Complex multi-step transformation",
            key_observations=["Pattern unclear from limited examples"],
            suggested_approach="Try multiple approaches",
            confidence="low",
        )

        assert result.confidence == "low"


class TestResponseParsing:
    """Tests for parsing LLM responses into AnalysisResult."""

    def test_parse_complete_analysis_response(self):
        """Test parsing a complete, well-formatted analysis."""
        response = """PATTERN: Fill the entire grid with the single non-zero color found in the input

OBSERVATIONS:
- Input contains exactly one non-zero cell
- Output is same size as input but completely filled
- The fill color matches the non-zero input color
- Grid size varies (2x2 to 10x10)

APPROACH: Locate the non-zero value using np.nonzero(), then create output grid filled with that value using np.full()

CONFIDENCE: high"""

        result = parse_analysis_response(response)

        assert "Fill the entire grid" in result.pattern_description
        assert len(result.key_observations) == 4
        assert "Input contains exactly one non-zero cell" in result.key_observations
        assert "np.nonzero()" in result.suggested_approach
        assert result.confidence == "high"

    def test_parse_analysis_with_missing_sections(self):
        """Test parsing when some sections are missing."""
        response = """PATTERN: Rotate grid 90 degrees clockwise

APPROACH: Use np.rot90() with k=-1 parameter

CONFIDENCE: medium"""

        result = parse_analysis_response(response)

        assert "Rotate grid 90 degrees" in result.pattern_description
        assert len(result.key_observations) == 0  # Missing observations
        assert "np.rot90()" in result.suggested_approach
        assert result.confidence == "medium"

    def test_parse_analysis_with_lowercase_keywords(self):
        """Test parsing with lowercase section keywords."""
        response = """pattern: Mirror the grid horizontally

observations:
- Left side becomes right side
- Right side becomes left side

approach: Use np.fliplr() to flip left-right

confidence: high"""

        result = parse_analysis_response(response)

        assert "Mirror the grid" in result.pattern_description
        assert len(result.key_observations) == 2
        assert result.confidence == "high"

    def test_parse_analysis_with_extra_whitespace(self):
        """Test parsing handles extra whitespace gracefully."""
        response = """  PATTERN:   Copy and extend pattern

OBSERVATIONS:
-   Input is 3x3
-   Output is 6x6

APPROACH:   Tile the input using np.tile()

CONFIDENCE:  high  """

        result = parse_analysis_response(response)

        assert "Copy and extend pattern" in result.pattern_description
        assert len(result.key_observations) == 2
        assert "np.tile()" in result.suggested_approach
        assert result.confidence == "high"

    def test_parse_analysis_with_multiple_line_sections(self):
        """Test parsing when sections span multiple lines."""
        response = """PATTERN: This is a complex pattern that
requires multiple lines to describe properly and
captures the essence of the transformation

OBSERVATIONS:
- First observation
- Second observation that is very long
  and spans multiple lines
- Third observation

APPROACH: Use a combination of operations including
numpy rotation, flipping, and filling to achieve
the desired transformation

CONFIDENCE: medium"""

        result = parse_analysis_response(response)

        assert "complex pattern" in result.pattern_description
        assert len(result.key_observations) == 3
        assert "combination of operations" in result.suggested_approach
        assert result.confidence == "medium"


class TestAnalystPromptCreation:
    """Tests for Analyst prompt generation."""

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    def test_create_analysis_prompt_basic(self, mock_get_api_key):
        """Test creating analysis prompt with simple task."""
        mock_get_api_key.return_value = "test-api-key"
        analyst = Analyst(model_name="gemini-2.5-flash-lite")

        task_json = {
            "train": [
                {"input": [[1, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
                {"input": [[0, 0], [4, 0]], "output": [[4, 4], [4, 4]]},
            ]
        }

        prompt = analyst._create_analysis_prompt(task_json)

        # Should include training examples
        assert (
            "training example" in prompt.lower() or "training pairs" in prompt.lower()
        )

        # Should include instructions
        assert "pattern" in prompt.lower()
        assert "transformation" in prompt.lower()

        # Should include output format requirements
        assert "PATTERN" in prompt
        assert "OBSERVATIONS" in prompt
        assert "APPROACH" in prompt
        assert "CONFIDENCE" in prompt

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    def test_create_analysis_prompt_includes_all_training_pairs(self, mock_get_api_key):
        """Test that prompt includes all training examples."""
        mock_get_api_key.return_value = "test-api-key"
        analyst = Analyst(model_name="gemini-2.5-flash-lite")

        task_json = {
            "train": [
                {"input": [[1]], "output": [[2]]},
                {"input": [[3]], "output": [[4]]},
                {"input": [[5]], "output": [[6]]},
            ]
        }

        prompt = analyst._create_analysis_prompt(task_json)

        # Should reference 3 training examples
        assert "3" in prompt or "three" in prompt.lower()


class TestAnalystAnalyzeTask:
    """Tests for Analyst.analyze_task method."""

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.analyst.genai")
    def test_analyze_task_with_mocked_api(self, mock_genai, mock_get_api_key):
        """Test analyze_task with mocked Gemini API response."""
        # Setup API key mock
        mock_get_api_key.return_value = "test-api-key"

        # Setup mock
        mock_response = Mock()
        mock_response.text = """PATTERN: Fill grid with non-zero color

OBSERVATIONS:
- Input has one colored cell
- Output fills entire grid

APPROACH: Find non-zero value, use np.full()

CONFIDENCE: high"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Create analyst and run analysis
        analyst = Analyst(model_name="gemini-2.5-flash-lite", use_cache=False)

        task_json = {"train": [{"input": [[1, 0]], "output": [[1, 1]]}]}

        result = analyst.analyze_task(task_json)

        # Verify result
        assert isinstance(result, AnalysisResult)
        assert "Fill grid" in result.pattern_description
        assert len(result.key_observations) == 2
        assert "np.full()" in result.suggested_approach
        assert result.confidence == "high"

        # Verify API was called
        mock_model.generate_content.assert_called_once()

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.analyst.genai")
    def test_analyze_task_with_temperature_setting(self, mock_genai, mock_get_api_key):
        """Test that analyze_task uses correct temperature."""
        mock_get_api_key.return_value = "test-api-key"

        mock_response = Mock()
        mock_response.text = """PATTERN: Test

OBSERVATIONS:
- Test

APPROACH: Test

CONFIDENCE: high"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        analyst = Analyst(
            model_name="gemini-2.5-flash-lite", temperature=0.2, use_cache=False
        )

        task_json = {"train": [{"input": [[1]], "output": [[2]]}]}

        analyst.analyze_task(task_json)

        # Verify temperature was passed to API
        call_kwargs = mock_model.generate_content.call_args[1]
        assert "generation_config" in call_kwargs
        assert call_kwargs["generation_config"]["temperature"] == 0.2

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.analyst.genai")
    def test_analyze_task_with_cache_disabled(self, mock_genai, mock_get_api_key):
        """Test that analyze_task respects use_cache=False."""
        mock_get_api_key.return_value = "test-api-key"

        mock_response = Mock()
        mock_response.text = """PATTERN: Test
OBSERVATIONS:
- Test
APPROACH: Test
CONFIDENCE: high"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        analyst = Analyst(use_cache=False)
        task_json = {"train": [{"input": [[1]], "output": [[2]]}]}

        # Call twice
        analyst.analyze_task(task_json)
        analyst.analyze_task(task_json)

        # Should call API twice (no caching)
        assert mock_model.generate_content.call_count == 2


class TestAnalystEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.analyst.genai")
    def test_analyze_task_with_single_training_example(
        self, mock_genai, mock_get_api_key
    ):
        """Test analysis with only one training example."""
        mock_get_api_key.return_value = "test-api-key"

        mock_response = Mock()
        mock_response.text = """PATTERN: Limited data

OBSERVATIONS:
- Only one example provided

APPROACH: Best guess approach

CONFIDENCE: low"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        analyst = Analyst(use_cache=False)
        task_json = {"train": [{"input": [[1]], "output": [[2]]}]}

        result = analyst.analyze_task(task_json)

        assert result.confidence == "low"

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.analyst.genai")
    def test_analyze_task_with_large_grids(self, mock_genai, mock_get_api_key):
        """Test analysis with large grid inputs."""
        mock_get_api_key.return_value = "test-api-key"

        mock_response = Mock()
        mock_response.text = """PATTERN: Large grid transformation

OBSERVATIONS:
- Grids are 30x30

APPROACH: Handle large arrays efficiently

CONFIDENCE: medium"""

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        analyst = Analyst(use_cache=False)

        # Create 30x30 grid
        large_grid = np.zeros((30, 30), dtype=int).tolist()
        task_json = {"train": [{"input": large_grid, "output": large_grid}]}

        result = analyst.analyze_task(task_json)

        assert result is not None
        assert "Large grid" in result.pattern_description

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.analyst.genai")
    def test_analyze_task_handles_api_timeout(self, mock_genai, mock_get_api_key):
        """Test that analyze_task handles API timeouts gracefully."""
        mock_get_api_key.return_value = "test-api-key"

        mock_model = Mock()
        mock_model.generate_content.side_effect = TimeoutError("API timeout")
        mock_genai.GenerativeModel.return_value = mock_model

        analyst = Analyst(use_cache=False)
        task_json = {"train": [{"input": [[1]], "output": [[2]]}]}

        with pytest.raises(TimeoutError):
            analyst.analyze_task(task_json)

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.analyst.genai")
    def test_analyze_task_with_incomplete_llm_response(
        self, mock_genai, mock_get_api_key
    ):
        """Test handling of incomplete LLM responses."""
        mock_get_api_key.return_value = "test-api-key"

        mock_response = Mock()
        mock_response.text = """PATTERN: Incomplete analysis
OBSERVATIONS:"""  # Missing required sections

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        analyst = Analyst(use_cache=False)
        task_json = {"train": [{"input": [[1]], "output": [[2]]}]}

        result = analyst.analyze_task(task_json)

        # Should handle gracefully with defaults
        assert result.pattern_description == "Incomplete analysis"
        assert len(result.key_observations) == 0
        assert result.suggested_approach == ""
        assert result.confidence == ""


class TestAnalystConfiguration:
    """Tests for Analyst configuration options."""

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    def test_analyst_default_configuration(self, mock_get_api_key):
        """Test Analyst with default configuration."""
        mock_get_api_key.return_value = "test-api-key"
        analyst = Analyst()

        assert analyst.model_name == "gemini-2.5-flash-lite"
        assert analyst.temperature == 0.3
        assert analyst.use_cache is True

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    def test_analyst_custom_model(self, mock_get_api_key):
        """Test Analyst with custom model."""
        mock_get_api_key.return_value = "test-api-key"
        analyst = Analyst(model_name="gemini-1.5-pro")

        assert analyst.model_name == "gemini-1.5-pro"

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    def test_analyst_custom_temperature(self, mock_get_api_key):
        """Test Analyst with custom temperature."""
        mock_get_api_key.return_value = "test-api-key"
        analyst = Analyst(temperature=0.7)

        assert analyst.temperature == 0.7

    @patch("arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    def test_analyst_cache_configuration(self, mock_get_api_key):
        """Test Analyst with caching enabled and disabled."""
        mock_get_api_key.return_value = "test-api-key"
        analyst_cached = Analyst(use_cache=True)
        analyst_no_cache = Analyst(use_cache=False)

        assert analyst_cached.use_cache is True
        assert analyst_no_cache.use_cache is False
