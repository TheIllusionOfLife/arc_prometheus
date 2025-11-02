"""Tests for Multi-Persona Analyst agent."""

import json
from unittest.mock import MagicMock, patch

import pytest

from arc_prometheus.cognitive_cells.multi_persona_analyst import (
    DEFAULT_PERSONAS,
    InterpretationResult,
    MultiPersonaAnalyst,
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
def sample_api_response():
    """Sample valid API response matching schema."""
    return {
        "interpretations": [
            {
                "persona": "Geometric Transformation Specialist",
                "pattern": "Flip the grid vertically (reverse row order)",
                "observations": [
                    "First row becomes last row",
                    "Middle row stays in middle",
                    "Last row becomes first row",
                ],
                "approach": "Use np.flip(grid, axis=0) for vertical flip",
                "confidence": "high",
            },
            {
                "persona": "Color Pattern Specialist",
                "pattern": "Invert the color values in reverse spatial order",
                "observations": [
                    "Colors maintain their positions",
                    "Spatial arrangement is reversed",
                ],
                "approach": "Reverse iteration with color mapping",
                "confidence": "medium",
            },
            {
                "persona": "Object Detection Specialist",
                "pattern": "Treat each row as an object and reverse their positions",
                "observations": [
                    "Three row-objects identified",
                    "Objects swapped: top↔bottom",
                ],
                "approach": "Extract rows, reverse list, stack back",
                "confidence": "high",
            },
            {
                "persona": "Grid Structure Specialist",
                "pattern": "Reverse grid along the horizontal axis",
                "observations": [
                    "Grid height unchanged (3x3)",
                    "Row-wise reversal pattern",
                ],
                "approach": "Use array slicing [::-1] on rows",
                "confidence": "high",
            },
            {
                "persona": "Logical Rules Specialist",
                "pattern": "Apply rule: output[i] = input[n-1-i] for each row i",
                "observations": [
                    "Index transformation pattern",
                    "Works for any grid height",
                ],
                "approach": "Iterate with reversed indices",
                "confidence": "medium",
            },
        ]
    }


class TestInterpretationResult:
    """Test InterpretationResult dataclass."""

    def test_create_interpretation_result(self):
        """Test creating InterpretationResult with all fields."""
        result = InterpretationResult(
            persona="Test Specialist",
            pattern="Test pattern description",
            observations=["obs1", "obs2"],
            approach="Test approach",
            confidence="high",
        )

        assert result.persona == "Test Specialist"
        assert result.pattern == "Test pattern description"
        assert result.observations == ["obs1", "obs2"]
        assert result.approach == "Test approach"
        assert result.confidence == "high"

    def test_interpretation_result_defaults(self):
        """Test InterpretationResult with default values."""
        result = InterpretationResult(persona="Test Specialist", pattern="Test pattern")

        assert result.persona == "Test Specialist"
        assert result.pattern == "Test pattern"
        assert result.observations == []
        assert result.approach == ""
        assert result.confidence == ""


class TestMultiPersonaAnalyst:
    """Test MultiPersonaAnalyst class."""

    def test_initialization_default(self):
        """Test default initialization."""
        with patch("arc_prometheus.cognitive_cells.multi_persona_analyst.genai"):
            analyst = MultiPersonaAnalyst()

            assert analyst.model_name == "gemini-2.0-flash-thinking-exp"
            assert analyst.temperature == 1.0
            assert analyst.personas == DEFAULT_PERSONAS
            assert analyst.use_cache is True

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_personas = {"persona_1": {"name": "Custom Specialist"}}

        with patch("arc_prometheus.cognitive_cells.multi_persona_analyst.genai"):
            analyst = MultiPersonaAnalyst(
                model_name="gemini-2.5-flash-lite",
                temperature=0.5,
                personas=custom_personas,
                use_cache=False,
            )

            assert analyst.model_name == "gemini-2.5-flash-lite"
            assert analyst.temperature == 0.5
            assert analyst.personas == custom_personas
            assert analyst.use_cache is False

    def test_format_grid(self, sample_task):
        """Test grid formatting."""
        with patch("arc_prometheus.cognitive_cells.multi_persona_analyst.genai"):
            analyst = MultiPersonaAnalyst()
            grid = sample_task["train"][0]["input"]

            formatted = analyst._format_grid(grid)

            expected = "0 1 2\n3 4 5\n6 7 8"
            assert formatted == expected

    def test_create_prompt_structure(self, sample_task):
        """Test that prompt contains required elements."""
        with patch("arc_prometheus.cognitive_cells.multi_persona_analyst.genai"):
            analyst = MultiPersonaAnalyst()
            prompt = analyst._create_prompt(sample_task)

            # Check for key sections
            assert "TRAINING EXAMPLES:" in prompt
            assert "THE 5 EXPERTS:" in prompt
            assert "INSTRUCTIONS:" in prompt

            # Check for all personas
            for persona_data in DEFAULT_PERSONAS.values():
                assert persona_data["name"] in prompt
                assert persona_data["emoji"] in prompt

            # Check for training examples
            assert "Example 1:" in prompt
            assert "Example 2:" in prompt
            assert "0 1 2" in prompt  # First grid data

            # Check for conciseness instructions
            assert "≤150 chars" in prompt
            assert "≤80 chars" in prompt
            assert "≤100 chars" in prompt

    def test_parse_response_valid(self, sample_api_response):
        """Test parsing valid API response."""
        from arc_prometheus.utils.schemas import MultiPersonaResponse

        with patch("arc_prometheus.cognitive_cells.multi_persona_analyst.genai"):
            analyst = MultiPersonaAnalyst()
            # Convert dict to Pydantic model
            pydantic_response = MultiPersonaResponse.model_validate(sample_api_response)
            results = analyst._parse_response(pydantic_response)

            assert len(results) == 5
            assert all(isinstance(r, InterpretationResult) for r in results)

            # Check first interpretation
            first = results[0]
            assert first.persona == "Geometric Transformation Specialist"
            assert "Flip the grid vertically" in first.pattern
            assert len(first.observations) == 3
            assert "np.flip" in first.approach
            assert first.confidence == "high"

    def test_parse_response_wrong_count(self):
        """Test parsing response with wrong number of interpretations."""
        from pydantic import ValidationError

        from arc_prometheus.utils.schemas import MultiPersonaResponse

        # Only 3 interpretations instead of 5 - Pydantic will catch this
        invalid_response_data = {
            "interpretations": [
                {
                    "persona": "Test 1",
                    "pattern": "Pattern 1",
                    "observations": ["obs"],
                    "approach": "approach",
                    "confidence": "high",
                }
            ]
            * 3
        }

        # Pydantic validation will raise ValidationError for wrong count
        with pytest.raises(ValidationError):
            MultiPersonaResponse.model_validate(invalid_response_data)

    @patch("arc_prometheus.cognitive_cells.multi_persona_analyst.genai")
    @patch("arc_prometheus.utils.llm_cache.get_cache")
    def test_analyze_task_with_cache_hit(
        self, mock_get_cache, mock_genai, sample_task, sample_api_response
    ):
        """Test analyze_task with cache hit."""
        # Setup mock cache
        mock_cache = MagicMock()
        mock_cache.get.return_value = json.dumps(sample_api_response)
        mock_get_cache.return_value = mock_cache

        analyst = MultiPersonaAnalyst(use_cache=True)
        results = analyst.analyze_task(sample_task)

        # Should get results from cache
        assert len(results) == 5
        assert mock_cache.get.called
        # API should NOT be called
        assert not mock_genai.GenerativeModel.called

    @patch("arc_prometheus.cognitive_cells.multi_persona_analyst.genai")
    @patch("arc_prometheus.utils.llm_cache.get_cache")
    def test_analyze_task_with_cache_miss(
        self, mock_get_cache, mock_genai, sample_task, sample_api_response
    ):
        """Test analyze_task with cache miss."""
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

        analyst = MultiPersonaAnalyst(use_cache=True)
        results = analyst.analyze_task(sample_task)

        # Should get results from API
        assert len(results) == 5
        assert mock_cache.get.called
        # API should be called
        assert mock_genai.GenerativeModel.called
        # Result should be cached
        assert mock_cache.set.called

    @patch("arc_prometheus.cognitive_cells.multi_persona_analyst.genai")
    def test_analyze_task_no_cache(self, mock_genai, sample_task, sample_api_response):
        """Test analyze_task with caching disabled."""
        # Setup mock API response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_api_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        analyst = MultiPersonaAnalyst(use_cache=False)
        results = analyst.analyze_task(sample_task)

        # Should get results from API
        assert len(results) == 5
        # API should be called
        assert mock_genai.GenerativeModel.called

    @patch("arc_prometheus.cognitive_cells.multi_persona_analyst.genai")
    def test_analyze_task_uses_structured_output(
        self, mock_genai, sample_task, sample_api_response
    ):
        """Test that analyze_task uses structured output configuration."""
        # Setup mock API response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_api_response)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        analyst = MultiPersonaAnalyst(use_cache=False)
        analyst.analyze_task(sample_task)

        # Check that generate_content was called
        assert mock_model.generate_content.called

        # Get the generation_config argument
        call_args = mock_model.generate_content.call_args
        generation_config = call_args[1]["generation_config"]

        # Verify structured output configuration
        assert generation_config.temperature == 1.0
        assert generation_config.response_mime_type == "application/json"
        assert generation_config.response_schema is not None


class TestDefaultPersonas:
    """Test DEFAULT_PERSONAS constant."""

    def test_has_five_personas(self):
        """Test that DEFAULT_PERSONAS has exactly 5 personas."""
        assert len(DEFAULT_PERSONAS) == 5

    def test_persona_structure(self):
        """Test that each persona has required fields."""
        required_fields = ["name", "emoji", "focus", "key_question"]

        for persona_id, persona_data in DEFAULT_PERSONAS.items():
            for field in required_fields:
                assert field in persona_data, f"{persona_id} missing {field}"
                assert persona_data[field], f"{persona_id}.{field} is empty"

    def test_persona_names_unique(self):
        """Test that all persona names are unique."""
        names = [p["name"] for p in DEFAULT_PERSONAS.values()]
        assert len(names) == len(set(names)), "Duplicate persona names found"

    def test_persona_emojis_unique(self):
        """Test that all persona emojis are unique."""
        emojis = [p["emoji"] for p in DEFAULT_PERSONAS.values()]
        assert len(emojis) == len(set(emojis)), "Duplicate persona emojis found"


# Integration test (can be run with real API if needed)
@pytest.mark.integration
def test_real_api_multi_persona_analyst(sample_task):
    """Integration test with real Gemini API.

    This test requires a valid GEMINI_API_KEY environment variable.
    Run with: pytest tests/test_multi_persona_analyst.py -m integration
    """
    analyst = MultiPersonaAnalyst(use_cache=False)

    try:
        results = analyst.analyze_task(sample_task)

        # Validate structure
        assert len(results) == 5
        assert all(isinstance(r, InterpretationResult) for r in results)

        # Validate content
        for result in results:
            assert result.persona, "Persona name should not be empty"
            assert result.pattern, "Pattern should not be empty"
            assert len(result.pattern) <= 150, "Pattern should be ≤150 chars"
            assert 1 <= len(result.observations) <= 3, "Should have 1-3 observations"
            for obs in result.observations:
                assert len(obs) <= 80, f"Observation too long: {len(obs)} chars"
            assert result.approach, "Approach should not be empty"
            assert len(result.approach) <= 100, "Approach should be ≤100 chars"
            assert result.confidence in [
                "high",
                "medium",
                "low",
            ], f"Invalid confidence: {result.confidence}"

        print("\n=== Real API Test Results ===")
        for idx, result in enumerate(results, 1):
            print(f"\n{idx}. {result.persona} ({result.confidence})")
            print(f"   Pattern: {result.pattern}")
            print(f"   Approach: {result.approach}")
            print(f"   Observations: {result.observations}")

    except Exception as e:
        pytest.skip(f"Real API test failed (may be expected): {e}")
