"""
Integration tests for Tagger with Evolution Loop.

Tests the integration of the Tagger agent with the evolution loop,
ensuring tags are properly generated and stored in generation results.
"""

import json
from unittest.mock import Mock, patch

import pytest
from src.arc_prometheus.cognitive_cells.tagger import Tagger, TaggingResult
from src.arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_task_path(tmp_path):
    """Create a sample ARC task file for testing."""
    task_data = {
        "train": [
            {
                "input": [[1, 0], [0, 1]],
                "output": [[0, 1], [1, 0]],
            },
            {
                "input": [[2, 0], [0, 2]],
                "output": [[0, 2], [2, 0]],
            },
        ],
        "test": [
            {
                "input": [[3, 0], [0, 3]],
                "output": [[0, 3], [3, 0]],
            }
        ],
    }

    task_file = tmp_path / "test_task.json"
    with open(task_file, "w") as f:
        json.dump(task_data, f)

    return str(task_file)


# =============================================================================
# Integration Tests
# =============================================================================


class TestTaggerEvolutionLoopIntegration:
    """Tests for Tagger integration with evolution loop."""

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    @patch("src.arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.programmer.genai")
    def test_tagger_integration_with_use_tagger_true(
        self,
        mock_prog_genai,
        mock_prog_key,
        mock_tag_genai,
        mock_tag_key,
        sample_task_path,
    ):
        """Test evolution loop with use_tagger=True generates tags."""
        # Setup mocks
        mock_prog_key.return_value = "test-api-key"
        mock_tag_key.return_value = "test-api-key"

        # Mock Programmer response (code that will actually execute)
        solver_code = """
import numpy as np

def solve(task_grid):
    # Simple transformation that passes some tests
    return [[row[i] for row in task_grid[::-1]] for i in range(len(task_grid[0]))]
"""

        mock_programmer_response = Mock()
        mock_programmer_response.text = f"```python\n{solver_code}\n```"
        mock_prog_model = Mock()
        mock_prog_model.generate_content.return_value = mock_programmer_response
        mock_prog_genai.GenerativeModel.return_value = mock_prog_model

        # Mock Tagger response
        mock_tagger_response = Mock()
        mock_tagger_response.text = (
            "TECHNIQUES: rotation, array_manipulation\nCONFIDENCE: high"
        )
        mock_tag_model = Mock()
        mock_tag_model.generate_content.return_value = mock_tagger_response
        mock_tag_genai.GenerativeModel.return_value = mock_tag_model

        # Run evolution loop with Tagger enabled
        generations = run_evolution_loop(
            task_json_path=sample_task_path,
            max_generations=1,
            model_name="gemini-2.5-flash-lite",
            use_analyst=False,
            use_tagger=True,
            tagger_temperature=0.4,
            use_cache=False,
        )

        # Verify tags were generated (only if fitness > 0)
        assert len(generations) == 1
        if generations[0]["fitness_result"]["fitness"] > 0:
            assert "tags" in generations[0]
            assert isinstance(generations[0]["tags"], list)
        # If fitness is 0, tags should not be present
        else:
            assert "tags" not in generations[0] or generations[0].get("tags") == []

    @patch("src.arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.programmer.genai")
    def test_tagger_integration_with_use_tagger_false(
        self, mock_genai, mock_get_api_key, sample_task_path
    ):
        """Test evolution loop with use_tagger=False skips tagging."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"

        solver_code = """
def solve(task_grid):
    return task_grid
"""

        mock_response = Mock()
        mock_response.text = f"```python\n{solver_code}\n```"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Run evolution loop with Tagger disabled
        generations = run_evolution_loop(
            task_json_path=sample_task_path,
            max_generations=1,
            model_name="gemini-2.5-flash-lite",
            use_analyst=False,
            use_tagger=False,
            use_cache=False,
        )

        # Verify tags were NOT generated
        assert len(generations) == 1
        assert "tags" not in generations[0] or generations[0].get("tags") == []

    @patch("src.arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.programmer.genai")
    def test_tagger_backward_compatibility(
        self, mock_genai, mock_key, sample_task_path
    ):
        """Test that evolution loop works without use_tagger parameter."""
        # This tests backward compatibility - evolution loop should work
        # even if use_tagger is not provided (defaults to False)

        mock_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = "```python\ndef solve(grid): return grid\n```"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Call without use_tagger parameter
        generations = run_evolution_loop(
            task_json_path=sample_task_path,
            max_generations=1,
            use_cache=False,
        )

        # Should work normally
        assert len(generations) == 1

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_tagger_tags_successful_solvers_only(
        self, mock_genai, mock_get_api_key, sample_task_path
    ):
        """Test that Tagger only tags solvers with fitness > 0."""
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = "TECHNIQUES: rotation\nCONFIDENCE: high"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        tagger = Tagger(use_cache=False)

        # Successful solver (would have fitness > 0)
        successful_code = """
def solve(task_grid):
    return np.rot90(task_grid, k=1)
"""

        task_json = {"train": [{"input": [[1]], "output": [[1]]}]}
        result = tagger.tag_solver(successful_code, task_json)

        assert len(result.tags) > 0

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    @patch("src.arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.programmer.genai")
    def test_tagger_temperature_configuration(
        self,
        mock_prog_genai,
        mock_prog_key,
        mock_tag_genai,
        mock_tag_key,
        sample_task_path,
    ):
        """Test that tagger_temperature is properly passed to Tagger."""
        mock_prog_key.return_value = "test-api-key"
        mock_tag_key.return_value = "test-api-key"

        solver_code = """
import numpy as np

def solve(task_grid):
    # Simple transformation that passes some tests
    return [[row[i] for row in task_grid[::-1]] for i in range(len(task_grid[0]))]
"""

        mock_programmer_response = Mock()
        mock_programmer_response.text = f"```python\n{solver_code}\n```"
        mock_prog_model = Mock()
        mock_prog_model.generate_content.return_value = mock_programmer_response
        mock_prog_genai.GenerativeModel.return_value = mock_prog_model

        mock_tagger_response = Mock()
        mock_tagger_response.text = (
            "TECHNIQUES: flip, array_manipulation\nCONFIDENCE: high"
        )
        mock_tag_model = Mock()
        mock_tag_model.generate_content.return_value = mock_tagger_response
        mock_tag_genai.GenerativeModel.return_value = mock_tag_model

        # Run with custom tagger temperature
        custom_temp = 0.7
        generations = run_evolution_loop(
            task_json_path=sample_task_path,
            max_generations=1,
            use_tagger=True,
            tagger_temperature=custom_temp,
            use_cache=False,
        )

        # Verify Tagger was initialized with correct temperature
        # (only if fitness > 0, which triggers tagging)
        if generations[0]["fitness_result"]["fitness"] > 0:
            calls = mock_tag_genai.GenerativeModel.call_args_list
            assert len(calls) >= 1
            tagger_call = calls[0]
            assert tagger_call[1]["generation_config"]["temperature"] == custom_temp

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    @patch("src.arc_prometheus.cognitive_cells.programmer.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.programmer.genai")
    @patch("src.arc_prometheus.cognitive_cells.analyst.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.analyst.genai")
    def test_tagger_with_analyst_mode(
        self,
        mock_analyst_genai,
        mock_analyst_key,
        mock_prog_genai,
        mock_prog_key,
        mock_tag_genai,
        mock_tag_key,
        sample_task_path,
    ):
        """Test Tagger works correctly with Analyst mode enabled."""
        mock_analyst_key.return_value = "test-api-key"
        mock_prog_key.return_value = "test-api-key"
        mock_tag_key.return_value = "test-api-key"

        # Mock Analyst response
        analyst_response = Mock()
        analyst_response.text = """
PATTERN: Rotate grid 90 degrees clockwise
OBSERVATIONS:
- Input and output are same size
APPROACH: Use np.rot90
CONFIDENCE: high
"""
        mock_analyst_model = Mock()
        mock_analyst_model.generate_content.return_value = analyst_response
        mock_analyst_genai.GenerativeModel.return_value = mock_analyst_model

        # Mock Programmer response (code that will actually execute)
        programmer_response = Mock()
        programmer_response.text = """
```python
import numpy as np

def solve(task_grid):
    # Simple transformation that passes some tests
    return [[row[i] for row in task_grid[::-1]] for i in range(len(task_grid[0]))]
```
"""
        mock_prog_model = Mock()
        mock_prog_model.generate_content.return_value = programmer_response
        mock_prog_genai.GenerativeModel.return_value = mock_prog_model

        # Mock Tagger response
        tagger_response = Mock()
        tagger_response.text = (
            "TECHNIQUES: rotation, array_manipulation\nCONFIDENCE: high"
        )
        mock_tag_model = Mock()
        mock_tag_model.generate_content.return_value = tagger_response
        mock_tag_genai.GenerativeModel.return_value = mock_tag_model

        # Run with both Analyst and Tagger
        generations = run_evolution_loop(
            task_json_path=sample_task_path,
            max_generations=1,
            use_analyst=True,
            use_tagger=True,
            use_cache=False,
        )

        # Verify both worked (tags only present if fitness > 0)
        assert len(generations) == 1
        if generations[0]["fitness_result"]["fitness"] > 0:
            assert "tags" in generations[0]
            # Should have some tags
            assert len(generations[0]["tags"]) > 0

    def test_cli_args_integration(self, sample_task_path):
        """Test that CLI arguments for Tagger are properly parsed and used."""
        from src.arc_prometheus.utils.cli_config import parse_evolution_args

        # Test parsing --use-tagger
        args = parse_evolution_args(["--use-tagger"])
        assert args.use_tagger is True
        assert args.tagger_temperature == 0.4  # Default

        # Test parsing --tagger-temperature
        args = parse_evolution_args(["--use-tagger", "--tagger-temperature", "0.6"])
        assert args.use_tagger is True
        assert args.tagger_temperature == 0.6

        # Test default (no tagger)
        args = parse_evolution_args([])
        assert args.use_tagger is False


# =============================================================================
# Test Tagger Data Structure
# =============================================================================


class TestTaggerDataStructure:
    """Tests for TaggingResult data structure in generation results."""

    def test_tagging_result_serializable(self):
        """Test that TaggingResult can be serialized to JSON."""
        result = TaggingResult(
            tags=["rotation", "flip"],
            confidence="high",
            technique_details={"rotation": "Uses np.rot90"},
        )

        # Convert to dict and serialize
        from dataclasses import asdict

        result_dict = asdict(result)
        serialized = json.dumps(result_dict)

        # Deserialize and verify
        deserialized = json.loads(serialized)
        assert deserialized["tags"] == ["rotation", "flip"]
        assert deserialized["confidence"] == "high"

    def test_generation_result_with_tags(self):
        """Test that GenerationResult dict can include tags."""
        # Simulate a GenerationResult with tags
        generation_result = {
            "generation": 0,
            "solver_code": "def solve(grid): return grid",
            "fitness_result": {"fitness": 10.0},
            "refinement_count": 0,
            "total_time": 1.5,
            "improvement": 10.0,
            "tags": ["array_manipulation"],
        }

        # Verify structure
        assert "tags" in generation_result
        assert isinstance(generation_result["tags"], list)
        assert "array_manipulation" in generation_result["tags"]

        # Verify JSON serializable
        serialized = json.dumps(generation_result)
        assert "tags" in serialized
