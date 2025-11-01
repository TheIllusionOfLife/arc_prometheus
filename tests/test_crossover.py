"""
Tests for Crossover Agent - Phase 3.4

Tests LLM-based technique fusion for population-based evolution.
TDD: Write tests first, then implement Crossover class.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from arc_prometheus.cognitive_cells.crossover import (
    Crossover,
    CrossoverResult,
)
from arc_prometheus.evolutionary_engine.solver_library import SolverRecord

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_solver_record_1():
    """Create sample SolverRecord with rotation + flip techniques."""
    return SolverRecord(
        solver_id="solver-rotation-flip",
        task_id="task-001",
        generation=0,
        code_str="""
def solve(grid):
    import numpy as np
    return np.rot90(np.flip(grid, axis=0))
""",
        fitness_score=15.0,
        train_correct=3,
        test_correct=1,
        parent_solver_id=None,
        tags=["rotation", "flip"],
        created_at=datetime.now(UTC).isoformat(),
    )


@pytest.fixture
def sample_solver_record_2():
    """Create sample SolverRecord with color_fill + grid_partition techniques."""
    return SolverRecord(
        solver_id="solver-fill-partition",
        task_id="task-001",
        generation=1,
        code_str="""
def solve(grid):
    import numpy as np
    # Partition and fill
    result = grid.copy()
    result[result == 0] = 3
    return result
""",
        fitness_score=12.0,
        train_correct=2,
        test_correct=1,
        parent_solver_id=None,
        tags=["color_fill", "grid_partition"],
        created_at=datetime.now(UTC).isoformat(),
    )


@pytest.fixture
def sample_task_json():
    """Create sample ARC task JSON."""
    return {
        "train": [
            {
                "input": [[0, 1], [2, 3]],
                "output": [[3, 2], [1, 0]],
            },
            {
                "input": [[1, 1], [0, 0]],
                "output": [[0, 0], [1, 1]],
            },
        ],
        "test": [
            {
                "input": [[2, 2], [3, 3]],
            }
        ],
    }


# =============================================================================
# Test CrossoverResult Dataclass
# =============================================================================


class TestCrossoverResult:
    """Test CrossoverResult dataclass creation and validation."""

    def test_crossover_result_creation(self):
        """Test basic CrossoverResult creation with all required fields."""
        result = CrossoverResult(
            fused_code="def solve(grid): return grid",
            parent_ids=["parent-1", "parent-2"],
            parent_techniques=[["rotation", "flip"], ["color_fill"]],
            compatibility_assessment="Geometric transforms complement region filling",
        )

        assert result.fused_code == "def solve(grid): return grid"
        assert result.parent_ids == ["parent-1", "parent-2"]
        assert result.parent_techniques == [["rotation", "flip"], ["color_fill"]]
        assert (
            result.compatibility_assessment
            == "Geometric transforms complement region filling"
        )
        assert result.confidence == "medium"  # Default

    def test_crossover_result_with_high_confidence(self):
        """Test CrossoverResult with explicit high confidence."""
        result = CrossoverResult(
            fused_code="def solve(grid): return grid * 2",
            parent_ids=["p1", "p2"],
            parent_techniques=[["rotation"], ["flip"]],
            compatibility_assessment="Highly compatible geometric operations",
            confidence="high",
        )

        assert result.confidence == "high"

    def test_crossover_result_with_multiple_parents(self):
        """Test CrossoverResult with 3 parent solvers."""
        result = CrossoverResult(
            fused_code="def solve(grid): return grid",
            parent_ids=["p1", "p2", "p3"],
            parent_techniques=[["rotation"], ["flip"], ["transpose"]],
            compatibility_assessment="All geometric transforms are compatible",
        )

        assert len(result.parent_ids) == 3
        assert len(result.parent_techniques) == 3


# =============================================================================
# Test Crossover Initialization
# =============================================================================


class TestCrossoverInitialization:
    """Test Crossover agent initialization and configuration."""

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_crossover_default_initialization(self, mock_genai, mock_get_api_key):
        """Test Crossover with default parameters."""
        mock_get_api_key.return_value = "fake-api-key"

        crossover = Crossover()

        assert crossover.model_name == "gemini-2.5-flash-lite"
        assert crossover.temperature == 0.5  # Default from config
        assert crossover.use_cache is True
        mock_get_api_key.assert_called_once()
        mock_genai.configure.assert_called_once_with(api_key="fake-api-key")

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_crossover_custom_model(self, mock_genai, mock_get_api_key):
        """Test Crossover with custom model name."""
        mock_get_api_key.return_value = "fake-api-key"

        crossover = Crossover(model_name="gemini-2.0-flash-thinking-exp")

        assert crossover.model_name == "gemini-2.0-flash-thinking-exp"

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_crossover_custom_temperature(self, mock_genai, mock_get_api_key):
        """Test Crossover with custom temperature."""
        mock_get_api_key.return_value = "fake-api-key"

        crossover = Crossover(temperature=0.7)

        assert crossover.temperature == 0.7

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_crossover_cache_disabled(self, mock_genai, mock_get_api_key):
        """Test Crossover with caching disabled."""
        mock_get_api_key.return_value = "fake-api-key"

        crossover = Crossover(use_cache=False)

        assert crossover.use_cache is False


# =============================================================================
# Test Prompt Construction
# =============================================================================


class TestCrossoverPromptConstruction:
    """Test prompt construction for LLM fusion."""

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_prompt_includes_parent_codes(
        self,
        mock_genai,
        mock_get_api_key,
        sample_solver_record_1,
        sample_solver_record_2,
        sample_task_json,
    ):
        """Test prompt includes both parent solver codes."""
        mock_get_api_key.return_value = "fake-api-key"

        crossover = Crossover()
        prompt = crossover._create_fusion_prompt(
            [sample_solver_record_1, sample_solver_record_2], sample_task_json
        )

        assert "np.rot90" in prompt  # From parent 1
        assert "result[result == 0] = 3" in prompt  # From parent 2

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_prompt_includes_parent_techniques(
        self,
        mock_genai,
        mock_get_api_key,
        sample_solver_record_1,
        sample_solver_record_2,
        sample_task_json,
    ):
        """Test prompt includes technique tags from both parents."""
        mock_get_api_key.return_value = "fake-api-key"

        crossover = Crossover()
        prompt = crossover._create_fusion_prompt(
            [sample_solver_record_1, sample_solver_record_2], sample_task_json
        )

        assert "rotation" in prompt
        assert "flip" in prompt
        assert "color_fill" in prompt
        assert "grid_partition" in prompt

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_prompt_includes_fitness_scores(
        self,
        mock_genai,
        mock_get_api_key,
        sample_solver_record_1,
        sample_solver_record_2,
        sample_task_json,
    ):
        """Test prompt includes parent fitness scores."""
        mock_get_api_key.return_value = "fake-api-key"

        crossover = Crossover()
        prompt = crossover._create_fusion_prompt(
            [sample_solver_record_1, sample_solver_record_2], sample_task_json
        )

        assert "15.0" in prompt  # Fitness of parent 1
        assert "12.0" in prompt  # Fitness of parent 2

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_prompt_includes_task_context(
        self,
        mock_genai,
        mock_get_api_key,
        sample_solver_record_1,
        sample_solver_record_2,
        sample_task_json,
    ):
        """Test prompt includes ARC task examples for context."""
        mock_get_api_key.return_value = "fake-api-key"

        crossover = Crossover()
        prompt = crossover._create_fusion_prompt(
            [sample_solver_record_1, sample_solver_record_2], sample_task_json
        )

        # Should include task dimensions
        assert "2x2" in prompt or "2" in prompt  # Grid dimensions

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_prompt_with_analyst_spec(
        self,
        mock_genai,
        mock_get_api_key,
        sample_solver_record_1,
        sample_solver_record_2,
        sample_task_json,
    ):
        """Test prompt includes analyst specification when provided."""
        from arc_prometheus.cognitive_cells.analyst import AnalysisResult

        mock_get_api_key.return_value = "fake-api-key"

        analyst_spec = AnalysisResult(
            pattern_description="Rotate grid and fill background",
            key_observations=["Rotation by 90 degrees", "Background filling"],
            suggested_approach="Combine rotation with region filling",
            confidence="high",
        )

        crossover = Crossover()
        prompt = crossover._create_fusion_prompt(
            [sample_solver_record_1, sample_solver_record_2],
            sample_task_json,
            analyst_spec=analyst_spec,
        )

        assert "Rotate grid and fill background" in prompt
        assert "Combine rotation with region filling" in prompt


# =============================================================================
# Test Code Parsing
# =============================================================================


class TestCrossoverCodeParsing:
    """Test parsing LLM-generated fused code."""

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_parse_code_with_markdown_block(self, mock_genai, mock_get_api_key):
        """Test parsing code wrapped in markdown ```python blocks."""
        mock_get_api_key.return_value = "fake-api-key"

        crossover = Crossover()
        llm_response = """
Here's the fused solver:

```python
def solve(grid):
    import numpy as np
    return np.rot90(grid)
```

This combines rotation from parent 1 with parent 2's approach.
"""
        code = crossover._parse_fused_code(llm_response)

        assert "def solve(grid):" in code
        assert "np.rot90" in code
        assert "```" not in code  # Markdown removed

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_parse_code_without_markdown(self, mock_genai, mock_get_api_key):
        """Test parsing raw code without markdown delimiters."""
        mock_get_api_key.return_value = "fake-api-key"

        crossover = Crossover()
        llm_response = """
def solve(grid):
    import numpy as np
    return grid * 2
"""
        code = crossover._parse_fused_code(llm_response)

        assert "def solve(grid):" in code
        assert "grid * 2" in code

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_parse_code_with_multiple_functions(self, mock_genai, mock_get_api_key):
        """Test parsing when LLM returns multiple functions (extract solve)."""
        mock_get_api_key.return_value = "fake-api-key"

        crossover = Crossover()
        llm_response = """
```python
def helper(x):
    return x + 1

def solve(grid):
    import numpy as np
    return helper(grid)
```
"""
        code = crossover._parse_fused_code(llm_response)

        # Should extract the solve function and its dependencies
        assert "def solve(grid):" in code

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_parse_code_strips_whitespace(self, mock_genai, mock_get_api_key):
        """Test parsing strips leading/trailing whitespace."""
        mock_get_api_key.return_value = "fake-api-key"

        crossover = Crossover()
        llm_response = """

        def solve(grid):
            return grid


"""
        code = crossover._parse_fused_code(llm_response)

        assert code.startswith("def solve(grid):")
        assert not code.startswith("\n")


# =============================================================================
# Test LLM Integration (Mocked)
# =============================================================================


class TestCrossoverLLMIntegration:
    """Test Crossover LLM integration with mocked API calls."""

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_fuse_solvers_basic(
        self,
        mock_genai,
        mock_get_api_key,
        sample_solver_record_1,
        sample_solver_record_2,
        sample_task_json,
    ):
        """Test basic fuse_solvers with mocked LLM response."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = """
COMPATIBILITY: High - geometric transforms complement each other
CONFIDENCE: high

FUSED CODE:
```python
def solve(grid):
    import numpy as np
    rotated = np.rot90(grid)
    flipped = np.flip(rotated, axis=0)
    return flipped
```
"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Execute
        crossover = Crossover()
        result = crossover.fuse_solvers(
            [sample_solver_record_1, sample_solver_record_2], sample_task_json
        )

        # Verify
        assert "def solve(grid):" in result.fused_code
        assert "np.rot90" in result.fused_code
        assert result.parent_ids == ["solver-rotation-flip", "solver-fill-partition"]
        assert result.parent_techniques == [
            ["rotation", "flip"],
            ["color_fill", "grid_partition"],
        ]
        assert "high" in result.compatibility_assessment.lower()
        assert result.confidence == "high"

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_fuse_solvers_with_analyst_context(
        self,
        mock_genai,
        mock_get_api_key,
        sample_solver_record_1,
        sample_solver_record_2,
        sample_task_json,
    ):
        """Test fuse_solvers includes analyst specification in prompt."""
        from arc_prometheus.cognitive_cells.analyst import AnalysisResult

        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = """
COMPATIBILITY: Medium
CONFIDENCE: medium

FUSED CODE:
```python
def solve(grid):
    return grid
```
"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        analyst_spec = AnalysisResult(
            pattern_description="Test pattern",
            key_observations=["Observation 1"],
            suggested_approach="Test approach",
            confidence="high",
        )

        # Execute (disable cache to ensure LLM is called for verification)
        crossover = Crossover(use_cache=False)
        crossover.fuse_solvers(
            [sample_solver_record_1, sample_solver_record_2],
            sample_task_json,
            analyst_spec=analyst_spec,
        )

        # Verify prompt included analyst spec
        call_args = mock_model.generate_content.call_args
        prompt = call_args[0][0]
        assert "Test pattern" in prompt
        assert "Test approach" in prompt

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_fuse_solvers_uses_cache(
        self,
        mock_genai,
        mock_get_api_key,
        sample_solver_record_1,
        sample_solver_record_2,
        sample_task_json,
    ):
        """Test fuse_solvers uses cache when enabled."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = """
COMPATIBILITY: High
CONFIDENCE: high
FUSED CODE:
```python
def solve(grid): return grid
```
"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Execute twice with cache enabled
        crossover = Crossover(use_cache=True)
        result1 = crossover.fuse_solvers(
            [sample_solver_record_1, sample_solver_record_2], sample_task_json
        )
        result2 = crossover.fuse_solvers(
            [sample_solver_record_1, sample_solver_record_2], sample_task_json
        )

        # Second call should use cache (same result)
        assert result1.fused_code == result2.fused_code

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_fuse_solvers_cache_disabled(
        self,
        mock_genai,
        mock_get_api_key,
        sample_solver_record_1,
        sample_solver_record_2,
        sample_task_json,
    ):
        """Test fuse_solvers skips cache when disabled."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = """
COMPATIBILITY: High
CONFIDENCE: high
FUSED CODE:
```python
def solve(grid): return grid
```
"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Execute with cache disabled
        crossover = Crossover(use_cache=False)
        crossover.fuse_solvers(
            [sample_solver_record_1, sample_solver_record_2], sample_task_json
        )

        # Should call LLM directly
        assert mock_model.generate_content.called


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestCrossoverEdgeCases:
    """Test edge cases and error handling."""

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_fuse_solvers_with_identical_techniques(
        self,
        mock_genai,
        mock_get_api_key,
        sample_solver_record_1,
        sample_task_json,
    ):
        """Test fusion when both parents have identical technique tags."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = """
COMPATIBILITY: Low - identical techniques, limited fusion potential
CONFIDENCE: low
FUSED CODE:
```python
def solve(grid): return grid
```
"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Both parents have same tags
        parent2 = SolverRecord(
            solver_id="solver-2",
            task_id="task-001",
            generation=1,
            code_str="def solve(grid): return grid",
            fitness_score=10.0,
            train_correct=2,
            test_correct=1,
            parent_solver_id=None,
            tags=["rotation", "flip"],  # Same as parent 1
            created_at=datetime.now(UTC).isoformat(),
        )

        crossover = Crossover()
        result = crossover.fuse_solvers(
            [sample_solver_record_1, parent2], sample_task_json
        )

        # Should still produce result, but with low confidence
        assert result.confidence == "low"

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_fuse_solvers_with_three_parents(
        self, mock_genai, mock_get_api_key, sample_task_json
    ):
        """Test fusion with 3 parent solvers."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = """
COMPATIBILITY: Medium - multiple techniques can be combined
CONFIDENCE: medium
FUSED CODE:
```python
def solve(grid): return grid
```
"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Create 3 parents
        parents = [
            SolverRecord(
                solver_id=f"solver-{i}",
                task_id="task-001",
                generation=i,
                code_str=f"def solve(grid): return grid * {i}",
                fitness_score=10.0 + i,
                train_correct=2,
                test_correct=1,
                parent_solver_id=None,
                tags=[f"technique-{i}"],
                created_at=datetime.now(UTC).isoformat(),
            )
            for i in range(3)
        ]

        crossover = Crossover()
        result = crossover.fuse_solvers(parents, sample_task_json)

        assert len(result.parent_ids) == 3
        assert len(result.parent_techniques) == 3

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_fuse_solvers_llm_error_handling(
        self,
        mock_genai,
        mock_get_api_key,
        sample_solver_record_1,
        sample_solver_record_2,
        sample_task_json,
    ):
        """Test graceful error handling when LLM fails."""
        # Setup mocks to raise error
        mock_get_api_key.return_value = "test-api-key"
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API timeout")
        mock_genai.GenerativeModel.return_value = mock_model

        # Disable cache to ensure LLM is called (and error is raised)
        crossover = Crossover(use_cache=False)

        # Should raise exception when LLM fails
        with pytest.raises(Exception, match="API timeout"):
            crossover.fuse_solvers(
                [sample_solver_record_1, sample_solver_record_2], sample_task_json
            )

    @patch("arc_prometheus.cognitive_cells.crossover.get_gemini_api_key")
    @patch("arc_prometheus.cognitive_cells.crossover.genai")
    def test_fuse_solvers_with_empty_tags(
        self, mock_genai, mock_get_api_key, sample_task_json
    ):
        """Test fusion when one parent has empty technique tags."""
        # Setup mocks
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = """
COMPATIBILITY: Low - one parent lacks identified techniques
CONFIDENCE: low
FUSED CODE:
```python
def solve(grid): return grid
```
"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        parent1 = SolverRecord(
            solver_id="solver-1",
            task_id="task-001",
            generation=0,
            code_str="def solve(grid): return grid",
            fitness_score=10.0,
            train_correct=2,
            test_correct=1,
            parent_solver_id=None,
            tags=["rotation"],
            created_at=datetime.now(UTC).isoformat(),
        )

        parent2 = SolverRecord(
            solver_id="solver-2",
            task_id="task-001",
            generation=1,
            code_str="def solve(grid): return grid * 2",
            fitness_score=5.0,
            train_correct=1,
            test_correct=0,
            parent_solver_id=None,
            tags=[],  # Empty tags
            created_at=datetime.now(UTC).isoformat(),
        )

        crossover = Crossover()
        result = crossover.fuse_solvers([parent1, parent2], sample_task_json)

        # Should handle empty tags gracefully
        assert result.parent_techniques == [["rotation"], []]


# =============================================================================
# Test Compatibility Assessment
# =============================================================================


class TestCrossoverCompatibility:
    """Test LLM-based compatibility assessment parsing."""

    def test_parse_compatibility_assessment(self):
        """Test parsing compatibility assessment from LLM response."""
        crossover = Crossover()
        llm_response = """
COMPATIBILITY: High - geometric transforms and color operations are complementary
CONFIDENCE: high

FUSED CODE:
```python
def solve(grid): return grid
```
"""
        compatibility, confidence = crossover._parse_assessment(llm_response)

        assert "geometric transforms and color operations" in compatibility.lower()
        assert confidence == "high"

    def test_parse_compatibility_medium_confidence(self):
        """Test parsing medium confidence assessment."""
        crossover = Crossover()
        llm_response = """
COMPATIBILITY: Moderate compatibility
CONFIDENCE: medium

FUSED CODE:
```python
def solve(grid): return grid
```
"""
        compatibility, confidence = crossover._parse_assessment(llm_response)

        assert "moderate" in compatibility.lower()
        assert confidence == "medium"

    def test_parse_compatibility_defaults_to_medium(self):
        """Test parsing defaults to medium confidence if not specified."""
        crossover = Crossover()
        llm_response = """
COMPATIBILITY: Some compatibility exists

FUSED CODE:
```python
def solve(grid): return grid
```
"""
        compatibility, confidence = crossover._parse_assessment(llm_response)

        assert confidence == "medium"  # Default when not specified
