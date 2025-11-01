"""
Tests for the Tagger cognitive cell.

The Tagger agent classifies successful solvers by technique (rotation, fill,
symmetry, etc.) to enable technique-based crossover in Phase 3.4.
"""

from dataclasses import asdict
from unittest.mock import Mock, patch

from src.arc_prometheus.cognitive_cells.tagger import (
    TECHNIQUE_TAXONOMY,
    Tagger,
    TaggingResult,
)

# =============================================================================
# Test Class 1: TaggingResult Dataclass Tests (3 tests)
# =============================================================================


class TestTaggingResult:
    """Tests for the TaggingResult dataclass."""

    def test_tagging_result_creation(self):
        """Test TaggingResult can be created with required fields."""
        result = TaggingResult(tags=["rotation", "symmetry"], confidence="high")

        assert result.tags == ["rotation", "symmetry"]
        assert result.confidence == "high"
        assert result.technique_details == {}

    def test_tagging_result_with_details(self):
        """Test TaggingResult with technique_details."""
        result = TaggingResult(
            tags=["rotation", "color_fill"],
            confidence="medium",
            technique_details={
                "rotation": "Uses np.rot90 for 90-degree rotation",
                "color_fill": "Fills regions with specific colors",
            },
        )

        assert len(result.tags) == 2
        assert result.confidence == "medium"
        assert "rotation" in result.technique_details
        assert "color_fill" in result.technique_details

    def test_tagging_result_default_values(self):
        """Test TaggingResult default values."""
        result = TaggingResult(tags=["flip"])

        assert result.tags == ["flip"]
        assert result.confidence == "medium"
        assert result.technique_details == {}

    def test_tagging_result_to_dict(self):
        """Test TaggingResult can be converted to dict."""
        result = TaggingResult(
            tags=["transpose"],
            confidence="low",
            technique_details={"transpose": "Matrix transpose operation"},
        )

        result_dict = asdict(result)
        assert result_dict["tags"] == ["transpose"]
        assert result_dict["confidence"] == "low"
        assert "transpose" in result_dict["technique_details"]


# =============================================================================
# Test Class 2: Static Analysis Tests (8 tests)
# =============================================================================


class TestStaticAnalysis:
    """Tests for static code analysis technique detection."""

    def test_detect_rotation_technique(self):
        """Test detection of rotation technique using np.rot90."""
        code = """
def solve(task_grid: np.ndarray) -> np.ndarray:
    return np.rot90(task_grid)
"""
        tagger = Tagger(use_cache=False)
        tags = tagger._static_analysis(code)

        assert "rotation" in tags

    def test_detect_flip_technique(self):
        """Test detection of flip technique using np.flip."""
        code = """
def solve(task_grid: np.ndarray) -> np.ndarray:
    return np.flip(task_grid, axis=0)
"""
        tagger = Tagger(use_cache=False)
        tags = tagger._static_analysis(code)

        assert "flip" in tags

    def test_detect_transpose_technique(self):
        """Test detection of transpose technique."""
        code = """
def solve(task_grid: np.ndarray) -> np.ndarray:
    return np.transpose(task_grid)
"""
        tagger = Tagger(use_cache=False)
        tags = tagger._static_analysis(code)

        assert "transpose" in tags

    def test_detect_color_fill_technique(self):
        """Test detection of color fill technique."""
        code = """
def solve(task_grid: np.ndarray) -> np.ndarray:
    result = task_grid.copy()
    # Fill regions with color
    result[result == 0] = 5
    return result
"""
        tagger = Tagger(use_cache=False)
        tags = tagger._static_analysis(code)

        assert "color_fill" in tags

    def test_detect_multiple_techniques(self):
        """Test detection of multiple techniques in one solver."""
        code = """
def solve(task_grid: np.ndarray) -> np.ndarray:
    # Rotate the grid
    rotated = np.rot90(task_grid)
    # Then flip it
    flipped = np.flip(rotated, axis=1)
    # Count non-zero elements
    count = len(flipped[flipped != 0])
    return flipped
"""
        tagger = Tagger(use_cache=False)
        tags = tagger._static_analysis(code)

        assert "rotation" in tags
        assert "flip" in tags
        assert "counting" in tags

    def test_detect_counting_technique(self):
        """Test detection of counting technique."""
        code = """
def solve(task_grid: np.ndarray) -> np.ndarray:
    count = len(task_grid[task_grid > 0])
    num_colors = len(np.unique(task_grid))
    return task_grid
"""
        tagger = Tagger(use_cache=False)
        tags = tagger._static_analysis(code)

        assert "counting" in tags

    def test_no_techniques_detected(self):
        """Test when no techniques are detected."""
        code = """
def solve(task_grid):
    return task_grid
"""
        tagger = Tagger(use_cache=False)
        tags = tagger._static_analysis(code)

        # Should detect minimal or no techniques
        # (array_manipulation might be detected from basic operations)
        assert len(tags) <= 1

    def test_complex_nested_calls(self):
        """Test detection in complex code with nested function calls."""
        code = """
def solve(task_grid: np.ndarray) -> np.ndarray:
    # Complex transformation
    result = np.rot90(np.transpose(task_grid), k=2)
    mask = result > 0
    if np.any(mask):
        result[mask] = 7
    return result
"""
        tagger = Tagger(use_cache=False)
        tags = tagger._static_analysis(code)

        assert "rotation" in tags
        assert "transpose" in tags
        assert "conditional_logic" in tags


# =============================================================================
# Test Class 3: LLM Analysis Tests (5 tests)
# =============================================================================


class TestLLMAnalysis:
    """Tests for LLM-based semantic technique analysis."""

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_llm_analysis_basic(self, mock_genai, mock_get_api_key):
        """Test basic LLM analysis with mocked response."""
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = """
TECHNIQUES: rotation, symmetry
CONFIDENCE: high
"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        tagger = Tagger(use_cache=False)
        code = "def solve(grid): return np.rot90(grid)"
        task_json = {"train": [{"input": [[1, 2]], "output": [[3, 4]]}]}

        tags = tagger._llm_analysis(code, task_json)

        assert "rotation" in tags
        assert "symmetry" in tags

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_llm_analysis_temperature_config(self, mock_genai, mock_get_api_key):
        """Test LLM analysis uses configured temperature."""
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = "TECHNIQUES: flip\nCONFIDENCE: medium"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        tagger = Tagger(temperature=0.5, use_cache=False)
        code = "def solve(grid): return grid"
        task_json = {"train": []}

        tagger._llm_analysis(code, task_json)

        # Check that GenerativeModel was called with correct config
        call_args = mock_genai.GenerativeModel.call_args
        assert call_args[1]["generation_config"]["temperature"] == 0.5

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_llm_analysis_with_caching(self, mock_genai, mock_get_api_key):
        """Test LLM analysis caching behavior."""
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = "TECHNIQUES: pattern_copy\nCONFIDENCE: high"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # First call with caching enabled
        tagger = Tagger(use_cache=True)
        code = "def solve(grid): return grid.copy()"
        task_json = {"train": [{"input": [[1]], "output": [[1]]}]}

        tags1 = tagger._llm_analysis(code, task_json)
        tags2 = tagger._llm_analysis(code, task_json)  # Should use cache

        assert tags1 == tags2
        # Note: We can't easily verify cache hits in unit tests,
        # but this tests that caching doesn't break functionality

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_llm_analysis_timeout_handling(self, mock_genai, mock_get_api_key):
        """Test LLM analysis handles timeout gracefully."""
        mock_get_api_key.return_value = "test-api-key"
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Timeout")
        mock_genai.GenerativeModel.return_value = mock_model

        tagger = Tagger(use_cache=False)
        code = "def solve(grid): return grid"
        task_json = {"train": []}

        tags = tagger._llm_analysis(code, task_json)

        # Should return empty list on error, not raise exception
        assert tags == []

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_llm_analysis_filters_invalid_techniques(
        self, mock_genai, mock_get_api_key
    ):
        """Test LLM analysis filters out techniques not in taxonomy."""
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        # Include invalid techniques
        mock_response.text = """
TECHNIQUES: rotation, invalid_technique, magic_transform, flip
CONFIDENCE: medium
"""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        tagger = Tagger(use_cache=False)
        code = "def solve(grid): return grid"
        task_json = {"train": []}

        tags = tagger._llm_analysis(code, task_json)

        # Only valid techniques should be returned
        assert "rotation" in tags
        assert "flip" in tags
        assert "invalid_technique" not in tags
        assert "magic_transform" not in tags


# =============================================================================
# Test Class 4: Tagger Combination Logic Tests (4 tests)
# =============================================================================


class TestTaggerCombination:
    """Tests for combining static and LLM analysis."""

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_combine_static_and_llm_results(self, mock_genai, mock_get_api_key):
        """Test combining static and LLM analysis results."""
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = "TECHNIQUES: symmetry, pattern_copy\nCONFIDENCE: high"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        code = """
def solve(grid):
    rotated = np.rot90(grid)  # Static will catch 'rotation'
    return rotated
"""
        task_json = {"train": [{"input": [[1]], "output": [[1]]}]}

        tagger = Tagger(use_cache=False)
        result = tagger.tag_solver(code, task_json)

        # Should have techniques from both static and LLM
        assert "rotation" in result.tags  # From static
        assert "symmetry" in result.tags  # From LLM
        assert "pattern_copy" in result.tags  # From LLM

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_deduplication_of_techniques(self, mock_genai, mock_get_api_key):
        """Test that duplicate techniques are removed."""
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        # LLM also detects 'rotation' (duplicate)
        mock_response.text = "TECHNIQUES: rotation, flip\nCONFIDENCE: high"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        code = "rotated = np.rot90(grid); flipped = np.flip(grid)"
        task_json = {"train": []}

        tagger = Tagger(use_cache=False)
        result = tagger.tag_solver(code, task_json)

        # Each technique should appear only once
        assert result.tags.count("rotation") == 1
        assert result.tags.count("flip") == 1

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_empty_code_handling(self, mock_genai, mock_get_api_key):
        """Test handling of empty code."""
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = "TECHNIQUES:\nCONFIDENCE: low"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        code = ""
        task_json = {"train": []}

        tagger = Tagger(use_cache=False)
        result = tagger.tag_solver(code, task_json)

        assert result.tags == []
        assert result.confidence == "low"

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_large_code_handling(self, mock_genai, mock_get_api_key):
        """Test handling of large code (>1000 lines)."""
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = "TECHNIQUES: array_manipulation\nCONFIDENCE: medium"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Generate large code
        large_code = "def solve(grid):\n" + "    x = 1\n" * 1000 + "    return grid"
        task_json = {"train": []}

        tagger = Tagger(use_cache=False)
        result = tagger.tag_solver(large_code, task_json)

        # Should handle large code without errors
        assert isinstance(result.tags, list)
        assert "array_manipulation" in result.tags


# =============================================================================
# Test Class 5: Tagger Configuration Tests (3 tests)
# =============================================================================


class TestTaggerConfiguration:
    """Tests for Tagger configuration options."""

    def test_custom_temperature(self):
        """Test Tagger with custom temperature."""
        tagger = Tagger(temperature=0.7, use_cache=False)

        assert tagger.temperature == 0.7

    def test_custom_model(self):
        """Test Tagger with custom model."""
        tagger = Tagger(model_name="gemini-2.0-flash", use_cache=False)

        assert tagger.model_name == "gemini-2.0-flash"

    def test_cache_enable_disable(self):
        """Test Tagger with cache enabled and disabled."""
        tagger_cached = Tagger(use_cache=True)
        tagger_no_cache = Tagger(use_cache=False)

        assert tagger_cached.use_cache is True
        assert tagger_no_cache.use_cache is False


# =============================================================================
# Test Class 6: Tagger Edge Cases Tests (3 tests)
# =============================================================================


class TestTaggerEdgeCases:
    """Tests for Tagger edge cases."""

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_invalid_code_syntax_error(self, mock_genai, mock_get_api_key):
        """Test handling of code with syntax errors."""
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = "TECHNIQUES:\nCONFIDENCE: low"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        code = "def solve(grid: this is invalid syntax"
        task_json = {"train": []}

        tagger = Tagger(use_cache=False)
        result = tagger.tag_solver(code, task_json)

        # Should not crash, just return results
        assert isinstance(result.tags, list)

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_minimal_code(self, mock_genai, mock_get_api_key):
        """Test handling of minimal code (2-3 lines)."""
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = "TECHNIQUES:\nCONFIDENCE: low"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        code = "def solve(grid):\n    return grid"
        task_json = {"train": []}

        tagger = Tagger(use_cache=False)
        result = tagger.tag_solver(code, task_json)

        assert isinstance(result.tags, list)
        assert isinstance(result.confidence, str)

    @patch("src.arc_prometheus.cognitive_cells.tagger.get_gemini_api_key")
    @patch("src.arc_prometheus.cognitive_cells.tagger.genai")
    def test_code_without_numpy(self, mock_genai, mock_get_api_key):
        """Test code that doesn't use numpy."""
        mock_get_api_key.return_value = "test-api-key"
        mock_response = Mock()
        mock_response.text = "TECHNIQUES: conditional_logic\nCONFIDENCE: medium"
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        code = """
def solve(grid):
    if len(grid) > 0:
        return grid
    return [[0]]
"""
        task_json = {"train": []}

        tagger = Tagger(use_cache=False)
        result = tagger.tag_solver(code, task_json)

        # Should still work, using LLM analysis
        assert "conditional_logic" in result.tags


# =============================================================================
# Test Technique Taxonomy Constant
# =============================================================================


def test_technique_taxonomy_exists():
    """Test that TECHNIQUE_TAXONOMY constant exists and is valid."""
    assert isinstance(TECHNIQUE_TAXONOMY, list)
    assert len(TECHNIQUE_TAXONOMY) >= 12  # Should have at least 12 techniques

    expected_techniques = [
        "rotation",
        "flip",
        "transpose",
        "color_fill",
        "pattern_copy",
        "symmetry",
        "grid_partition",
        "object_detection",
        "counting",
        "conditional_logic",
        "array_manipulation",
        "neighborhood_analysis",
    ]

    for technique in expected_techniques:
        assert technique in TECHNIQUE_TAXONOMY
