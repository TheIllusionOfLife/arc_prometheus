"""Tests for evolution loop - Multi-generation solver evolution (Phase 2.3)."""

import json
from unittest.mock import patch

import pytest


class TestEvolutionLoopBasics:
    """Test basic evolution loop functionality."""

    @patch("arc_prometheus.evolutionary_engine.evolution_loop.refine_solver")
    @patch("arc_prometheus.evolutionary_engine.evolution_loop.generate_solver")
    def test_single_generation_improvement(self, mock_generate, mock_refine, tmp_path):
        """Test that evolution improves fitness in single generation."""
        from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop

        # Create simple task: multiply by 2
        task_data = {
            "train": [
                {"input": [[1]], "output": [[2]]},
                {"input": [[2]], "output": [[4]]},
                {"input": [[3]], "output": [[6]]},
            ],
            "test": [{"input": [[4]], "output": [[8]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Mock initial bad code (adds instead of multiplies)
        mock_generate.return_value = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1  # Wrong - should multiply by 2
"""

        # Mock refined good code
        mock_refine.return_value = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid * 2  # Correct
"""

        # Run evolution for 2 generations
        results = run_evolution_loop(str(task_file), max_generations=2, verbose=False)

        # Verify structure
        assert len(results) == 2
        assert results[0]["generation"] == 0
        assert results[1]["generation"] == 1

        # Verify improvement
        fitness_gen0 = results[0]["fitness_result"]["fitness"]
        fitness_gen1 = results[1]["fitness_result"]["fitness"]
        assert fitness_gen1 > fitness_gen0

        # Verify refinement was called
        assert mock_refine.called

    @patch("arc_prometheus.evolutionary_engine.evolution_loop.refine_solver")
    @patch("arc_prometheus.evolutionary_engine.evolution_loop.generate_solver")
    def test_multi_generation_convergence(self, mock_generate, mock_refine, tmp_path):
        """Test fitness improves across multiple generations."""
        from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop

        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[6]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Simulate gradual improvement through refinement
        codes = [
            "def solve(x): return x + 0",  # Gen 0: fitness = 0 (both wrong)
            "def solve(x): return x + 1",  # Gen 1: fitness = 1 (train correct, test wrong)
            "def solve(x): return x * 2",  # Gen 2: fitness = 11 (both correct)
        ]

        mock_generate.return_value = f"""
import numpy as np

{codes[0]}
"""

        # Mock refine returns progressively better code
        refine_results = [
            f"import numpy as np\n\n{codes[1]}",
            f"import numpy as np\n\n{codes[2]}",
        ]
        mock_refine.side_effect = refine_results

        results = run_evolution_loop(str(task_file), max_generations=3, verbose=False)

        assert len(results) == 3

        # Verify monotonic improvement (or at least non-decreasing)
        for i in range(1, len(results)):
            assert (
                results[i]["fitness_result"]["fitness"]
                >= results[i - 1]["fitness_result"]["fitness"]
            )

    @patch("arc_prometheus.evolutionary_engine.evolution_loop.refine_solver")
    @patch("arc_prometheus.evolutionary_engine.evolution_loop.generate_solver")
    def test_early_termination_on_target_fitness(
        self, mock_generate, mock_refine, tmp_path
    ):
        """Test evolution stops when target_fitness is reached."""
        from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop

        task_data = {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]], "output": [[2]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Mock perfect code from start
        mock_generate.return_value = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid.copy()
"""

        # Run with target fitness = 11 (perfect = 1 train + 1 test * 10 = 11)
        results = run_evolution_loop(
            str(task_file), max_generations=5, target_fitness=11, verbose=False
        )

        # Should stop after generation 0 (already perfect)
        assert len(results) == 1
        assert results[0]["fitness_result"]["fitness"] == 11

        # Refiner should NOT be called (already optimal)
        assert not mock_refine.called

    @patch("arc_prometheus.evolutionary_engine.evolution_loop.refine_solver")
    @patch("arc_prometheus.evolutionary_engine.evolution_loop.generate_solver")
    def test_max_generations_limit(self, mock_generate, mock_refine, tmp_path):
        """Test evolution stops at max_generations even if not converged."""
        from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop

        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[6]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Mock code that never improves (refiner fails)
        bad_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 0  # Always wrong (returns wrong values)
"""

        mock_generate.return_value = bad_code
        mock_refine.return_value = bad_code  # Refiner doesn't help

        results = run_evolution_loop(
            str(task_file), max_generations=3, target_fitness=11, verbose=False
        )

        # Should run exactly 3 generations
        assert len(results) == 3

        # Fitness stays at 0 (no improvement)
        for result in results:
            assert result["fitness_result"]["fitness"] == 0

    @patch("arc_prometheus.evolutionary_engine.evolution_loop.refine_solver")
    @patch("arc_prometheus.evolutionary_engine.evolution_loop.generate_solver")
    def test_perfect_initial_solver_no_refinement(
        self, mock_generate, mock_refine, tmp_path
    ):
        """Test that perfect initial solver doesn't trigger refinement."""
        from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop

        task_data = {
            "train": [
                {"input": [[1, 2]], "output": [[1, 2]]},
                {"input": [[3, 4]], "output": [[3, 4]]},
            ],
            "test": [{"input": [[5, 6]], "output": [[5, 6]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Perfect solver
        mock_generate.return_value = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid.copy()
"""

        results = run_evolution_loop(
            str(task_file), max_generations=3, target_fitness=12, verbose=False
        )

        # Should stop at generation 1 (perfect fitness)
        assert len(results) == 1
        assert results[0]["fitness_result"]["fitness"] == 12  # 2*1 + 1*10

        # No refinement needed
        assert not mock_refine.called


class TestEvolutionLoopMetadata:
    """Test generation metadata tracking."""

    @patch("arc_prometheus.evolutionary_engine.evolution_loop.refine_solver")
    @patch("arc_prometheus.evolutionary_engine.evolution_loop.generate_solver")
    def test_generation_metadata_tracking(self, mock_generate, mock_refine, tmp_path):
        """Test that all GenerationResult fields are populated correctly."""
        from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop

        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[6]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        mock_generate.return_value = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
"""

        mock_refine.return_value = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid * 2
"""

        results = run_evolution_loop(str(task_file), max_generations=2, verbose=False)

        # Verify all required fields present
        required_fields = {
            "generation",
            "solver_code",
            "fitness_result",
            "refinement_count",
            "total_time",
            "improvement",
        }

        for result in results:
            assert set(result.keys()) == required_fields

            # Verify types
            assert isinstance(result["generation"], int)
            assert isinstance(result["solver_code"], str)
            assert isinstance(result["fitness_result"], dict)
            assert isinstance(result["refinement_count"], int)
            assert isinstance(result["total_time"], float)
            assert isinstance(result["improvement"], float)

            # Verify fitness_result structure
            assert "fitness" in result["fitness_result"]
            assert "train_correct" in result["fitness_result"]
            assert "test_correct" in result["fitness_result"]

    @patch("arc_prometheus.evolutionary_engine.evolution_loop.refine_solver")
    @patch("arc_prometheus.evolutionary_engine.evolution_loop.generate_solver")
    def test_improvement_calculation(self, mock_generate, mock_refine, tmp_path):
        """Test that improvement deltas are calculated correctly."""
        from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop

        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[2]], "output": [[4]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Fitness progression: 1 → 0 → 11 (actual values after execution)
        codes = [
            "def solve(x): return x + 1",  # fitness = 1 (train correct: 1→2, test wrong: 2→3 expected 4)
            "def solve(x): return x * 1",  # fitness = 0 (train wrong: 1→1 expected 2, test wrong: 2→2 expected 4)
            "def solve(x): return x * 2",  # fitness = 11 (train correct: 1→2, test correct: 2→4)
        ]

        mock_generate.return_value = f"import numpy as np\n\n{codes[0]}"
        mock_refine.side_effect = [
            f"import numpy as np\n\n{codes[1]}",
            f"import numpy as np\n\n{codes[2]}",
        ]

        results = run_evolution_loop(str(task_file), max_generations=3, verbose=False)

        # Gen 0: improvement = 0 (baseline)
        assert results[0]["improvement"] == 0.0

        # Gen 1: improvement = 0 - 1 = -1
        assert results[1]["improvement"] == -1.0

        # Gen 2: improvement = 11 - 0 = 11
        assert results[2]["improvement"] == 11.0

    @patch("arc_prometheus.evolutionary_engine.evolution_loop.refine_solver")
    @patch("arc_prometheus.evolutionary_engine.evolution_loop.generate_solver")
    def test_refinement_count_tracking(self, mock_generate, mock_refine, tmp_path):
        """Test that refinement_count is tracked correctly."""
        from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop

        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[6]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # First generation: bad code
        mock_generate.return_value = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
"""

        # Subsequent generations: improved code
        mock_refine.return_value = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid * 2
"""

        results = run_evolution_loop(
            str(task_file), max_generations=3, target_fitness=11, verbose=False
        )

        # Gen 0: 0 refinements (initial generation)
        assert results[0]["refinement_count"] == 0

        # Gen 1: 1 refinement (refined once, hit target)
        assert results[1]["refinement_count"] == 1

        # Should stop at gen 1 (target reached)
        assert len(results) == 2


class TestEvolutionLoopErrorHandling:
    """Test error handling in evolution loop."""

    @patch("arc_prometheus.evolutionary_engine.evolution_loop.refine_solver")
    @patch("arc_prometheus.evolutionary_engine.evolution_loop.generate_solver")
    def test_error_handling_llm_failure(self, mock_generate, mock_refine, tmp_path):
        """Test graceful handling when LLM fails."""
        from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop

        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[6]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Generate fails
        mock_generate.side_effect = Exception("Gemini API failure")

        # Should raise exception
        with pytest.raises(Exception, match="Gemini API failure"):
            run_evolution_loop(str(task_file), max_generations=2, verbose=False)

    @patch("arc_prometheus.evolutionary_engine.evolution_loop.refine_solver")
    @patch("arc_prometheus.evolutionary_engine.evolution_loop.generate_solver")
    def test_error_handling_refiner_failure(self, mock_generate, mock_refine, tmp_path):
        """Test handling when refiner fails but initial generation works."""
        from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop

        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[6]]}],
        }

        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data))

        # Generate works
        mock_generate.return_value = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
"""

        # Refine fails
        mock_refine.side_effect = Exception("Refiner API failure")

        # Should raise exception when trying to refine
        with pytest.raises(Exception, match="Refiner API failure"):
            run_evolution_loop(str(task_file), max_generations=2, verbose=False)

    @patch("arc_prometheus.evolutionary_engine.evolution_loop.refine_solver")
    @patch("arc_prometheus.evolutionary_engine.evolution_loop.generate_solver")
    def test_handles_invalid_task_file(self, mock_generate, mock_refine, tmp_path):
        """Test handling of invalid task file."""
        from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop

        # Non-existent file
        with pytest.raises(FileNotFoundError):
            run_evolution_loop(
                str(tmp_path / "nonexistent.json"), max_generations=2, verbose=False
            )


class TestEvolutionLoopIntegration:
    """Integration tests with real dataset."""

    @pytest.mark.integration
    def test_real_arc_task_evolution(self, tmp_path):
        """Test evolution loop with actual ARC task and real API.

        This test is marked as integration and will be skipped if:
        - GEMINI_API_KEY is not set in environment
        - Running without --run-integration flag
        """
        from arc_prometheus.evolutionary_engine.evolution_loop import run_evolution_loop
        from arc_prometheus.utils.config import get_gemini_api_key

        # Skip if no API key
        try:
            api_key = get_gemini_api_key()
            if not api_key:
                pytest.skip("GEMINI_API_KEY not configured")
        except ValueError:
            pytest.skip("GEMINI_API_KEY not configured")

        # Create simple synthetic task
        task_data = {
            "train": [
                {"input": [[1, 2]], "output": [[2, 4]]},
                {"input": [[3, 4]], "output": [[6, 8]]},
            ],
            "test": [{"input": [[5, 6]], "output": [[10, 12]]}],
        }

        task_file = tmp_path / "real_task.json"
        task_file.write_text(json.dumps(task_data))

        # Run evolution with real API (limit to 2 generations to save API calls)
        results = run_evolution_loop(
            str(task_file), max_generations=2, target_fitness=12, verbose=True
        )

        # Verify basic structure
        assert len(results) > 0
        assert len(results) <= 2

        # Verify each generation has valid structure
        for result in results:
            assert "generation" in result
            assert "solver_code" in result
            assert "fitness_result" in result
            assert "def solve(" in result["solver_code"]

            # Verify fitness calculation worked
            fitness = result["fitness_result"]
            assert "fitness" in fitness
            assert "train_correct" in fitness
            assert "test_correct" in fitness
            assert fitness["fitness"] >= 0

        # If multiple generations, verify refinement was attempted
        if len(results) > 1:
            assert results[0]["refinement_count"] == 0
            assert results[1]["refinement_count"] == 1
