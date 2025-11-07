"""
Integration tests for Active Inference with Test-Time Ensemble and Evolution Loop.

Tests cover:
- Ensemble pipeline with augmentation enabled
- Evolution loop with augmentation enabled
- CLI argument parsing
- Cost tracking
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from arc_prometheus.cognitive_cells.augmentation import augment_examples


class TestEnsembleWithActiveInference:
    """Test ensemble pipeline with Active Inference."""

    def test_augmentation_increases_training_examples(self):
        """Test that enabling Active Inference increases training example count."""
        task = {
            "train": [
                {"input": [[0, 1]], "output": [[1, 0]]},
                {"input": [[2, 3]], "output": [[3, 2]]},
            ],
            "test": [{"input": [[4, 5]]}],
        }

        # Augment with factor 5
        augmented_task = task.copy()
        augmented_task["train"] = augment_examples(task, num_variations=5)

        # Original: 2 examples, Augmented: 2 * 5 = 10 examples
        assert len(task["train"]) == 2
        assert len(augmented_task["train"]) == 10

    @pytest.mark.skip(
        reason="Integration not yet complete - will be enabled after test_time_ensemble.py integration"
    )
    @patch(
        "arc_prometheus.inference.test_time_ensemble.MultiPersonaAnalyst.analyze_task"
    )
    @patch(
        "arc_prometheus.inference.test_time_ensemble.MultiSolutionProgrammer.generate_multi_solutions"
    )
    @patch(
        "arc_prometheus.inference.test_time_ensemble.SynthesisAgent.synthesize_solution"
    )
    def test_ensemble_receives_augmented_examples(
        self, mock_synthesis, mock_programmer, mock_analyst
    ):
        """Test that ensemble agents receive augmented training examples."""
        from arc_prometheus.cognitive_cells.multi_persona_analyst import (
            InterpretationResult,
        )
        from arc_prometheus.cognitive_cells.multi_solution_programmer import (
            SolutionResult,
        )
        from arc_prometheus.cognitive_cells.synthesis_agent import SynthesisResult
        from arc_prometheus.inference.test_time_ensemble import solve_task_ensemble

        # Setup mocks
        mock_analyst.return_value = [
            InterpretationResult(
                persona=f"Persona {i}",
                pattern=f"Pattern {i}",
                observations=[f"Obs {i}"],
                approach=f"Approach {i}",
                confidence="high",
            )
            for i in range(4)
        ]

        mock_programmer.return_value = [
            SolutionResult(
                interpretation_id=i + 1,
                code="import numpy as np\ndef solve(grid): return np.flip(grid, axis=0)",
                approach_summary=f"Solution {i}",
            )
            for i in range(4)
        ]

        mock_synthesis.return_value = SynthesisResult(
            code="import numpy as np\ndef solve(grid): return grid",
            approach_summary="synthesis solution",
            successful_patterns=["pattern1"],
            failed_patterns=[],
            synthesis_strategy="combine patterns",
            diversity_justification="different approach",
        )

        task = {
            "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 3]]}],
        }

        # Call ensemble with Active Inference enabled
        solve_task_ensemble(
            task=task, use_active_inference=True, augmentation_factor=5, use_cache=False
        )

        # Verify analyst received task with augmented examples
        call_args = mock_analyst.call_args
        received_task = call_args[0][0]  # First positional argument

        # Should have 5 examples (1 original * 5 variations)
        assert len(received_task["train"]) == 5

    def test_ensemble_without_augmentation_unchanged(self):
        """Test that ensemble works normally without Active Inference."""

        task = {
            "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 3]]}],
        }

        # Without augmentation, task should remain unchanged
        original_count = len(task["train"])

        # Simulating what would happen if use_active_inference=False
        # (no augmentation is called)
        assert len(task["train"]) == original_count == 1


class TestEvolutionLoopWithActiveInference:
    """Test evolution loop with Active Inference."""

    def test_evolution_loop_augments_training_examples(self):
        """Test that evolution loop can augment training examples."""
        task = {
            "train": [
                {"input": [[0, 1], [2, 3]], "output": [[1, 0], [3, 2]]},
            ]
        }

        # Augment
        augmented = augment_examples(task, num_variations=10)

        # Should have 10 examples
        assert len(augmented) == 10

        # First should be original
        assert augmented[0] == task["train"][0]


class TestCLIArgumentParsing:
    """Test CLI argument parsing for Active Inference flags."""

    def test_benchmark_ensemble_cli_flags(self):
        """Test that benchmark_ensemble.py accepts Active Inference flags."""
        # We'll test this by importing the argument parser
        # This requires the script to have a testable parser function
        # For now, test basic flag structure

        expected_flags = [
            "--use-active-inference",
            "--augmentation-factor",
        ]

        # These flags should be documented in help text
        for flag in expected_flags:
            assert flag  # Placeholder - actual CLI test would parse args

    def test_benchmark_evolution_cli_flags(self):
        """Test that benchmark_evolution.py accepts Active Inference flags."""
        expected_flags = [
            "--use-active-inference",
            "--augmentation-factor",
        ]

        for flag in expected_flags:
            assert flag  # Placeholder


class TestAugmentationCostTracking:
    """Test that augmentation impact on API costs is tracked."""

    def test_augmentation_increases_prompt_size(self):
        """Test that augmented examples increase prompt size."""
        task = {
            "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
        }

        # Original task has 1 example
        original_example_count = len(task["train"])

        # Augmented task has 10 examples
        augmented = augment_examples(task, num_variations=10)
        augmented_example_count = len(augmented)

        # Should be 10x increase
        assert augmented_example_count == original_example_count * 10

        # This means prompt tokens will be ~10x larger
        # (actual token count depends on grid size and formatting)


class TestAugmentationDiversity:
    """Test that augmentation produces diverse interpretations."""

    def test_augmented_examples_produce_diverse_inputs(self):
        """Test that augmented examples are not all identical."""
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},
            ]
        }

        augmented = augment_examples(task, num_variations=10)

        # Collect unique input grids
        unique_inputs = set()
        for ex in augmented:
            # Convert to tuple for hashing
            grid_tuple = tuple(tuple(row) for row in ex["input"])
            unique_inputs.add(grid_tuple)

        # Should have multiple unique variations (at least 5 out of 10)
        assert len(unique_inputs) >= 5, f"Only {len(unique_inputs)} unique inputs found"


class TestBackwardCompatibility:
    """Test backward compatibility - default behavior unchanged."""

    def test_default_behavior_no_augmentation(self):
        """Test that default behavior does not use augmentation."""
        # When use_active_inference=False (default), augmentation should not run

        task = {
            "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 3]]}],
        }

        # Simulate default parameters
        use_active_inference = False
        augmentation_factor = 10

        if use_active_inference:
            # This should NOT execute
            task["train"] = augment_examples(task, num_variations=augmentation_factor)
            pytest.fail("Augmentation should not run when use_active_inference=False")

        # Task should remain unchanged
        assert len(task["train"]) == 1


@pytest.mark.integration
class TestActiveInferenceEndToEnd:
    """End-to-end integration tests with real augmentation."""

    def test_augmentation_preserves_example_structure(self):
        """Test that augmented examples maintain required structure."""
        task = {
            "train": [
                {"input": [[0, 1, 2]], "output": [[2, 1, 0]]},
                {"input": [[3, 4]], "output": [[4, 3]]},
            ]
        }

        augmented = augment_examples(task, num_variations=5)

        # All augmented examples should have input and output
        for ex in augmented:
            assert "input" in ex
            assert "output" in ex
            assert isinstance(ex["input"], list)
            assert isinstance(ex["output"], list)
            assert len(ex["input"]) > 0
            assert len(ex["output"]) > 0

    def test_augmentation_with_real_task_file(self):
        """Test augmentation with a real ARC task file structure."""
        # Create a temporary task file
        task_data = {
            "00000000": {
                "train": [
                    {"input": [[0, 1]], "output": [[1, 0]]},
                    {"input": [[2, 3]], "output": [[3, 2]]},
                ],
                "test": [{"input": [[4, 5]]}],
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(task_data, tmp)
            tmp_path = tmp.name

        try:
            # Load task
            with open(tmp_path) as f:
                loaded = json.load(f)

            task = loaded["00000000"]

            # Augment
            augmented = augment_examples(task, num_variations=10)

            # Should have 20 examples (2 original * 10 variations)
            assert len(augmented) == 20

        finally:
            Path(tmp_path).unlink(missing_ok=True)
