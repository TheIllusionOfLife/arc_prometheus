"""Tests for prepare_evaluation_data script.

This module tests the preprocessing script that merges ARC evaluation challenges
with their solutions to create a unified format compatible with the training data.
"""

import json
import tempfile
from pathlib import Path

import pytest


def test_merge_evaluation_data():
    """Test basic merging of evaluation challenges and solutions."""
    # Sample evaluation challenges (no test outputs)
    challenges = {
        "task1": {
            "train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}],
            "test": [{"input": [[0, 0], [0, 0]]}],
        },
        "task2": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        },
    }

    # Sample evaluation solutions (test outputs by task ID)
    solutions = {
        "task1": [[[1, 1], [1, 1]]],  # One test output
        "task2": [[[4]]],  # One test output
    }

    # Write temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        challenges_path = Path(tmpdir) / "challenges.json"
        solutions_path = Path(tmpdir) / "solutions.json"
        output_path = Path(tmpdir) / "merged.json"

        with open(challenges_path, "w") as f:
            json.dump(challenges, f)
        with open(solutions_path, "w") as f:
            json.dump(solutions, f)

        # Import and run merge function
        from scripts.prepare_evaluation_data import merge_evaluation_data

        merge_evaluation_data(
            str(challenges_path), str(solutions_path), str(output_path)
        )

        # Verify merged output
        with open(output_path) as f:
            merged = json.load(f)

        # Check structure
        assert "task1" in merged
        assert "task2" in merged

        # Check task1 test output merged
        assert len(merged["task1"]["test"]) == 1
        assert "output" in merged["task1"]["test"][0]
        assert merged["task1"]["test"][0]["output"] == [[1, 1], [1, 1]]

        # Check task2 test output merged
        assert len(merged["task2"]["test"]) == 1
        assert "output" in merged["task2"]["test"][0]
        assert merged["task2"]["test"][0]["output"] == [[4]]


def test_merge_preserves_train_data():
    """Test that merging preserves all training data unchanged."""
    challenges = {
        "task1": {
            "train": [
                {"input": [[0]], "output": [[1]]},
                {"input": [[2]], "output": [[3]]},
            ],
            "test": [{"input": [[4]]}],
        }
    }

    solutions = {"task1": [[[5]]]}

    with tempfile.TemporaryDirectory() as tmpdir:
        challenges_path = Path(tmpdir) / "challenges.json"
        solutions_path = Path(tmpdir) / "solutions.json"
        output_path = Path(tmpdir) / "merged.json"

        with open(challenges_path, "w") as f:
            json.dump(challenges, f)
        with open(solutions_path, "w") as f:
            json.dump(solutions, f)

        from scripts.prepare_evaluation_data import merge_evaluation_data

        merge_evaluation_data(
            str(challenges_path), str(solutions_path), str(output_path)
        )

        with open(output_path) as f:
            merged = json.load(f)

        # Train data should be identical
        assert merged["task1"]["train"] == challenges["task1"]["train"]
        assert len(merged["task1"]["train"]) == 2


def test_merge_handles_multiple_test_inputs():
    """Test merging tasks with multiple test inputs (ARC allows 1-2)."""
    challenges = {
        "task1": {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]]}, {"input": [[3]]}],
        }
    }

    solutions = {
        "task1": [[[4]], [[5]]]  # Two test outputs
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        challenges_path = Path(tmpdir) / "challenges.json"
        solutions_path = Path(tmpdir) / "solutions.json"
        output_path = Path(tmpdir) / "merged.json"

        with open(challenges_path, "w") as f:
            json.dump(challenges, f)
        with open(solutions_path, "w") as f:
            json.dump(solutions, f)

        from scripts.prepare_evaluation_data import merge_evaluation_data

        merge_evaluation_data(
            str(challenges_path), str(solutions_path), str(output_path)
        )

        with open(output_path) as f:
            merged = json.load(f)

        # Both test outputs should be merged
        assert len(merged["task1"]["test"]) == 2
        assert merged["task1"]["test"][0]["output"] == [[4]]
        assert merged["task1"]["test"][1]["output"] == [[5]]


def test_merge_validates_count_mismatch():
    """Test that merging errors if test input/output counts differ."""
    challenges = {
        "task1": {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [
                {"input": [[2]]},
                {"input": [[3]]},  # 2 test inputs
            ],
        }
    }

    solutions = {
        "task1": [[[4]]]  # Only 1 test output - MISMATCH
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        challenges_path = Path(tmpdir) / "challenges.json"
        solutions_path = Path(tmpdir) / "solutions.json"
        output_path = Path(tmpdir) / "merged.json"

        with open(challenges_path, "w") as f:
            json.dump(challenges, f)
        with open(solutions_path, "w") as f:
            json.dump(solutions, f)

        from scripts.prepare_evaluation_data import merge_evaluation_data

        # Should raise ValueError with clear message
        with pytest.raises(ValueError, match="task1.*2 test inputs.*1 solutions"):
            merge_evaluation_data(
                str(challenges_path), str(solutions_path), str(output_path)
            )


def test_merge_validates_missing_task_ids():
    """Test that merging errors if task in challenges but not in solutions."""
    challenges = {
        "task1": {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]]}],
        },
        "task2": {  # Missing from solutions
            "train": [{"input": [[3]], "output": [[4]]}],
            "test": [{"input": [[5]]}],
        },
    }

    solutions = {
        "task1": [[[6]]]
        # task2 missing - MISMATCH
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        challenges_path = Path(tmpdir) / "challenges.json"
        solutions_path = Path(tmpdir) / "solutions.json"
        output_path = Path(tmpdir) / "merged.json"

        with open(challenges_path, "w") as f:
            json.dump(challenges, f)
        with open(solutions_path, "w") as f:
            json.dump(solutions, f)

        from scripts.prepare_evaluation_data import merge_evaluation_data

        # Should raise ValueError with clear message
        with pytest.raises(ValueError, match="task2.*not found in solutions"):
            merge_evaluation_data(
                str(challenges_path), str(solutions_path), str(output_path)
            )
