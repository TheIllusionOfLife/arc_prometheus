"""
Unit tests for training example augmentation (Active Inference).

Tests cover:
- Basic augmentation functionality
- Rotations (90°, 180°, 270°)
- Flips (horizontal, vertical)
- Color permutations
- Edge cases and error handling
"""

import numpy as np
import pytest

from arc_prometheus.cognitive_cells.augmentation import (
    apply_color_map,
    augment_examples,
    generate_color_permutations,
)


class TestBasicAugmentation:
    """Test basic augmentation functionality."""

    def test_augment_examples_increases_count(self):
        """Test that augmentation increases example count."""
        task = {
            "train": [
                {"input": [[0, 1], [2, 3]], "output": [[1, 0], [3, 2]]},
                {"input": [[4, 5]], "output": [[5, 4]]},
            ]
        }

        augmented = augment_examples(task, num_variations=5)

        # 2 original examples * 5 variations = 10 total
        assert len(augmented) == 10
        assert all(isinstance(ex, dict) for ex in augmented)
        assert all("input" in ex and "output" in ex for ex in augmented)

    def test_augment_single_example(self):
        """Test augmentation with single training example."""
        task = {"train": [{"input": [[0, 1]], "output": [[1, 0]]}]}

        augmented = augment_examples(task, num_variations=3)

        assert len(augmented) == 3
        # Original example should be included
        assert augmented[0] == task["train"][0]

    def test_augmentation_factor_zero(self):
        """Test that augmentation_factor=1 returns only original examples."""
        task = {
            "train": [
                {"input": [[0, 1]], "output": [[1, 0]]},
                {"input": [[2, 3]], "output": [[3, 2]]},
            ]
        }

        augmented = augment_examples(task, num_variations=1)

        # Should return original examples only
        assert len(augmented) == 2
        assert augmented == task["train"]

    def test_empty_train_examples(self):
        """Test error handling for empty training set."""
        task = {"train": []}

        with pytest.raises(ValueError, match="must have at least one training example"):
            augment_examples(task)

    def test_missing_train_key(self):
        """Test error handling for missing 'train' key."""
        task = {"test": [{"input": [[0]], "output": [[1]]}]}

        with pytest.raises(ValueError, match="must have a 'train' key"):
            augment_examples(task)


class TestRotations:
    """Test rotation augmentations."""

    def test_rotation_90(self):
        """Test 90-degree rotation."""
        task = {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[5, 6], [7, 8]],
                }
            ]
        }

        # Request all variations (10) to ensure all rotations are included
        augmented = augment_examples(task, num_variations=10, seed=42)

        # Find rotated version (should have rotated input)
        # 90° rotation of [[1,2],[3,4]] = [[3,1],[4,2]]
        found_rotation = False
        for ex in augmented:
            if ex["input"] == [[3, 1], [4, 2]]:
                found_rotation = True
                # Output should also be rotated
                assert ex["output"] == [[7, 5], [8, 6]]
                break

        assert found_rotation, "90-degree rotation not found in augmented examples"

    def test_rotation_180(self):
        """Test 180-degree rotation."""
        task = {"train": [{"input": [[1, 2]], "output": [[3, 4]]}]}

        # Request all variations to ensure all rotations are included
        augmented = augment_examples(task, num_variations=10, seed=42)

        # 180° rotation of [[1,2]] = [[2,1]] (flipped both ways)
        found_rotation = False
        for ex in augmented:
            # 180° is equivalent to flip both horizontally and vertically
            input_array = np.array(ex["input"])
            if np.array_equal(input_array, [[2, 1]]):
                found_rotation = True
                assert ex["output"] == [[4, 3]]
                break

        assert found_rotation, "180-degree rotation not found"

    def test_rotation_270(self):
        """Test 270-degree rotation."""
        task = {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[5, 6], [7, 8]],
                }
            ]
        }

        # Request all variations to ensure all rotations are included
        augmented = augment_examples(task, num_variations=10, seed=42)

        # 270° rotation of [[1,2],[3,4]] = [[2,4],[1,3]]
        found_rotation = False
        for ex in augmented:
            if ex["input"] == [[2, 4], [1, 3]]:
                found_rotation = True
                assert ex["output"] == [[6, 8], [5, 7]]
                break

        assert found_rotation, "270-degree rotation not found"


class TestFlips:
    """Test flip augmentations."""

    def test_flip_horizontal(self):
        """Test horizontal flip."""
        task = {"train": [{"input": [[1, 2, 3]], "output": [[4, 5, 6]]}]}

        # Request all variations to ensure all flips are included
        augmented = augment_examples(task, num_variations=10, seed=42)

        # Horizontal flip: [[1,2,3]] → [[3,2,1]]
        found_flip = False
        for ex in augmented:
            if ex["input"] == [[3, 2, 1]]:
                found_flip = True
                assert ex["output"] == [[6, 5, 4]]
                break

        assert found_flip, "Horizontal flip not found"

    def test_flip_vertical(self):
        """Test vertical flip."""
        task = {
            "train": [
                {
                    "input": [[1, 2], [3, 4], [5, 6]],
                    "output": [[7, 8], [9, 10], [11, 12]],
                }
            ]
        }

        augmented = augment_examples(task, num_variations=10)

        # Vertical flip: reverse row order
        found_flip = False
        for ex in augmented:
            if ex["input"] == [[5, 6], [3, 4], [1, 2]]:
                found_flip = True
                assert ex["output"] == [[11, 12], [9, 10], [7, 8]]
                break

        assert found_flip, "Vertical flip not found"


class TestColorPermutations:
    """Test color permutation augmentations."""

    def test_generate_color_permutations(self):
        """Test color permutation generation."""
        perms = generate_color_permutations(limit=3)

        assert len(perms) == 3
        assert all(isinstance(p, dict) for p in perms)

        # Each permutation should map colors 0-9
        for perm in perms:
            assert len(perm) == 10  # ARC has 10 colors (0-9)
            assert set(perm.keys()) == set(range(10))
            assert set(perm.values()) == set(range(10))  # Bijective mapping

    def test_apply_color_map(self):
        """Test color map application."""
        grid = [[0, 1, 2], [3, 4, 5]]
        perm = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}

        mapped = apply_color_map(grid, perm)

        assert mapped == [[5, 6, 7], [8, 9, 0]]

    def test_color_permutation_in_augmentation(self):
        """Test that color permutations are included in augmented examples."""
        task = {"train": [{"input": [[0, 1], [2, 3]], "output": [[1, 0], [3, 2]]}]}

        augmented = augment_examples(task, num_variations=10)

        # Should have some examples with different colors
        original_colors = {0, 1, 2, 3}
        found_color_change = False

        for ex in augmented[1:]:  # Skip first (original)
            ex_colors = set()
            for row in ex["input"]:
                ex_colors.update(row)

            # If colors are different but count is same, likely a permutation
            if ex_colors != original_colors and len(ex_colors) == len(original_colors):
                found_color_change = True
                break

        assert found_color_change, "No color permutations found in augmented examples"


class TestJSONSerialization:
    """Test JSON serialization compatibility."""

    def test_output_is_json_serializable(self):
        """Test that augmented examples are JSON-serializable."""
        import json

        task = {
            "train": [
                {"input": [[0, 1], [2, 3]], "output": [[1, 0], [3, 2]]},
            ]
        }

        augmented = augment_examples(task, num_variations=3)

        # Should be JSON-serializable (lists, not numpy arrays)
        try:
            json.dumps(augmented)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Augmented examples not JSON-serializable: {e}")

    def test_grid_elements_are_lists(self):
        """Test that grids are lists, not numpy arrays."""
        task = {"train": [{"input": [[1, 2]], "output": [[3, 4]]}]}

        augmented = augment_examples(task, num_variations=2)

        for ex in augmented:
            assert isinstance(ex["input"], list)
            assert isinstance(ex["output"], list)
            assert all(isinstance(row, list) for row in ex["input"])
            assert all(isinstance(row, list) for row in ex["output"])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_cell_grid(self):
        """Test augmentation with 1x1 grid."""
        task = {"train": [{"input": [[5]], "output": [[7]]}]}

        augmented = augment_examples(task, num_variations=2)

        # 1x1 grid should still work (rotations/flips don't change it)
        assert len(augmented) >= 1
        assert all(ex["input"] == [[5]] or ex["input"][0][0] != 5 for ex in augmented)

    def test_large_grid(self):
        """Test augmentation with 30x30 grid."""
        large_grid = [list(range(30)) for _ in range(30)]
        task = {"train": [{"input": large_grid, "output": large_grid}]}

        augmented = augment_examples(task, num_variations=2)

        assert len(augmented) >= 1
        # Check that large grids are handled without errors
        for ex in augmented:
            assert len(ex["input"]) == 30
            assert len(ex["input"][0]) == 30

    def test_non_square_grid(self):
        """Test augmentation with non-square grid."""
        task = {
            "train": [
                {
                    "input": [[1, 2, 3], [4, 5, 6]],  # 2x3 grid
                    "output": [[7, 8, 9], [10, 11, 12]],
                }
            ]
        }

        augmented = augment_examples(task, num_variations=3)

        # Non-square grids should work
        assert len(augmented) >= 1
        # Check that dimensions are preserved or transposed correctly
        for ex in augmented:
            rows = len(ex["input"])
            cols = len(ex["input"][0])
            # Either same dimensions or transposed
            assert (rows == 2 and cols == 3) or (rows == 3 and cols == 2)

    def test_high_augmentation_factor(self):
        """Test with high augmentation factor."""
        task = {"train": [{"input": [[0]], "output": [[1]]}]}

        augmented = augment_examples(task, num_variations=20)

        # Should return up to 20 variations
        assert 1 <= len(augmented) <= 20

    def test_malformed_example_missing_input(self):
        """Test error handling for malformed example (missing input)."""
        task = {"train": [{"output": [[1, 2]]}]}

        with pytest.raises(ValueError, match="must have 'input' and 'output' keys"):
            augment_examples(task)

    def test_malformed_example_missing_output(self):
        """Test error handling for malformed example (missing output)."""
        task = {"train": [{"input": [[1, 2]]}]}

        with pytest.raises(ValueError, match="must have 'input' and 'output' keys"):
            augment_examples(task)


class TestDiversityAndCount:
    """Test augmentation diversity and count guarantees."""

    def test_augmentation_produces_diverse_examples(self):
        """Test that augmented examples are diverse (not all identical)."""
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},
            ]
        }

        augmented = augment_examples(task, num_variations=5)

        # Should have multiple unique examples
        unique_inputs = set()
        for ex in augmented:
            # Convert to tuple for hashing
            unique_inputs.add(str(ex["input"]))

        assert len(unique_inputs) >= 3, "Augmented examples not diverse enough"

    def test_original_example_included(self):
        """Test that original example is always included."""
        task = {"train": [{"input": [[1, 2]], "output": [[3, 4]]}]}

        augmented = augment_examples(task, num_variations=3)

        # First example should be original
        assert augmented[0] == task["train"][0]


class TestSeedReproducibility:
    """Test seed parameter for reproducible augmentation."""

    def test_same_seed_produces_identical_augmentations(self):
        """Test that same seed produces identical augmentations across runs."""
        task = {"train": [{"input": [[0, 1], [2, 3]], "output": [[1, 0], [3, 2]]}]}

        # Run twice with same seed
        aug1 = augment_examples(task, num_variations=10, seed=42)
        aug2 = augment_examples(task, num_variations=10, seed=42)

        # Should produce identical results
        assert len(aug1) == len(aug2)
        for ex1, ex2 in zip(aug1, aug2, strict=True):
            assert ex1["input"] == ex2["input"]
            assert ex1["output"] == ex2["output"]

    def test_different_seeds_produce_different_augmentations(self):
        """Test that different seeds produce different color permutations."""
        task = {"train": [{"input": [[0, 1], [2, 3]], "output": [[1, 0], [3, 2]]}]}

        # Run with different seeds
        aug1 = augment_examples(task, num_variations=10, seed=42)
        aug2 = augment_examples(task, num_variations=10, seed=123)

        # Should have same count and structure
        assert len(aug1) == len(aug2)

        # But color permutations should differ (check last few examples which are color perms)
        # Note: First 6 examples are geometric transforms (deterministic), last 4 are color perms
        color_perm_start = 6  # After original + 3 rotations + 2 flips
        found_difference = False
        for i in range(color_perm_start, len(aug1)):
            if aug1[i]["input"] != aug2[i]["input"]:
                found_difference = True
                break

        assert found_difference, (
            "Different seeds should produce different color permutations"
        )

    def test_color_permutation_diversity_with_seed(self):
        """Test that color permutations with seed have minimum diversity."""
        # Generate permutations with seed
        perms = generate_color_permutations(limit=5, seed=42, min_swaps=5)

        assert len(perms) == 5

        # Each permutation should have at least 5 colors changed
        for perm in perms:
            swaps = sum(1 for i in range(10) if perm[i] != i)
            assert swaps >= 5, f"Permutation only has {swaps} swaps, expected >= 5"
