"""
Training example augmentation for Active Inference.

This module implements data augmentation techniques to generate diverse
variations of ARC training examples while preserving the transformation rules.

Augmentation techniques:
- Rotations: 90°, 180°, 270°
- Flips: horizontal, vertical
- Color permutations: swap colors while preserving pattern

The goal is to give LLMs more "experience" with task patterns by providing
30+ examples instead of just 3, improving code generation accuracy.

IMPORTANT - Semantic Preservation Limitation:
Geometric transformations (rotation, flip) may not preserve task semantics for
all ARC tasks. For example, if a task involves directional concepts like "move
up" or "find the top object", rotating the grid 90° changes the meaning from
"up" to "right". Similarly, tasks involving left/right asymmetry may be broken
by horizontal flips. Color permutations are generally safer but may still fail
for tasks that assign semantic meaning to specific colors (e.g., "red means
obstacle"). Use augmentation judiciously and monitor for accuracy degradation
on specific task types.
"""

import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Constants
MAX_COLOR_PERMUTATIONS = 5  # Number of random color permutations to generate


def augment_examples(
    task_data: dict[str, Any], num_variations: int = 10, seed: int | None = None
) -> list[dict[str, Any]]:
    """
    Generate augmented variations of training examples.

    Args:
        task_data: ARC task dict with "train" key containing training examples
        num_variations: Target number of total examples per original example
        seed: Random seed for reproducible augmentation (default: None)

    Returns:
        List of augmented training examples (input/output dicts)

    Raises:
        ValueError: If task_data is invalid or missing required keys

    Example:
        >>> task = {"train": [{"input": [[0,1]], "output": [[1,0]]}]}
        >>> augmented = augment_examples(task, num_variations=5)
        >>> len(augmented)
        5
    """
    # Validate input
    if "train" not in task_data:
        raise ValueError("Task data must have a 'train' key")

    train_examples = task_data["train"]

    if not train_examples:
        raise ValueError("Task data must have at least one training example")

    # Validate each example
    for ex in train_examples:
        if "input" not in ex or "output" not in ex:
            raise ValueError("Each example must have 'input' and 'output' keys")

    # If num_variations is 1, return original examples (cast for type checker)
    if num_variations == 1:
        return list(train_examples)

    augmented = []

    for example in train_examples:
        # Add original example first
        augmented.append(example)

        # Calculate how many variations to generate per example
        variations_per_example = num_variations - 1

        # Generate augmented variations
        example_augmentations = _generate_variations(
            example, variations_per_example, seed
        )
        augmented.extend(example_augmentations)

    # Return exactly num_variations * len(train_examples) examples
    target_count = num_variations * len(train_examples)
    return augmented[:target_count]


def _generate_variations(
    example: dict[str, Any], count: int, seed: int | None = None
) -> list[dict[str, Any]]:
    """
    Generate variations of a single example.

    Variations are returned in deterministic order for reproducibility:
    1. Rotations (90°, 180°, 270°) - 3 variations
    2. Flips (horizontal, vertical) - 2 variations
    3. Color permutations - up to MAX_COLOR_PERMUTATIONS (5) variations

    This means requesting count=9 will return the first 9 transformations in order.
    If you want all color permutations, use count >= 10 (3 + 2 + 5).

    Args:
        example: Single training example with input/output
        count: Number of variations to generate
        seed: Random seed for reproducible color permutations

    Returns:
        List of augmented variations (deterministic order, sliced to count)
    """
    all_variations = []

    input_grid = np.array(example["input"], dtype=np.int64)
    output_grid = np.array(example["output"], dtype=np.int64)

    # Generate all transformation variations
    # Rotations: 90°, 180°, 270°
    for k in [1, 2, 3]:
        try:
            aug_input = np.rot90(input_grid, k=k)
            aug_output = np.rot90(output_grid, k=k)
            all_variations.append(
                {
                    "input": aug_input.tolist(),
                    "output": aug_output.tolist(),
                }
            )
        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"Rotation {k * 90}° failed: {e}")

    # Horizontal flip
    try:
        aug_input = np.fliplr(input_grid)
        aug_output = np.fliplr(output_grid)
        all_variations.append(
            {
                "input": aug_input.tolist(),
                "output": aug_output.tolist(),
            }
        )
    except (ValueError, TypeError, IndexError) as e:
        logger.warning(f"Horizontal flip failed: {e}")

    # Vertical flip
    try:
        aug_input = np.flipud(input_grid)
        aug_output = np.flipud(output_grid)
        all_variations.append(
            {
                "input": aug_input.tolist(),
                "output": aug_output.tolist(),
            }
        )
    except (ValueError, TypeError, IndexError) as e:
        logger.warning(f"Vertical flip failed: {e}")

    # Color permutations
    color_perms = generate_color_permutations(limit=MAX_COLOR_PERMUTATIONS, seed=seed)
    for perm in color_perms:
        try:
            aug_input = _apply_color_map_numpy(input_grid, perm)
            aug_output = _apply_color_map_numpy(output_grid, perm)
            all_variations.append(
                {
                    "input": aug_input.tolist(),
                    "output": aug_output.tolist(),
                }
            )
        except (ValueError, TypeError, IndexError, KeyError) as e:
            logger.warning(f"Color permutation failed: {e}")

    # Return requested count (deterministic order for testing)
    # Note: In production, shuffle could be added for diversity
    return all_variations[:count]


def generate_color_permutations(
    limit: int = 3, seed: int | None = None, min_swaps: int = 5
) -> list[dict[int, int]]:
    """
    Generate diverse random color permutations for ARC grids.

    ARC has 10 colors (0-9). This generates random bijective mappings with
    guaranteed diversity (minimum number of color swaps) to avoid near-identity
    permutations that add little augmentation value.

    Args:
        limit: Number of permutations to generate
        seed: Random seed for reproducibility (default: None)
        min_swaps: Minimum number of colors that must change (default: 5)

    Returns:
        List of color permutation dictionaries (old_color -> new_color)

    Example:
        >>> perms = generate_color_permutations(limit=2, seed=42)
        >>> len(perms)
        2
        >>> all(len(p) == 10 for p in perms)
        True
        >>> # Each permutation should have at least min_swaps colors changed
        >>> all(sum(1 for i in range(10) if p[i] != i) >= 5 for p in perms)
        True
    """
    # Create Random instance for reproducibility
    # noqa: S311 - Not used for cryptographic purposes
    rng = random.Random(seed) if seed is not None else random.Random()  # noqa: S311

    permutations: list[dict[int, int]] = []
    max_attempts = limit * 10  # Prevent infinite loop if min_swaps too high

    attempts = 0
    while len(permutations) < limit and attempts < max_attempts:
        attempts += 1

        # Create a random permutation of colors 0-9
        colors = list(range(10))
        shuffled = colors.copy()
        rng.shuffle(shuffled)

        # Ensure minimum diversity (at least min_swaps colors changed)
        swaps = sum(1 for i in range(10) if colors[i] != shuffled[i])
        if swaps < min_swaps:
            continue  # Regenerate - not diverse enough

        # Create mapping
        perm = dict(zip(colors, shuffled, strict=True))
        permutations.append(perm)

    return permutations


def apply_color_map(
    grid: list[list[int]], permutation: dict[int, int]
) -> list[list[int]]:
    """
    Apply color permutation to a grid.

    Args:
        grid: 2D grid as list of lists
        permutation: Mapping from old color to new color

    Returns:
        Grid with colors swapped according to permutation

    Example:
        >>> grid = [[0, 1], [2, 3]]
        >>> perm = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        >>> apply_color_map(grid, perm)
        [[1, 0], [3, 2]]
    """
    return [[permutation[cell] for cell in row] for row in grid]


def _apply_color_map_numpy(grid: np.ndarray, permutation: dict[int, int]) -> np.ndarray:
    """
    Apply color permutation to a numpy array using vectorized lookup table.

    This optimized implementation uses numpy indexing (O(n)) instead of looping
    over colors (O(10*n)), providing ~5-10x speedup for typical ARC grids.

    Args:
        grid: 2D grid as numpy array
        permutation: Mapping from old color to new color

    Returns:
        Grid with colors swapped according to permutation
    """
    # Create lookup table for vectorized mapping (O(10) setup, O(n) apply)
    # ARC colors are always 0-9, so we can use direct indexing
    lut = np.array([permutation[i] for i in range(10)], dtype=grid.dtype)
    result: np.ndarray = lut[grid]
    return result
