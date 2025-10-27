"""ARC grid evaluation functionality.

Provides functions to compare grids for correctness evaluation.
"""

from typing import cast

import numpy as np


def evaluate_grids(grid_a: np.ndarray, grid_b: np.ndarray) -> bool:
    """Check if two grids are identical.

    Compares both shape and values of two grids to determine
    if they represent the same solution.

    Args:
        grid_a: First grid (2D numpy array)
        grid_b: Second grid (2D numpy array)

    Returns:
        True if grids are identical in shape and values, False otherwise

    Examples:
        >>> grid1 = np.array([[1, 2], [3, 4]])
        >>> grid2 = np.array([[1, 2], [3, 4]])
        >>> evaluate_grids(grid1, grid2)
        True

        >>> grid3 = np.array([[1, 2], [3, 5]])
        >>> evaluate_grids(grid1, grid3)
        False
    """
    # Check if shapes match
    if grid_a.shape != grid_b.shape:
        return False

    # Check if all values match
    return cast(bool, np.array_equal(grid_a, grid_b))
