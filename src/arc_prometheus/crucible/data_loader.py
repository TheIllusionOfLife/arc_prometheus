"""ARC task data loading and visualization.

Provides functions to load ARC tasks from JSON files and display grids.
"""

import json
from pathlib import Path

import numpy as np


def load_task(
    filepath: str | Path, task_id: str | None = None
) -> dict[str, list[dict[str, np.ndarray]]]:
    """Load ARC task from JSON file.

    Args:
        filepath: Path to ARC task JSON file (single task or collection)
        task_id: Optional task ID to extract from a collection file.
                If provided, filepath should be a collection (e.g., arc-agi_training_challenges.json).
                If not provided, filepath should be a single task file.

    Returns:
        Dictionary with structure:
        {
            "train": [{"input": np.ndarray, "output": np.ndarray}, ...],
            "test": [{"input": np.ndarray, "output": np.ndarray (optional)}, ...]
        }

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON structure is invalid, missing required keys, or task_id not found
    """
    filepath_obj = Path(filepath)

    if not filepath_obj.exists():
        raise FileNotFoundError(f"ARC task file not found: {filepath_obj}")

    try:
        with open(filepath_obj) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {filepath_obj}: {e}") from e

    # If task_id provided, extract from collection
    if task_id is not None:
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected collection format (dict of tasks), got {type(data)}"
            )
        if task_id not in data:
            raise ValueError(f"Task ID '{task_id}' not found in {filepath_obj}")
        data = data[task_id]

    # Validate structure
    if not isinstance(data, dict):
        raise ValueError(f"Task file must contain a JSON object, got {type(data)}")

    if "train" not in data or "test" not in data:
        raise ValueError("Task file must contain 'train' and 'test' keys")

    if not isinstance(data["train"], list) or not isinstance(data["test"], list):
        raise ValueError("'train' and 'test' must be lists")

    if len(data["train"]) == 0:
        raise ValueError("Task must have at least one train example")

    # Convert lists to numpy arrays
    result: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "test": []}

    for example in data["train"]:
        if "input" not in example or "output" not in example:
            raise ValueError("Each train example must have 'input' and 'output' keys")

        result["train"].append(
            {
                "input": np.array(example["input"], dtype=int),
                "output": np.array(example["output"], dtype=int),
            }
        )

    for example in data["test"]:
        if "input" not in example:
            raise ValueError("Each test example must have 'input' key")

        test_item = {"input": np.array(example["input"], dtype=int)}
        # Test examples may or may not have output (for evaluation vs prediction)
        if "output" in example:
            test_item["output"] = np.array(example["output"], dtype=int)

        result["test"].append(test_item)

    return result


def print_grid(grid: np.ndarray, label: str = "") -> None:
    """Print grid to console with visual formatting.

    Args:
        grid: 2D numpy array of integers (0-9) representing colors
        label: Optional label to print before grid
    """
    if label:
        print(f"\n{label}")
        print("=" * max(len(label), grid.shape[1] * 2))

    # Color mapping for visual distinction (using terminal colors)
    # Colors 0-9 map to different terminal color codes
    color_map = {
        0: "\033[40m",  # Black background
        1: "\033[44m",  # Blue background
        2: "\033[41m",  # Red background
        3: "\033[42m",  # Green background
        4: "\033[43m",  # Yellow background
        5: "\033[45m",  # Magenta background
        6: "\033[46m",  # Cyan background
        7: "\033[47m",  # White background
        8: "\033[100m",  # Bright black background
        9: "\033[103m",  # Bright yellow background
    }
    reset = "\033[0m"

    # Print grid with colors
    for row in grid:
        row_str = ""
        for cell in row:
            color = color_map.get(cell, "")
            row_str += f"{color}{cell:2d}{reset}"
        print(row_str)

    print()  # Empty line after grid
