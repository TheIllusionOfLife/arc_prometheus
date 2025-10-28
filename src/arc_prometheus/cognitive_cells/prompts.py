"""Prompt templates for LLM-based code generation.

This module provides prompt construction for the Analyst+Programmer
unified pipeline that analyzes ARC tasks and generates solver code.
"""

import numpy as np


def _format_grid_as_ascii(grid: np.ndarray) -> str:
    """Format grid as ASCII art for LLM prompt.

    Converts numpy array to readable text format:
    - Each row on a new line
    - Values separated by spaces
    - Preserves visual structure

    Args:
        grid: 2D numpy array of integers (0-9)

    Returns:
        String representation of grid

    Example:
        >>> grid = np.array([[1, 0, 2], [0, 3, 0]])
        >>> print(_format_grid_as_ascii(grid))
        1 0 2
        0 3 0
    """
    rows = []
    for row in grid:
        rows.append(" ".join(str(cell) for cell in row))
    return "\n".join(rows)


def create_solver_prompt(train_pairs: list[dict[str, np.ndarray]]) -> str:
    """Create prompt for Gemini to generate ARC solver code.

    This prompt combines Analyst and Programmer roles:
    1. Analyzes train examples to infer transformation rule
    2. Generates Python code implementing the rule

    Args:
        train_pairs: List of {"input": np.ndarray, "output": np.ndarray}

    Returns:
        Formatted prompt string ready for Gemini API

    Prompt Structure:
        - Task description and goals
        - All train examples (input/output pairs) as ASCII art
        - Analysis instructions (what patterns to look for)
        - Code generation instructions with constraints
        - Example code structure
        - Required function signature

    Example:
        >>> train_pairs = [
        ...     {"input": np.array([[1, 2]]), "output": np.array([[2, 3]])}
        ... ]
        >>> prompt = create_solver_prompt(train_pairs)
        >>> "def solve(" in prompt
        True
    """
    prompt_parts = [
        "You are an AI system analyzing Abstract Reasoning Corpus (ARC) puzzles.",
        "",
        "## Task",
        "Analyze the input-output examples below, infer the transformation rule,",
        "and implement it as a Python function using only numpy.",
        "",
        "## Examples",
    ]

    # Add each train pair
    for idx, pair in enumerate(train_pairs, 1):
        prompt_parts.extend(
            [
                "",
                f"### Example {idx}",
                "Input:",
                _format_grid_as_ascii(pair["input"]),
                "",
                "Output:",
                _format_grid_as_ascii(pair["output"]),
            ]
        )

    # Add instructions
    prompt_parts.extend(
        [
            "",
            "## Instructions",
            "1. Analyze the patterns: what changes between input and output?",
            "2. Consider: rotations, reflections, color changes, shape detection, filling patterns, etc.",
            "3. Implement a function with this EXACT signature:",
            "   def solve(task_grid: np.ndarray) -> np.ndarray:",
            "",
            "## Requirements",
            "- Use ONLY numpy for array operations (no other libraries)",
            "- Function must be named 'solve' (lowercase)",
            "- Must accept one parameter: task_grid (np.ndarray)",
            "- Must return np.ndarray (the transformed grid)",
            "- Include 'import numpy as np' at the top",
            "- Handle edge cases (empty grids, varying sizes if applicable)",
            "- The output grid may have different dimensions than input",
            "",
            "## Output Format",
            "Return ONLY the Python code, starting with 'import numpy as np'.",
            "Do NOT include explanations, markdown formatting, or code blocks.",
            "Just raw Python code that can be executed directly.",
        ]
    )

    return "\n".join(prompt_parts)
