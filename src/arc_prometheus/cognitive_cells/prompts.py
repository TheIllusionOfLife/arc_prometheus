"""Prompt templates for LLM-based code generation.

This module provides prompt construction for the Analyst+Programmer
unified pipeline that analyzes ARC tasks and generates solver code.
"""

from typing import TYPE_CHECKING

import numpy as np

from ..evolutionary_engine.fitness import FitnessResult

if TYPE_CHECKING:
    from ..evolutionary_engine.error_classifier import ErrorType

# Constants for prompt formatting
MAX_ERRORS_TO_SHOW = 5  # Limit execution errors shown to prevent token overflow


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
            "Do NOT include explanations or debugging commentary.",
            "You may optionally wrap code in ```python blocks, but raw code is preferred.",
            "Just the code that can be executed directly.",
        ]
    )

    return "\n".join(prompt_parts)


def create_refiner_prompt(
    failed_code: str,
    task_data: dict,
    fitness_result: FitnessResult,
    max_examples: int = 3,
    error_type: "ErrorType | None" = None,
) -> str:
    """Create prompt for Refiner to debug failed solver code.

    This prompt provides failure analysis and asks the LLM to fix the bugs.

    Args:
        failed_code: Python code that failed execution or produced wrong results
        task_data: ARC task dict with {"train": [...], "test": [...]}
        fitness_result: Result from calculate_fitness() containing:
            - train_correct, train_total: Train performance
            - test_correct, test_total: Test performance
            - execution_errors: List of error messages
        max_examples: Maximum number of train examples to include (default: 3)
            Lower values save API tokens, higher values provide more context

    Returns:
        Formatted prompt string for debugging

    Prompt Structure:
        - Role and debugging goal
        - Original task examples (up to max_examples train pairs for context)
        - Failed code with analysis
        - Failure details (which examples failed, errors)
        - Debugging instructions
        - Output format requirements

    Example:
        >>> failed_code = "def solve(x): return x"  # Missing signature
        >>> task_data = {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
        >>> fitness_result = {"train_correct": 0, "train_total": 1,
        ...                   "test_correct": 0, "test_total": 0,
        ...                   "execution_errors": ["Train example 0: Execution failed"]}
        >>> prompt = create_refiner_prompt(failed_code, task_data, fitness_result)
        >>> "debug" in prompt.lower() or "fix" in prompt.lower()
        True
    """
    prompt_parts = [
        "You are an expert Python debugger for Abstract Reasoning Corpus (ARC) solvers.",
        "",
        "## Goal",
        "Debug and fix the provided solver code that failed to solve the ARC task correctly.",
        "",
        "## Original Task Examples",
        "Here are some examples from the task (for context):",
    ]

    # Show up to max_examples train examples (not all, to save tokens)
    train_examples = task_data.get("train", [])
    examples_to_show = min(max_examples, len(train_examples))

    for idx in range(examples_to_show):
        example = train_examples[idx]
        input_grid = np.array(example["input"], dtype=np.int64)
        output_grid = np.array(example["output"], dtype=np.int64)

        prompt_parts.extend(
            [
                "",
                f"### Example {idx + 1}",
                "Input:",
                _format_grid_as_ascii(input_grid),
                "",
                "Output:",
                _format_grid_as_ascii(output_grid),
            ]
        )

    # Show the failed code
    prompt_parts.extend(
        [
            "",
            "## Failed Code",
            "This code attempted to solve the task but failed:",
            "```python",
            failed_code.strip(),
            "```",
            "",
            "## Failure Analysis",
        ]
    )

    # Add performance stats
    train_correct = fitness_result.get("train_correct", 0)
    train_total = fitness_result.get("train_total", 0)
    test_correct = fitness_result.get("test_correct", 0)
    test_total = fitness_result.get("test_total", 0)

    prompt_parts.append(
        f"Performance: {train_correct}/{train_total} train correct, "
        f"{test_correct}/{test_total} test correct"
    )

    # Add execution errors (limit to prevent token overflow)
    execution_errors = fitness_result.get("execution_errors", [])
    if execution_errors:
        prompt_parts.append("")
        prompt_parts.append("Execution errors:")
        for error in execution_errors[:MAX_ERRORS_TO_SHOW]:
            prompt_parts.append(f"- {error}")
        if len(execution_errors) > MAX_ERRORS_TO_SHOW:
            remaining = len(execution_errors) - MAX_ERRORS_TO_SHOW
            prompt_parts.append(f"... and {remaining} more error(s)")

    # Add error-type-specific debugging strategy
    prompt_parts.extend(["", "## Debugging Strategy"])

    if error_type:
        from ..evolutionary_engine.error_classifier import get_debugging_strategy

        prompt_parts.append(f"**Error Type Detected: {error_type.value.upper()}**")
        prompt_parts.append("")
        prompt_parts.append(get_debugging_strategy(error_type))
    else:
        # Fallback: show all strategies
        prompt_parts.extend([
            "1. Identify the root cause of failure:",
            "   - Syntax errors (missing colons, parentheses, indentation)",
            "   - Runtime errors (division by zero, index out of bounds, type mismatches)",
            "   - Logic errors (wrong algorithm, incorrect transformations)",
            "   - Performance issues (infinite loops, excessive computation)",
        ])

    # Requirements (always show)
    prompt_parts.extend(
        [
            "",
            "## Requirements",
            "Fix the bugs while maintaining these requirements:",
            "- Use ONLY numpy for array operations (no other libraries)",
            "- Function must be named 'solve' with signature: def solve(task_grid: np.ndarray) -> np.ndarray:",
            "- Must return np.ndarray",
            "- Include 'import numpy as np' at the top",
            "- Handle edge cases properly",
            "- Ensure the fixed code correctly transforms the inputs to match the expected outputs",
            "",
            "## Output Format",
            "Return ONLY the corrected Python code, starting with 'import numpy as np'.",
            "Do NOT include explanations or debugging commentary.",
            "You may optionally wrap code in ```python blocks, but raw code is preferred.",
            "Just the fixed code that can be executed directly.",
        ]
    )

    return "\n".join(prompt_parts)
