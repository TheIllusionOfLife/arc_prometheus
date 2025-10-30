"""
Error classification for solver failures.

This module provides error type detection and classification to enable targeted
debugging strategies in the Refiner agent.
"""

from enum import Enum
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from .fitness import FitnessResult


class ErrorType(str, Enum):
    """Classification of solver execution errors."""

    SYNTAX = "syntax"  # SyntaxError, IndentationError
    RUNTIME = "runtime"  # TypeError, ValueError, IndexError, etc.
    TIMEOUT = "timeout"  # Execution exceeded time limit
    LOGIC = "logic"  # Wrong output (correct execution, wrong result)
    VALIDATION = "validation"  # Missing solve(), invalid return type


class ErrorDetail(TypedDict):
    """Structured error information for a single example."""

    example_id: str  # "train_0", "test_1", etc.
    error_type: ErrorType
    error_message: str
    exception_class: str | None  # "SyntaxError", "TypeError", etc. (None for timeout)


def classify_error(fitness_result: "FitnessResult") -> ErrorType | None:
    """
    Determine primary error type from fitness result.

    Args:
        fitness_result: Fitness evaluation result with error details

    Returns:
        Most common error type, or None if solver is perfect

    Examples:
        >>> # Syntax error case
        >>> result = {"error_summary": {"syntax": 3}, "train_correct": 0, "train_total": 3}
        >>> classify_error(result)
        <ErrorType.SYNTAX: 'syntax'>

        >>> # Perfect solver case
        >>> result = {"error_details": [], "train_correct": 3, "train_total": 3}
        >>> classify_error(result)
        None

        >>> # Logic error case (no execution errors but wrong output)
        >>> result = {"error_details": [], "train_correct": 2, "train_total": 3}
        >>> classify_error(result)
        <ErrorType.LOGIC: 'logic'>
    """
    error_details = fitness_result.get("error_details", [])

    if not error_details:
        # No execution errors - check for logic errors
        if fitness_result["train_correct"] < fitness_result["train_total"]:
            return ErrorType.LOGIC
        # Perfect solver or only test failures (still a form of logic error)
        if fitness_result.get("test_correct", 0) < fitness_result.get("test_total", 0):
            return ErrorType.LOGIC
        return None  # No errors at all

    # Return most common error type
    error_summary = fitness_result.get("error_summary", {})
    if error_summary:
        most_common_type = max(error_summary.items(), key=lambda x: x[1])[0]
        return ErrorType(most_common_type)

    # Fallback: return first error type
    first_error_type = error_details[0]["error_type"]
    if isinstance(first_error_type, ErrorType):
        return first_error_type
    return ErrorType(first_error_type)


def get_debugging_strategy(error_type: ErrorType) -> str:
    """
    Return error-specific debugging instructions.

    Args:
        error_type: Type of error detected

    Returns:
        Detailed debugging strategy for the given error type

    Examples:
        >>> strategy = get_debugging_strategy(ErrorType.SYNTAX)
        >>> "colon" in strategy.lower()
        True

        >>> strategy = get_debugging_strategy(ErrorType.TIMEOUT)
        >>> "loop" in strategy.lower()
        True
    """
    strategies = {
        ErrorType.SYNTAX: """Focus on Python syntax issues:
- Check for missing colons after function definitions and control flow statements
- Verify indentation consistency (use 4 spaces, not tabs)
- Ensure all brackets/parentheses are balanced
- Look for invalid operators or keywords
- Check for proper string quote matching""",
        ErrorType.RUNTIME: """Focus on runtime error prevention:
- Add bounds checking before array indexing (0 <= idx < len(array))
- Validate array shapes before operations
- Check for division by zero before arithmetic operations
- Ensure type consistency (all operations use np.ndarray)
- Verify function arguments match expected types""",
        ErrorType.TIMEOUT: """Focus on performance optimization:
- Remove infinite loops (while True without break, unbounded recursion)
- Avoid nested loops over large arrays when possible
- Use vectorized numpy operations instead of Python loops
- Check for redundant computations that can be cached
- Consider early termination conditions in loops""",
        ErrorType.LOGIC: """Focus on transformation logic:
- Carefully analyze input→output patterns in ALL train examples
- Verify the algorithm correctly implements the underlying rule
- Test edge cases: empty grids, single cells, uniform grids
- Check if output shape should match input shape or be different
- Mentally trace through the algorithm with example data""",
        ErrorType.VALIDATION: """Focus on function structure:
- Ensure function is named exactly 'solve' (lowercase)
- Verify signature: def solve(task_grid: np.ndarray) -> np.ndarray
- Confirm return type is np.ndarray (use np.array(...) for conversions)
- Import numpy as np at module level
- Ensure the function is defined at module level (not nested)""",
    }
    return strategies.get(error_type, "Review code carefully for any issues.")
