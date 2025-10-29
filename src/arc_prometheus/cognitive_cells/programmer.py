"""LLM-based solver code generation using Google Gemini API.

This module provides functions to:
1. Generate solver code from ARC train examples using Gemini API
2. Extract Python code from LLM responses (handling various formats)
"""

import re

import google.generativeai as genai
import numpy as np

from ..utils.config import (
    MODEL_NAME,
    PROGRAMMER_GENERATION_CONFIG,
    get_gemini_api_key,
)
from .prompts import create_solver_prompt


def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from LLM response.

    Handles various response formats:
    - Code in ```python ... ``` blocks
    - Code in ``` ... ``` blocks (without python keyword)
    - Raw code without delimiters
    - Multiple code blocks (extracts the one with solve())
    - Markdown formatting artifacts

    Args:
        response_text: Raw text response from LLM

    Returns:
        Extracted Python code string

    Raises:
        ValueError: If no valid solve() function found in response

    Examples:
        >>> response = '''```python
        ... import numpy as np
        ... def solve(task_grid: np.ndarray) -> np.ndarray:
        ...     return task_grid + 1
        ... ```'''
        >>> code = extract_code_from_response(response)
        >>> "def solve(" in code
        True
        >>> "```" in code
        False
    """
    # Strategy 1: Try to extract from code blocks with ```
    code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
    code_blocks = re.findall(code_block_pattern, response_text, re.DOTALL)

    if code_blocks:
        # Find block containing solve() function
        for block in code_blocks:
            if "def solve(" in block:
                result: str = block.strip()
                return result

    # Strategy 2: Try to extract raw code without delimiters
    # Look for lines starting with 'import' or 'def solve('
    lines = response_text.split("\n")

    # Find start: first line with 'import numpy' or 'def solve('
    start_idx = None
    for i, line in enumerate(lines):
        if "import numpy" in line or "def solve(" in line:
            start_idx = i
            break

    if start_idx is not None:
        # Find end: last line that looks like code
        # Strategy: Stop at first unindented line after function that doesn't start with
        # Python keywords (import/from/def/class) or blank line
        end_idx = len(lines)

        # Scan forward from start to find where code ends
        in_function = False
        for i in range(start_idx, len(lines)):
            line = lines[i]

            # Track if we're inside the solve function
            if "def solve(" in line:
                in_function = True
                continue

            # If we're in function, look for dedented non-code line
            if in_function:
                stripped = line.strip()

                # Empty lines are OK
                if not stripped:
                    continue

                # Indented lines (code inside function) are OK
                if line.startswith(" ") or line.startswith("\t"):
                    continue

                # Check if this is a top-level Python statement (another function/import/decorator)
                # Use lstrip() to handle any leading whitespace
                if any(
                    line.lstrip().startswith(kw)
                    for kw in ["import ", "from ", "def ", "class ", "@"]
                ):
                    continue

                # We found an unindented line that's not a Python statement
                # This is likely explanation text - stop here
                end_idx = i
                break

        code = "\n".join(lines[start_idx:end_idx]).strip()
        if "def solve(" in code:
            return code

    # Strategy 3: Failed to extract
    raise ValueError(
        "solve() function not found in LLM response. "
        "Response must contain 'def solve(task_grid: np.ndarray) -> np.ndarray:'"
    )


def generate_solver(
    train_pairs: list[dict[str, np.ndarray]],
    model_name: str | None = None,
    temperature: float | None = None,
    timeout: int = 60,
) -> str:
    """Generate solver code using Gemini API.

    Uses gemini-2.5-flash-lite by default, Google's fastest and latest flash model.
    Optimized for cost-efficiency and high throughput with thinking capabilities.

    Args:
        train_pairs: List of {"input": np.ndarray, "output": np.ndarray}
        model_name: LLM model name (default: from config.py)
        temperature: LLM temperature 0.0-2.0 (default: from config.py)
        timeout: API request timeout in seconds (default: 60)

    Returns:
        Python code string containing solve() function

    Raises:
        ValueError: If API key not configured or response unparseable
        Exception: If Gemini API call fails

    Example:
        >>> train_pairs = [
        ...     {"input": np.array([[1, 2]]), "output": np.array([[2, 3]])}
        ... ]
        >>> code = generate_solver(train_pairs)  # doctest: +SKIP
        >>> "import numpy as np" in code  # doctest: +SKIP
        True
        >>> "def solve(" in code  # doctest: +SKIP
        True
    """
    # Get API key (will raise ValueError if not configured)
    api_key = get_gemini_api_key()

    # Configure Gemini
    genai.configure(api_key=api_key)

    # Use provided model or fall back to config
    model_to_use = model_name if model_name is not None else MODEL_NAME
    model = genai.GenerativeModel(model_to_use)

    # Build generation config (merge custom temperature if provided)
    generation_config = dict(PROGRAMMER_GENERATION_CONFIG)
    if temperature is not None:
        generation_config["temperature"] = temperature

    # Create prompt
    prompt = create_solver_prompt(train_pairs)

    # Generate response with timeout
    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={"timeout": timeout},
        )

        response_text = response.text

    except Exception as e:
        raise Exception(f"Gemini API call failed: {e}") from e

    # Extract code from response
    try:
        code = extract_code_from_response(response_text)
        return code
    except ValueError as e:
        # Include response text in error for debugging with improved preview
        max_preview_length = 1000
        if len(response_text) > max_preview_length:
            half = max_preview_length // 2
            chars_truncated = len(response_text) - max_preview_length
            preview = (
                f"{response_text[:half]}\n\n"
                f"... [{chars_truncated} chars truncated] ...\n\n"
                f"{response_text[-half:]}"
            )
        else:
            preview = response_text
        raise ValueError(
            f"Failed to parse LLM response: {e}\nResponse was:\n{preview}"
        ) from e
