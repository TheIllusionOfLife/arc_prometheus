"""LLM-based solver code debugging and refinement (Phase 2.2).

This module provides the Refiner agent - the first evolutionary mechanism
(Mutation) that improves failed solver code through automated debugging.
"""

import google.generativeai as genai

from ..crucible.data_loader import load_task
from ..utils.config import get_gemini_api_key
from .programmer import extract_code_from_response
from .prompts import create_refiner_prompt


def refine_solver(
    failed_code: str, task_json_path: str, fitness_result: dict, timeout: int = 60
) -> str:
    """Debug and improve failed solver code using Gemini API.

    This function is the core of the Mutation evolutionary mechanism.
    It analyzes why a solver failed and generates an improved version.

    Args:
        failed_code: Original solver code that failed
        task_json_path: Path to ARC task JSON file
        fitness_result: Result from calculate_fitness() containing:
            - train_correct, train_total: Train performance
            - test_correct, test_total: Test performance
            - execution_errors: List of error messages
        timeout: API request timeout in seconds (default: 60)

    Returns:
        Improved solver code string with bugs fixed

    Raises:
        ValueError: If API key not configured or response unparseable
        FileNotFoundError: If task file not found
        Exception: If Gemini API call fails

    Process:
        1. Load task data for context
        2. Create refiner prompt with failure analysis
        3. Call Gemini API with temperature 0.4 (debugging creativity)
        4. Extract corrected code from response
        5. Return refined code (ready for re-evaluation)

    Example:
        >>> failed_code = '''
        ... import numpy as np
        ... def solve(task_grid: np.ndarray) -> np.ndarray:
        ...     return task_grid + 1  # Wrong logic
        ... '''
        >>> fitness_result = {
        ...     "train_correct": 0, "train_total": 3,
        ...     "test_correct": 0, "test_total": 1,
        ...     "execution_errors": []
        ... }
        >>> refined = refine_solver(failed_code, "task.json", fitness_result)  # doctest: +SKIP
        >>> "def solve(" in refined  # doctest: +SKIP
        True
    """
    # Get API key (will raise ValueError if not configured)
    api_key = get_gemini_api_key()

    # Load task data for context
    task_data = load_task(task_json_path)

    # Configure Gemini
    genai.configure(api_key=api_key)

    # Use gemini-2.5-flash-lite - same model as programmer
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # Create refiner prompt with failure analysis
    prompt = create_refiner_prompt(failed_code, task_data, fitness_result)

    # Generate refined code
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.4,  # Slightly higher than programmer (0.3)
                # for debugging creativity
                "max_output_tokens": 3048,  # More than programmer (2048)
                # to allow detailed fixes
            },
            request_options={"timeout": timeout},
        )

        response_text = response.text

    except Exception as e:
        raise Exception(f"Gemini API call failed: {e}") from e

    # Extract code from response (reuse programmer's parser)
    try:
        code = extract_code_from_response(response_text)
        return code
    except ValueError as e:
        # Include response text in error for debugging
        if len(response_text) > 500:
            preview = f"{response_text[:300]}\n\n... [truncated] ...\n\n{response_text[-200:]}"
        else:
            preview = response_text
        raise ValueError(
            f"Failed to parse LLM response: {e}\nResponse was:\n{preview}"
        ) from e
