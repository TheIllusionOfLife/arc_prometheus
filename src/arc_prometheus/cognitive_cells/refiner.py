"""LLM-based solver code debugging and refinement (Phase 2.2).

This module provides the Refiner agent - the first evolutionary mechanism
(Mutation) that improves failed solver code through automated debugging.
"""

from typing import Any

import google.generativeai as genai

from ..crucible.data_loader import load_task
from ..evolutionary_engine.error_classifier import classify_error
from ..evolutionary_engine.fitness import FitnessResult
from ..utils.config import (
    MODEL_NAME,
    REFINER_GENERATION_CONFIG,
    get_gemini_api_key,
)
from .programmer import extract_code_from_response
from .prompts import create_refiner_prompt


def refine_solver(
    failed_code: str,
    task_json_path: str,
    fitness_result: FitnessResult,
    model_name: str | None = None,
    temperature: float | None = None,
    timeout: int = 60,
    use_cache: bool = True,
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
        model_name: LLM model name (default: from config.py)
        temperature: LLM temperature 0.0-2.0 (default: from config.py)
        timeout: API request timeout in seconds (default: 60)
        use_cache: If True, use LLM response cache (default: True)

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

    # Use provided model or fall back to config
    model_to_use = model_name if model_name is not None else MODEL_NAME
    model = genai.GenerativeModel(model_to_use)

    # Build generation config (merge custom temperature if provided)
    # Type as Any to satisfy mypy while maintaining runtime correctness
    generation_config: Any = dict(REFINER_GENERATION_CONFIG)
    if temperature is not None:
        generation_config["temperature"] = temperature

    # Determine actual temperature for cache key
    temp_to_use = (
        temperature
        if temperature is not None
        else REFINER_GENERATION_CONFIG["temperature"]
    )

    # Classify error type for targeted debugging
    error_type = classify_error(fitness_result)

    # Create refiner prompt with failure analysis and error classification
    prompt = create_refiner_prompt(
        failed_code, task_data, fitness_result, error_type=error_type
    )

    # Check cache if enabled
    if use_cache:
        from ..utils.llm_cache import get_cache

        cache = get_cache()
        cached_response = cache.get(prompt, model_to_use, temp_to_use)
        if cached_response is not None:
            # Parse and return cached response
            return extract_code_from_response(cached_response)

    # Generate refined code (cache miss or cache disabled)
    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={"timeout": timeout},
        )

        response_text = response.text

    except Exception as e:
        raise Exception(f"Gemini API call failed: {e}") from e

    # Store in cache if enabled
    if use_cache:
        from ..utils.llm_cache import get_cache

        cache = get_cache()
        cache.set(prompt, response_text, model_to_use, temp_to_use)

    # Extract code from response (reuse programmer's parser)
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
