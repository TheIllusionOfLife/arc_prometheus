"""Response validation utilities for Gemini API.

This module provides utilities for validating and processing Gemini API responses,
including finish_reason checking and markdown unwrapping.
"""

from typing import Any


def validate_finish_reason(response: Any) -> None:
    """Check that response completed successfully.

    Args:
        response: Gemini API response object

    Raises:
        ValueError: If response was truncated, blocked, or failed
    """
    if not response.candidates:
        raise ValueError("No candidates in API response")

    candidate = response.candidates[0]

    # Skip validation for non-Gemini objects (e.g., test mocks)
    if not hasattr(type(candidate), "FinishReason"):
        return

    finish_reason_enum = type(candidate).FinishReason

    # Check for various termination conditions
    if candidate.finish_reason == finish_reason_enum.MAX_TOKENS:
        raise ValueError(
            "Response truncated: MAX_TOKENS reached. "
            "The response was cut off mid-generation due to token limit. "
            "Try increasing max_output_tokens in GenerationConfig."
        )
    elif candidate.finish_reason == finish_reason_enum.SAFETY:
        raise ValueError(
            f"Response blocked by safety filters. "
            f"Safety ratings: {candidate.safety_ratings}"
        )
    elif candidate.finish_reason == finish_reason_enum.RECITATION:
        raise ValueError(
            "Response blocked: contains recited copyrighted content. "
            "The model detected content that matches copyrighted material."
        )
    elif candidate.finish_reason == finish_reason_enum.PROHIBITED_CONTENT:
        raise ValueError(
            "Response blocked: prohibited content detected. "
            "The response violated content policies."
        )
    elif candidate.finish_reason == finish_reason_enum.BLOCKLIST:
        raise ValueError(
            "Response blocked: matched safety blocklist. "
            "The response triggered safety blocklist filters."
        )
    elif candidate.finish_reason == finish_reason_enum.MALFORMED_FUNCTION_CALL:
        raise ValueError(
            "Response blocked: malformed function call detected. "
            "The model generated an invalid function call."
        )
    elif candidate.finish_reason != finish_reason_enum.STOP:
        # Catch any other unexpected finish reasons
        raise ValueError(
            f"Unexpected finish_reason: {candidate.finish_reason} "
            f"(expected STOP=1). This may indicate an API issue."
        )

    # If we reach here, finish_reason == STOP (normal completion)


def unwrap_markdown_json(text: str) -> str:
    """Remove markdown code fences if present.

    Some models occasionally wrap JSON responses in ```json ... ```,
    despite response_mime_type="application/json" in GenerationConfig.
    This function defensively removes such wrapping.

    Args:
        text: Response text that may be wrapped in markdown

    Returns:
        Clean text with markdown fences removed
    """
    text = text.strip()

    # Remove opening fence
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    elif text.startswith("```"):
        text = text[3:]  # Remove ```

    # Remove closing fence
    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def fix_truncated_json(text: str) -> str:
    """Attempt to fix truncated JSON by closing open structures.

    When MAX_TOKENS is reached, JSON is often cut off mid-string or mid-object.
    This function attempts to close common truncation patterns to allow parsing.

    Args:
        text: Potentially truncated JSON string

    Returns:
        JSON string with missing closures added

    Note:
        This is best-effort - complex truncations may still fail to parse.
        The goal is to recover partial results (e.g., 4/5 solutions) when possible.
    """
    result = text.strip()

    # Check if truncated mid-string (most common case)
    # Count quotes to see if we're inside a string
    quote_count = result.count('"') - result.count('\\"')  # Exclude escaped quotes
    if quote_count % 2 == 1:
        # Odd number of quotes = truncated inside string
        result += '"'

    # Count opening/closing brackets and braces
    open_braces = result.count("{") - result.count("}")
    open_brackets = result.count("[") - result.count("]")

    # Close any open structures
    # Order matters: close innermost first (brackets before braces)
    result += "]" * open_brackets
    result += "}" * open_braces

    return result
