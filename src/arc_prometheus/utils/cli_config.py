"""CLI configuration argument parsing for evolution experiments.

This module provides command-line argument parsing for evolution loop experiments,
allowing users to customize model settings, temperatures, timeouts, and other
parameters without modifying code.

Example:
    Run evolution with custom configuration:

    $ python scripts/demo_phase2_3_evolution.py \\
        --model gemini-2.0-flash-thinking-exp \\
        --programmer-temperature 0.5 \\
        --refiner-temperature 0.6 \\
        --max-generations 10
"""

import argparse

from .config import (
    DEFAULT_TIMEOUT_SECONDS,
    MODEL_NAME,
    PROGRAMMER_GENERATION_CONFIG,
    REFINER_GENERATION_CONFIG,
)


def _validate_temperature(value: str) -> float:
    """Validate temperature is within Gemini API range (0.0-2.0).

    Args:
        value: Temperature value as string

    Returns:
        Temperature as float

    Raises:
        argparse.ArgumentTypeError: If temperature is out of range
    """
    try:
        temp = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Temperature must be a number, got: {value}"
        ) from e

    if temp < 0.0 or temp > 2.0:
        raise argparse.ArgumentTypeError(
            f"Temperature must be between 0.0 and 2.0, got: {temp}"
        )

    return temp


def _validate_positive_int(value: str) -> int:
    """Validate value is a positive integer.

    Args:
        value: Integer value as string

    Returns:
        Value as positive int

    Raises:
        argparse.ArgumentTypeError: If value is not positive
    """
    try:
        val = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Must be an integer, got: {value}") from e

    if val <= 0:
        raise argparse.ArgumentTypeError(f"Must be positive, got: {val}")

    return val


def parse_evolution_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for evolution experiments.

    Args:
        args: Command-line arguments to parse. If None, uses sys.argv.

    Returns:
        Parsed arguments as Namespace with fields:
        - model: LLM model name
        - programmer_temperature: Temperature for code generation
        - refiner_temperature: Temperature for debugging
        - timeout_llm: LLM API call timeout in seconds
        - timeout_eval: Sandbox execution timeout in seconds
        - max_generations: Maximum evolution generations
        - target_fitness: Target fitness for early stopping (optional)
        - verbose: Enable verbose output
        - use_cache: Enable LLM response caching
        - cache_stats: Show cache statistics and exit
        - clear_cache: Clear all cache entries and exit
        - clear_expired_cache: Clear expired cache entries and exit
        - cache_ttl: Cache TTL in days

    Example:
        >>> args = parse_evolution_args(['--model', 'gemini-2.0-flash-thinking-exp'])
        >>> args.model
        'gemini-2.0-flash-thinking-exp'
    """
    parser = argparse.ArgumentParser(
        description="Run ARC solver evolution with custom configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use different model
  %(prog)s --model gemini-2.0-flash-thinking-exp

  # Adjust creativity (temperature)
  %(prog)s --programmer-temperature 0.5 --refiner-temperature 0.6

  # Longer timeouts for complex tasks
  %(prog)s --timeout-llm 120 --timeout-eval 10

  # Early stopping when fitness reaches 15
  %(prog)s --target-fitness 15 --max-generations 20

  # Cache management
  %(prog)s --cache-stats              # View cache performance
  %(prog)s --clear-cache              # Clear all cache entries
  %(prog)s --clear-expired-cache      # Clear only expired entries
  %(prog)s --no-cache                 # Disable cache for this run

  # Quiet mode
  %(prog)s --no-verbose
        """,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"LLM model name (default: {MODEL_NAME})",
    )

    # Temperature controls
    parser.add_argument(
        "--programmer-temperature",
        type=_validate_temperature,
        default=PROGRAMMER_GENERATION_CONFIG["temperature"],
        help=f"Temperature for code generation (0.0-2.0, default: {PROGRAMMER_GENERATION_CONFIG['temperature']})",
    )

    parser.add_argument(
        "--refiner-temperature",
        type=_validate_temperature,
        default=REFINER_GENERATION_CONFIG["temperature"],
        help=f"Temperature for debugging (0.0-2.0, default: {REFINER_GENERATION_CONFIG['temperature']})",
    )

    # Timeout controls
    parser.add_argument(
        "--timeout-llm",
        type=_validate_positive_int,
        default=60,
        help="LLM API call timeout in seconds (default: 60)",
    )

    parser.add_argument(
        "--timeout-eval",
        type=_validate_positive_int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Sandbox execution timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )

    # Evolution controls
    parser.add_argument(
        "--max-generations",
        type=_validate_positive_int,
        default=5,
        help="Maximum evolution generations (default: 5)",
    )

    parser.add_argument(
        "--target-fitness",
        type=float,
        default=None,
        help="Target fitness for early stopping (optional)",
    )

    # Output controls
    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Disable verbose output (default: enabled)",
    )

    # Cache controls
    parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Disable LLM response caching (default: cache enabled)",
    )

    parser.add_argument(
        "--cache-stats",
        action="store_true",
        help="Show cache statistics and exit",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cache entries and exit",
    )

    parser.add_argument(
        "--clear-expired-cache",
        action="store_true",
        help="Clear expired cache entries and exit",
    )

    parser.add_argument(
        "--cache-ttl",
        type=_validate_positive_int,
        default=7,
        help="Cache TTL in days (default: 7)",
    )

    return parser.parse_args(args)
