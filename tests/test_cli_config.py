"""Tests for CLI configuration argument parsing.

This module tests the CLI argument parser for evolution experiments,
ensuring proper validation and default value handling.
"""

import pytest

from arc_prometheus.utils.cli_config import parse_evolution_args


class TestCLIArgumentParsing:
    """Test CLI argument parsing for evolution experiments."""

    def test_default_values(self) -> None:
        """Test that default values match config.py."""
        args = parse_evolution_args([])

        assert args.model == "gemini-2.5-flash-lite"
        assert args.programmer_temperature == 0.3
        assert args.refiner_temperature == 0.4
        assert args.timeout_llm == 60
        assert args.timeout_eval == 5
        assert args.max_generations == 5
        assert args.target_fitness is None
        assert args.verbose is True

    def test_custom_model(self) -> None:
        """Test custom model specification."""
        args = parse_evolution_args(["--model", "gemini-2.0-flash-thinking-exp"])

        assert args.model == "gemini-2.0-flash-thinking-exp"
        # Other values should remain default
        assert args.programmer_temperature == 0.3
        assert args.refiner_temperature == 0.4

    def test_custom_temperatures(self) -> None:
        """Test custom temperature settings for programmer and refiner."""
        args = parse_evolution_args(
            ["--programmer-temperature", "0.5", "--refiner-temperature", "0.6"]
        )

        assert args.programmer_temperature == 0.5
        assert args.refiner_temperature == 0.6

    def test_custom_timeouts(self) -> None:
        """Test custom timeout settings."""
        args = parse_evolution_args(
            ["--timeout-llm", "120", "--timeout-eval", "10"]
        )

        assert args.timeout_llm == 120
        assert args.timeout_eval == 10

    def test_custom_generations_and_target(self) -> None:
        """Test custom max generations and target fitness."""
        args = parse_evolution_args(
            ["--max-generations", "10", "--target-fitness", "15.0"]
        )

        assert args.max_generations == 10
        assert args.target_fitness == 15.0

    def test_no_verbose_flag(self) -> None:
        """Test disabling verbose output."""
        args = parse_evolution_args(["--no-verbose"])

        assert args.verbose is False

    def test_all_custom_args(self) -> None:
        """Test specifying all arguments at once."""
        args = parse_evolution_args(
            [
                "--model",
                "gemini-2.0-flash-thinking-exp",
                "--programmer-temperature",
                "0.1",
                "--refiner-temperature",
                "0.9",
                "--timeout-llm",
                "90",
                "--timeout-eval",
                "8",
                "--max-generations",
                "20",
                "--target-fitness",
                "12.5",
                "--no-verbose",
            ]
        )

        assert args.model == "gemini-2.0-flash-thinking-exp"
        assert args.programmer_temperature == 0.1
        assert args.refiner_temperature == 0.9
        assert args.timeout_llm == 90
        assert args.timeout_eval == 8
        assert args.max_generations == 20
        assert args.target_fitness == 12.5
        assert args.verbose is False

    def test_invalid_temperature_too_low(self) -> None:
        """Test that temperature below 0.0 raises error."""
        with pytest.raises(SystemExit):
            parse_evolution_args(["--programmer-temperature", "-0.1"])

    def test_invalid_temperature_too_high(self) -> None:
        """Test that temperature above 2.0 raises error."""
        with pytest.raises(SystemExit):
            parse_evolution_args(["--refiner-temperature", "2.1"])

    def test_invalid_timeout_zero(self) -> None:
        """Test that timeout of 0 raises error."""
        with pytest.raises(SystemExit):
            parse_evolution_args(["--timeout-llm", "0"])

    def test_invalid_timeout_negative(self) -> None:
        """Test that negative timeout raises error."""
        with pytest.raises(SystemExit):
            parse_evolution_args(["--timeout-eval", "-5"])

    def test_invalid_max_generations_zero(self) -> None:
        """Test that max_generations of 0 raises error."""
        with pytest.raises(SystemExit):
            parse_evolution_args(["--max-generations", "0"])

    def test_invalid_max_generations_negative(self) -> None:
        """Test that negative max_generations raises error."""
        with pytest.raises(SystemExit):
            parse_evolution_args(["--max-generations", "-3"])

    def test_help_text_generation(self) -> None:
        """Test that --help generates help text (exits with 0)."""
        with pytest.raises(SystemExit) as exc_info:
            parse_evolution_args(["--help"])

        assert exc_info.value.code == 0


class TestCLIConfigIntegration:
    """Test CLI config integration with evolution components."""

    def test_args_namespace_has_all_required_fields(self) -> None:
        """Test that parsed args namespace has all expected fields."""
        args = parse_evolution_args([])

        # Verify all expected attributes exist
        assert hasattr(args, "model")
        assert hasattr(args, "programmer_temperature")
        assert hasattr(args, "refiner_temperature")
        assert hasattr(args, "timeout_llm")
        assert hasattr(args, "timeout_eval")
        assert hasattr(args, "max_generations")
        assert hasattr(args, "target_fitness")
        assert hasattr(args, "verbose")

    def test_temperature_edge_cases(self) -> None:
        """Test temperature boundary values (0.0 and 2.0)."""
        # Test minimum valid temperature
        args_min = parse_evolution_args(["--programmer-temperature", "0.0"])
        assert args_min.programmer_temperature == 0.0

        # Test maximum valid temperature
        args_max = parse_evolution_args(["--refiner-temperature", "2.0"])
        assert args_max.refiner_temperature == 2.0

    def test_float_parsing_precision(self) -> None:
        """Test that float values are parsed with correct precision."""
        args = parse_evolution_args(
            [
                "--programmer-temperature",
                "0.123",
                "--target-fitness",
                "13.456",
            ]
        )

        assert args.programmer_temperature == pytest.approx(0.123)
        assert args.target_fitness == pytest.approx(13.456)
