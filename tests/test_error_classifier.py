"""
Tests for error classification functionality.

This module tests error type detection, classification, and debugging strategy
selection for solver failures.
"""

from arc_prometheus.evolutionary_engine.error_classifier import (
    ErrorDetail,
    ErrorType,
    classify_error,
    get_debugging_strategy,
)


class TestErrorType:
    """Tests for ErrorType enum."""

    def test_error_type_values(self):
        """Test that ErrorType enum has all required values."""
        assert ErrorType.SYNTAX == "syntax"
        assert ErrorType.RUNTIME == "runtime"
        assert ErrorType.TIMEOUT == "timeout"
        assert ErrorType.LOGIC == "logic"
        assert ErrorType.VALIDATION == "validation"

    def test_error_type_membership(self):
        """Test that all error types are valid enum members."""
        valid_types = {ErrorType.SYNTAX, ErrorType.RUNTIME, ErrorType.TIMEOUT, ErrorType.LOGIC, ErrorType.VALIDATION}
        assert len(valid_types) == 5

    def test_error_type_string_conversion(self):
        """Test that ErrorType can be converted to string."""
        assert str(ErrorType.SYNTAX) == "ErrorType.SYNTAX"
        assert ErrorType.SYNTAX.value == "syntax"


class TestErrorDetail:
    """Tests for ErrorDetail TypedDict structure."""

    def test_error_detail_structure(self):
        """Test that ErrorDetail can be created with required fields."""
        detail: ErrorDetail = {
            "example_id": "train_0",
            "error_type": ErrorType.SYNTAX,
            "error_message": "SyntaxError: invalid syntax",
            "exception_class": "SyntaxError",
        }
        assert detail["example_id"] == "train_0"
        assert detail["error_type"] == ErrorType.SYNTAX
        assert detail["error_message"] == "SyntaxError: invalid syntax"
        assert detail["exception_class"] == "SyntaxError"

    def test_error_detail_with_none_exception(self):
        """Test that ErrorDetail works with None exception_class (e.g., timeout)."""
        detail: ErrorDetail = {
            "example_id": "test_1",
            "error_type": ErrorType.TIMEOUT,
            "error_message": "Execution exceeded 5s timeout",
            "exception_class": None,
        }
        assert detail["exception_class"] is None

    def test_error_detail_with_empty_message(self):
        """Test that ErrorDetail handles empty error messages."""
        detail: ErrorDetail = {
            "example_id": "train_2",
            "error_type": ErrorType.RUNTIME,
            "error_message": "",
            "exception_class": "ValueError",
        }
        assert detail["error_message"] == ""


class TestClassifyError:
    """Tests for classify_error() function."""

    def test_classify_syntax_error(self):
        """Test classification of syntax errors."""
        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 3,
            "test_correct": 0,
            "test_total": 1,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "execution_errors": ["Train example 0: syntax"],
            "error_details": [
                {
                    "example_id": "train_0",
                    "error_type": ErrorType.SYNTAX,
                    "error_message": "SyntaxError: invalid syntax",
                    "exception_class": "SyntaxError",
                }
            ],
            "error_summary": {"syntax": 1},
        }
        assert classify_error(fitness_result) == ErrorType.SYNTAX

    def test_classify_timeout_error(self):
        """Test classification of timeout errors."""
        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 3,
            "test_correct": 0,
            "test_total": 1,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "execution_errors": ["Train example 0: timeout"],
            "error_details": [
                {
                    "example_id": "train_0",
                    "error_type": ErrorType.TIMEOUT,
                    "error_message": "Execution exceeded 5s timeout",
                    "exception_class": None,
                }
            ],
            "error_summary": {"timeout": 1},
        }
        assert classify_error(fitness_result) == ErrorType.TIMEOUT

    def test_classify_runtime_error(self):
        """Test classification of runtime errors."""
        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 3,
            "test_correct": 0,
            "test_total": 1,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "execution_errors": ["Train example 0: runtime"],
            "error_details": [
                {
                    "example_id": "train_0",
                    "error_type": ErrorType.RUNTIME,
                    "error_message": "TypeError: unsupported operand type(s)",
                    "exception_class": "TypeError",
                }
            ],
            "error_summary": {"runtime": 1},
        }
        assert classify_error(fitness_result) == ErrorType.RUNTIME

    def test_classify_logic_error_no_execution_errors(self):
        """Test classification of logic errors (code runs but wrong output)."""
        fitness_result = {
            "fitness": 3,
            "train_correct": 3,
            "train_total": 3,
            "test_correct": 0,
            "test_total": 1,
            "train_accuracy": 1.0,
            "test_accuracy": 0.0,
            "execution_errors": [],
            "error_details": [],
            "error_summary": {},
        }
        assert classify_error(fitness_result) == ErrorType.LOGIC

    def test_classify_logic_error_partial_train_correct(self):
        """Test logic error when some train examples work but not all."""
        fitness_result = {
            "fitness": 1,
            "train_correct": 1,
            "train_total": 3,
            "test_correct": 0,
            "test_total": 1,
            "train_accuracy": 0.33,
            "test_accuracy": 0.0,
            "execution_errors": [],
            "error_details": [],
            "error_summary": {},
        }
        assert classify_error(fitness_result) == ErrorType.LOGIC

    def test_classify_no_errors(self):
        """Test classification when solver is perfect."""
        fitness_result = {
            "fitness": 13,
            "train_correct": 3,
            "train_total": 3,
            "test_correct": 1,
            "test_total": 1,
            "train_accuracy": 1.0,
            "test_accuracy": 1.0,
            "execution_errors": [],
            "error_details": [],
            "error_summary": {},
        }
        assert classify_error(fitness_result) is None

    def test_classify_mixed_error_types_returns_most_common(self):
        """Test that classification returns the most common error type."""
        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 5,
            "test_correct": 0,
            "test_total": 1,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "execution_errors": ["Train example 0: timeout", "Train example 1: timeout", "Train example 2: runtime"],
            "error_details": [
                {
                    "example_id": "train_0",
                    "error_type": ErrorType.TIMEOUT,
                    "error_message": "Timeout",
                    "exception_class": None,
                },
                {
                    "example_id": "train_1",
                    "error_type": ErrorType.TIMEOUT,
                    "error_message": "Timeout",
                    "exception_class": None,
                },
                {
                    "example_id": "train_2",
                    "error_type": ErrorType.RUNTIME,
                    "error_message": "TypeError",
                    "exception_class": "TypeError",
                },
            ],
            "error_summary": {"timeout": 2, "runtime": 1},
        }
        assert classify_error(fitness_result) == ErrorType.TIMEOUT

    def test_classify_fallback_to_first_error(self):
        """Test fallback when error_summary is missing but error_details exist."""
        fitness_result = {
            "fitness": 0,
            "train_correct": 0,
            "train_total": 3,
            "test_correct": 0,
            "test_total": 1,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "execution_errors": ["Train example 0: syntax"],
            "error_details": [
                {
                    "example_id": "train_0",
                    "error_type": ErrorType.SYNTAX,
                    "error_message": "SyntaxError",
                    "exception_class": "SyntaxError",
                }
            ],
            # No error_summary
        }
        assert classify_error(fitness_result) == ErrorType.SYNTAX


class TestGetDebuggingStrategy:
    """Tests for get_debugging_strategy() function."""

    def test_syntax_error_strategy(self):
        """Test debugging strategy for syntax errors."""
        strategy = get_debugging_strategy(ErrorType.SYNTAX)
        assert "syntax" in strategy.lower()
        assert "colon" in strategy.lower()
        assert "indentation" in strategy.lower()

    def test_runtime_error_strategy(self):
        """Test debugging strategy for runtime errors."""
        strategy = get_debugging_strategy(ErrorType.RUNTIME)
        assert "runtime" in strategy.lower()
        assert "bounds checking" in strategy.lower() or "index" in strategy.lower()

    def test_timeout_error_strategy(self):
        """Test debugging strategy for timeout errors."""
        strategy = get_debugging_strategy(ErrorType.TIMEOUT)
        assert "performance" in strategy.lower() or "timeout" in strategy.lower()
        assert "loop" in strategy.lower()

    def test_logic_error_strategy(self):
        """Test debugging strategy for logic errors."""
        strategy = get_debugging_strategy(ErrorType.LOGIC)
        assert "logic" in strategy.lower() or "algorithm" in strategy.lower()
        assert "pattern" in strategy.lower() or "transformation" in strategy.lower()

    def test_validation_error_strategy(self):
        """Test debugging strategy for validation errors."""
        strategy = get_debugging_strategy(ErrorType.VALIDATION)
        assert "solve" in strategy.lower()
        assert "signature" in strategy.lower() or "function" in strategy.lower()

    def test_strategy_returns_non_empty_string(self):
        """Test that all strategies return non-empty strings."""
        for error_type in ErrorType:
            strategy = get_debugging_strategy(error_type)
            assert isinstance(strategy, str)
            assert len(strategy) > 0
