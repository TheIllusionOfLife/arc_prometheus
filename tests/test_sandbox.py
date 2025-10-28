"""Tests for safe execution sandbox functionality.

This test suite validates the sandbox's ability to:
1. Execute valid solver code successfully
2. Enforce timeout on long-running code
3. Handle runtime exceptions gracefully
4. Handle syntax errors
5. Validate return types
6. Integrate with Phase 1.2 manual solver
"""

import contextlib
import os

import numpy as np

from arc_prometheus.crucible.sandbox import safe_execute


class TestBasicExecution:
    """Tests for basic successful execution scenarios."""

    def test_successful_execution(self):
        """Test that simple valid solver executes and returns correct result."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Simple transformation: add 1 to all values
    return task_grid + 1
"""
        input_grid = np.array([[1, 2], [3, 4]])
        expected_output = np.array([[2, 3], [4, 5]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is True
        assert result is not None
        assert np.array_equal(result, expected_output)

    def test_solver_with_numpy_operations(self):
        """Test solver using various numpy operations."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Use numpy operations: transpose and multiply
    return task_grid.T * 2
"""
        input_grid = np.array([[1, 2, 3], [4, 5, 6]])
        expected_output = np.array([[2, 8], [4, 10], [6, 12]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is True
        assert result is not None
        assert np.array_equal(result, expected_output)

    def test_return_format(self):
        """Verify return format is (bool, Optional[np.ndarray])."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return np.zeros((3, 3), dtype=int)
"""
        input_grid = np.array([[1]])

        result = safe_execute(solver_code, input_grid)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert result[1] is None or isinstance(result[1], np.ndarray)


class TestTimeoutEnforcement:
    """Tests for timeout handling."""

    def test_timeout_enforcement(self):
        """Test that infinite loop triggers timeout after 5 seconds."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Intentional infinite loop
    while True:
        pass
    return task_grid
"""
        input_grid = np.array([[1]])

        success, result = safe_execute(solver_code, input_grid, timeout=2)

        assert success is False
        assert result is None

    def test_custom_timeout(self):
        """Test that custom timeout parameter works."""
        solver_code = """
import time
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    time.sleep(3)  # Sleep for 3 seconds
    return task_grid
"""
        input_grid = np.array([[1]])

        # Should timeout with 1 second
        success, result = safe_execute(solver_code, input_grid, timeout=1)
        assert success is False
        assert result is None

        # Should succeed with 5 seconds
        success, result = safe_execute(solver_code, input_grid, timeout=5)
        assert success is True
        assert result is not None


class TestExceptionHandling:
    """Tests for runtime exception handling."""

    def test_runtime_exception_zero_division(self):
        """Test that ZeroDivisionError is caught and handled."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Intentional error
    x = 1 / 0
    return task_grid
"""
        input_grid = np.array([[1]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is False
        assert result is None

    def test_runtime_exception_index_error(self):
        """Test that IndexError is caught and handled."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Intentional index error
    return task_grid[100, 100]
"""
        input_grid = np.array([[1, 2], [3, 4]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is False
        assert result is None

    def test_syntax_error(self):
        """Test that syntax errors are caught and handled."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Intentional syntax error
    return task_grid +
"""
        input_grid = np.array([[1]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is False
        assert result is None


class TestInvalidReturnTypes:
    """Tests for invalid return type handling."""

    def test_invalid_return_type_string(self):
        """Test that solver returning string is handled."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return "not an array"
"""
        input_grid = np.array([[1]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is False
        assert result is None

    def test_invalid_return_type_none(self):
        """Test that solver returning None is handled."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return None
"""
        input_grid = np.array([[1]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is False
        assert result is None

    def test_invalid_return_type_list(self):
        """Test that solver returning list is handled."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return [[1, 2], [3, 4]]
"""
        input_grid = np.array([[1]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is False
        assert result is None


class TestMissingOrInvalidSolveFunction:
    """Tests for missing or invalid solve() function."""

    def test_missing_solve_function(self):
        """Test that code without solve() function fails gracefully."""
        solver_code = """
import numpy as np

def other_function():
    return 42
"""
        input_grid = np.array([[1]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is False
        assert result is None

    def test_solve_function_wrong_signature(self):
        """Test that solve() with wrong parameters fails gracefully."""
        solver_code = """
import numpy as np

def solve():  # Missing parameter
    return np.array([[1]])
"""
        input_grid = np.array([[1]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is False
        assert result is None


class TestPhase12Integration:  # noqa: N801
    """Integration tests with Phase 1.2 manual solver."""

    def test_with_phase1_2_manual_solver(self):
        """Test sandbox with actual Phase 1.2 manual solver code."""
        # Simplified version of Phase 1.2 solver for testing
        solver_code = """
import numpy as np
from collections import OrderedDict

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Extract diagonal pattern (simplified)
    diagonals = OrderedDict()

    for i in range(task_grid.shape[0]):
        for j in range(task_grid.shape[1]):
            if task_grid[i, j] != 0:
                diag_idx = i + j
                if diag_idx not in diagonals:
                    diagonals[diag_idx] = task_grid[i, j]

    if len(diagonals) == 0:
        return np.zeros_like(task_grid)

    diagonal_values = list(diagonals.values())
    pattern_len = len(diagonal_values)

    # Fill grid with repeating pattern
    output = np.zeros_like(task_grid)
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            pattern_idx = (col + row) % pattern_len
            output[row, col] = diagonal_values[pattern_idx]

    return output
"""
        # Simple 3x3 input with diagonal pattern
        input_grid = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is True
        assert result is not None
        assert result.shape == (3, 3)


class TestLargeGrids:
    """Tests for large grid processing."""

    def test_large_grid_execution(self):
        """Test processing of 30x30 grid (ARC maximum size)."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Fill with constant value
    return np.full_like(task_grid, 5)
"""
        input_grid = np.ones((30, 30), dtype=int)
        expected_output = np.full((30, 30), 5, dtype=int)

        success, result = safe_execute(solver_code, input_grid)

        assert success is True
        assert result is not None
        assert np.array_equal(result, expected_output)


class TestSecurityRestrictions:
    """Tests for security restrictions and documentation of limitations."""

    def test_restricted_builtins_eval(self):
        """Test that eval() is not available in restricted environment."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    eval("1 + 1")  # Should fail
    return task_grid
"""
        input_grid = np.array([[1]])

        success, result = safe_execute(solver_code, input_grid)

        # Should fail due to eval not being available
        assert success is False
        assert result is None

    def test_restricted_builtins_exec(self):
        """Test that exec() is not available in restricted environment."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    exec("x = 1")  # Should fail
    return task_grid
"""
        input_grid = np.array([[1]])

        success, result = safe_execute(solver_code, input_grid)

        # Should fail due to exec not being available
        assert success is False
        assert result is None

    def test_restricted_builtins_compile(self):
        """Test that compile() is not available in restricted environment."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    compile("1 + 1", "<string>", "eval")  # Should fail
    return task_grid
"""
        input_grid = np.array([[1]])

        success, result = safe_execute(solver_code, input_grid)

        # Should fail due to compile not being available
        assert success is False
        assert result is None

    def test_builtins_bypass_prevented(self):
        """Test that 'import builtins' bypass is prevented."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Try to bypass restrictions via "import builtins"
    import builtins
    builtins.eval("1 + 1")  # Should fail
    return task_grid
"""
        input_grid = np.array([[1]])

        success, result = safe_execute(solver_code, input_grid)

        # Should fail - the imported builtins module should also be restricted
        assert success is False
        assert result is None

    def test_filesystem_access_limitation_documented(self):
        """Document that multiprocessing doesn't prevent filesystem access.

        NOTE: This test documents a LIMITATION of the current sandbox.
        Python multiprocessing does NOT prevent filesystem access.
        For production use, consider Docker with read-only filesystem.
        """
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # This WILL execute (limitation of multiprocessing)
    # For production, use Docker with read-only filesystem
    try:
        with open('/tmp/test_sandbox_write.txt', 'w') as f:
            f.write('test')
    except:
        pass
    return task_grid
"""
        input_grid = np.array([[1]])

        # This test documents the limitation - it will succeed
        # because multiprocessing doesn't prevent file I/O
        success, result = safe_execute(solver_code, input_grid)

        # Clean up test file if it exists
        with contextlib.suppress(OSError):
            os.remove("/tmp/test_sandbox_write.txt")  # noqa: S108 - Test cleanup

        # Test succeeds, demonstrating the limitation
        assert success is True


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_grid(self):
        """Test handling of empty grid."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
"""
        input_grid = np.array([[]]).reshape(0, 0)

        success, result = safe_execute(solver_code, input_grid)

        assert success is True
        assert result is not None
        assert result.shape == (0, 0)

    def test_single_cell_grid(self):
        """Test handling of 1x1 grid."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid * 2
"""
        input_grid = np.array([[5]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is True
        assert result is not None
        assert np.array_equal(result, np.array([[10]]))

    def test_different_output_shape(self):
        """Test solver that returns different shape (should be valid)."""
        solver_code = """
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    # Return larger grid
    return np.zeros((5, 5), dtype=int)
"""
        input_grid = np.array([[1, 2], [3, 4]])

        success, result = safe_execute(solver_code, input_grid)

        assert success is True
        assert result is not None
        assert result.shape == (5, 5)
