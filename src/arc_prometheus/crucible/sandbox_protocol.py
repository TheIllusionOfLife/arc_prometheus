"""
Protocol definition for sandbox execution environments.

This module defines the ExecutionEnvironment protocol that all sandbox
implementations must follow, enabling pluggable execution backends
(multiprocessing, Docker, etc.) with a consistent interface.
"""

from typing import Any, Protocol
import numpy as np


class ExecutionEnvironment(Protocol):
    """
    Protocol for sandbox execution environments.

    All sandbox implementations (multiprocessing, Docker, etc.) must
    implement this protocol to ensure consistent behavior and interface.

    Example:
        >>> # Multiprocessing implementation
        >>> sandbox = MultiprocessSandbox()
        >>> success, result, error = sandbox.execute(code, grid, timeout=5)

        >>> # Docker implementation
        >>> sandbox = DockerSandbox()
        >>> success, result, error = sandbox.execute(code, grid, timeout=5)
    """

    def execute(
        self,
        solver_code: str,
        task_grid: np.ndarray,
        timeout: int,
    ) -> tuple[bool, np.ndarray | None, dict[str, Any] | None]:
        """
        Execute solver code in an isolated environment.

        Args:
            solver_code: Python code containing solve() function
            task_grid: Input grid as numpy array (dtype=np.int64)
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (success, result_grid, error_detail):

            On success:
                - success: True
                - result_grid: Output grid as numpy array
                - error_detail: None

            On failure:
                - success: False
                - result_grid: None
                - error_detail: Dict with keys:
                    - error_type: str (ErrorType enum value)
                    - error_message: str (human-readable description)
                    - exception_class: str | None (Python exception class name)

        Example:
            >>> code = "def solve(grid): return grid * 2"
            >>> grid = np.array([[1, 2], [3, 4]], dtype=np.int64)
            >>> success, result, error = sandbox.execute(code, grid, timeout=5)
            >>> if success:
            ...     print(f"Result: {result}")
            ... else:
            ...     print(f"Error: {error['error_message']}")
        """
        ...
