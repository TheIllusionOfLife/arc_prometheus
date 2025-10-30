"""Safe execution sandbox for untrusted LLM-generated solver code.

Provides isolated execution environment using multiprocessing with:
- Process isolation (separate memory space)
- Timeout enforcement
- Restricted builtins (no eval, exec, compile, open)
- Prevention of builtins bypass via "import builtins"
- Exception handling

IMPORTANT SECURITY LIMITATIONS:
- Multiprocessing does NOT prevent filesystem access
- Multiprocessing does NOT prevent network access
- For production use, consider Docker with:
  - Read-only filesystem
  - Network disabled
  - Resource limits (CPU, memory)
"""

import multiprocessing as mp
import queue
from multiprocessing import Queue
from typing import Any

import numpy as np

# Dangerous builtins that must be blocked in sandbox
DANGEROUS_BUILTINS = frozenset(
    [
        "eval",
        "exec",
        "compile",
        "open",
        "__loader__",
        "__build_class__",
    ]
)


def _worker_execute(code_str: str, task_grid: np.ndarray, result_queue: Queue) -> None:
    """Worker function that executes solver code in isolated process.

    This function runs in a separate process with restricted builtins.
    Any exceptions are caught and sent back through the queue.

    Args:
        code_str: Python code containing solve() function
        task_grid: Input grid to pass to solve()
        result_queue: Queue for returning (success, result, error_detail) tuple

    Returns:
        None (results sent through queue)
        Result format: (bool, np.ndarray | None, dict | None)
        - success: True if execution succeeded
        - result: Output grid on success, None on failure
        - error_detail: Structured error info on failure, None on success
    """
    try:
        # Create restricted execution environment
        # Block dangerous builtins AND prevent bypassing via "import builtins"
        # Note: We cannot fully sandbox file I/O or network access with multiprocessing alone

        import builtins
        import sys
        from types import ModuleType

        # Create restricted builtins module to prevent bypass via "import builtins"
        restricted_builtins_module = ModuleType("builtins")
        restricted_builtins = {
            name: getattr(builtins, name)
            for name in dir(builtins)
            if name not in DANGEROUS_BUILTINS
        }

        # Set attributes on the restricted module
        for name, value in restricted_builtins.items():
            setattr(restricted_builtins_module, name, value)

        # Replace builtins in sys.modules to prevent "import builtins" bypass
        original_builtins = sys.modules["builtins"]
        sys.modules["builtins"] = restricted_builtins_module

        try:
            # Create execution namespace with restricted builtins
            exec_globals = {
                "__builtins__": restricted_builtins,
            }

            # Execute the solver code to define solve() function
            exec(code_str, exec_globals)  # nosec B102 # noqa: S102

            # Check if solve() function exists
            if "solve" not in exec_globals:
                from ..evolutionary_engine.error_classifier import ErrorType

                error_detail = {
                    "error_type": ErrorType.VALIDATION,
                    "error_message": "solve() function not found",
                    "exception_class": None,
                }
                result_queue.put((False, None, error_detail))
                return

            # Get the solve function
            # Use Any type because exec_globals dict contains dynamic content
            solve_func: Any = exec_globals["solve"]

            # Execute solve() with the input grid
            result: Any = solve_func(task_grid)

            # Validate result type
            if not isinstance(result, np.ndarray):
                from ..evolutionary_engine.error_classifier import ErrorType

                error_detail = {
                    "error_type": ErrorType.VALIDATION,
                    "error_message": f"Invalid return type: {type(result).__name__} (expected np.ndarray)",
                    "exception_class": None,
                }
                result_queue.put((False, None, error_detail))
                return

            # Success - send result back
            result_queue.put((True, result, None))

        finally:
            # Restore original builtins module
            sys.modules["builtins"] = original_builtins

    except SyntaxError as e:
        from ..evolutionary_engine.error_classifier import ErrorType

        error_detail = {
            "error_type": ErrorType.SYNTAX,
            "error_message": f"SyntaxError: {str(e)}",
            "exception_class": "SyntaxError",
        }
        result_queue.put((False, None, error_detail))
    except TypeError as e:
        # Catch wrong function signature errors and type mismatches
        from ..evolutionary_engine.error_classifier import ErrorType

        error_detail = {
            "error_type": ErrorType.RUNTIME,
            "error_message": f"TypeError: {str(e)}",
            "exception_class": "TypeError",
        }
        result_queue.put((False, None, error_detail))
    except Exception as e:
        # Catch all other runtime exceptions
        from ..evolutionary_engine.error_classifier import ErrorType

        error_detail = {
            "error_type": ErrorType.RUNTIME,
            "error_message": f"{type(e).__name__}: {str(e)}",
            "exception_class": type(e).__name__,
        }
        result_queue.put((False, None, error_detail))


def safe_execute(
    solver_code: str, task_grid: np.ndarray, timeout: int = 5
) -> tuple[bool, np.ndarray | None, dict[str, Any] | None]:
    """Execute LLM-generated solver code safely in isolated process.

    Runs untrusted code in a separate process with:
    - Timeout enforcement
    - Restricted builtins (no eval, exec, compile, open)
    - Prevention of builtins bypass via "import builtins"
    - Exception handling
    - Return type validation
    - Structured error reporting

    Args:
        solver_code: Python code containing solve() function
        task_grid: Input grid to pass to solve()
        timeout: Maximum execution time in seconds (default: 5)

    Returns:
        Tuple of (success, result, error_detail):
        - (True, result_grid, None) on successful execution
        - (False, None, error_detail) on failure/timeout/exception

        error_detail dict contains:
        - error_type: "syntax" | "runtime" | "timeout" | "validation"
        - error_message: Human-readable error description
        - exception_class: Python exception class name (or None for timeout)

    Security Notes:
        - Runs in isolated multiprocessing.Process (separate memory space)
        - Restricted builtins prevent eval, exec, compile, open
        - sys.modules["builtins"] replaced to prevent "import builtins" bypass
        - Timeout enforcement with process termination
        - Error messages logged to stderr for debugging
        - LIMITATION: Does not prevent filesystem/network access
          (multiprocessing isolation is not a security sandbox)
        - For production: Use Docker with read-only filesystem and network disabled

    Examples:
        >>> solver_code = '''
        ... import numpy as np
        ... def solve(task_grid: np.ndarray) -> np.ndarray:
        ...     return task_grid + 1
        ... '''
        >>> input_grid = np.array([[1, 2], [3, 4]])
        >>> success, result, error_detail = safe_execute(solver_code, input_grid)
        >>> success
        True
        >>> result
        array([[2, 3],
               [4, 5]])
        >>> error_detail is None
        True

        >>> # Timeout example
        >>> infinite_loop = '''
        ... import numpy as np
        ... def solve(task_grid: np.ndarray) -> np.ndarray:
        ...     while True:
        ...         pass
        ... '''
        >>> success, result, error_detail = safe_execute(infinite_loop, input_grid, timeout=1)
        >>> success
        False
        >>> result is None
        True
        >>> error_detail is not None
        True
    """
    # Create queue for inter-process communication
    result_queue: Queue = mp.Queue()

    # Create and start worker process
    process = mp.Process(
        target=_worker_execute, args=(solver_code, task_grid, result_queue)
    )
    process.start()

    # Wait for process to complete with timeout
    process.join(timeout=timeout)

    # Check if process is still running (timeout occurred)
    if process.is_alive():
        # Timeout - terminate the process
        process.terminate()
        process.join(timeout=1)  # Wait for termination

        # Force kill if still alive
        if process.is_alive():
            process.kill()
            process.join()

        # Return timeout error detail
        from ..evolutionary_engine.error_classifier import ErrorType

        error_detail = {
            "error_type": ErrorType.TIMEOUT,
            "error_message": f"Execution exceeded {timeout}s timeout",
            "exception_class": None,
        }
        return (False, None, error_detail)

    # Process completed - check for results using get_nowait() instead of empty()
    # empty() is unreliable due to multiprocessing race conditions
    try:
        success, result, error_detail = result_queue.get_nowait()
        if success:
            return (True, result, None)
        else:
            # Error occurred during execution - log to stderr for debugging
            import sys

            if error_detail and "error_message" in error_detail:
                print(
                    f"Sandbox execution error: {error_detail['error_message']}",
                    file=sys.stderr,
                )
            return (False, None, error_detail)
    except queue.Empty:
        # No result in queue (unexpected) - process may have crashed
        from ..evolutionary_engine.error_classifier import ErrorType

        error_detail = {
            "error_type": ErrorType.RUNTIME,
            "error_message": "Process crashed unexpectedly (no result in queue)",
            "exception_class": None,
        }
        return (False, None, error_detail)
