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
        result_queue: Queue for returning (success, result, error_msg) tuple

    Returns:
        None (results sent through queue)
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
                result_queue.put((False, None, "solve() function not found"))
                return

            # Get the solve function
            # Use Any type because exec_globals dict contains dynamic content
            solve_func: Any = exec_globals["solve"]

            # Execute solve() with the input grid
            result: Any = solve_func(task_grid)

            # Validate result type
            if not isinstance(result, np.ndarray):
                result_queue.put(
                    (False, None, f"Invalid return type: {type(result).__name__}")
                )
                return

            # Success - send result back
            result_queue.put((True, result, None))

        finally:
            # Restore original builtins module
            sys.modules["builtins"] = original_builtins

    except SyntaxError as e:
        result_queue.put((False, None, f"SyntaxError: {str(e)}"))
    except TypeError as e:
        # Catch wrong function signature errors
        result_queue.put((False, None, f"TypeError: {str(e)}"))
    except Exception as e:
        # Catch all other runtime exceptions
        result_queue.put((False, None, f"{type(e).__name__}: {str(e)}"))


def safe_execute(
    solver_code: str, task_grid: np.ndarray, timeout: int = 5
) -> tuple[bool, np.ndarray | None]:
    """Execute LLM-generated solver code safely in isolated process.

    Runs untrusted code in a separate process with:
    - Timeout enforcement
    - Restricted builtins (no eval, exec, compile, open)
    - Prevention of builtins bypass via "import builtins"
    - Exception handling
    - Return type validation

    Args:
        solver_code: Python code containing solve() function
        task_grid: Input grid to pass to solve()
        timeout: Maximum execution time in seconds (default: 5)

    Returns:
        (True, result_grid) on successful execution
        (False, None) on failure/timeout/exception

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
        >>> success, result = safe_execute(solver_code, input_grid)
        >>> success
        True
        >>> result
        array([[2, 3],
               [4, 5]])

        >>> # Timeout example
        >>> infinite_loop = '''
        ... import numpy as np
        ... def solve(task_grid: np.ndarray) -> np.ndarray:
        ...     while True:
        ...         pass
        ... '''
        >>> success, result = safe_execute(infinite_loop, input_grid, timeout=1)
        >>> success
        False
        >>> result is None
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

        return (False, None)

    # Process completed - check for results using get_nowait() instead of empty()
    # empty() is unreliable due to multiprocessing race conditions
    try:
        success, result, error_msg = result_queue.get_nowait()
        if success:
            return (True, result)
        else:
            # Error occurred during execution - log to stderr for debugging
            import sys

            if error_msg:
                print(f"Sandbox execution error: {error_msg}", file=sys.stderr)
            return (False, None)
    except Exception:
        # No result in queue (unexpected) - process may have crashed
        return (False, None)
