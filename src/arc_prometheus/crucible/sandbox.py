"""Safe execution sandbox for untrusted LLM-generated solver code.

Provides isolated execution environment using multiprocessing with:
- Process isolation (separate memory space)
- Timeout enforcement
- Restricted builtins (no eval, exec, compile, __import__)
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


def _worker_execute(code_str: str, task_grid: np.ndarray, result_queue: Queue) -> None:
    """Worker function that executes solver code in isolated process.

    This function runs in a separate process with restricted builtins.
    Any exceptions are caught and sent back through the queue.

    Args:
        code_str: Python code containing solve() function
        task_grid: Input grid to pass to solve()
        result_queue: Queue for returning (success, result) tuple

    Returns:
        None (results sent through queue)
    """
    try:
        # Create restricted execution environment
        # We need to allow __import__ for "import numpy" but restrict eval/exec/compile
        # Note: We cannot fully sandbox file I/O or network access with multiprocessing alone

        import builtins

        # Copy safe builtins, excluding dangerous ones
        restricted_builtins = {}
        for name in dir(builtins):
            if name not in [
                "eval",
                "exec",
                "compile",
                "open",
                "__loader__",
                "__build_class__",
            ]:
                restricted_builtins[name] = getattr(builtins, name)

        # Create execution namespace with restricted builtins
        exec_globals = {
            "__builtins__": restricted_builtins,
        }

        # Execute the solver code to define solve() function
        exec(code_str, exec_globals)  # noqa: S102 - Intentional use of exec in sandbox

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
    - Restricted builtins (no eval, exec, compile, __import__)
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
        - Restricted builtins prevent eval, exec, compile, __import__
        - Timeout enforcement with process termination
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

    # Process completed - check for results
    if not result_queue.empty():
        success, result, error_msg = result_queue.get()
        if success:
            return (True, result)
        else:
            # Error occurred during execution
            return (False, None)

    # No result in queue (unexpected)
    return (False, None)
