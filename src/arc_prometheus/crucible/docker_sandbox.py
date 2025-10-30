"""
Docker-based execution sandbox for production-grade security.

Provides isolated execution environment using Docker containers with:
- Network disabled (no external communication)
- Read-only filesystem (except /tmp for numpy operations)
- Resource limits (CPU, memory, processes)
- Timeout enforcement at container level
- Non-root user execution (UID 1000)

Security guarantees superior to multiprocessing sandbox.
"""

import base64
import pickle
import textwrap
from typing import Any, cast

import numpy as np

try:
    import docker
    from docker.errors import ContainerError, DockerException, ImageNotFound
except ImportError:
    # Docker not installed - will raise error in __init__
    pass

from ..evolutionary_engine.error_classifier import ErrorType


class DockerSandbox:
    """
    Docker-based secure execution sandbox.

    Provides production-grade security with container isolation,
    resource limits, and network/filesystem restrictions.

    Security Features:
        - Network completely disabled
        - Filesystem read-only (except /tmp tmpfs)
        - Memory limit: 512MB (configurable)
        - CPU limit: 50% of one core (configurable)
        - Process limit: 100 (prevents fork bombs)
        - Container-level timeout enforcement
        - Non-root user execution

    Example:
        >>> sandbox = DockerSandbox()
        >>> code = "def solve(grid): return grid * 2"
        >>> grid = np.array([[1, 2]], dtype=np.int64)
        >>> success, result, error = sandbox.execute(code, grid, timeout=5)
        >>> if success:
        ...     print(f"Result: {result}")

    Args:
        memory_limit: Memory limit string (e.g., "512m", "1g")
        cpu_quota: CPU quota in microseconds (50000 = 50% of one core)
        pids_limit: Maximum number of processes
        image_name: Docker image to use (default: arc-prometheus-sandbox:latest)
    """

    def __init__(
        self,
        memory_limit: str = "512m",
        cpu_quota: int = 50000,
        pids_limit: int = 100,
        image_name: str = "arc-prometheus-sandbox:latest",
    ):
        """Initialize Docker sandbox with resource limits."""
        try:
            self.client = docker.from_env()  # type: ignore[attr-defined]
            # Verify Docker daemon is accessible
            self.client.ping()
        except DockerException as e:
            raise RuntimeError(
                f"Docker daemon not accessible: {e}. "
                "Ensure Docker is installed and running."
            ) from e

        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.pids_limit = pids_limit
        self.image_name = image_name

        # Verify image exists
        try:
            self.client.images.get(self.image_name)
        except ImageNotFound as e:
            raise RuntimeError(
                f"Docker image '{self.image_name}' not found. "
                f"Build it with: docker build -t {self.image_name} "
                "-f docker/sandbox.Dockerfile ."
            ) from e

    def execute(
        self,
        solver_code: str,
        task_grid: np.ndarray,
        timeout: int,
    ) -> tuple[bool, np.ndarray | None, dict[str, Any] | None]:
        """
        Execute solver code in isolated Docker container.

        Args:
            solver_code: Python code containing solve() function
            task_grid: Input grid as numpy array
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (success, result_grid, error_detail):
            - (True, result_grid, None) on success
            - (False, None, error_detail) on failure

        Security:
            - Executes in isolated container
            - Network disabled
            - Read-only filesystem (except /tmp tmpfs)
            - Resource limits enforced
            - Container destroyed after execution

        Example:
            >>> sandbox = DockerSandbox()
            >>> code = '''
            ... import numpy as np
            ... def solve(grid):
            ...     return grid + 1
            ... '''
            >>> grid = np.array([[1, 2]], dtype=np.int64)
            >>> success, result, error = sandbox.execute(code, grid, 5)
        """
        container = None
        try:
            # Create execution script with result serialization
            exec_script = self._create_execution_script(solver_code, task_grid)

            # Create container with security restrictions
            container = self.client.containers.create(
                image=self.image_name,
                command=["python", "-c", exec_script],
                # Security: Network disabled
                network_disabled=True,
                # Security: Read-only filesystem
                read_only=True,
                # Security: /tmp as writable tmpfs (numpy needs temp space)
                tmpfs={"/tmp": "size=100m,uid=1000"},  # noqa: S108  # nosec
                # Resource limits
                mem_limit=self.memory_limit,
                memswap_limit=self.memory_limit,  # Prevent swap usage
                cpu_period=100000,  # 100ms period
                cpu_quota=self.cpu_quota,  # 50% of one core by default
                pids_limit=self.pids_limit,  # Prevent fork bombs
                # Cleanup (we'll remove manually after getting logs)
                detach=True,
            )

            # Start container and wait for completion
            container.start()

            try:
                # Wait with timeout
                exit_info = container.wait(timeout=timeout)
                exit_code = exit_info["StatusCode"]
            except Exception:
                # Timeout occurred
                container.kill()
                container.wait()  # Ensure container stopped

                error_detail: dict[str, Any] = {
                    "error_type": ErrorType.TIMEOUT,
                    "error_message": f"Execution exceeded {timeout}s timeout",
                    "exception_class": None,
                }
                return (False, None, error_detail)

            # Get container logs (stdout + stderr)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8")

            # Check for OOM kill
            inspect = container.attrs
            if inspect["State"].get("OOMKilled", False):
                error_detail = {
                    "error_type": ErrorType.RUNTIME,
                    "error_message": f"Memory limit exceeded ({self.memory_limit})",
                    "exception_class": "MemoryError",
                }
                return (False, None, error_detail)

            # Parse result or error from logs
            return self._parse_result(logs, exit_code)

        except ContainerError as e:
            # Container execution error
            error_detail = {
                "error_type": ErrorType.RUNTIME,
                "error_message": f"Container error: {str(e)}",
                "exception_class": "ContainerError",
            }
            return (False, None, error_detail)

        except DockerException as e:
            # Docker API error
            error_detail = {
                "error_type": ErrorType.RUNTIME,
                "error_message": f"Docker error: {str(e)}",
                "exception_class": "DockerException",
            }
            return (False, None, error_detail)

        finally:
            # Cleanup: Always remove container
            if container is not None:
                try:  # noqa: SIM105
                    container.remove(force=True)
                except Exception:  # noqa: S110
                    # Ignore cleanup errors - container may already be removed
                    pass

    def _create_execution_script(self, solver_code: str, task_grid: np.ndarray) -> str:
        """
        Create Python script for container execution.

        Embeds solver code, input grid, and result serialization logic.

        Args:
            solver_code: User's solve() function code
            task_grid: Input grid to process

        Returns:
            Complete Python script as string
        """
        # Serialize input grid to pickle + base64
        grid_bytes = pickle.dumps(task_grid)
        grid_b64 = base64.b64encode(grid_bytes).decode("ascii")

        # Create execution script
        # This runs inside the container and must be completely self-contained
        script = textwrap.dedent(
            f"""
            import sys
            import pickle
            import base64
            import numpy as np

            try:
                # Deserialize input grid
                grid_b64 = {repr(grid_b64)}
                grid_bytes = base64.b64decode(grid_b64)
                task_grid = pickle.loads(grid_bytes)

                # Execute user's solver code
                exec_globals = {{}}
                solver_code = {repr(solver_code)}
                exec(solver_code, exec_globals)

                # Validate solve() function exists
                if "solve" not in exec_globals:
                    print("ERROR:VALIDATION:solve() function not found", file=sys.stderr)
                    sys.exit(1)

                # Get solve function
                solve_func = exec_globals["solve"]

                # Execute solve()
                result = solve_func(task_grid)

                # Validate return type
                if not isinstance(result, np.ndarray):
                    print(f"ERROR:VALIDATION:Invalid return type: {{type(result).__name__}} (expected np.ndarray)", file=sys.stderr)
                    sys.exit(1)

                # Serialize result
                result_bytes = pickle.dumps(result)
                result_b64 = base64.b64encode(result_bytes).decode("ascii")

                # Output result marker
                print(f"RESULT:{{result_b64}}")

            except SyntaxError as e:
                print(f"ERROR:SYNTAX:SyntaxError: {{str(e)}}", file=sys.stderr)
                sys.exit(1)

            except TypeError as e:
                print(f"ERROR:RUNTIME:TypeError: {{str(e)}}", file=sys.stderr)
                sys.exit(1)

            except Exception as e:
                print(f"ERROR:RUNTIME:{{type(e).__name__}}: {{str(e)}}", file=sys.stderr)
                sys.exit(1)
            """
        )

        return script

    def _parse_result(
        self, logs: str, exit_code: int
    ) -> tuple[bool, np.ndarray | None, dict[str, Any] | None]:
        """
        Parse container logs to extract result or error.

        Args:
            logs: Combined stdout + stderr from container
            exit_code: Container exit code

        Returns:
            Tuple of (success, result, error_detail)
        """
        # Success case: Look for RESULT: marker
        if "RESULT:" in logs:
            try:
                # Extract base64-encoded result
                result_line = [line for line in logs.split("\n") if "RESULT:" in line][
                    0
                ]
                result_b64 = result_line.split("RESULT:")[1].strip()

                # Deserialize result
                result_bytes = base64.b64decode(result_b64)
                result_grid = pickle.loads(result_bytes)  # noqa: S301  # nosec

                return (True, result_grid, None)

            except Exception as e:
                # Result parsing failed
                error_detail = {
                    "error_type": ErrorType.RUNTIME,
                    "error_message": f"Failed to parse result: {str(e)}",
                    "exception_class": "ResultParsingError",
                }
                return (False, None, error_detail)

        # Error case: Parse ERROR: markers from stderr
        if "ERROR:" in logs:
            error_lines = [line for line in logs.split("\n") if "ERROR:" in line]
            if error_lines:
                error_line = error_lines[0]
                # Format: ERROR:TYPE:message
                parts = error_line.split(":", 2)
                if len(parts) >= 3:
                    error_type_str = parts[1].strip()
                    error_message = parts[2].strip()

                    # Map error type string to ErrorType enum
                    if error_type_str == "SYNTAX":
                        error_type = ErrorType.SYNTAX
                        exception_class: str | None = "SyntaxError"
                    elif error_type_str == "VALIDATION":
                        error_type = ErrorType.VALIDATION
                        exception_class = None
                    else:  # RUNTIME or unknown
                        error_type = ErrorType.RUNTIME
                        # Extract exception class from message if present
                        if ":" in error_message:
                            exception_class = error_message.split(":", 1)[0].strip()
                        else:
                            exception_class = "RuntimeError"

                    error_detail = cast(
                        dict[str, Any],
                        {
                            "error_type": error_type,
                            "error_message": error_message,
                            "exception_class": exception_class,
                        },
                    )
                    return (False, None, error_detail)

        # Exit code != 0 but no parseable error
        if exit_code != 0:
            error_detail = cast(
                dict[str, Any],
                {
                    "error_type": ErrorType.RUNTIME,
                    "error_message": f"Container exited with code {exit_code}. Logs: {logs[:500]}",
                    "exception_class": None,
                },
            )
            return (False, None, error_detail)

        # Unexpected: Exit code 0 but no result
        error_detail = cast(
            dict[str, Any],
            {
                "error_type": ErrorType.RUNTIME,
                "error_message": "Container succeeded but produced no result",
                "exception_class": None,
            },
        )
        return (False, None, error_detail)
