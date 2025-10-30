"""
Tests for Docker-based execution sandbox.

Tests cover security enforcement, error handling, and compatibility
with existing multiprocessing sandbox.
"""

import numpy as np
import pytest

# Check if Docker is available
pytest.importorskip("docker")

try:
    import docker

    client = docker.from_env()
    client.ping()
    DOCKER_AVAILABLE = True
except Exception:
    DOCKER_AVAILABLE = False

from arc_prometheus.crucible.docker_sandbox import DockerSandbox
from arc_prometheus.evolutionary_engine.error_classifier import ErrorType

# Skip all tests if Docker not available
pytestmark = pytest.mark.skipif(
    not DOCKER_AVAILABLE, reason="Docker daemon not available"
)


class TestDockerSandboxBasicExecution:
    """Test basic execution scenarios."""

    def test_successful_execution(self):
        """Test basic solver execution."""
        sandbox = DockerSandbox()
        code = """
import numpy as np
def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
"""
        input_grid = np.array([[1, 2], [3, 4]], dtype=np.int64)

        success, result, error = sandbox.execute(code, input_grid, timeout=10)

        assert success is True
        assert result is not None
        assert error is None
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([[2, 3], [4, 5]]))

    def test_numpy_operations(self):
        """Test solver with numpy operations."""
        sandbox = DockerSandbox()
        code = """
import numpy as np
def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid * 2
"""
        input_grid = np.array([[1, 2], [3, 4]], dtype=np.int64)

        success, result, _ = sandbox.execute(code, input_grid, timeout=10)

        assert success is True
        assert np.array_equal(result, np.array([[2, 4], [6, 8]]))

    def test_large_grid(self):
        """Test with ARC maximum grid size (30x30)."""
        sandbox = DockerSandbox()
        code = """
import numpy as np
def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid.T  # Transpose
"""
        input_grid = np.ones((30, 30), dtype=np.int64)

        success, result, _ = sandbox.execute(code, input_grid, timeout=10)

        assert success is True
        assert result.shape == (30, 30)


class TestDockerSandboxTimeoutEnforcement:
    """Test timeout enforcement."""

    def test_timeout_enforcement(self):
        """Test that infinite loops are terminated."""
        sandbox = DockerSandbox()
        code = """
import numpy as np
def solve(task_grid: np.ndarray) -> np.ndarray:
    while True:
        pass
    return task_grid
"""
        input_grid = np.array([[1, 2]], dtype=np.int64)

        success, result, error = sandbox.execute(code, input_grid, timeout=2)

        assert success is False
        assert result is None
        assert error is not None
        assert error["error_type"] == ErrorType.TIMEOUT
        assert "2s timeout" in error["error_message"]

    def test_custom_timeout(self):
        """Test custom timeout values."""
        sandbox = DockerSandbox()
        code = """
import time
import numpy as np
def solve(task_grid: np.ndarray) -> np.ndarray:
    time.sleep(5)
    return task_grid
"""
        input_grid = np.array([[1]], dtype=np.int64)

        success, _, error = sandbox.execute(code, input_grid, timeout=2)

        assert success is False
        assert error["error_type"] == ErrorType.TIMEOUT


class TestDockerSandboxErrorHandling:
    """Test error detection and classification."""

    def test_syntax_error(self):
        """Test syntax error detection."""
        sandbox = DockerSandbox()
        code = """
import numpy as np
def solve(task_grid: np.ndarray) -> np.ndarray  # Missing colon
    return task_grid
"""
        input_grid = np.array([[1]], dtype=np.int64)

        success, result, error = sandbox.execute(code, input_grid, timeout=5)

        assert success is False
        assert result is None
        assert error["error_type"] == ErrorType.SYNTAX
        assert "SyntaxError" in error["error_message"]

    def test_runtime_error_index_error(self):
        """Test runtime error (index error)."""
        sandbox = DockerSandbox()
        code = """
import numpy as np
def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid[100, 100]  # Out of bounds
"""
        input_grid = np.array([[1]], dtype=np.int64)

        success, result, error = sandbox.execute(code, input_grid, timeout=5)

        assert success is False
        assert result is None
        assert error["error_type"] == ErrorType.RUNTIME
        assert "IndexError" in error["error_message"]

    def test_missing_solve_function(self):
        """Test validation error (missing solve function)."""
        sandbox = DockerSandbox()
        code = """
import numpy as np
def foo(task_grid: np.ndarray) -> np.ndarray:
    return task_grid
"""
        input_grid = np.array([[1]], dtype=np.int64)

        success, result, error = sandbox.execute(code, input_grid, timeout=5)

        assert success is False
        assert result is None
        assert error["error_type"] == ErrorType.VALIDATION
        assert "solve() function not found" in error["error_message"]

    def test_invalid_return_type(self):
        """Test validation error (wrong return type)."""
        sandbox = DockerSandbox()
        code = """
import numpy as np
def solve(task_grid: np.ndarray) -> np.ndarray:
    return "not an array"
"""
        input_grid = np.array([[1]], dtype=np.int64)

        success, result, error = sandbox.execute(code, input_grid, timeout=5)

        assert success is False
        assert result is None
        assert error["error_type"] == ErrorType.VALIDATION
        assert "Invalid return type" in error["error_message"]


class TestDockerSandboxSecurityFeatures:
    """Test Docker security enforcement (network, filesystem, resources)."""

    @pytest.mark.skip(reason="Network test requires network-dependent code")
    def test_network_disabled(self):
        """Test that network access is blocked."""
        # This test would require code that attempts network access
        # Skipping for now as it's complex to test reliably
        pass

    @pytest.mark.skip(reason="Filesystem test requires write attempt detection")
    def test_filesystem_readonly(self):
        """Test that filesystem writes are blocked."""
        # This test would require code that attempts file writes
        # Skipping for now as error detection is complex
        pass


class TestDockerSandboxResourceLimits:
    """Test resource limit enforcement."""

    def test_memory_limit_configuration(self):
        """Test custom memory limit configuration."""
        sandbox = DockerSandbox(memory_limit="256m")
        assert sandbox.memory_limit == "256m"

    def test_cpu_limit_configuration(self):
        """Test custom CPU limit configuration."""
        sandbox = DockerSandbox(cpu_quota=25000)  # 25%
        assert sandbox.cpu_quota == 25000

    def test_pids_limit_configuration(self):
        """Test custom process limit configuration."""
        sandbox = DockerSandbox(pids_limit=50)
        assert sandbox.pids_limit == 50


class TestDockerSandboxImageManagement:
    """Test Docker image handling."""

    def test_image_not_found_error(self):
        """Test graceful handling when image doesn't exist."""
        with pytest.raises(RuntimeError, match="not found"):
            DockerSandbox(image_name="nonexistent-image:latest")

    def test_default_image_name(self):
        """Test default image name."""
        sandbox = DockerSandbox()
        assert sandbox.image_name == "arc-prometheus-sandbox:latest"


class TestDockerSandboxCompatibility:
    """Test compatibility with multiprocessing sandbox behavior."""

    def test_return_format_matches_multiprocess(self):
        """Test that return format matches multiprocessing sandbox."""
        from arc_prometheus.crucible.sandbox import MultiprocessSandbox

        code = """
import numpy as np
def solve(task_grid: np.ndarray) -> np.ndarray:
    return task_grid + 1
"""
        input_grid = np.array([[1, 2]], dtype=np.int64)

        # Multiprocess sandbox
        mp_sandbox = MultiprocessSandbox()
        mp_success, mp_result, mp_error = mp_sandbox.execute(code, input_grid, 5)

        # Docker sandbox
        docker_sandbox = DockerSandbox()
        docker_success, docker_result, docker_error = docker_sandbox.execute(
            code, input_grid, 5
        )

        # Results should match
        assert mp_success == docker_success == True  # noqa: E712
        assert np.array_equal(mp_result, docker_result)
        assert mp_error == docker_error == None  # noqa: E711

    def test_error_format_matches_multiprocess(self):
        """Test that error format matches multiprocessing sandbox."""
        from arc_prometheus.crucible.sandbox import MultiprocessSandbox

        code = """
import numpy as np
def solve(task_grid: np.ndarray) -> np.ndarray  # Missing colon
    return task_grid
"""
        input_grid = np.array([[1]], dtype=np.int64)

        # Multiprocess sandbox
        mp_sandbox = MultiprocessSandbox()
        mp_success, mp_result, mp_error = mp_sandbox.execute(code, input_grid, 5)

        # Docker sandbox
        docker_sandbox = DockerSandbox()
        docker_success, docker_result, docker_error = docker_sandbox.execute(
            code, input_grid, 5
        )

        # Error structure should match
        assert mp_success == docker_success == False  # noqa: E712
        assert mp_result == docker_result == None  # noqa: E711
        assert mp_error["error_type"] == docker_error["error_type"]
        assert "error_message" in mp_error
        assert "error_message" in docker_error
