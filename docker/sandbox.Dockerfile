# ARC-Prometheus Sandbox Container
#
# Minimal Python environment for secure execution of untrusted solver code.
# Provides production-grade isolation with:
# - Read-only filesystem
# - Network disabled
# - Resource limits (CPU, memory, processes)
# - Non-root user execution

FROM python:3.13.0-slim

# Install numpy only (minimal dependencies)
# Pin version to match project requirements
RUN pip install --no-cache-dir numpy==2.2.1

# Create non-root user for sandbox execution
# UID 1000 ensures consistent permissions
RUN useradd -m -u 1000 sandbox

# Switch to non-root user
USER sandbox

# Set working directory
WORKDIR /workspace

# Python runs unbuffered for immediate output
# This ensures logs are available immediately for debugging
ENV PYTHONUNBUFFERED=1

# No ENTRYPOINT - command passed at runtime
# This allows flexible execution via docker.containers.run()
