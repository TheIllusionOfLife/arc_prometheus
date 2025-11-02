"""Test-time inference module for ARC-Prometheus.

This module provides test-time ensemble functionality that orchestrates
multiple agents to generate diverse predictions for ARC tasks.
"""

from .test_time_ensemble import solve_task_ensemble

__all__ = ["solve_task_ensemble"]
