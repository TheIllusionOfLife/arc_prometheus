"""
Tests for SolverLibrary - Phase 3.5

Tests SQLite-based solver storage and retrieval for population-based evolution.
TDD: Write tests first, then implement SolverLibrary class.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pytest

from arc_prometheus.evolutionary_engine.solver_library import (
    SolverLibrary,
    SolverRecord,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def solver_library(temp_db_path):
    """Create SolverLibrary instance with temporary database."""
    return SolverLibrary(db_path=temp_db_path)


@pytest.fixture
def sample_solver_record():
    """Create sample SolverRecord for testing."""
    return SolverRecord(
        solver_id="test-solver-001",
        task_id="task-12345",
        generation=0,
        code_str="def solve(grid): return grid",
        fitness_score=10.5,
        train_correct=2,
        test_correct=1,
        parent_solver_id=None,
        tags=["rotation", "flip"],
        created_at=datetime.utcnow().isoformat(),
    )


# =============================================================================
# Test SolverRecord Dataclass
# =============================================================================


class TestSolverRecord:
    """Test SolverRecord dataclass creation and validation."""

    def test_solver_record_creation(self, sample_solver_record):
        """Test basic SolverRecord creation with all fields."""
        assert sample_solver_record.solver_id == "test-solver-001"
        assert sample_solver_record.task_id == "task-12345"
        assert sample_solver_record.generation == 0
        assert sample_solver_record.fitness_score == 10.5
        assert sample_solver_record.tags == ["rotation", "flip"]

    def test_solver_record_with_parent(self):
        """Test SolverRecord with parent_solver_id (lineage tracking)."""
        record = SolverRecord(
            solver_id="child-solver",
            task_id="task-001",
            generation=1,
            code_str="def solve(grid): return grid * 2",
            fitness_score=15.0,
            train_correct=3,
            test_correct=1,
            parent_solver_id="parent-solver-001",
            tags=["rotation"],
            created_at=datetime.utcnow().isoformat(),
        )
        assert record.parent_solver_id == "parent-solver-001"

    def test_solver_record_empty_tags(self):
        """Test SolverRecord with empty tags list."""
        record = SolverRecord(
            solver_id="no-tags-solver",
            task_id="task-001",
            generation=0,
            code_str="def solve(grid): return grid",
            fitness_score=5.0,
            train_correct=1,
            test_correct=0,
            parent_solver_id=None,
            tags=[],
            created_at=datetime.utcnow().isoformat(),
        )
        assert record.tags == []


# =============================================================================
# Test SolverLibrary Initialization
# =============================================================================


class TestSolverLibraryInitialization:
    """Test SolverLibrary database creation and schema."""

    def test_library_initialization(self, solver_library):
        """Test SolverLibrary creates database with correct schema."""
        assert solver_library is not None
        # Database file should exist
        assert os.path.exists(solver_library.db_path)

    def test_default_db_path(self):
        """Test SolverLibrary uses default path if none provided."""
        library = SolverLibrary()
        # Should create in ~/.arc_prometheus/solver_library.db
        assert library.db_path.endswith("solver_library.db")
        assert ".arc_prometheus" in library.db_path

    def test_multiple_instances_same_db(self, temp_db_path):
        """Test multiple SolverLibrary instances can access same database."""
        library1 = SolverLibrary(db_path=temp_db_path)
        library2 = SolverLibrary(db_path=temp_db_path)

        # Add solver via library1
        record = SolverRecord(
            solver_id="shared-solver",
            task_id="task-001",
            generation=0,
            code_str="def solve(grid): return grid",
            fitness_score=10.0,
            train_correct=2,
            test_correct=1,
            parent_solver_id=None,
            tags=["rotation"],
            created_at=datetime.utcnow().isoformat(),
        )
        library1.add_solver(record)

        # Retrieve via library2
        retrieved = library2.get_solver("shared-solver")
        assert retrieved is not None
        assert retrieved.solver_id == "shared-solver"


# =============================================================================
# Test CRUD Operations
# =============================================================================


class TestSolverLibraryCRUD:
    """Test Create, Read, Update, Delete operations."""

    def test_add_solver(self, solver_library, sample_solver_record):
        """Test adding solver to library."""
        solver_id = solver_library.add_solver(sample_solver_record)
        assert solver_id == "test-solver-001"

    def test_get_solver_exists(self, solver_library, sample_solver_record):
        """Test retrieving existing solver by ID."""
        solver_library.add_solver(sample_solver_record)
        retrieved = solver_library.get_solver("test-solver-001")

        assert retrieved is not None
        assert retrieved.solver_id == "test-solver-001"
        assert retrieved.task_id == "task-12345"
        assert retrieved.fitness_score == 10.5
        assert retrieved.tags == ["rotation", "flip"]

    def test_get_solver_not_exists(self, solver_library):
        """Test retrieving non-existent solver returns None."""
        retrieved = solver_library.get_solver("nonexistent-id")
        assert retrieved is None

    def test_add_duplicate_solver_id_raises_error(
        self, solver_library, sample_solver_record
    ):
        """Test adding solver with duplicate ID raises error."""
        solver_library.add_solver(sample_solver_record)

        # Attempt to add again with same ID
        with pytest.raises(ValueError, match="already exists"):
            solver_library.add_solver(sample_solver_record)

    def test_clear_task_solvers(self, solver_library):
        """Test clearing all solvers for a specific task."""
        # Add multiple solvers for task-001
        for i in range(3):
            record = SolverRecord(
                solver_id=f"solver-{i}",
                task_id="task-001",
                generation=i,
                code_str=f"def solve(grid): return grid * {i}",
                fitness_score=10.0 + i,
                train_correct=2,
                test_correct=1,
                parent_solver_id=None,
                tags=["rotation"],
                created_at=datetime.utcnow().isoformat(),
            )
            solver_library.add_solver(record)

        # Add solver for different task
        other_task_record = SolverRecord(
            solver_id="other-task-solver",
            task_id="task-002",
            generation=0,
            code_str="def solve(grid): return grid",
            fitness_score=5.0,
            train_correct=1,
            test_correct=0,
            parent_solver_id=None,
            tags=[],
            created_at=datetime.utcnow().isoformat(),
        )
        solver_library.add_solver(other_task_record)

        # Clear task-001 solvers
        solver_library.clear_task_solvers("task-001")

        # task-001 solvers should be gone
        assert solver_library.get_solver("solver-0") is None
        assert solver_library.get_solver("solver-1") is None
        assert solver_library.get_solver("solver-2") is None

        # task-002 solver should remain
        assert solver_library.get_solver("other-task-solver") is not None


# =============================================================================
# Test Query Operations
# =============================================================================


class TestSolverLibraryQueries:
    """Test solver retrieval and filtering queries."""

    def test_get_solvers_by_task(self, solver_library):
        """Test retrieving all solvers for a task."""
        # Add 3 solvers for task-001
        for i in range(3):
            record = SolverRecord(
                solver_id=f"task1-solver-{i}",
                task_id="task-001",
                generation=i,
                code_str=f"def solve(grid): return grid * {i}",
                fitness_score=10.0 + i,
                train_correct=2,
                test_correct=1,
                parent_solver_id=None,
                tags=["rotation"],
                created_at=datetime.utcnow().isoformat(),
            )
            solver_library.add_solver(record)

        solvers = solver_library.get_solvers_by_task("task-001")
        assert len(solvers) == 3
        assert all(s.task_id == "task-001" for s in solvers)

    def test_get_solvers_by_task_min_fitness(self, solver_library):
        """Test filtering solvers by minimum fitness."""
        # Add solvers with varying fitness
        for i in range(5):
            record = SolverRecord(
                solver_id=f"solver-{i}",
                task_id="task-001",
                generation=i,
                code_str=f"def solve(grid): return grid * {i}",
                fitness_score=float(i * 5),  # 0, 5, 10, 15, 20
                train_correct=i,
                test_correct=i // 2,
                parent_solver_id=None,
                tags=["rotation"],
                created_at=datetime.utcnow().isoformat(),
            )
            solver_library.add_solver(record)

        # Get solvers with fitness >= 10
        high_fitness_solvers = solver_library.get_solvers_by_task(
            "task-001", min_fitness=10.0
        )
        assert len(high_fitness_solvers) == 3  # fitness: 10, 15, 20
        assert all(s.fitness_score >= 10.0 for s in high_fitness_solvers)

    def test_get_solvers_by_task_sorted_by_fitness_desc(self, solver_library):
        """Test solvers are returned sorted by fitness (descending)."""
        # Add solvers in random fitness order
        fitness_scores = [15.0, 5.0, 20.0, 10.0]
        for i, fitness in enumerate(fitness_scores):
            record = SolverRecord(
                solver_id=f"solver-{i}",
                task_id="task-001",
                generation=i,
                code_str="def solve(grid): return grid",
                fitness_score=fitness,
                train_correct=2,
                test_correct=1,
                parent_solver_id=None,
                tags=["rotation"],
                created_at=datetime.utcnow().isoformat(),
            )
            solver_library.add_solver(record)

        solvers = solver_library.get_solvers_by_task("task-001")
        fitness_list = [s.fitness_score for s in solvers]
        assert fitness_list == sorted(fitness_list, reverse=True)  # [20, 15, 10, 5]

    def test_get_solvers_empty_task(self, solver_library):
        """Test retrieving solvers for task with no entries."""
        solvers = solver_library.get_solvers_by_task("nonexistent-task")
        assert solvers == []


# =============================================================================
# Test Diverse Solver Selection
# =============================================================================


class TestDiverseSolverSelection:
    """Test get_diverse_solvers() algorithm for crossover."""

    def test_get_diverse_solvers_basic(self, solver_library):
        """Test selecting 2 solvers with different techniques."""
        # Solver 1: rotation, flip
        solver1 = SolverRecord(
            solver_id="solver-1",
            task_id="task-001",
            generation=0,
            code_str="def solve(grid): return np.rot90(grid)",
            fitness_score=15.0,
            train_correct=3,
            test_correct=1,
            parent_solver_id=None,
            tags=["rotation", "flip"],
            created_at=datetime.utcnow().isoformat(),
        )

        # Solver 2: color_fill, grid_partition
        solver2 = SolverRecord(
            solver_id="solver-2",
            task_id="task-001",
            generation=1,
            code_str="def solve(grid): return flood_fill(grid)",
            fitness_score=12.0,
            train_correct=2,
            test_correct=1,
            parent_solver_id=None,
            tags=["color_fill", "grid_partition"],
            created_at=datetime.utcnow().isoformat(),
        )

        solver_library.add_solver(solver1)
        solver_library.add_solver(solver2)

        # Get 2 diverse solvers
        diverse = solver_library.get_diverse_solvers("task-001", num_solvers=2)
        assert len(diverse) == 2

        # Should have different tags
        tags1 = set(diverse[0].tags)
        tags2 = set(diverse[1].tags)
        assert tags1 != tags2

    def test_get_diverse_solvers_prioritizes_high_fitness(self, solver_library):
        """Test diverse selection prioritizes higher fitness solvers."""
        # Add 3 solvers with different fitness and tags
        solvers = [
            SolverRecord(
                solver_id="low-fitness",
                task_id="task-001",
                generation=0,
                code_str="def solve(grid): return grid",
                fitness_score=5.0,
                train_correct=1,
                test_correct=0,
                parent_solver_id=None,
                tags=["rotation"],
                created_at=datetime.utcnow().isoformat(),
            ),
            SolverRecord(
                solver_id="high-fitness",
                task_id="task-001",
                generation=1,
                code_str="def solve(grid): return grid * 2",
                fitness_score=20.0,
                train_correct=3,
                test_correct=1,
                parent_solver_id=None,
                tags=["color_fill"],
                created_at=datetime.utcnow().isoformat(),
            ),
            SolverRecord(
                solver_id="medium-fitness",
                task_id="task-001",
                generation=2,
                code_str="def solve(grid): return grid + 1",
                fitness_score=12.0,
                train_correct=2,
                test_correct=1,
                parent_solver_id=None,
                tags=["symmetry"],
                created_at=datetime.utcnow().isoformat(),
            ),
        ]

        for solver in solvers:
            solver_library.add_solver(solver)

        # Get 2 diverse solvers
        diverse = solver_library.get_diverse_solvers("task-001", num_solvers=2)
        assert len(diverse) == 2

        # Should include high-fitness solver
        solver_ids = {s.solver_id for s in diverse}
        assert "high-fitness" in solver_ids

    def test_get_diverse_solvers_insufficient_solvers(self, solver_library):
        """Test requesting more diverse solvers than available."""
        # Add only 1 solver
        solver = SolverRecord(
            solver_id="only-solver",
            task_id="task-001",
            generation=0,
            code_str="def solve(grid): return grid",
            fitness_score=10.0,
            train_correct=2,
            test_correct=1,
            parent_solver_id=None,
            tags=["rotation"],
            created_at=datetime.utcnow().isoformat(),
        )
        solver_library.add_solver(solver)

        # Request 2 solvers but only 1 available
        diverse = solver_library.get_diverse_solvers("task-001", num_solvers=2)
        assert len(diverse) == 1  # Return what's available

    def test_get_diverse_solvers_empty_task(self, solver_library):
        """Test diverse selection on task with no solvers."""
        diverse = solver_library.get_diverse_solvers("nonexistent-task", num_solvers=2)
        assert diverse == []

    def test_get_diverse_solvers_all_same_tags(self, solver_library):
        """Test diverse selection when all solvers have identical tags."""
        # Add 3 solvers with same tags but different fitness
        for i in range(3):
            solver = SolverRecord(
                solver_id=f"solver-{i}",
                task_id="task-001",
                generation=i,
                code_str=f"def solve(grid): return grid * {i}",
                fitness_score=10.0 + i,
                train_correct=2,
                test_correct=1,
                parent_solver_id=None,
                tags=["rotation", "flip"],  # All same tags
                created_at=datetime.utcnow().isoformat(),
            )
            solver_library.add_solver(solver)

        # Should still return 2 solvers (highest fitness)
        diverse = solver_library.get_diverse_solvers("task-001", num_solvers=2)
        assert len(diverse) == 2
        # Should be top 2 by fitness
        assert diverse[0].fitness_score == 12.0
        assert diverse[1].fitness_score == 11.0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestSolverLibraryEdgeCases:
    """Test edge cases and error handling."""

    def test_add_solver_with_none_parent(self, solver_library):
        """Test adding solver with explicit None parent_id."""
        record = SolverRecord(
            solver_id="root-solver",
            task_id="task-001",
            generation=0,
            code_str="def solve(grid): return grid",
            fitness_score=10.0,
            train_correct=2,
            test_correct=1,
            parent_solver_id=None,
            tags=["rotation"],
            created_at=datetime.utcnow().isoformat(),
        )
        solver_id = solver_library.add_solver(record)
        assert solver_id == "root-solver"

        retrieved = solver_library.get_solver("root-solver")
        assert retrieved.parent_solver_id is None

    def test_solver_with_large_code_string(self, solver_library):
        """Test storing solver with very large code string."""
        large_code = (
            "def solve(grid):\n" + "    " + "x = 1\n" * 1000 + "    return grid"
        )
        record = SolverRecord(
            solver_id="large-code-solver",
            task_id="task-001",
            generation=0,
            code_str=large_code,
            fitness_score=10.0,
            train_correct=2,
            test_correct=1,
            parent_solver_id=None,
            tags=["rotation"],
            created_at=datetime.utcnow().isoformat(),
        )
        solver_library.add_solver(record)

        retrieved = solver_library.get_solver("large-code-solver")
        assert retrieved.code_str == large_code

    def test_solver_with_zero_fitness(self, solver_library):
        """Test solver with fitness = 0 (failed solver)."""
        record = SolverRecord(
            solver_id="failed-solver",
            task_id="task-001",
            generation=0,
            code_str="def solve(grid): return None",
            fitness_score=0.0,
            train_correct=0,
            test_correct=0,
            parent_solver_id=None,
            tags=[],  # No tags for failed solver
            created_at=datetime.utcnow().isoformat(),
        )
        solver_library.add_solver(record)

        # Should be retrievable but filtered by min_fitness queries
        assert solver_library.get_solver("failed-solver") is not None

        # Should not appear in min_fitness=0.1 query
        solvers = solver_library.get_solvers_by_task("task-001", min_fitness=0.1)
        assert len(solvers) == 0
