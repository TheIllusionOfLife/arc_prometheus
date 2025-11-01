"""
Solver Library - Phase 3.5

SQLite-based storage for solver population in evolutionary loop.
Enables population-based crossover by tracking successful solvers.

Key Features:
- Persistent storage across evolution runs
- Fitness-based retrieval and sorting
- Tag-based diversity selection for crossover
- Thread-safe WAL mode (like llm_cache.py)
- Solver lineage tracking via parent_solver_id
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SolverRecord:
    """
    Solver record for storage in solver library.

    Attributes:
        solver_id: Unique solver identifier (UUID)
        task_id: ARC task ID this solver targets
        generation: Evolution generation number
        code_str: Python solver code
        fitness_score: Fitness score (train*1 + test*10)
        train_correct: Number of correct training examples
        test_correct: Number of correct test examples
        parent_solver_id: Parent solver ID (None for generation 0)
        tags: List of technique tags from Tagger agent
        created_at: ISO timestamp
    """

    solver_id: str
    task_id: str
    generation: int
    code_str: str
    fitness_score: float
    train_correct: int
    test_correct: int
    parent_solver_id: str | None
    tags: list[str]
    created_at: str


class SolverLibrary:
    """
    SQLite-based solver library for population tracking.

    Stores successful solvers across evolution runs to enable:
    - Population-based crossover (select diverse parents)
    - Lineage tracking (evolutionary history)
    - Fitness analysis (compare generations)

    Example:
        >>> library = SolverLibrary()
        >>> record = SolverRecord(...)
        >>> solver_id = library.add_solver(record)
        >>> diverse_parents = library.get_diverse_solvers("task-001", num_solvers=2)
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize solver library with SQLite database.

        Args:
            db_path: Path to SQLite database file. If None, uses default
                     ~/.arc_prometheus/solver_library.db
        """
        if db_path is None:
            # Use default path in ~/.arc_prometheus/
            home = Path.home()
            cache_dir = home / ".arc_prometheus"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "solver_library.db")

        self._db_path = db_path
        self._init_database()

    @property
    def db_path(self) -> str:
        """Get database file path."""
        return self._db_path

    def _init_database(self) -> None:
        """Create database schema if not exists."""
        conn = sqlite3.connect(self._db_path)

        # Enable WAL mode for concurrent access (like llm_cache.py)
        conn.execute("PRAGMA journal_mode=WAL")

        # Create solvers table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS solvers (
                solver_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                code_str TEXT NOT NULL,
                fitness_score REAL NOT NULL,
                train_correct INTEGER NOT NULL,
                test_correct INTEGER NOT NULL,
                parent_solver_id TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_solver_id) REFERENCES solvers(solver_id)
            )
        """)

        # Create indexes for efficient queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_fitness
            ON solvers(task_id, fitness_score DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags
            ON solvers(tags)
        """)

        conn.commit()
        conn.close()

    def add_solver(self, solver: SolverRecord) -> str:
        """
        Add solver to library.

        Args:
            solver: SolverRecord to store

        Returns:
            solver_id of added solver

        Raises:
            ValueError: If solver_id already exists
        """
        conn = sqlite3.connect(self._db_path)

        try:
            # Check for duplicate
            cursor = conn.execute(
                "SELECT solver_id FROM solvers WHERE solver_id = ?", (solver.solver_id,)
            )
            if cursor.fetchone() is not None:
                raise ValueError(f"Solver with ID {solver.solver_id} already exists")

            # Insert solver
            conn.execute(
                """
                INSERT INTO solvers
                (solver_id, task_id, generation, code_str, fitness_score,
                 train_correct, test_correct, parent_solver_id, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    solver.solver_id,
                    solver.task_id,
                    solver.generation,
                    solver.code_str,
                    solver.fitness_score,
                    solver.train_correct,
                    solver.test_correct,
                    solver.parent_solver_id,
                    json.dumps(solver.tags),  # Store tags as JSON
                    solver.created_at,
                ),
            )

            conn.commit()
            return solver.solver_id

        finally:
            conn.close()

    def get_solver(self, solver_id: str) -> SolverRecord | None:
        """
        Get solver by ID.

        Args:
            solver_id: Unique solver identifier

        Returns:
            SolverRecord if found, None otherwise
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(
                "SELECT * FROM solvers WHERE solver_id = ?", (solver_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_record(row)

        finally:
            conn.close()

    def get_solvers_by_task(
        self, task_id: str, min_fitness: float = 0.0
    ) -> list[SolverRecord]:
        """
        Get all solvers for a task, sorted by fitness (descending).

        Args:
            task_id: ARC task identifier
            min_fitness: Minimum fitness threshold (default: 0.0)

        Returns:
            List of SolverRecords sorted by fitness (highest first)
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(
                """
                SELECT * FROM solvers
                WHERE task_id = ? AND fitness_score >= ?
                ORDER BY fitness_score DESC
            """,
                (task_id, min_fitness),
            )

            return [self._row_to_record(row) for row in cursor.fetchall()]

        finally:
            conn.close()

    def get_diverse_solvers(
        self, task_id: str, num_solvers: int = 2
    ) -> list[SolverRecord]:
        """
        Get diverse solvers for crossover (different techniques).

        Selection algorithm:
        1. Get all solvers for task sorted by fitness (descending)
        2. Select solvers with maximum tag diversity
        3. Prioritize higher fitness when diversity is equal

        Args:
            task_id: ARC task identifier
            num_solvers: Number of diverse solvers to return

        Returns:
            List of SolverRecords with diverse tags, up to num_solvers
        """
        # Get all solvers sorted by fitness
        all_solvers = self.get_solvers_by_task(task_id, min_fitness=0.0)

        if len(all_solvers) == 0:
            return []

        if len(all_solvers) <= num_solvers:
            return all_solvers

        # Greedy diversity selection
        selected: list[SolverRecord] = []
        selected_tags: set[str] = set()

        # Start with highest fitness solver
        selected.append(all_solvers[0])
        selected_tags.update(all_solvers[0].tags)

        # Greedily select solvers with most new tags
        remaining = all_solvers[1:]

        while len(selected) < num_solvers and remaining:
            # Score each remaining solver by number of new tags
            best_score = -1
            best_idx = 0

            for idx, solver in enumerate(remaining):
                solver_tags = set(solver.tags)
                new_tags = solver_tags - selected_tags
                score = len(new_tags)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            # Add best solver
            selected.append(remaining[best_idx])
            selected_tags.update(remaining[best_idx].tags)
            remaining.pop(best_idx)

        return selected

    def clear_task_solvers(self, task_id: str) -> None:
        """
        Clear all solvers for a task (for testing).

        Args:
            task_id: ARC task identifier
        """
        conn = sqlite3.connect(self._db_path)

        try:
            conn.execute("DELETE FROM solvers WHERE task_id = ?", (task_id,))
            conn.commit()

        finally:
            conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> SolverRecord:
        """Convert SQLite row to SolverRecord."""
        return SolverRecord(
            solver_id=row["solver_id"],
            task_id=row["task_id"],
            generation=row["generation"],
            code_str=row["code_str"],
            fitness_score=row["fitness_score"],
            train_correct=row["train_correct"],
            test_correct=row["test_correct"],
            parent_solver_id=row["parent_solver_id"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            created_at=row["created_at"],
        )
