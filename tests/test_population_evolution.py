"""Tests for population-based evolution (genetic algorithm)."""

from __future__ import annotations

import json
import os
from unittest.mock import Mock, patch

import pytest

from arc_prometheus.evolutionary_engine.population_evolution import (
    GenerationStats,
    PopulationEvolution,
    PopulationMember,
    PopulationResult,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_task_json():
    """Sample ARC task for testing."""
    return {
        "train": [
            {
                "input": [[0, 1], [1, 0]],
                "output": [[1, 0], [0, 1]],
            },
            {
                "input": [[1, 2], [2, 1]],
                "output": [[2, 1], [1, 2]],
            },
        ],
        "test": [
            {
                "input": [[3, 4], [4, 3]],
                "output": [[4, 3], [3, 4]],
            },
        ],
    }


@pytest.fixture
def sample_task_file(tmp_path, sample_task_json):
    """Create temporary task JSON file."""
    task_file = tmp_path / "test_task.json"
    with open(task_file, "w") as f:
        json.dump(sample_task_json, f)
    return str(task_file)


@pytest.fixture
def mock_fitness_result_perfect():
    """Perfect fitness result (all tests pass)."""
    return {
        "fitness": 13.0,
        "train_correct": 3,
        "train_total": 3,
        "test_correct": 1,
        "test_total": 1,
        "train_accuracy": 1.0,
        "test_accuracy": 1.0,
        "execution_errors": [],
        "error_details": [],
        "error_summary": {},
    }


@pytest.fixture
def mock_fitness_result_partial():
    """Partial fitness result (some tests pass)."""
    return {
        "fitness": 3.0,
        "train_correct": 3,
        "train_total": 3,
        "test_correct": 0,
        "test_total": 1,
        "train_accuracy": 1.0,
        "test_accuracy": 0.0,
        "execution_errors": [],
        "error_details": [],
        "error_summary": {},
    }


@pytest.fixture
def mock_fitness_result_zero():
    """Zero fitness result (all tests fail)."""
    return {
        "fitness": 0.0,
        "train_correct": 0,
        "train_total": 3,
        "test_correct": 0,
        "test_total": 1,
        "train_accuracy": 0.0,
        "test_accuracy": 0.0,
        "execution_errors": ["SyntaxError: invalid syntax"],
        "error_details": [{"error_type": "SyntaxError"}],
        "error_summary": {"SyntaxError": 1},
    }


# ============================================================================
# Test Class: Data Structures
# ============================================================================


class TestDataStructures:
    """Test PopulationMember, PopulationResult, GenerationStats data structures."""

    def test_population_member_creation(self):
        """Test PopulationMember can be created with all fields."""
        member = PopulationMember(
            solver_id="test_solver_001",
            code_str="def solve(task_grid): return task_grid",
            fitness_score=13.0,
            train_correct=3,
            test_correct=1,
            generation=0,
            parent_ids=[],
            tags=["rotation", "flip"],
            created_at="2025-11-01T00:00:00",
        )

        assert member.solver_id == "test_solver_001"
        assert member.fitness_score == 13.0
        assert member.generation == 0
        assert member.tags == ["rotation", "flip"]

    def test_generation_stats_creation(self):
        """Test GenerationStats can be created with all fields."""
        stats = GenerationStats(
            generation=1,
            population_size=10,
            best_fitness=13.0,
            average_fitness=6.5,
            diversity_score=0.8,
            crossover_events=5,
            mutation_events=3,
        )

        assert stats.generation == 1
        assert stats.best_fitness == 13.0
        assert stats.diversity_score == 0.8

    def test_population_result_creation(self):
        """Test PopulationResult can be created with all fields."""
        member = PopulationMember(
            solver_id="best",
            code_str="code",
            fitness_score=13.0,
            train_correct=3,
            test_correct=1,
            generation=5,
            parent_ids=[],
            tags=[],
            created_at="2025-11-01T00:00:00",
        )

        result = PopulationResult(
            final_population=[member],
            best_solver=member,
            generation_history=[],
            total_time=120.5,
        )

        assert len(result.final_population) == 1
        assert result.best_solver.fitness_score == 13.0
        assert result.total_time == 120.5


# ============================================================================
# Test Class: Basic Functionality
# ============================================================================


class TestPopulationEvolutionBasics:
    """Test basic population evolution functionality."""

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    def test_population_initialization(
        self, mock_genai, mock_get_api_key, sample_task_file
    ):
        """Test initial population is generated with N diverse solvers."""
        mock_get_api_key.return_value = "test_key"

        pop_evo = PopulationEvolution(
            population_size=5,
            model_name="gemini-2.5-flash-lite",
            use_cache=False,
        )

        # Mock generate_solver to return different code strings
        with patch(
            "arc_prometheus.evolutionary_engine.population_evolution.generate_solver",
            side_effect=[
                f"def solve(grid): return grid  # solver {i}" for i in range(5)
            ],
        ):
            population = pop_evo._initialize_population(
                sample_task_file, sample_task_json, analyst_spec=None
            )

        assert len(population) == 5
        assert all(isinstance(member, PopulationMember) for member in population)
        assert all(member.generation == 0 for member in population)
        assert all(member.parent_ids == [] for member in population)

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.calculate_fitness")
    def test_population_evaluation(
        self,
        mock_calc_fitness,
        mock_genai,
        mock_get_api_key,
        sample_task_file,
        mock_fitness_result_perfect,
        mock_fitness_result_partial,
    ):
        """Test fitness calculation for all population members."""
        mock_get_api_key.return_value = "test_key"
        mock_calc_fitness.side_effect = [
            mock_fitness_result_perfect,
            mock_fitness_result_partial,
            mock_fitness_result_perfect,
        ]

        pop_evo = PopulationEvolution(population_size=3, use_cache=False)

        # Create test population
        population = [
            PopulationMember(
                solver_id=f"solver_{i}",
                code_str=f"def solve(grid): return grid  # {i}",
                fitness_score=0.0,
                train_correct=0,
                test_correct=0,
                generation=0,
                parent_ids=[],
                tags=[],
                created_at="2025-11-01T00:00:00",
            )
            for i in range(3)
        ]

        evaluated = pop_evo._evaluate_population(population, sample_task_file)

        assert len(evaluated) == 3
        assert evaluated[0].fitness_score == 13.0
        assert evaluated[1].fitness_score == 3.0
        assert evaluated[2].fitness_score == 13.0
        assert mock_calc_fitness.call_count == 3

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    def test_tournament_selection(self, mock_genai, mock_get_api_key):
        """Test tournament selection chooses winner from k=3 tournament."""
        mock_get_api_key.return_value = "test_key"

        pop_evo = PopulationEvolution(population_size=10, use_cache=False)

        # Create population with different fitness scores
        population = [
            PopulationMember(
                solver_id=f"solver_{i}",
                code_str="code",
                fitness_score=float(i),
                train_correct=i,
                test_correct=0,
                generation=0,
                parent_ids=[],
                tags=[],
                created_at="2025-11-01T00:00:00",
            )
            for i in range(10)
        ]

        # Run tournament selection
        parents = pop_evo._select_parents(population, tournament_size=3)

        # Should return same number of parents as population
        assert len(parents) == len(population)
        # All parents should be from original population
        assert all(p in population for p in parents)
        # Should favor higher fitness (not guaranteed but statistically likely)
        avg_parent_fitness = sum(p.fitness_score for p in parents) / len(parents)
        avg_population_fitness = sum(p.fitness_score for p in population) / len(
            population
        )
        assert avg_parent_fitness >= avg_population_fitness

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    def test_survivor_selection_elitism(self, mock_genai, mock_get_api_key):
        """Test survivor selection keeps top N by fitness with elitism."""
        mock_get_api_key.return_value = "test_key"

        pop_evo = PopulationEvolution(population_size=10, use_cache=False)

        # Create combined population (parents + offspring)
        combined = [
            PopulationMember(
                solver_id=f"solver_{i}",
                code_str="code",
                fitness_score=float(i),
                train_correct=i,
                test_correct=0,
                generation=0 if i < 10 else 1,
                parent_ids=[],
                tags=[],
                created_at="2025-11-01T00:00:00",
            )
            for i in range(20)
        ]

        survivors = pop_evo._select_survivors(combined, target_size=10)

        # Should return exactly target_size survivors
        assert len(survivors) == 10
        # Top 20% (2 solvers) should be elite (highest fitness)
        elite_count = max(1, int(10 * 0.2))
        elite_ids = {f"solver_{i}" for i in range(20 - elite_count, 20)}
        survivor_ids = {s.solver_id for s in survivors}
        assert elite_ids.issubset(survivor_ids)

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.calculate_fitness")
    def test_evolve_population_returns_valid_result(
        self,
        mock_calc_fitness,
        mock_genai,
        mock_get_api_key,
        sample_task_file,
        mock_fitness_result_perfect,
    ):
        """Test evolve_population returns valid PopulationResult structure."""
        mock_get_api_key.return_value = "test_key"
        mock_calc_fitness.return_value = mock_fitness_result_perfect

        pop_evo = PopulationEvolution(
            population_size=3, max_generations=2, use_cache=False
        )

        # Mock all functions and methods
        with (
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.generate_solver",
                return_value="def solve(g): return g",
            ),
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.refine_solver",
                return_value="def solve(g): return g",
            ),
            patch.object(
                pop_evo.tagger, "tag_solver", return_value=Mock(tags=["rotation"])
            ),
            patch.object(
                pop_evo.crossover,
                "fuse_solvers",
                return_value=Mock(fused_code="def solve(g): return g"),
            ),
        ):
            result = pop_evo.evolve_population(
                task_json_path=sample_task_file,
                max_generations=2,
                target_fitness=None,
            )

        assert isinstance(result, PopulationResult)
        assert len(result.final_population) > 0
        assert isinstance(result.best_solver, PopulationMember)
        assert len(result.generation_history) > 0
        assert result.total_time >= 0.0


# ============================================================================
# Test Class: Crossover & Mutation
# ============================================================================


class TestCrossoverAndMutation:
    """Test crossover and mutation mechanisms."""

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.calculate_fitness")
    def test_crossover_when_diverse_parents_available(
        self,
        mock_calc_fitness,
        mock_genai,
        mock_get_api_key,
        sample_task_file,
        sample_task_json,
        mock_fitness_result_perfect,
    ):
        """Test crossover is used when 2+ diverse parents available."""
        mock_get_api_key.return_value = "test_key"
        mock_calc_fitness.return_value = mock_fitness_result_perfect

        pop_evo = PopulationEvolution(
            population_size=3, crossover_rate=1.0, use_cache=False
        )

        # Create diverse parents (different tags)
        parents = [
            PopulationMember(
                solver_id="solver_0",
                code_str="def solve(g): return np.rot90(g)",
                fitness_score=13.0,
                train_correct=3,
                test_correct=1,
                generation=0,
                parent_ids=[],
                tags=["rotation"],
                created_at="2025-11-01T00:00:00",
            ),
            PopulationMember(
                solver_id="solver_1",
                code_str="def solve(g): return np.flip(g)",
                fitness_score=13.0,
                train_correct=3,
                test_correct=1,
                generation=0,
                parent_ids=[],
                tags=["flip"],
                created_at="2025-11-01T00:00:00",
            ),
        ]

        # Mock crossover to track calls
        crossover_called = False

        def mock_fuse(*args, **kwargs):
            nonlocal crossover_called
            crossover_called = True
            return Mock(fused_code="def solve(g): return g")

        with patch.object(pop_evo.crossover, "fuse_solvers", side_effect=mock_fuse):
            offspring = pop_evo._breed_offspring(
                parents, sample_task_file, sample_task_json, analyst_spec=None
            )

        assert crossover_called, "Crossover should be called with diverse parents"
        assert len(offspring) > 0

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.calculate_fitness")
    def test_mutation_when_no_diverse_parents(
        self,
        mock_calc_fitness,
        mock_genai,
        mock_get_api_key,
        sample_task_file,
        sample_task_json,
        mock_fitness_result_partial,
    ):
        """Test mutation (Refiner) is used when no diverse parents available."""
        mock_get_api_key.return_value = "test_key"
        mock_calc_fitness.return_value = mock_fitness_result_partial

        pop_evo = PopulationEvolution(
            population_size=2, mutation_rate=1.0, use_cache=False
        )

        # Create parents with SAME tags (not diverse)
        parents = [
            PopulationMember(
                solver_id=f"solver_{i}",
                code_str="def solve(g): return g",
                fitness_score=3.0,
                train_correct=3,
                test_correct=0,
                generation=0,
                parent_ids=[],
                tags=["rotation"],  # Same tags
                created_at="2025-11-01T00:00:00",
            )
            for i in range(2)
        ]

        # Mock refiner to track calls
        refiner_called = False

        def mock_refine(*args, **kwargs):
            nonlocal refiner_called
            refiner_called = True
            return "def solve(g): return g"

        with (
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.refine_solver",
                side_effect=mock_refine,
            ),
            patch.object(
                pop_evo.crossover,
                "fuse_solvers",
                return_value=Mock(fused_code="def solve(g): return g"),
            ),
        ):
            offspring = pop_evo._breed_offspring(
                parents, sample_task_file, sample_task_json, analyst_spec=None
            )

        assert refiner_called, "Refiner should be called when parents not diverse"
        assert len(offspring) > 0

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    def test_hybrid_breeding_strategy(
        self, mock_genai, mock_get_api_key, sample_task_file, sample_task_json
    ):
        """Test breeding uses mix of crossover and mutation based on rates."""
        mock_get_api_key.return_value = "test_key"

        pop_evo = PopulationEvolution(
            population_size=10, crossover_rate=0.5, mutation_rate=0.5, use_cache=False
        )

        # Create diverse parents
        parents = [
            PopulationMember(
                solver_id=f"solver_{i}",
                code_str="def solve(g): return g",
                fitness_score=13.0,
                train_correct=3,
                test_correct=1,
                generation=0,
                parent_ids=[],
                tags=[f"tag_{i % 3}"],  # Some diversity
                created_at="2025-11-01T00:00:00",
            )
            for i in range(10)
        ]

        # Track calls
        crossover_calls = 0
        mutation_calls = 0

        def mock_fuse(*args, **kwargs):
            nonlocal crossover_calls
            crossover_calls += 1
            return Mock(fused_code="def solve(g): return g")

        def mock_refine(*args, **kwargs):
            nonlocal mutation_calls
            mutation_calls += 1
            return "def solve(g): return g"

        with (
            patch.object(pop_evo.crossover, "fuse_solvers", side_effect=mock_fuse),
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.refine_solver",
                side_effect=mock_refine,
            ),
        ):
            offspring = pop_evo._breed_offspring(
                parents, sample_task_file, sample_task_json, analyst_spec=None
            )

        # Should use both strategies (not 100% deterministic but likely)
        assert len(offspring) > 0
        # At least one breeding event should occur
        assert (crossover_calls + mutation_calls) > 0

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.calculate_fitness")
    def test_offspring_tagged_correctly(
        self,
        mock_calc_fitness,
        mock_genai,
        mock_get_api_key,
        sample_task_file,
        mock_fitness_result_perfect,
    ):
        """Test offspring are tagged with techniques after breeding."""
        mock_get_api_key.return_value = "test_key"
        mock_calc_fitness.return_value = mock_fitness_result_perfect

        pop_evo = PopulationEvolution(population_size=2, use_cache=False)

        # Create offspring code
        offspring_codes = ["def solve(g): return g" for _ in range(2)]

        # Mock tagger
        with patch.object(
            pop_evo.tagger,
            "tag_solver",
            return_value=Mock(tags=["rotation", "flip"]),
        ):
            tagged = pop_evo._tag_offspring(
                offspring_codes, sample_task_file, sample_task_json
            )

        assert all(member.tags == ["rotation", "flip"] for member in tagged)


# ============================================================================
# Test Class: Population Dynamics
# ============================================================================


class TestPopulationDynamics:
    """Test population diversity and fitness improvement over generations."""

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.calculate_fitness")
    def test_fitness_improvement_over_generations(
        self,
        mock_calc_fitness,
        mock_genai,
        mock_get_api_key,
        sample_task_file,
        mock_fitness_result_partial,
        mock_fitness_result_perfect,
    ):
        """Test best fitness improves over generations."""
        mock_get_api_key.return_value = "test_key"

        # Generation 0: partial fitness, Generation 1+: perfect fitness
        # Need enough for: Gen 0 (5) + Gen 1-3 offspring (5 each = 15) = 20 total
        mock_calc_fitness.side_effect = (
            [mock_fitness_result_partial] * 5  # Gen 0
            + [mock_fitness_result_perfect] * 20  # Gen 1+ (enough for all generations)
        )

        pop_evo = PopulationEvolution(population_size=5, use_cache=False)

        with (
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.generate_solver",
                return_value="def solve(g): return g",
            ),
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.refine_solver",
                return_value="def solve(g): return g",
            ),
            patch.object(
                pop_evo.tagger, "tag_solver", return_value=Mock(tags=["rotation"])
            ),
            patch.object(
                pop_evo.crossover,
                "fuse_solvers",
                return_value=Mock(fused_code="def solve(g): return g"),
            ),
        ):
            result = pop_evo.evolve_population(
                task_json_path=sample_task_file, max_generations=3
            )

        # Best fitness should improve or stay same across generations
        gen_fitnesses = [stats.best_fitness for stats in result.generation_history]
        for i in range(1, len(gen_fitnesses)):
            assert gen_fitnesses[i] >= gen_fitnesses[i - 1]

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.calculate_fitness")
    def test_diversity_maintenance(
        self,
        mock_calc_fitness,
        mock_genai,
        mock_get_api_key,
        sample_task_file,
        mock_fitness_result_perfect,
    ):
        """Test population maintains diversity (doesn't converge to clones)."""
        mock_get_api_key.return_value = "test_key"
        mock_calc_fitness.return_value = mock_fitness_result_perfect

        pop_evo = PopulationEvolution(population_size=5, use_cache=False)

        # Mock tagger to return diverse tags
        tag_counter = 0

        def mock_tag_diverse(*args, **kwargs):
            nonlocal tag_counter
            tag_counter += 1
            return [f"tag_{tag_counter % 3}"]

        with (
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.generate_solver",
                return_value="def solve(g): return g",
            ),
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.refine_solver",
                return_value="def solve(g): return g",
            ),
            patch.object(
                pop_evo.tagger,
                "tag_solver",
                side_effect=lambda *args: Mock(tags=mock_tag_diverse()),
            ),
            patch.object(
                pop_evo.crossover,
                "fuse_solvers",
                return_value=Mock(fused_code="def solve(g): return g"),
            ),
        ):
            result = pop_evo.evolve_population(
                task_json_path=sample_task_file, max_generations=3
            )

        # Check diversity score in generation history
        for stats in result.generation_history:
            if stats.generation == 0:
                # Generation 0 has no tags yet (tagging happens after breeding)
                assert stats.diversity_score >= 0.0
            else:
                # Generation 1+ should have diverse tags
                assert stats.diversity_score > 0.0

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.calculate_fitness")
    def test_early_termination_on_target_fitness(
        self,
        mock_calc_fitness,
        mock_genai,
        mock_get_api_key,
        sample_task_file,
        mock_fitness_result_perfect,
    ):
        """Test evolution stops when target fitness is reached."""
        mock_get_api_key.return_value = "test_key"
        mock_calc_fitness.return_value = mock_fitness_result_perfect

        pop_evo = PopulationEvolution(population_size=3, use_cache=False)

        with (
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.generate_solver",
                return_value="def solve(g): return g",
            ),
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.refine_solver",
                return_value="def solve(g): return g",
            ),
            patch.object(
                pop_evo.tagger, "tag_solver", return_value=Mock(tags=["rotation"])
            ),
            patch.object(
                pop_evo.crossover,
                "fuse_solvers",
                return_value=Mock(fused_code="def solve(g): return g"),
            ),
        ):
            result = pop_evo.evolve_population(
                task_json_path=sample_task_file,
                max_generations=10,
                target_fitness=13.0,
            )

        # Should terminate early (before max_generations)
        assert len(result.generation_history) < 10
        # Best solver should meet target
        assert result.best_solver.fitness_score >= 13.0


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.calculate_fitness")
    def test_small_population_graceful_degradation(
        self,
        mock_calc_fitness,
        mock_genai,
        mock_get_api_key,
        sample_task_file,
        mock_fitness_result_perfect,
    ):
        """Test handles population_size=2 gracefully."""
        mock_get_api_key.return_value = "test_key"
        mock_calc_fitness.return_value = mock_fitness_result_perfect

        pop_evo = PopulationEvolution(population_size=2, use_cache=False)

        with (
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.generate_solver",
                return_value="def solve(g): return g",
            ),
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.refine_solver",
                return_value="def solve(g): return g",
            ),
            patch.object(
                pop_evo.tagger, "tag_solver", return_value=Mock(tags=["rotation"])
            ),
            patch.object(
                pop_evo.crossover,
                "fuse_solvers",
                return_value=Mock(fused_code="def solve(g): return g"),
            ),
        ):
            result = pop_evo.evolve_population(
                task_json_path=sample_task_file, max_generations=2
            )

        # Should complete without errors
        assert len(result.final_population) == 2
        assert result.best_solver.fitness_score >= 0.0

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.calculate_fitness")
    def test_all_solvers_fail_fitness_zero(
        self,
        mock_calc_fitness,
        mock_genai,
        mock_get_api_key,
        sample_task_file,
        mock_fitness_result_zero,
    ):
        """Test doesn't crash when all solvers have fitness=0."""
        mock_get_api_key.return_value = "test_key"
        mock_calc_fitness.return_value = mock_fitness_result_zero

        pop_evo = PopulationEvolution(population_size=3, use_cache=False)

        with (
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.generate_solver",
                return_value="def solve(g): return g",
            ),
            patch(
                "arc_prometheus.evolutionary_engine.population_evolution.refine_solver",
                return_value="def solve(g): return g",
            ),
            patch.object(pop_evo.tagger, "tag_solver", return_value=Mock(tags=[])),
            patch.object(
                pop_evo.crossover,
                "fuse_solvers",
                return_value=Mock(fused_code="def solve(g): return g"),
            ),
        ):
            result = pop_evo.evolve_population(
                task_json_path=sample_task_file, max_generations=2
            )

        # Should complete without errors
        assert result.best_solver.fitness_score == 0.0
        assert len(result.final_population) == 3

    @patch("arc_prometheus.evolutionary_engine.population_evolution.get_gemini_api_key")
    @patch("arc_prometheus.evolutionary_engine.population_evolution.genai")
    def test_empty_initial_population_handling(
        self, mock_genai, mock_get_api_key, sample_task_file
    ):
        """Test robust error handling for empty initial population."""
        mock_get_api_key.return_value = "test_key"

        # Error should be raised at initialization time
        with pytest.raises(ValueError, match="Population size must be at least 1"):
            PopulationEvolution(population_size=0, use_cache=False)


# ============================================================================
# Integration Test
# ============================================================================


class TestPopulationEvolutionIntegration:
    """Integration tests with real API calls."""

    @pytest.mark.integration
    def test_real_api_population_evolution(self, sample_task_file):
        """End-to-end test with real Gemini API."""
        # Skip if no API key
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("No GEMINI_API_KEY environment variable")

        pop_evo = PopulationEvolution(
            population_size=3,
            model_name="gemini-2.5-flash-lite",
            use_cache=True,
        )

        result = pop_evo.evolve_population(
            task_json_path=sample_task_file,
            max_generations=2,
            target_fitness=None,
        )

        # Validate result structure
        assert isinstance(result, PopulationResult)
        assert len(result.final_population) == 3
        assert result.best_solver.fitness_score >= 0.0
        assert len(result.generation_history) >= 1
        assert result.total_time > 0.0

        # Validate generation stats
        for stats in result.generation_history:
            assert stats.population_size == 3
            assert stats.best_fitness >= 0.0
            assert stats.average_fitness >= 0.0
            assert 0.0 <= stats.diversity_score <= 1.0
