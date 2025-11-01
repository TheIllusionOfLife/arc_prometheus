"""Population-based evolution using genetic algorithm with multi-agent cognitive cells."""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import google.generativeai as genai

from arc_prometheus.cognitive_cells.analyst import Analyst
from arc_prometheus.cognitive_cells.crossover import Crossover
from arc_prometheus.cognitive_cells.programmer import generate_solver
from arc_prometheus.cognitive_cells.refiner import refine_solver
from arc_prometheus.cognitive_cells.tagger import Tagger
from arc_prometheus.evolutionary_engine.fitness import calculate_fitness
from arc_prometheus.evolutionary_engine.solver_library import (
    SolverLibrary,
    SolverRecord,
)
from arc_prometheus.utils.config import (
    ANALYST_DEFAULT_TEMPERATURE,
    CROSSOVER_DEFAULT_TEMPERATURE,
    MODEL_NAME,
    TAGGER_DEFAULT_TEMPERATURE,
    get_gemini_api_key,
)

# Default temperatures for programmer and refiner (from their generation configs)
PROGRAMMER_DEFAULT_TEMPERATURE = 0.3
REFINER_DEFAULT_TEMPERATURE = 0.4


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class PopulationMember:
    """Individual solver in the population."""

    solver_id: str
    code_str: str
    fitness_score: float
    train_correct: int
    test_correct: int
    generation: int
    parent_ids: list[str]
    tags: list[str]
    created_at: str


@dataclass
class GenerationStats:
    """Statistics for a single generation."""

    generation: int
    population_size: int
    best_fitness: float
    average_fitness: float
    diversity_score: float  # Unique tags / total solvers
    crossover_events: int
    mutation_events: int


@dataclass
class PopulationResult:
    """Final result of population evolution."""

    final_population: list[PopulationMember]
    best_solver: PopulationMember
    generation_history: list[GenerationStats]
    total_time: float


# ============================================================================
# PopulationEvolution Class
# ============================================================================


class PopulationEvolution:
    """Genetic algorithm with multi-agent cognitive cells for ARC solving."""

    def __init__(
        self,
        population_size: int = 10,
        selection_pressure: float = 0.3,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.5,
        max_generations: int = 10,
        model_name: str | None = None,
        programmer_temperature: float | None = None,
        refiner_temperature: float | None = None,
        analyst_temperature: float | None = None,
        tagger_temperature: float | None = None,
        crossover_temperature: float | None = None,
        use_cache: bool = True,
        timeout_per_eval: int = 5,
        timeout_per_llm: int = 60,
        sandbox_mode: str = "multiprocess",
        verbose: bool = True,
    ):
        """
        Initialize population evolution with all cognitive agents.

        Args:
            population_size: Number of solvers in population
            selection_pressure: Tournament selection pressure (0.0-1.0)
            mutation_rate: Probability of mutation (0.0-1.0)
            crossover_rate: Probability of crossover (0.0-1.0)
            max_generations: Maximum evolution generations
            model_name: LLM model name (default: gemini-2.5-flash-lite)
            programmer_temperature: Temperature for code generation
            refiner_temperature: Temperature for debugging
            analyst_temperature: Temperature for pattern analysis
            tagger_temperature: Temperature for technique classification
            crossover_temperature: Temperature for technique fusion
            use_cache: Enable LLM response caching
            timeout_per_eval: Timeout for each fitness evaluation (seconds)
            timeout_per_llm: Timeout for each LLM call (seconds)
            sandbox_mode: Sandbox type ("multiprocess" or "docker")
            verbose: Enable verbose output
        """
        if population_size < 1:
            raise ValueError("Population size must be at least 1")

        self.population_size = population_size
        self.selection_pressure = selection_pressure
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.timeout_per_eval = timeout_per_eval
        self.timeout_per_llm = timeout_per_llm
        self.sandbox_mode = sandbox_mode
        self.verbose = verbose

        # Configure Gemini API
        api_key = get_gemini_api_key()
        genai.configure(api_key=api_key)

        # Determine model
        self.model_name = model_name if model_name else MODEL_NAME

        # Store temperatures for function calls
        self.programmer_temperature = (
            programmer_temperature or PROGRAMMER_DEFAULT_TEMPERATURE
        )
        self.refiner_temperature = refiner_temperature or REFINER_DEFAULT_TEMPERATURE
        self.analyst_temperature = analyst_temperature or ANALYST_DEFAULT_TEMPERATURE
        self.tagger_temperature = tagger_temperature or TAGGER_DEFAULT_TEMPERATURE
        self.crossover_temperature = (
            crossover_temperature or CROSSOVER_DEFAULT_TEMPERATURE
        )
        self.use_cache = use_cache

        # Initialize class-based agents (they don't all accept timeout)
        self.analyst = Analyst(
            model_name=self.model_name,
            temperature=self.analyst_temperature,
            use_cache=use_cache,
        )

        self.tagger = Tagger(
            model_name=self.model_name,
            temperature=self.tagger_temperature,
            use_cache=use_cache,
        )

        self.crossover = Crossover(
            model_name=self.model_name,
            temperature=self.crossover_temperature,
            use_cache=use_cache,
        )

        # Initialize solver library
        self.solver_library = SolverLibrary()

    def evolve_population(
        self,
        task_json_path: str,
        max_generations: int | None = None,
        target_fitness: float | None = None,
    ) -> PopulationResult:
        """
        Evolve population of solvers using genetic algorithm.

        Args:
            task_json_path: Path to ARC task JSON file
            max_generations: Override max generations (default: use constructor value)
            target_fitness: Early stopping fitness threshold (optional)

        Returns:
            PopulationResult with final population, best solver, and statistics
        """
        start_time = time.time()
        max_gen = (
            max_generations if max_generations is not None else self.max_generations
        )

        # Load task data
        with open(task_json_path) as f:
            task_json = json.load(f)

        # Extract task ID
        task_id = Path(task_json_path).stem

        if self.verbose:
            print(f"\n=== Population Evolution: Task {task_id} ===")
            print(f"Population Size: {self.population_size}")
            print(f"Max Generations: {max_gen}")
            if target_fitness:
                print(f"Target Fitness: {target_fitness}")

        # Phase 0: Analyst - Understand the task (optional but recommended)
        analyst_spec = None
        if self.verbose:
            print("\nPhase 0: Analyst - Pattern Analysis")
        analyst_result = self.analyst.analyze_task(task_json)
        analyst_spec = analyst_result

        # Phase 1: Initialize population
        if self.verbose:
            print("\nGeneration 0: Initialization")
        population = self._initialize_population(
            task_json_path, task_json, analyst_spec
        )

        # Phase 2: Evaluate initial fitness
        population = self._evaluate_population(population, task_json_path)

        # Track statistics
        generation_history = []
        stats = self._compute_generation_stats(population, generation=0)
        generation_history.append(stats)

        if self.verbose:
            self._print_generation_stats(stats)

        # Phase 3: Evolution loop
        for gen in range(1, max_gen + 1):
            if self.verbose:
                print(f"\nGeneration {gen}: Evolution")

            # Step 1: Selection
            parents = self._select_parents(population)

            # Step 2: Breeding (crossover + mutation)
            offspring_codes = self._breed_offspring(
                parents, task_json_path, task_json, analyst_spec
            )

            # Step 3: Tag offspring
            offspring = self._tag_offspring(offspring_codes, task_json_path, task_json)

            # Step 4: Evaluate offspring
            offspring = self._evaluate_population(offspring, task_json_path)

            # Step 5: Combine and select survivors
            combined = population + offspring
            population = self._select_survivors(
                combined, target_size=self.population_size
            )

            # Step 6: Compute statistics
            stats = self._compute_generation_stats(population, generation=gen)
            generation_history.append(stats)

            if self.verbose:
                self._print_generation_stats(stats)

            # Step 7: Check termination
            if target_fitness and stats.best_fitness >= target_fitness:
                if self.verbose:
                    print(f"\n🎉 Target fitness {target_fitness} reached!")
                break

        # Final result
        total_time = time.time() - start_time
        best_solver = max(population, key=lambda s: s.fitness_score)

        if self.verbose:
            print("\n=== Evolution Complete ===")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Best Fitness: {best_solver.fitness_score}")
            print(f"Techniques: {best_solver.tags}")

        return PopulationResult(
            final_population=population,
            best_solver=best_solver,
            generation_history=generation_history,
            total_time=total_time,
        )

    def _initialize_population(
        self,
        task_json_path: str,
        task_json: dict[str, Any],
        analyst_spec: Any = None,
    ) -> list[PopulationMember]:
        """Generate initial population using Programmer."""
        population = []
        task_id = Path(task_json_path).stem

        for i in range(self.population_size):
            # Generate with increasing temperature for diversity
            temp_offset = i * 0.02
            code_str = generate_solver(
                train_pairs=task_json["train"],
                analyst_spec=analyst_spec,
                model_name=self.model_name,
                temperature=self.programmer_temperature + temp_offset,
                timeout=self.timeout_per_llm,
                use_cache=self.use_cache,
            )

            # Create population member
            solver_id = f"{task_id}_gen0_{uuid.uuid4().hex[:8]}"
            member = PopulationMember(
                solver_id=solver_id,
                code_str=code_str,
                fitness_score=0.0,  # Not evaluated yet
                train_correct=0,
                test_correct=0,
                generation=0,
                parent_ids=[],
                tags=[],
                created_at=datetime.now(UTC).isoformat(),
            )

            population.append(member)

        return population

    def _evaluate_population(
        self, population: list[PopulationMember], task_json_path: str
    ) -> list[PopulationMember]:
        """Calculate fitness for all population members."""
        evaluated = []

        for member in population:
            # Calculate fitness
            fitness_result = calculate_fitness(
                task_json_path,
                member.code_str,
                timeout=self.timeout_per_eval,
                sandbox_mode=self.sandbox_mode,
            )

            # Update member with fitness data
            member.fitness_score = fitness_result["fitness"]
            member.train_correct = fitness_result["train_correct"]
            member.test_correct = fitness_result["test_correct"]

            evaluated.append(member)

        return evaluated

    def _select_parents(
        self, population: list[PopulationMember], tournament_size: int = 3
    ) -> list[PopulationMember]:
        """Tournament selection to choose parents for breeding."""
        parents = []

        for _ in range(len(population)):
            # Random tournament
            tournament = random.sample(
                population, min(tournament_size, len(population))
            )
            # Winner = highest fitness
            winner = max(tournament, key=lambda s: s.fitness_score)
            parents.append(winner)

        return parents

    def _breed_offspring(
        self,
        parents: list[PopulationMember],
        task_json_path: str,
        task_json: dict[str, Any],
        analyst_spec: Any = None,
    ) -> list[str]:
        """Create offspring using crossover and mutation."""
        offspring_codes = []
        task_id = Path(task_json_path).stem

        # Track breeding events
        self._crossover_events = 0
        self._mutation_events = 0

        i = 0
        while i < len(parents):
            # Decide: crossover or mutation?
            if random.random() < self.crossover_rate and i + 1 < len(parents):  # noqa: S311
                # Attempt crossover
                parent1 = parents[i]
                parent2 = parents[i + 1]

                # Check technique diversity
                if self._has_technique_diversity(parent1, parent2):
                    # Convert to SolverRecord for crossover
                    record1 = self._member_to_record(parent1, task_id)
                    record2 = self._member_to_record(parent2, task_id)

                    # Fuse solvers
                    crossover_result = self.crossover.fuse_solvers(
                        [record1, record2],
                        task_json,
                        analyst_spec=analyst_spec,
                    )

                    offspring_codes.append(crossover_result.fused_code)
                    self._crossover_events += 1
                    i += 2  # Used both parents
                    continue

            # Fall through to mutation
            parent = parents[i]

            # Mutation via Refiner (if fitness < perfect)
            if random.random() < self.mutation_rate and parent.fitness_score < 13.0:  # noqa: S311
                # Create complete FitnessResult for refiner
                fitness_result = {
                    "fitness": parent.fitness_score,
                    "train_correct": parent.train_correct,
                    "test_correct": parent.test_correct,
                    "train_total": 3,  # Standard ARC task size
                    "test_total": 1,
                    "train_accuracy": parent.train_correct / 3.0,
                    "test_accuracy": float(parent.test_correct),
                    "execution_errors": [],
                    "error_details": [],
                    "error_summary": {},
                }
                improved_code = refine_solver(
                    failed_code=parent.code_str,
                    task_json_path=task_json_path,
                    fitness_result=fitness_result,  # type: ignore[arg-type]
                    analyst_spec=analyst_spec,
                    model_name=self.model_name,
                    temperature=self.refiner_temperature,
                    timeout=self.timeout_per_llm,
                    use_cache=self.use_cache,
                )

                offspring_codes.append(improved_code)
                self._mutation_events += 1
            else:
                # Direct copy (no mutation)
                offspring_codes.append(parent.code_str)

            i += 1

        return offspring_codes

    def _tag_offspring(
        self,
        offspring_codes: list[str],
        task_json_path: str,
        task_json: dict[str, Any],
    ) -> list[PopulationMember]:
        """Tag offspring with techniques."""
        offspring = []
        task_id = Path(task_json_path).stem

        for _i, code_str in enumerate(offspring_codes):
            # Tag solver
            tagging_result = self.tagger.tag_solver(code_str, task_json)
            tags = tagging_result.tags if tagging_result else []

            # Create population member
            solver_id = f"{task_id}_offspring_{uuid.uuid4().hex[:8]}"
            member = PopulationMember(
                solver_id=solver_id,
                code_str=code_str,
                fitness_score=0.0,  # Not evaluated yet
                train_correct=0,
                test_correct=0,
                generation=1,  # Placeholder (will be updated)
                parent_ids=[],  # Placeholder
                tags=tags,
                created_at=datetime.now(UTC).isoformat(),
            )

            offspring.append(member)

        return offspring

    def _select_survivors(
        self, combined_population: list[PopulationMember], target_size: int
    ) -> list[PopulationMember]:
        """Select survivors using elitism + fitness-based selection."""
        # Sort by fitness descending
        sorted_pop = sorted(
            combined_population, key=lambda s: s.fitness_score, reverse=True
        )

        # Elitism: Keep top 20%
        elite_count = max(1, int(target_size * 0.2))
        elites = sorted_pop[:elite_count]

        # Fill remaining slots with fitness-proportionate selection
        remaining = sorted_pop[elite_count:]
        survivors = elites + self._fitness_proportionate_select(
            remaining, target_size - elite_count
        )

        return survivors[:target_size]

    def _fitness_proportionate_select(
        self, population: list[PopulationMember], count: int
    ) -> list[PopulationMember]:
        """Select solvers with probability proportional to fitness."""
        if not population or count <= 0:
            return []

        # Handle all-zero fitness case
        total_fitness = sum(s.fitness_score for s in population)
        if total_fitness == 0:
            # Random selection
            return random.sample(population, min(count, len(population)))

        # Fitness-proportionate selection
        selected = []
        for _ in range(min(count, len(population))):
            rand = random.uniform(0, total_fitness)  # noqa: S311
            cumulative = 0.0

            for solver in population:
                cumulative += solver.fitness_score
                if cumulative >= rand:
                    selected.append(solver)
                    break

        return selected

    def _compute_generation_stats(
        self, population: list[PopulationMember], generation: int
    ) -> GenerationStats:
        """Compute statistics for current generation."""
        best_fitness = max((s.fitness_score for s in population), default=0.0)
        average_fitness = sum(s.fitness_score for s in population) / len(population)

        # Diversity: unique tags / total solvers
        all_tags = {tag for solver in population for tag in solver.tags}
        diversity_score = len(all_tags) / len(population) if population else 0.0

        # Breeding events (only available after generation 0)
        crossover_events = getattr(self, "_crossover_events", 0)
        mutation_events = getattr(self, "_mutation_events", 0)

        return GenerationStats(
            generation=generation,
            population_size=len(population),
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            diversity_score=diversity_score,
            crossover_events=crossover_events,
            mutation_events=mutation_events,
        )

    def _print_generation_stats(self, stats: GenerationStats) -> None:
        """Print generation statistics."""
        print(f"  Population Size: {stats.population_size}")
        print(f"  Best Fitness: {stats.best_fitness:.1f}")
        print(f"  Average Fitness: {stats.average_fitness:.2f}")
        print(f"  Diversity: {stats.diversity_score:.2f}")

        if stats.generation > 0:
            print(f"  Crossover Events: {stats.crossover_events}")
            print(f"  Mutation Events: {stats.mutation_events}")

    def _has_technique_diversity(
        self, parent1: PopulationMember, parent2: PopulationMember
    ) -> bool:
        """Check if two parents have diverse techniques (for crossover)."""
        tags1 = set(parent1.tags)
        tags2 = set(parent2.tags)

        # Diverse if tags differ or both have at least 1 tag
        if not tags1 and not tags2:
            return False  # No techniques identified

        overlap = len(tags1 & tags2)
        total = len(tags1 | tags2)

        # Diversity threshold: < 50% overlap
        if total == 0:
            return False

        diversity = 1 - (overlap / total)
        return diversity >= 0.3  # At least 30% different

    def _member_to_record(self, member: PopulationMember, task_id: str) -> SolverRecord:
        """Convert PopulationMember to SolverRecord for crossover."""
        return SolverRecord(
            solver_id=member.solver_id,
            task_id=task_id,
            generation=member.generation,
            code_str=member.code_str,
            fitness_score=member.fitness_score,
            train_correct=member.train_correct,
            test_correct=member.test_correct,
            parent_solver_id=member.parent_ids[0] if member.parent_ids else None,
            tags=member.tags,
            created_at=member.created_at,
        )
