"""Multi-generation evolution loop - iterative solver improvement (Phase 2.3/3.4/3.5).

This module implements the complete evolutionary cycle:
1. Generate initial solver (Programmer)
2. Evaluate fitness
3. Refine if below target (Refiner - Mutation OR Crossover - Genetic Innovation)
4. Track improvement across generations
5. Terminate when target fitness reached or max generations hit

Phase 3.4/3.5 adds population-based crossover evolution:
- Solver Library: Persistent storage of successful solvers
- Crossover Agent: LLM-based technique fusion from diverse parents
- Hybrid strategy: Crossover when 2+ diverse solvers exist, else Refiner
"""

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any, TypedDict

from ..cognitive_cells.analyst import Analyst
from ..cognitive_cells.crossover import Crossover
from ..cognitive_cells.programmer import generate_solver
from ..cognitive_cells.refiner import refine_solver
from ..cognitive_cells.tagger import Tagger
from ..crucible.data_loader import load_task
from ..utils.config import (
    ANALYST_DEFAULT_TEMPERATURE,
    CROSSOVER_DEFAULT_TEMPERATURE,
    MODEL_NAME,
    TAGGER_DEFAULT_TEMPERATURE,
)
from .fitness import FitnessResult, calculate_fitness
from .solver_library import SolverLibrary, SolverRecord


class GenerationResult(TypedDict, total=False):
    """Result from a single evolution generation.

    Attributes:
        generation: Generation number (0-indexed)
        solver_code: Current solver code for this generation
        fitness_result: Complete fitness evaluation result
        refinement_count: Number of refinements applied this generation (0 or 1)
        total_time: Time taken for this generation in seconds
        improvement: Fitness improvement from previous generation
        tags: List of technique tags (only present when use_tagger=True)
        solver_id: Unique solver identifier (UUID, Phase 3.5)
        parent_solver_id: Parent solver ID for lineage tracking (Phase 3.5)
        crossover_used: Whether crossover was used this generation (Phase 3.4)
    """

    generation: int
    solver_code: str
    fitness_result: FitnessResult
    refinement_count: int
    total_time: float
    improvement: float
    tags: list[str]  # Optional field, only present when use_tagger=True
    solver_id: str  # Optional field, Phase 3.5
    parent_solver_id: str | None  # Optional field, Phase 3.5
    crossover_used: bool  # Optional field, Phase 3.4


def run_evolution_loop(
    task_json_path: str,
    max_generations: int = 5,
    target_fitness: float | None = None,
    timeout_per_eval: int = 5,
    timeout_per_llm: int = 60,
    verbose: bool = True,
    sandbox_mode: str = "multiprocess",
    model_name: str | None = None,
    programmer_temperature: float | None = None,
    refiner_temperature: float | None = None,
    use_cache: bool = True,
    use_analyst: bool = False,
    analyst_temperature: float | None = None,
    use_tagger: bool = False,
    tagger_temperature: float | None = None,
    use_crossover: bool = False,
    crossover_temperature: float | None = None,
    solver_library: SolverLibrary | None = None,
) -> list[GenerationResult]:
    """Run multi-generation evolution loop on ARC task.

    Process:
        1. (Optional) Analyze task with Analyst agent (if use_analyst=True)
        2. Generate initial solver from train examples (Programmer)
        3. Evaluate fitness with calculate_fitness()
        4. If fitness < target_fitness, refine code (Refiner - Mutation)
        5. Repeat for max_generations or until target_fitness reached
        6. Track and return all generation results

    Args:
        task_json_path: Path to ARC task JSON file
        max_generations: Maximum evolution generations (default: 5)
        target_fitness: Stop when fitness >= this value (default: None = never stop early)
        timeout_per_eval: Timeout for sandbox execution per example in seconds (default: 5)
        timeout_per_llm: Timeout for LLM API calls in seconds (default: 60)
        verbose: Print progress information (default: True)
        sandbox_mode: "multiprocess" or "docker" (default: "multiprocess")
        model_name: LLM model name (default: from config.py)
        programmer_temperature: Temperature for code generation (default: from config.py)
        refiner_temperature: Temperature for debugging (default: from config.py)
        use_cache: If True, use LLM response cache (default: True)
        use_analyst: If True, use Analyst agent for pattern analysis (AI Civilization mode) (default: False)
        analyst_temperature: Temperature for Analyst (default: 0.3, only used if use_analyst=True)
        use_tagger: If True, use Tagger agent for technique classification (Phase 3.3) (default: False)
        tagger_temperature: Temperature for Tagger (default: 0.4, only used if use_tagger=True)
        use_crossover: If True, use Crossover agent for technique fusion (Phase 3.4) (default: False)
        crossover_temperature: Temperature for Crossover (default: 0.5, only used if use_crossover=True)
        solver_library: SolverLibrary instance for population tracking (Phase 3.5) (default: None = create new)

    Returns:
        List of GenerationResult dicts, one per generation

    Raises:
        FileNotFoundError: If task file not found
        ValueError: If API key not configured
        Exception: If LLM API call fails

    Example:
        >>> # Direct mode (Phase 2 - backward compatible)
        >>> results = run_evolution_loop("task.json", max_generations=3, target_fitness=11)  # doctest: +SKIP
        >>> print(f"Final fitness: {results[-1]['fitness_result']['fitness']}")  # doctest: +SKIP
        Final fitness: 13

        >>> # AI Civilization mode (Phase 3 - with Analyst)
        >>> results = run_evolution_loop("task.json", use_analyst=True, max_generations=3)  # doctest: +SKIP
        >>> # Analyst analyzes pattern first, guides Programmer and Refiner

        >>> # Check improvement over generations
        >>> for r in results:  # doctest: +SKIP
        ...     print(f"Gen {r['generation']}: fitness = {r['fitness_result']['fitness']}")
        Gen 0: fitness = 3
        Gen 1: fitness = 13

    Notes:
        - Generation 0 is always initial generation (no refinement)
        - Subsequent generations refine if fitness < target_fitness
        - Early termination when target_fitness reached saves API calls
        - Each generation tracks its own timing for performance analysis
        - When use_analyst=True, Analyst runs once in Generation 0 and its output
          is reused for Programmer and Refiner (pattern analysis is task-specific, not code-specific)
    """
    # Load task once (used for prompt creation)
    task_data = load_task(task_json_path)
    train_pairs = task_data.get("train", [])

    if not train_pairs:
        raise ValueError(f"Task {task_json_path} has no train examples")

    results: list[GenerationResult] = []
    current_code: str = ""
    previous_fitness: float = 0.0
    analyst_spec: Any = None  # Store analyst result for reuse
    tagger: Tagger | None = None  # Store tagger instance for reuse
    crossover_agent: Crossover | None = None  # Store crossover instance for reuse
    library: SolverLibrary = (
        solver_library or SolverLibrary()
    )  # Use provided or create new
    current_solver_id: str | None = None  # Track current solver ID for lineage
    task_id: str = task_json_path.split("/")[-1].replace(".json", "")  # Extract task ID

    # Phase 0a: Initialize Solver Library (if crossover enabled)
    if use_crossover and verbose:
        print(f"\n{'=' * 70}")
        print(" Solver Library: Population Tracking")
        print(f"{'=' * 70}")
        print(f"✅ Solver library initialized (DB: {library.db_path})")

    # Phase 0b: Initialize Crossover (if enabled)
    if use_crossover:
        crossover_agent = Crossover(
            model_name=model_name or MODEL_NAME,
            temperature=(
                crossover_temperature
                if crossover_temperature is not None
                else CROSSOVER_DEFAULT_TEMPERATURE
            ),
            use_cache=use_cache,
        )
        if verbose:
            print(f"\n{'=' * 70}")
            print(" Crossover Agent: Technique Fusion")
            print(f"{'=' * 70}")
            print(
                "✅ Crossover initialized (fusion will be used when 2+ diverse solvers exist)"
            )

    # Phase 0c: Initialize Tagger (if enabled)
    if use_tagger:
        tagger = Tagger(
            model_name=model_name or MODEL_NAME,
            temperature=(
                tagger_temperature
                if tagger_temperature is not None
                else TAGGER_DEFAULT_TEMPERATURE
            ),
            use_cache=use_cache,
        )
        if verbose:
            print(f"\n{'=' * 70}")
            print(" Tagger Agent: Technique Classification")
            print(f"{'=' * 70}")
            print(
                "✅ Tagger initialized (tags will be generated for successful solvers)"
            )

    # Phase 0d: Analyst analysis (AI Civilization mode only)
    if use_analyst:
        if verbose:
            print(f"\n{'=' * 70}")
            print(" AI Civilization Mode: Analyst Agent")
            print(f"{'=' * 70}")
            print("\n🔍 Analyzing task patterns...")

        analyst = Analyst(
            model_name=model_name or MODEL_NAME,  # Ensure string type for mypy
            temperature=(
                analyst_temperature
                if analyst_temperature is not None
                else ANALYST_DEFAULT_TEMPERATURE
            ),
            use_cache=use_cache,
        )

        # Load full task for Analyst
        with open(task_json_path) as f:
            task_json = json.load(f)

        analyst_spec = analyst.analyze_task(task_json)

        if verbose:
            print("✅ Pattern analysis complete")
            print(f"  Pattern: {analyst_spec.pattern_description}")
            print(f"  Confidence: {analyst_spec.confidence}")
            print(
                f"  Observations: {len(analyst_spec.key_observations)} key observations"
            )

    # Phase 2: Evolution loop
    for generation in range(max_generations):
        gen_start_time = time.time()

        if verbose:
            print(f"\n{'=' * 70}")
            print(f" Generation {generation}")
            print(f"{'=' * 70}")

        # Generation 0: Generate initial solver
        if generation == 0:
            if verbose:
                mode = "AI Civilization" if use_analyst else "Direct"
                print(
                    f"\n📝 Generating initial solver from train examples ({mode} mode)..."
                )

            current_code = generate_solver(
                train_pairs,
                model_name=model_name,
                temperature=programmer_temperature,
                timeout=timeout_per_llm,
                use_cache=use_cache,
                analyst_spec=analyst_spec,  # Pass analyst result to Programmer
            )

            if verbose:
                print(f"✅ Initial solver generated ({len(current_code)} characters)")

            refinement_count = 0

        # Subsequent generations: Refine if needed
        else:
            # Check if refinement needed
            if target_fitness is not None and previous_fitness >= target_fitness:
                # Target already reached, stop evolution
                if verbose:
                    print(
                        f"\n🎯 Target fitness {target_fitness} reached in generation {generation - 1}"
                    )
                    print("Evolution complete!")
                break

            # DECISION: Crossover vs Mutation (Refiner)
            crossover_used = False

            # Check if crossover is available and feasible
            if crossover_agent is not None:
                # Get diverse parent solvers from library
                diverse_solvers = library.get_diverse_solvers(task_id, num_solvers=2)

                if len(diverse_solvers) >= 2:
                    # USE CROSSOVER: Fusion of diverse techniques
                    if verbose:
                        print(
                            f"\n🧬 Crossover: Fusing {len(diverse_solvers)} diverse solvers..."
                        )
                        parent_techniques = [
                            ", ".join(s.tags) if s.tags else "none"
                            for s in diverse_solvers
                        ]
                        for i, (solver, techniques) in enumerate(
                            zip(diverse_solvers, parent_techniques, strict=True), 1
                        ):
                            print(
                                f"  Parent {i} (fitness {solver.fitness_score:.1f}): {techniques}"
                            )

                    # Load task JSON for crossover context
                    with open(task_json_path) as f:
                        task_json = json.load(f)

                    crossover_result = crossover_agent.fuse_solvers(
                        diverse_solvers, task_json, analyst_spec=analyst_spec
                    )
                    current_code = crossover_result.fused_code
                    crossover_used = True

                    if verbose:
                        print(
                            f"✅ Solvers fused ({len(current_code)} characters) - {crossover_result.compatibility_assessment[:50]}..."
                        )
                        print(f"   Confidence: {crossover_result.confidence}")

                else:
                    # Fall back to mutation - insufficient diverse solvers
                    if verbose:
                        print(
                            f"\n🔧 Mutation: Refining solver (only {len(diverse_solvers)} diverse solver(s) available)..."
                        )

            # USE MUTATION (Refiner) - either crossover not enabled or not feasible
            if not crossover_used:
                if verbose and crossover_agent is None:
                    print(
                        f"\n🔧 Mutation: Refining solver (fitness {previous_fitness:.1f} < target {target_fitness if target_fitness else 'N/A'})..."
                    )

                # Get previous fitness result for refiner context
                prev_result = results[-1]["fitness_result"]
                current_code = refine_solver(
                    current_code,
                    task_json_path,
                    prev_result,
                    model_name=model_name,
                    temperature=refiner_temperature,
                    timeout=timeout_per_llm,
                    use_cache=use_cache,
                    analyst_spec=analyst_spec,  # Pass analyst result to Refiner
                )

                if verbose:
                    print(f"✅ Solver refined ({len(current_code)} characters)")

            refinement_count = 1

        # Evaluate fitness
        if verbose:
            print("\n📊 Evaluating fitness...")

        fitness_result = calculate_fitness(
            task_json_path,
            current_code,
            timeout=timeout_per_eval,
            sandbox_mode=sandbox_mode,
        )

        current_fitness = fitness_result["fitness"]

        if verbose:
            print(f"Fitness: {current_fitness:.1f}")
            print(
                f"  Train: {fitness_result['train_correct']}/{fitness_result['train_total']} "
                f"({fitness_result['train_accuracy']:.0%})"
            )
            print(
                f"  Test: {fitness_result['test_correct']}/{fitness_result['test_total']} "
                f"({fitness_result['test_accuracy']:.0%})"
            )

            if generation > 0:
                improvement = current_fitness - previous_fitness
                print(f"  Improvement: {improvement:+.1f}")

        # Tag solver techniques (if Tagger enabled and fitness > 0)
        tags: list[str] = []
        if tagger is not None and current_fitness > 0:
            if verbose:
                print("\n🏷️  Tagging solver techniques...")

            # Load task JSON for Tagger context
            with open(task_json_path) as f:
                task_json = json.load(f)

            tagging_result = tagger.tag_solver(current_code, task_json)
            tags = tagging_result.tags

            if verbose:
                if tags:
                    print(f"✅ Techniques identified: {', '.join(tags)}")
                    print(f"   Confidence: {tagging_result.confidence}")
                else:
                    print("  No specific techniques identified")

        # Store solver in library (Phase 3.5 - only when crossover enabled)
        new_solver_id: str | None = None
        parent_solver_id: str | None = None

        if use_crossover:
            parent_solver_id = current_solver_id  # Previous generation's solver
            new_solver_id = str(uuid.uuid4())

            solver_record = SolverRecord(
                solver_id=new_solver_id,
                task_id=task_id,
                generation=generation,
                code_str=current_code,
                fitness_score=current_fitness,
                train_correct=fitness_result["train_correct"],
                test_correct=fitness_result["test_correct"],
                parent_solver_id=parent_solver_id if generation > 0 else None,
                tags=tags,
                created_at=datetime.now(UTC).isoformat(),
            )
            library.add_solver(solver_record)
            current_solver_id = new_solver_id  # Update for next generation

        # Calculate metrics
        gen_total_time = time.time() - gen_start_time
        improvement = (
            float(current_fitness - previous_fitness) if generation > 0 else 0.0
        )

        # Record generation result
        generation_result: GenerationResult = {
            "generation": generation,
            "solver_code": current_code,
            "fitness_result": fitness_result,
            "refinement_count": refinement_count,
            "total_time": gen_total_time,
            "improvement": improvement,
        }

        # Add solver tracking fields (only when crossover enabled)
        if use_crossover and new_solver_id is not None:
            generation_result["solver_id"] = new_solver_id
            generation_result["parent_solver_id"] = parent_solver_id

        # Add tags if generated
        if tags:
            generation_result["tags"] = tags

        # Add crossover flag if used (Phase 3.4)
        if generation > 0 and use_crossover:
            generation_result["crossover_used"] = crossover_used

        results.append(generation_result)

        # Update for next iteration
        previous_fitness = current_fitness

        if verbose:
            print(f"⏱️  Generation time: {gen_total_time:.2f}s")

        # Check early termination
        if target_fitness is not None and current_fitness >= target_fitness:
            if verbose:
                print(f"\n🎯 Target fitness {target_fitness} reached!")
                print("Evolution complete!")
            break

    if verbose and results:
        print(f"\n{'=' * 70}")
        print(" Evolution Summary")
        print(f"{'=' * 70}")
        print(f"Generations completed: {len(results)}")
        print(f"Initial fitness: {results[0]['fitness_result']['fitness']:.1f}")
        print(f"Final fitness: {results[-1]['fitness_result']['fitness']:.1f}")
        total_improvement = (
            results[-1]["fitness_result"]["fitness"]
            - results[0]["fitness_result"]["fitness"]
        )
        print(f"Total improvement: {total_improvement:+.1f}")
        total_time = sum(r["total_time"] for r in results)
        print(f"Total time: {total_time:.2f}s")

    return results
