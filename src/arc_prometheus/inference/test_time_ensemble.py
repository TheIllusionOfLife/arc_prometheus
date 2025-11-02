"""Test-Time Ensemble Pipeline - Orchestrates agents for diverse predictions.

This module implements the test-time ensemble approach that combines:
1. Multi-Persona Analyst (5 diverse interpretations)
2. Multi-Solution Programmer (5 diverse solutions)
3. Synthesis Agent (6th meta-learning solution)

The pipeline returns pass@2 predictions (best + synthesis) for all test inputs.

Example:
    ```python
    from arc_prometheus.inference import solve_task_ensemble

    task = load_task("data/arc-prize-2025/training/00576224.json")
    predictions = solve_task_ensemble(task)

    # predictions = [(best_pred, synthesis_pred), ...]
    # One tuple per test input
    for i, (best, synth) in enumerate(predictions):
        print(f"Test {i}: best shape={best.shape}, synth shape={synth.shape}")
    ```
"""

import logging

import numpy as np

from ..cognitive_cells.multi_persona_analyst import (
    InterpretationResult,
    MultiPersonaAnalyst,
)
from ..cognitive_cells.multi_solution_programmer import (
    MultiSolutionProgrammer,
    SolutionResult,
)
from ..cognitive_cells.synthesis_agent import SynthesisAgent
from ..crucible.sandbox import MultiprocessSandbox
from ..crucible.sandbox_protocol import ExecutionEnvironment

logger = logging.getLogger(__name__)


def _calculate_accuracies(
    task: dict,
    solutions: list[SolutionResult],
    sandbox: ExecutionEnvironment,
    timeout: int,
) -> list[float]:
    """Calculate train accuracy for each solution.

    Args:
        task: ARC task dictionary with 'train' examples
        solutions: List of SolutionResult objects to evaluate
        sandbox: Sandbox environment for safe execution
        timeout: Timeout in seconds for each execution

    Returns:
        List of accuracy floats (0.0-1.0), one per solution
    """
    accuracies = []

    for solution in solutions:
        correct = 0
        total = len(task["train"])

        for example in task["train"]:
            input_grid = np.array(example["input"], dtype=np.int64)
            expected = np.array(example["output"], dtype=np.int64)

            success, result, _ = sandbox.execute(solution.code, input_grid, timeout)

            if (
                success
                and result is not None
                and result.shape == expected.shape
                and np.array_equal(result, expected)
            ):
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        accuracies.append(accuracy)

    return accuracies


def _select_best_solution(
    solutions: list[SolutionResult], accuracies: list[float]
) -> tuple[SolutionResult, float]:
    """Select solution with highest train accuracy.

    Args:
        solutions: List of SolutionResult objects
        accuracies: List of accuracy scores (same length as solutions)

    Returns:
        Tuple of (best_solution, best_accuracy)

    Raises:
        ValueError: If solutions or accuracies are empty
    """
    if not solutions or not accuracies:
        raise ValueError("Cannot select best solution from empty lists")

    if len(solutions) != len(accuracies):
        raise ValueError(
            f"Solutions ({len(solutions)}) and accuracies ({len(accuracies)}) length mismatch"
        )

    # Find index of maximum accuracy (first occurrence if tie)
    best_idx = accuracies.index(max(accuracies))
    return solutions[best_idx], accuracies[best_idx]


def _pad_solutions(
    solutions: list[SolutionResult], interpretations: list[InterpretationResult]
) -> tuple[list[SolutionResult], list[InterpretationResult]]:
    """Pad solutions to exactly 5 for Synthesis agent compatibility.

    If fewer than 5 solutions are provided, duplicates the best solution
    (first in list) and its interpretation to reach 5 total.

    Args:
        solutions: List of 1-5 SolutionResult objects
        interpretations: List of 5 InterpretationResult objects

    Returns:
        Tuple of (padded_solutions, matched_interpretations) both length 5

    Raises:
        ValueError: If solutions list is empty or interpretations != 5
    """
    if not solutions:
        raise ValueError("Cannot pad empty solutions list")

    if len(interpretations) != 5:
        raise ValueError(f"Expected 5 interpretations, got {len(interpretations)}")

    if len(solutions) >= 5:
        # Take first 5 solutions
        return solutions[:5], interpretations[:5]

    # Need to pad to 5
    padded_solutions = solutions.copy()
    matched_interpretations = []

    # Match existing solutions to their interpretations
    for solution in solutions:
        interp_idx = solution.interpretation_id - 1  # IDs are 1-indexed
        if 0 <= interp_idx < 5:
            matched_interpretations.append(interpretations[interp_idx])
        else:
            # Fallback to first interpretation if ID out of range
            matched_interpretations.append(interpretations[0])

    # Duplicate first solution and interpretation to reach 5
    while len(padded_solutions) < 5:
        padded_solutions.append(solutions[0])
        matched_interpretations.append(
            interpretations[solutions[0].interpretation_id - 1]
        )

    logger.warning(
        f"Padded {len(solutions)} solutions to 5 by duplicating best solution"
    )

    return padded_solutions, matched_interpretations


def _execute_on_test_input(
    code: str,
    test_input: np.ndarray,
    sandbox: ExecutionEnvironment,
    timeout: int,
) -> np.ndarray:
    """Execute solution code on a single test input.

    Args:
        code: Python code string with solve() function
        test_input: Test input grid as numpy array
        sandbox: Sandbox environment for safe execution
        timeout: Timeout in seconds

    Returns:
        Result grid as numpy array, or placeholder [[0, 0], [0, 0]] on failure
    """
    success, result, _ = sandbox.execute(code, test_input, timeout)

    if success and result is not None:
        return result
    else:
        # Return placeholder on failure
        logger.warning("Execution failed, returning placeholder grid")
        return np.array([[0, 0], [0, 0]], dtype=np.int64)


def solve_task_ensemble(
    task: dict,
    model_name: str = "gemini-2.0-flash-thinking-exp",
    analyst_temperature: float = 1.0,
    programmer_temperature: float = 0.7,
    synthesis_temperature: float = 0.5,
    use_cache: bool = True,
    timeout: int = 5,
    sandbox_mode: str = "multiprocess",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate pass@2 predictions for all test inputs using ensemble.

    Orchestrates three agents to generate diverse predictions:
    1. Multi-Persona Analyst analyzes the task from 5 perspectives
    2. Multi-Solution Programmer generates 5 diverse solutions
    3. Synthesis Agent creates a 6th solution via meta-learning

    Returns the best-performing solution and synthesis solution predictions
    for all test inputs in the task.

    Args:
        task: ARC task dictionary with 'train' and 'test' sections
        model_name: Gemini model to use (default: gemini-2.0-flash-thinking-exp)
        analyst_temperature: Temperature for Analyst (default 1.0 for diversity)
        programmer_temperature: Temperature for Programmer (default 0.7)
        synthesis_temperature: Temperature for Synthesis (default 0.5)
        use_cache: Whether to use LLM response caching (default True)
        timeout: Timeout in seconds for solution execution (default 5)
        sandbox_mode: Sandbox mode ("multiprocess" or "docker", default "multiprocess")

    Returns:
        List of (best_prediction, synthesis_prediction) tuples as numpy arrays,
        one tuple per test input in the task

    Raises:
        ValueError: If task has no test inputs, or if any agent fails

    Example:
        ```python
        task = {
            "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 3]]}]
        }
        predictions = solve_task_ensemble(task)
        # predictions = [(best_pred, synthesis_pred)]
        ```
    """
    # Validate task has test inputs
    if "test" not in task or not task["test"]:
        raise ValueError("Task must have at least one test input")

    test_inputs = task["test"]
    logger.info(
        f"Starting test-time ensemble for task with {len(test_inputs)} test inputs"
    )

    # Step 1: Multi-Persona Analysis
    logger.info("Starting Multi-Persona Analyst...")
    analyst = MultiPersonaAnalyst(
        model_name=model_name,
        temperature=analyst_temperature,
        use_cache=use_cache,
    )
    interpretations = analyst.analyze_task(task)
    logger.info(f"Generated {len(interpretations)} interpretations")

    # Step 2: Multi-Solution Generation
    logger.info("Starting Multi-Solution Programmer...")
    programmer = MultiSolutionProgrammer(
        model_name=model_name,
        temperature=programmer_temperature,
        use_cache=use_cache,
    )
    solutions = programmer.generate_multi_solutions(task, interpretations)
    logger.info(f"Generated {len(solutions)}/5 valid solutions")

    # Step 3: Pad solutions to 5 if needed (Synthesis requires exactly 5)
    if len(solutions) < 5:
        solutions, matched_interpretations = _pad_solutions(solutions, interpretations)
        logger.info("Padded to 5 solutions for Synthesis agent")
    else:
        # Use first 5 solutions and match interpretations
        solutions = solutions[:5]
        matched_interpretations = [
            interpretations[sol.interpretation_id - 1] for sol in solutions
        ]

    # Step 4: Calculate train accuracies and select best
    logger.info("Calculating solution accuracies on training examples...")
    sandbox: ExecutionEnvironment
    if sandbox_mode == "multiprocess":
        sandbox = MultiprocessSandbox()
    else:
        # Docker sandbox would be imported here
        from ..crucible.docker_sandbox import DockerSandbox

        sandbox = DockerSandbox()

    accuracies = _calculate_accuracies(task, solutions, sandbox, timeout)
    best_solution, best_accuracy = _select_best_solution(solutions, accuracies)
    logger.info(
        f"Best solution: {best_accuracy * 100:.1f}% accuracy on train "
        f"(interpretation {best_solution.interpretation_id})"
    )

    # Step 5: Synthesis - Create 6th solution
    logger.info("Starting Synthesis Agent...")
    synthesis_agent = SynthesisAgent(
        model_name=model_name,
        temperature=synthesis_temperature,
        use_cache=use_cache,
        timeout=timeout,
        sandbox_mode=sandbox_mode,
    )
    synthesis_result = synthesis_agent.synthesize_solution(
        task, solutions, matched_interpretations
    )
    logger.info(f"Generated synthesis solution: {synthesis_result.approach_summary}")

    # Handle case where all solutions failed on train
    if best_accuracy == 0.0:
        logger.warning(
            "All solutions failed on train examples - using synthesis for both attempts"
        )
        best_solution = SolutionResult(
            interpretation_id=0,  # Dummy ID
            code=synthesis_result.code,
            approach_summary=synthesis_result.approach_summary,
        )

    # Step 6: Execute best + synthesis on all test inputs
    logger.info(f"Executing on {len(test_inputs)} test inputs...")
    predictions = []

    for i, test_example in enumerate(test_inputs):
        test_input_grid = np.array(test_example["input"], dtype=np.int64)

        # Execute best solution
        best_pred = _execute_on_test_input(
            best_solution.code, test_input_grid, sandbox, timeout
        )

        # Execute synthesis solution
        synthesis_pred = _execute_on_test_input(
            synthesis_result.code, test_input_grid, sandbox, timeout
        )

        predictions.append((best_pred, synthesis_pred))
        logger.debug(
            f"Test input {i + 1}/{len(test_inputs)}: "
            f"best={best_pred.shape}, synthesis={synthesis_pred.shape}"
        )

    logger.info(f"Ensemble complete: {len(predictions)} predictions generated")
    return predictions
