"""End-to-End Pipeline for Phase 1.5: Complete ARC Task Solving

Orchestrates all Phase 1 components:
1. Load ARC task from dataset
2. Generate solver code with LLM (Gemini)
3. Execute solver in sandbox on train examples
4. Evaluate correctness and report results
5. Save successful solvers

Usage:
    python scripts/run_phase1_test.py <task_id>

    Example:
    python scripts/run_phase1_test.py 00576224

Requirements:
    - GEMINI_API_KEY environment variable set
    - ARC dataset in data/arc-prize-2025/
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_prometheus.cognitive_cells.programmer import generate_solver
from arc_prometheus.crucible.data_loader import load_task, print_grid
from arc_prometheus.crucible.evaluator import evaluate_grids
from arc_prometheus.crucible.sandbox import safe_execute
from arc_prometheus.utils.config import DATA_DIR


def print_header(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print(f"\n{char * 70}")
    print(title)
    print(f"{char * 70}")


def print_code_section(code: str, max_lines: int = 30) -> None:
    """Print generated code with line numbers."""
    lines = code.split("\n")
    if len(lines) > max_lines:
        # Show first 20 and last 10 lines
        for i, line in enumerate(lines[:20], 1):
            print(f"{i:3d} | {line}")
        print(f"... ({len(lines) - 30} lines omitted) ...")
        for i, line in enumerate(lines[-10:], len(lines) - 9):
            print(f"{i:3d} | {line}")
    else:
        for i, line in enumerate(lines, 1):
            print(f"{i:3d} | {line}")


def save_solver(task_id: str, solver_code: str, success_rate: float) -> Path:
    """Save successful solver to output directory."""
    output_dir = Path("output") / "solvers"
    output_dir.mkdir(parents=True, exist_ok=True)

    solver_file = output_dir / f"solver_{task_id}.py"

    # Add header comment
    header = f'''"""
ARC-Prometheus Generated Solver
Task ID: {task_id}
Success Rate: {success_rate:.0f}%
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

import numpy as np

'''

    with open(solver_file, "w") as f:
        f.write(header + solver_code)

    return solver_file


def run_e2e_pipeline(task_id: str) -> dict:
    """Run complete E2E pipeline for a task.

    Returns:
        dict with keys: success, correct_count, total_count,
                       solver_code, error_message
    """
    result = {
        "success": False,
        "correct_count": 0,
        "total_count": 0,
        "solver_code": None,
        "error_message": None,
    }

    # Step 1: Load task
    print_header("STEP 1: Load ARC Task")
    challenges_file = DATA_DIR / "arc-agi_training_challenges.json"

    if not challenges_file.exists():
        result["error_message"] = f"Dataset not found at {challenges_file}"
        return result

    try:
        task_data = load_task(str(challenges_file), task_id=task_id)
        print(f"✅ Task {task_id} loaded successfully")
        print(f"   Train examples: {len(task_data['train'])}")
        print(f"   Test examples: {len(task_data['test'])}")
    except Exception as e:
        result["error_message"] = f"Failed to load task: {e}"
        return result

    # Step 2: Generate solver code
    print_header("STEP 2: Generate Solver Code with LLM")
    print("Calling Gemini API to generate solver...")

    try:
        start_time = time.time()
        solver_code = generate_solver(task_data["train"], timeout=60)
        generation_time = time.time() - start_time

        print(f"✅ Code generation successful in {generation_time:.1f}s")
        print(f"\n{'─' * 70}")
        print("Generated Code:")
        print(f"{'─' * 70}")
        print_code_section(solver_code)

        result["solver_code"] = solver_code

    except Exception as e:
        result["error_message"] = f"Code generation failed: {e}"
        return result

    # Step 3: Execute and evaluate on train examples
    print_header("STEP 3: Execute Solver on Train Examples")

    print("\n⚠️  SECURITY WARNING:")
    print("    Multiprocessing sandbox does NOT prevent filesystem/network access.")
    print("    LLM-generated code could potentially access system resources.")
    print("    For production: Use Docker with read-only filesystem.\n")

    correct_count = 0
    total_count = len(task_data["train"])
    result["total_count"] = total_count

    for idx, example in enumerate(task_data["train"], 1):
        print(f"\n{'─' * 70}")
        print(f"Train Example {idx}/{total_count}")
        print(f"{'─' * 70}")

        # Show input
        print("\nInput Grid:")
        print_grid(example["input"], label="")

        # Execute in sandbox
        print("\nExecuting solver in sandbox (timeout: 5s)...")
        start_time = time.time()
        success, result_grid = safe_execute(solver_code, example["input"], timeout=5)
        execution_time = time.time() - start_time

        if not success:
            print(f"❌ EXECUTION FAILED in {execution_time:.2f}s")
            print("   Reason: Timeout, exception, or invalid return type")
            continue

        print(f"✓ Execution completed in {execution_time:.3f}s")

        # Evaluate correctness
        is_correct = evaluate_grids(result_grid, example["output"])

        if is_correct:
            print("\n✅ MATCH: Predicted output matches expected output!")
            correct_count += 1
        else:
            print("\n❌ MISMATCH: Predicted output does NOT match expected")
            print("\nPredicted Output:")
            print_grid(result_grid, label="")
            print("\nExpected Output:")
            print_grid(example["output"], label="")

            # Show first difference
            if result_grid.shape == example["output"].shape:
                diff_mask = result_grid != example["output"]
                if np.any(diff_mask):
                    diff_positions = np.argwhere(diff_mask)
                    first_diff = diff_positions[0]
                    i, j = first_diff[0], first_diff[1]
                    print(f"\n   First difference at position ({i}, {j}):")
                    print(
                        f"   Predicted: {result_grid[i, j]}, Expected: {example['output'][i, j]}"
                    )
            else:
                print(
                    f"\n   Shape mismatch: Predicted {result_grid.shape} vs Expected {example['output'].shape}"
                )

    result["correct_count"] = correct_count
    result["success"] = True

    return result


def main():
    """Main entry point."""
    print_header("ARC-Prometheus Phase 1.5: End-to-End Pipeline", "=")
    print("Complete AI Solver Generation and Evaluation")
    print("=" * 70)

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("\n❌ ERROR: Task ID required")
        print("\nUsage:")
        print("  python scripts/run_phase1_test.py <task_id>")
        print("\nExample:")
        print("  python scripts/run_phase1_test.py 00576224")
        sys.exit(1)

    task_id = sys.argv[1]
    print(f"\nTask ID: {task_id}")

    # Run pipeline
    result = run_e2e_pipeline(task_id)

    # Print results
    print_header("FINAL RESULTS", "=")

    if not result["success"]:
        print("\n❌ PIPELINE FAILED")
        print(f"   Error: {result['error_message']}")
        sys.exit(1)

    correct_count = result["correct_count"]
    total_count = result["total_count"]
    success_rate = (correct_count / total_count * 100) if total_count > 0 else 0

    print(
        f"\nSolver Performance: {correct_count}/{total_count} correct ({success_rate:.0f}%)"
    )

    if correct_count > 0:
        print("\n🎉 SUCCESS: AI-generated solver solved train examples!")

        # Save solver
        solver_file = save_solver(task_id, result["solver_code"], success_rate)
        print(f"\n✓ Saved solver to: {solver_file}")

        print("\n" + "=" * 70)
        print("Phase 1 Milestone Achieved!")
        print("AI successfully generated code that solves ARC train pairs")
        print("=" * 70)
    else:
        print("\n⚠️  PARTIAL SUCCESS: Pipeline executed but solver failed all examples")
        print("   This demonstrates:")
        print("   - LLM code generation works")
        print("   - Sandbox execution works")
        print("   - Evaluation logic works")
        print("   Next: Try different task or implement Phase 2 (Refiner agent)")


if __name__ == "__main__":
    main()
