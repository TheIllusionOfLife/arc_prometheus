"""Demo script for Phase 1.4: LLM Code Generation

This script demonstrates:
1. Loading ARC task
2. Generating solver code with Gemini API (gemini-2.5-flash-lite)
3. Executing generated code in sandbox
4. Evaluating results

Usage:
    python scripts/demo_phase1_4_llm.py [task_id]

    If no task_id provided, uses a simple task (e.g., 007bbfb7)
"""

import sys

from arc_prometheus.cognitive_cells.programmer import generate_solver
from arc_prometheus.crucible.data_loader import load_task, print_grid
from arc_prometheus.crucible.evaluator import evaluate_grids
from arc_prometheus.crucible.sandbox import safe_execute
from arc_prometheus.utils.config import DATA_DIR


def main():
    """Run LLM code generation demo."""
    print("=" * 70)
    print("ARC-Prometheus Phase 1.4 Demo: LLM Code Generation")
    print("=" * 70)
    print("\nUsing: gemini-2.5-flash-lite (Google's latest, fastest model)")

    # Select task
    task_id = sys.argv[1] if len(sys.argv) > 1 else "007bbfb7"
    print(f"Task ID: {task_id}")

    # Load task
    challenges_file = DATA_DIR / "arc-agi_training_challenges.json"

    if not challenges_file.exists():
        print(f"\n❌ ERROR: Dataset not found at {challenges_file}")
        print("\nPlease download the ARC Prize 2025 dataset from:")
        print("https://www.kaggle.com/competitions/arc-prize-2025/data")
        print(f"And place it in: {DATA_DIR}")
        sys.exit(1)

    try:
        task_data = load_task(str(challenges_file), task_id=task_id)
    except Exception as e:
        print(f"\n❌ ERROR loading task: {e}")
        sys.exit(1)

    print(f"\n✓ Loaded task: {len(task_data['train'])} train examples")

    # Display first train example
    print("\n" + "─" * 70)
    print("Sample Train Example")
    print("─" * 70)
    print_grid(task_data["train"][0]["input"], label="Input")
    print_grid(task_data["train"][0]["output"], label="Expected Output")

    # Generate solver code
    print("\n" + "=" * 70)
    print("GENERATING SOLVER CODE WITH GEMINI")
    print("=" * 70)
    print("\n⏳ Calling Gemini API (this may take 10-30 seconds)...")

    try:
        solver_code = generate_solver(task_data["train"])
        print("\n✅ Code generation successful!")
    except Exception as e:
        print(f"\n❌ ERROR during code generation: {e}")
        sys.exit(1)

    # Display generated code
    print("\n" + "─" * 70)
    print("Generated Solver Code")
    print("─" * 70)
    print(solver_code)
    print("─" * 70)

    # Test solver on train examples
    print("\n" + "=" * 70)
    print("TESTING GENERATED SOLVER")
    print("=" * 70)

    correct_count = 0
    total_count = len(task_data["train"])

    for idx, example in enumerate(task_data["train"], 1):
        print(f"\n{'─' * 70}")
        print(f"Train Example {idx}/{total_count}")
        print(f"{'─' * 70}")

        # Execute in sandbox
        print("⏳ Executing in sandbox...")
        success, result = safe_execute(solver_code, example["input"], timeout=5)

        if not success:
            print("❌ EXECUTION FAILED: Timeout or error occurred")
            continue

        # Evaluate
        is_correct = evaluate_grids(result, example["output"])

        if is_correct:
            print("✅ MATCH: Generated solver solved this example!")
            correct_count += 1
        else:
            print("❌ MISMATCH: Output does not match expected")
            print_grid(result, label="Predicted")
            print_grid(example["output"], label="Expected")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    success_rate = (correct_count / total_count) * 100
    print(f"\nSolver Performance: {correct_count}/{total_count} ({success_rate:.0f}%)")

    if correct_count > 0:
        print("\n🎉 FIRST VICTORY: AI-generated code solved ARC train pair(s)!")
        print("\n✓ Phase 1.4 Complete:")
        print("  - Gemini API integration working (gemini-2.5-flash-lite)")
        print("  - Code parser handles LLM responses")
        print("  - Generated solver executed successfully")
        print(f"  - {correct_count} train example(s) solved correctly")
    else:
        print("\n⚠️  No train examples solved yet")
        print("   Try a different task or refine prompts")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\nPhase 1.5: End-to-End Pipeline")
    print("  - Create run_phase1_test.py orchestrator")
    print("  - Test on multiple ARC tasks")
    print("  - Calculate overall success rate")
    print("  - Complete Phase 1 milestone!")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
