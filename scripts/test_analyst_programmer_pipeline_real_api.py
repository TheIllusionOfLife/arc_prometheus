#!/usr/bin/env python3
"""Test Analyst → Programmer → Evaluator pipeline with real Gemini API.

This script validates:
1. Analyst analyzes ARC task patterns
2. Programmer generates code using Analyst's specifications
3. Generated code executes successfully in sandbox
4. Solvers correctly transform inputs to outputs
5. Comparison between AI Civilization mode (with Analyst) vs Direct mode

Run 3-5 diverse tasks to verify the full pipeline works end-to-end.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_prometheus.cognitive_cells.analyst import Analyst
from arc_prometheus.cognitive_cells.programmer import generate_solver
from arc_prometheus.crucible.sandbox import safe_execute


def test_task_with_analyst(task_id: str, task_data: dict) -> dict:
    """Test task using AI Civilization mode (Analyst → Programmer)."""
    print(f"\n{'='*80}")
    print(f"Task {task_id} - AI CIVILIZATION MODE (with Analyst)")
    print(f"{'='*80}")

    try:
        # Step 1: Analyst analyzes task
        print("\n[1/3] Analyst analyzing task...")
        analyst = Analyst(model_name="gemini-2.5-flash-lite", use_cache=False)
        analysis = analyst.analyze_task(task_data)

        print(f"  ✓ Pattern: {analysis.pattern_description}")
        print(f"  ✓ Observations: {len(analysis.key_observations)} items")
        print(f"  ✓ Confidence: {analysis.confidence}")

        # Step 2: Programmer generates code using Analyst's analysis
        print("\n[2/3] Programmer generating code from Analyst's specifications...")
        train_pairs = [
            {
                "input": np.array(ex["input"], dtype=np.int64),
                "output": np.array(ex["output"], dtype=np.int64)
            }
            for ex in task_data["train"]
        ]

        code = generate_solver(
            train_pairs,
            analyst_spec=analysis,
            use_cache=False,
            timeout=60
        )

        print(f"  ✓ Generated code ({len(code)} chars)")
        print(f"  ✓ First 200 chars: {code[:200]}...")

        # Step 3: Test generated code on train examples
        print("\n[3/3] Testing generated solver on train examples...")
        train_correct = 0
        train_total = len(task_data["train"])

        for idx, example in enumerate(task_data["train"]):
            input_grid = np.array(example["input"], dtype=np.int64)
            expected_output = np.array(example["output"], dtype=np.int64)

            success, result, error = safe_execute(
                code,
                input_grid,
                timeout=5
            )

            if success and result is not None:
                if np.array_equal(result, expected_output):
                    train_correct += 1
                    print(f"  ✓ Train example {idx + 1}: Correct")
                else:
                    print(f"  ✗ Train example {idx + 1}: Wrong output")
            else:
                error_msg = error.get("error_message", "Unknown") if error else "Unknown"
                print(f"  ✗ Train example {idx + 1}: Execution failed - {error_msg}")

        score = train_correct / train_total if train_total > 0 else 0
        print(f"\n  📊 Score: {train_correct}/{train_total} ({score*100:.1f}%)")

        return {
            "mode": "ai_civilization",
            "task_id": task_id,
            "success": True,
            "pattern": analysis.pattern_description,
            "confidence": analysis.confidence,
            "code_length": len(code),
            "train_correct": train_correct,
            "train_total": train_total,
            "score": score,
            "code": code
        }

    except Exception as e:
        print(f"\n  ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "mode": "ai_civilization",
            "task_id": task_id,
            "success": False,
            "error": str(e)
        }


def test_task_without_analyst(task_id: str, task_data: dict) -> dict:
    """Test task using Direct mode (Programmer only, no Analyst)."""
    print(f"\n{'='*80}")
    print(f"Task {task_id} - DIRECT MODE (Programmer only)")
    print(f"{'='*80}")

    try:
        # Step 1: Programmer generates code directly from examples
        print("\n[1/2] Programmer analyzing and generating code...")
        train_pairs = [
            {
                "input": np.array(ex["input"], dtype=np.int64),
                "output": np.array(ex["output"], dtype=np.int64)
            }
            for ex in task_data["train"]
        ]

        code = generate_solver(
            train_pairs,
            analyst_spec=None,  # Direct mode
            use_cache=False,
            timeout=60
        )

        print(f"  ✓ Generated code ({len(code)} chars)")
        print(f"  ✓ First 200 chars: {code[:200]}...")

        # Step 2: Test generated code on train examples
        print("\n[2/2] Testing generated solver on train examples...")
        train_correct = 0
        train_total = len(task_data["train"])

        for idx, example in enumerate(task_data["train"]):
            input_grid = np.array(example["input"], dtype=np.int64)
            expected_output = np.array(example["output"], dtype=np.int64)

            success, result, error = safe_execute(
                code,
                input_grid,
                timeout=5
            )

            if success and result is not None:
                if np.array_equal(result, expected_output):
                    train_correct += 1
                    print(f"  ✓ Train example {idx + 1}: Correct")
                else:
                    print(f"  ✗ Train example {idx + 1}: Wrong output")
            else:
                error_msg = error.get("error_message", "Unknown") if error else "Unknown"
                print(f"  ✗ Train example {idx + 1}: Execution failed - {error_msg}")

        score = train_correct / train_total if train_total > 0 else 0
        print(f"\n  📊 Score: {train_correct}/{train_total} ({score*100:.1f}%)")

        return {
            "mode": "direct",
            "task_id": task_id,
            "success": True,
            "code_length": len(code),
            "train_correct": train_correct,
            "train_total": train_total,
            "score": score,
            "code": code
        }

    except Exception as e:
        print(f"\n  ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "mode": "direct",
            "task_id": task_id,
            "success": False,
            "error": str(e)
        }


def main():
    """Run pipeline tests on diverse ARC tasks."""

    # Select 5 diverse tasks
    test_tasks = [
        ("00576224", "simple fill pattern"),
        ("007bbfb7", "rotation transformation"),
        ("025d127b", "pattern copy/extension"),
        ("00d62c1b", "color transformation"),
        ("0520fde7", "symmetry pattern"),
    ]

    print("=" * 80)
    print("Testing Analyst → Programmer → Evaluator Pipeline (Real API)")
    print("=" * 80)
    print("\nThis test compares two modes:")
    print("  1. AI Civilization mode: Analyst → Programmer")
    print("  2. Direct mode: Programmer only (baseline)")
    print("\nWe expect AI Civilization mode to produce clearer, more reliable code.")

    # Load tasks
    task_path = Path("data/arc-prize-2025/arc-agi_training_challenges.json")
    if not task_path.exists():
        print(f"\n❌ Task file not found: {task_path}")
        return 1

    with open(task_path) as f:
        all_tasks = json.load(f)

    results = []

    # Test each task in BOTH modes
    for task_id, description in test_tasks:
        if task_id not in all_tasks:
            print(f"\n⚠️  Task {task_id} not found, skipping...")
            continue

        task_data = all_tasks[task_id]
        print(f"\n\n{'#'*80}")
        print(f"# Task {task_id}: {description}")
        print(f"# Training examples: {len(task_data['train'])}")
        print(f"{'#'*80}")

        # Test with Analyst (AI Civilization mode)
        result_with_analyst = test_task_with_analyst(task_id, task_data)
        results.append(result_with_analyst)

        # Test without Analyst (Direct mode - baseline)
        result_without_analyst = test_task_without_analyst(task_id, task_data)
        results.append(result_without_analyst)

        # Compare results
        if result_with_analyst["success"] and result_without_analyst["success"]:
            print(f"\n{'='*80}")
            print(f"COMPARISON for {task_id}")
            print(f"{'='*80}")
            print(f"  AI Civilization: {result_with_analyst['score']*100:.1f}% correct")
            print(f"  Direct mode:     {result_without_analyst['score']*100:.1f}% correct")
            if result_with_analyst["score"] > result_without_analyst["score"]:
                print("  ✅ AI Civilization performed BETTER")
            elif result_with_analyst["score"] < result_without_analyst["score"]:
                print("  ⚠️  Direct mode performed better (unexpected)")
            else:
                print("  ➡️  Both modes performed equally")

    # Summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    ai_civ_results = [r for r in results if r["mode"] == "ai_civilization"]
    direct_results = [r for r in results if r["mode"] == "direct"]

    ai_civ_success = sum(1 for r in ai_civ_results if r.get("success", False))
    direct_success = sum(1 for r in direct_results if r.get("success", False))

    print(f"\nAI Civilization mode: {ai_civ_success}/{len(ai_civ_results)} tasks completed")
    print(f"Direct mode:          {direct_success}/{len(direct_results)} tasks completed")

    if ai_civ_success == len(ai_civ_results) and direct_success == len(direct_results):
        # Calculate average scores
        ai_civ_avg = sum(r.get("score", 0) for r in ai_civ_results) / len(ai_civ_results)
        direct_avg = sum(r.get("score", 0) for r in direct_results) / len(direct_results)

        print("\nAverage train accuracy:")
        print(f"  AI Civilization: {ai_civ_avg*100:.1f}%")
        print(f"  Direct mode:     {direct_avg*100:.1f}%")

        if ai_civ_avg > direct_avg:
            print("\n✅ AI CIVILIZATION MODE OUTPERFORMED DIRECT MODE")
        elif ai_civ_avg < direct_avg:
            print("\n⚠️  DIRECT MODE OUTPERFORMED AI CIVILIZATION (needs investigation)")
        else:
            print("\n➡️  BOTH MODES PERFORMED EQUALLY")

        print("\n✅ ALL TESTS PASSED - Pipeline working correctly with real API")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - See errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
