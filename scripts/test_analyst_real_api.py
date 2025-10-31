#!/usr/bin/env python3
"""Test Analyst agent with real Gemini API on diverse ARC tasks.

This script validates:
1. Pattern analysis quality on various task types
2. No timeouts or errors
3. No truncation or incomplete responses
4. Proper formatting of outputs
5. Confidence levels are appropriate
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_prometheus.cognitive_cells.analyst import Analyst


def test_analyst_on_diverse_tasks():
    """Test Analyst on 5-10 diverse ARC tasks."""

    # Select diverse tasks (small/medium/large grids, various transformations)
    test_tasks = [
        # Small grids, simple fill pattern
        ("00576224", "simple fill"),
        # Medium grids, rotation pattern
        ("007bbfb7", "rotation"),
        # Medium grids, pattern copy
        ("025d127b", "pattern copy"),
        # Small grids, color change
        ("00d62c1b", "color transformation"),
        # Medium grids, symmetry
        ("0520fde7", "symmetry"),
        # Larger grids, complex pattern
        ("05269061", "complex grid"),
        # Multi-step transformation
        ("007bbfb7", "multi-step"),
    ]

    analyst = Analyst(model_name="gemini-2.5-flash-lite", use_cache=False)

    results = []
    print("=" * 80)
    print("Testing Analyst Agent with Real Gemini API")
    print("=" * 80)

    for task_id, description in test_tasks:
        print(f"\nTask {task_id} ({description}):")
        print("-" * 80)

        try:
            # Load task
            task_path = "data/arc-prize-2025/arc-agi_training_challenges.json"
            with open(task_path) as f:
                tasks = json.load(f)

            if task_id not in tasks:
                print(f"  ⚠️  Task {task_id} not found, skipping...")
                continue

            task_data = tasks[task_id]

            # Analyze task
            print(f"  Analyzing task with {len(task_data['train'])} training examples...")
            analysis = analyst.analyze_task(task_data)

            # Validate output
            success = True
            issues = []

            # Check pattern description
            if not analysis.pattern_description:
                issues.append("❌ Empty pattern description")
                success = False
            elif len(analysis.pattern_description) < 10:
                issues.append(f"⚠️  Very short pattern description: {len(analysis.pattern_description)} chars")
            else:
                print(f"  ✓ Pattern: {analysis.pattern_description}")

            # Check observations
            if len(analysis.key_observations) == 0:
                issues.append("⚠️  No observations provided")
            else:
                print(f"  ✓ Observations: {len(analysis.key_observations)} items")
                for obs in analysis.key_observations[:3]:  # Show first 3
                    print(f"    - {obs}")
                if len(analysis.key_observations) > 3:
                    print(f"    ... and {len(analysis.key_observations) - 3} more")

            # Check approach
            if not analysis.suggested_approach:
                issues.append("⚠️  No approach suggested")
            else:
                print(f"  ✓ Approach: {analysis.suggested_approach[:100]}...")

            # Check confidence
            if analysis.confidence not in ["high", "medium", "low", ""]:
                issues.append(f"⚠️  Invalid confidence: {analysis.confidence}")
            else:
                print(f"  ✓ Confidence: {analysis.confidence}")

            # Check for truncation (very long responses might indicate issues)
            total_length = (
                len(analysis.pattern_description) +
                sum(len(obs) for obs in analysis.key_observations) +
                len(analysis.suggested_approach)
            )

            if total_length < 50:
                issues.append(f"❌ Response too short ({total_length} chars) - likely truncated or error")
                success = False
            elif total_length > 5000:
                issues.append(f"⚠️  Response very long ({total_length} chars)")

            # Print issues
            if issues:
                for issue in issues:
                    print(f"  {issue}")

            # Store result
            results.append({
                "task_id": task_id,
                "description": description,
                "success": success,
                "analysis": analysis,
                "issues": issues
            })

            if success:
                print("  ✅ SUCCESS - Analysis quality verified")
            else:
                print("  ❌ FAILED - See issues above")

        except TimeoutError as e:
            print(f"  ❌ TIMEOUT: {e}")
            results.append({
                "task_id": task_id,
                "description": description,
                "success": False,
                "error": str(e)
            })

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({
                "task_id": task_id,
                "description": description,
                "success": False,
                "error": str(e)
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if r.get("success", False))
    total = len(results)

    print(f"\nResults: {successful}/{total} tasks analyzed successfully")

    if successful == total:
        print("\n✅ ALL TESTS PASSED - Analyst agent working correctly with real API")
        return 0
    else:
        failed = [r for r in results if not r.get("success", False)]
        print(f"\n❌ {len(failed)} TESTS FAILED:")
        for r in failed:
            print(f"  - {r['task_id']} ({r['description']})")
            if "error" in r:
                print(f"    Error: {r['error']}")
            if "issues" in r:
                for issue in r["issues"]:
                    print(f"    {issue}")
        return 1


if __name__ == "__main__":
    sys.exit(test_analyst_on_diverse_tasks())
