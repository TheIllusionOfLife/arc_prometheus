#!/usr/bin/env python3
"""Quick test script for Tagger with real API."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arc_prometheus.cognitive_cells.tagger import Tagger

# Test code with clear techniques
test_code = """
import numpy as np

def solve(task_grid):
    # Rotate the grid 90 degrees
    rotated = np.rot90(task_grid)

    # Fill zeros with a specific color
    result = rotated.copy()
    result[result == 0] = 5

    # Count non-zero elements
    count = len(result[result != 0])

    return result
"""

# Simple task context
task_json = {"train": [{"input": [[1, 0], [0, 1]], "output": [[5, 1], [1, 5]]}]}

print("Testing Tagger with real Gemini API...\n")
print("Code to analyze:")
print(test_code)
print("\n" + "=" * 70)

# Create Tagger instance
tagger = Tagger(use_cache=False)

# Run tagging
result = tagger.tag_solver(test_code, task_json)

print("\n✅ Tagging complete!\n")
print(f"Tags identified: {result.tags}")
print(f"Confidence: {result.confidence}")

print("\n" + "=" * 70)
print("Expected tags: rotation, color_fill, counting, array_manipulation")
print(f"Actual tags:   {', '.join(result.tags)}")

# Verify some expected tags are present
expected_tags = ["rotation", "color_fill", "counting"]
found_tags = [tag for tag in expected_tags if tag in result.tags]

print(f"\nTags found: {len(found_tags)}/{len(expected_tags)}")
for tag in found_tags:
    print(f"  ✓ {tag}")

for tag in expected_tags:
    if tag not in found_tags:
        print(f"  ✗ {tag} (not detected)")

if len(found_tags) >= 2:
    print("\n✅ Test PASSED: Tagger detected multiple techniques correctly!")
else:
    print("\n⚠️  Test WARNING: Expected more techniques to be detected")
