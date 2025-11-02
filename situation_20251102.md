# Situation Report: LLM Code Generation Truncation Issue
**Date:** November 2, 2025
**Project:** ARC-Prometheus (AI Civilization for ARC Prize)
**Status:** 🔴 Critical Issue - 0% Test Accuracy Across All Experiments

---

## Problem Statement

**All population-based evolution experiments are producing 0% test accuracy** despite 100% success rate (no crashes). Root cause identified: **LLM-generated solver code is being truncated mid-function**, resulting in incomplete code that returns `None` instead of `np.ndarray`.

### Evidence
- 4 overnight experiments (5 tasks each, various population/generation configs)
- **All produced fitness=0.0** (0 correct test examples out of all attempts)
- **All experiments completed without crashes** (success_rate=1.0)
- Generated code examination shows truncation patterns

---

## Background

### System Architecture
- **Programmer Agent**: Uses Gemini LLM to generate numpy-based solver code from ARC task examples
- **Refiner Agent**: Debugs and fixes failed solver code (mutation operator)
- **Evolution Loop**: Iteratively improves solvers through selection + mutation/crossover
- **Fitness Function**: Evaluates accuracy (test_correct × 10 + train_correct × 1)

### ARC Task Complexity
- Abstract reasoning puzzles with 2D grid transformations
- Only ~3 training examples provided per task
- Solvers must generalize to unseen test examples
- Tasks are intentionally difficult (designed to be hard for AI)

---

## Experimental Results

### Overnight Population Experiments (All Failed)
| Experiment | Population | Generations | Avg Time/Task | Fitness | Test Accuracy |
|------------|-----------|-------------|---------------|---------|---------------|
| exp1 (low/low) | 5 | 3 | 94.3s | 0.0 | **0%** |
| exp2 (high/low) | 20 | 3 | 338.3s | 0.0 | **0%** |
| exp3 (low/high) | 5 | 10 | 133.6s | 0.0 | **0%** |
| exp4 (high/high) | 20 | 10 | 625.9s | 0.0 | **0%** |

**Key Insight**: Increasing population or generations made NO difference → **not a hyperparameter problem**

---

## Root Cause Analysis

### Investigation Timeline

#### 1. Initial Hypothesis: Insufficient Token Limit
- **Config found**: `max_output_tokens: 2048` (too low)
- **Evidence**: Generated code for task `13e47133` was 25 lines, ended with:
  ```python
  # "3. The 'new' colors are those in the output that are not in the input."
  # [CODE CUTS OFF - NO RETURN STATEMENT]
  ```

#### 2. First Fix Attempt: Increase Token Limit to 8192
**Changes made:**
- Updated `PROGRAMMER_GENERATION_CONFIG["max_output_tokens"]` from 2048 → 8192
- Added conciseness constraints to prompts ("Keep under 150 lines", "MINIMAL comments")

**Result:** ❌ **FAILED** - Still truncated at 8192 tokens

**New evidence:** Generated 24,474 characters of **repetitive code loops**:
```python
# Same pattern repeated 10+ times:
rows_nz, cols_nz = np.where(task_grid != 1)
min_row, max_row = np.min(rows_nz), np.max(rows_nz)
min_col, max_col = np.min(cols_nz), np.max(cols_nz)
non_one_region = task_grid[min_row : max_row + 1, min_col : max_col + 1]
# ... (repeats verbatim) ...
```

**Conclusion**: Problem is **LLM behavior** (repetitive loops), not just token limit.

---

## Solutions Attempted

### ✅ What Worked
1. **Added MAX_TOKENS handler** in `programmer.py`:
   - Detects `finish_reason=2` (MAX_TOKENS)
   - Extracts partial response via `response.parts`
   - Emits warning about truncation
   - Prevents crashes, allows evolution to continue

2. **Added truncation detection**:
   - Checks for missing `return` statements
   - Detects incomplete last lines
   - Provides useful debugging warnings

### ❌ What Didn't Work

#### Attempt 1: Switch to `gemini-2.5-pro`
**Rationale:** Pro model has better reasoning, supports 65k output tokens

**Result:** ❌ **FAILED**
- Hit **504 DeadlineExceeded** timeout (120 seconds)
- Model too slow for production use
- **ALSO** hit MAX_TOKENS limit at 8192 (same repetitive behavior)

#### Attempt 2: Enhanced Prompts with Few-Shot Examples
**Changes made:**
- Added concise code example (9 lines) showing good style
- Added explicit instructions: "Aim for 20-50 lines maximum"
- Added early stopping: "DO NOT explore multiple approaches - commit to ONE solution quickly"
- Added fallback instruction: "If stuck after ONE attempt, return a simple fallback"

**Result:** ❌ **FAILED** - Model **completely ignored** instructions

**Evidence:** Generated 122+ lines of repetitive code:
```python
output_grid[min_row+2:...] = np.array([[4,4,3,3]])
output_grid[min_row+3:...] = np.array([[4,4,3,3]])
# ... (repeated 120+ times) ...
output_grid[min_row+121:...] = np.array([[4,4,3,3]])
```

---

## Current State

### Code Changes Made
1. **config.py**:
   - Reverted to `gemini-2.5-flash-lite` (Pro too slow)
   - Kept `max_output_tokens: 8192` (up from 2048)
   - Added model selection documentation for easy switching

2. **programmer.py**:
   - Added MAX_TOKENS handler (lines 229-244)
   - Added truncation detection (lines 245-270)
   - Extracts partial responses instead of crashing

3. **prompts.py**:
   - Added "Code Style - CRITICAL" section with brevity requirements
   - Added few-shot concise example
   - Added early stopping instructions

### What We Know
- ✅ Truncation handler works (no more crashes)
- ✅ Warnings are emitted when truncation occurs
- ❌ LLM generates repetitive code regardless of instructions
- ❌ Both `gemini-2.5-flash-lite` and `gemini-2.5-pro` exhibit same behavior
- ❌ Stronger prompts do NOT reduce verbosity

---

## Options Being Considered

### Option A: Accept Truncation, Rely on Refiner ⚠️ CURRENT APPROACH
**Strategy:** Let Programmer generate truncated code, rely on Refiner to fix it in subsequent generations

**Pros:**
- No additional changes needed
- Refiner agent specifically designed to fix broken code
- Evolution loop should naturally filter out bad solutions

**Cons:**
- Wastes first generation (fitness=0 for all initial solvers)
- Refiner may struggle with severely broken code
- Slower evolution (needs more generations to converge)

**Status:** Background experiments still running - will show if this works

### Option B: Increase Token Limit Further (16k or 32k) 🤔 UNTESTED
**Strategy:** Accept verbosity, just allow model to complete its response

**Pros:**
- Both models support much higher limits (flash-lite: 65k, pro: 65k)
- Would allow code completion even with repetition
- Simple config change

**Cons:**
- Higher API costs (4x-8x token usage)
- Doesn't fix underlying verbosity issue
- May still hit limits on very complex tasks

**Implementation:**
```python
PROGRAMMER_GENERATION_CONFIG: Any = {
    "temperature": 0.3,
    "max_output_tokens": 16384,  # or 32768
}
```

### Option C: Try Different Model (gemini-2.0-flash-thinking-exp) 🤔 UNTESTED
**Strategy:** Use experimental thinking model with different generation behavior

**Pros:**
- Designed for reasoning tasks (good fit for ARC)
- May have different verbosity characteristics
- Faster than Pro, potentially more reliable than flash-lite

**Cons:**
- Experimental/unstable API
- Unknown behavior characteristics
- May have same repetition issue

### Option D: Post-Processing Filter 🤔 IDEA ONLY
**Strategy:** Detect repetitive patterns in generated code and truncate intelligently

**Pros:**
- Could salvage some repetitive but valid code
- Removes obvious redundancy
- Keeps token limits reasonable

**Cons:**
- Complex to implement correctly
- Risk of breaking valid code
- Doesn't fix root cause

**Implementation sketch:**
```python
def detect_repetition(code: str, threshold: float = 0.5) -> str:
    lines = code.split('\n')
    # Find repeating pattern
    # Truncate at first repetition
    # Ensure function is still valid
    return deduplicated_code
```

---

## Questions for Advisors

1. **Is repetitive LLM code generation a known issue with Gemini models on complex reasoning tasks?**
   - Have others encountered this?
   - Are there documented mitigation strategies?

2. **Should we switch to a completely different LLM provider?**
   - Would OpenAI GPT-4 / Claude / other models behave differently?
   - Trade-offs in cost/latency/API stability?

3. **Is our prompt design fundamentally flawed?**
   - Are we asking the wrong kind of question?
   - Should we break down task into smaller sub-prompts?

4. **Should we accept low initial fitness and trust evolution to fix it?**
   - Is 0% first-generation accuracy acceptable if Refiner can iterate?
   - How many generations typically needed for convergence?

5. **Alternative architectures?**
   - Should we use LLM for analysis only, not code generation?
   - Could we combine LLM analysis with template-based code generation?
   - Should we use a different approach entirely (non-LLM)?

---

## Technical Details

### Reproduction Steps
```bash
# Generate solver for task a25697e4 (demonstrates truncation)
cd /Users/yuyamukai/dev/arc_prometheus
uv run python << 'EOF'
import sys
sys.path.insert(0, 'src')
from arc_prometheus.cognitive_cells.programmer import generate_solver
import numpy as np
import json

with open('data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json') as f:
    tasks = json.load(f)

task = tasks['a25697e4']
train_pairs = [{"input": np.array(ex["input"], dtype=np.int64),
                "output": np.array(ex["output"], dtype=np.int64)}
               for ex in task["train"]]

code = generate_solver(train_pairs, use_cache=False)
print(f"Code length: {len(code)} chars")
print(f"Contains 'return': {code.count('return')} occurrences")
print("\nLast 10 lines:")
print('\n'.join(code.split('\n')[-10:]))
EOF
```

### Expected vs Actual Behavior
**Expected:**
- Concise numpy code (20-50 lines)
- Complete `solve()` function with return statement
- Diverse attempts across different tasks

**Actual:**
- Repetitive code (100-500 lines before truncation)
- Truncated mid-expression (no closing, no return)
- Same repetitive pattern across ALL tasks

### Environment
- **Model:** `gemini-2.5-flash-lite` (also tested: `gemini-2.5-pro`)
- **max_output_tokens:** 8192 (also tested: 2048)
- **Temperature:** 0.3 (low for determinism)
- **Python:** 3.14
- **google-generativeai:** Latest version

---

## Files Changed (for reference)

1. `src/arc_prometheus/utils/config.py` - Model selection, token limits
2. `src/arc_prometheus/cognitive_cells/programmer.py` - MAX_TOKENS handling
3. `src/arc_prometheus/cognitive_cells/prompts.py` - Enhanced prompts with examples

All changes are backward compatible and can be reverted easily.

---

## Next Steps

**Immediate (waiting on background experiments):**
1. Monitor 4 running population experiments to completion
2. Check if ANY achieve fitness > 0 with current fixes
3. Examine Refiner agent's ability to fix truncated code

**If experiments fail (fitness still 0):**
1. Try Option B: Increase tokens to 16k or 32k
2. If that fails, try Option C: Different model (thinking-exp)
3. If all fail, reconsider architecture (Option D or advisor suggestions)

**Timeline:**
- Background experiments: Running (started ~Nov 1, expected completion TBD)
- Decision point: After experiments complete
- Next iteration: Within 24 hours

---

## Contact & Discussion

For questions or suggestions, please discuss:
- What has worked for you with LLM code generation on complex tasks?
- Recommended alternative models or approaches?
- Similar experiences with Gemini API repetition issues?

Thank you for any insights! 🙏
