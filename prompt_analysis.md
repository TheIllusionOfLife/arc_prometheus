# AI Civilization Prompt Analysis & Improvement Suggestions

## Current Prompts

### 1. ANALYST PROMPT (analyst.py:191-224)

**Current:**
```
Analyze this ARC task and infer the transformation rule.
You will see N training example(s).

[Examples shown as ASCII grids]

Your task:
1. Observe patterns across all training examples
2. Identify the core transformation rule that applies to ALL examples
3. Describe the rule in one clear sentence
4. Note key features (colors, shapes, positions, symmetries, patterns)
5. Suggest high-level implementation approach using numpy operations

Output format (REQUIRED - use exact keywords):
PATTERN: [One clear sentence]
OBSERVATIONS:
- [Key observation 1]
- [Key observation 2]
APPROACH: [High-level numpy strategy]
CONFIDENCE: [high/medium/low]

Important:
- Focus on ABSTRACT RULE, not implementation details
- Rule must work for ALL training examples
- Be specific about what changes
- Suggest numpy operations (np.rot90, np.flip, np.tile, etc.)
```

**Assessment:** ✅ Working well - produces good analysis with high confidence

---

### 2. PROGRAMMER PROMPT (prompts.py:48-246)

**Current (AI Civilization mode with Analyst):**
```
You are a Programmer agent in an AI civilization working to solve ARC puzzles.

## Pattern Analysis (from Analyst Agent)
**Transformation Rule:** [Analyst's pattern description]

**Key Observations:**
- [Analyst's observations]

**Suggested Implementation Approach:**
[Analyst's suggested approach]

**Analyst Confidence:** [high/medium/low]

## Task
Implement the transformation rule described above as a Python function.
Use the training examples below to verify your understanding.

## Training Examples (for verification)
[Examples as ASCII grids]

## Instructions
1. Implement the transformation rule described by the Analyst
2. Use the suggested approach as guidance for numpy operations
3. Verify your implementation matches all training examples
4. Implement with this EXACT signature:
   def solve(task_grid: np.ndarray) -> np.ndarray:

## Requirements
- Use ONLY numpy (no other libraries)
- Function must be named 'solve' (lowercase)
- Must accept one parameter: task_grid (np.ndarray)
- Must return np.ndarray
- Include 'import numpy as np' at the top
- Handle edge cases
- Output grid may have different dimensions than input

## Code Style - CRITICAL
- Write CONCISE code with MINIMAL comments
- Aim for 20-50 lines maximum
- DO NOT explore multiple approaches - commit to ONE solution quickly
- DO NOT add explanatory comments about what you're trying
- If stuck after ONE attempt, return a simple fallback
- Response will be truncated at 8000 tokens - brevity is essential

## Example of GOOD Concise Code
```python
import numpy as np

def solve(task_grid: np.ndarray) -> np.ndarray:
    if task_grid.size == 0:
        return task_grid
    # Detect pattern: if all values < 5, add 1; else subtract 1
    if np.all(task_grid < 5):
        return task_grid + 1
    return task_grid - 1
```

## Output Format
Return ONLY the Python code, starting with 'import numpy as np'.
Do NOT include explanations, debugging commentary, or multiple attempts.
Just the code that can be executed directly.
```

**Assessment:** ⚠️ Produces syntactically valid code, but logic is often wrong (wrong shapes, wrong transformations)

---

### 3. REFINER PROMPT (prompts.py:249-446)

**Current:**
```
You are an expert Python debugger for ARC solvers.

## Goal
Debug and fix the provided solver code that failed to solve the ARC task correctly.

[Optional: Original Pattern Analysis from Analyst if available]

## Original Task Examples
Here are some examples from the task (for context):
[Shows first 3 training examples]

## Failed Code
This code attempted to solve the task but failed:
```python
[The broken code]
```

## Failure Analysis
Performance: X/Y train correct, X/Y test correct

Execution errors:
- [Error message 1]
- [Error message 2]
...

[Optional: Error-type-specific debugging strategy based on error classifier]

## Requirements
Fix the bugs while maintaining these requirements:
- Use ONLY numpy
- Function must be named 'solve'
- Must return np.ndarray
- Include 'import numpy as np'
- Handle edge cases properly
- Ensure fixed code correctly transforms inputs to match expected outputs

## Output Format
Return ONLY the corrected Python code, starting with 'import numpy as np'.
Do NOT include explanations or debugging commentary.
Just the fixed code that can be executed directly.
```

**Assessment:** ❌ CRITICAL PROBLEM - Rewrites code from scratch instead of debugging the specific bugs!

---

## Problems Identified

### Programmer Issues:
1. **Wrong Logic** - Generates code that runs but produces incorrect results (shape mismatches, wrong transformations)
2. **Doesn't Verify** - Prompt says "verify implementation matches" but model doesn't actually test
3. **Too Abstract** - Analyst's high-level description doesn't translate well to specific numpy operations

### Refiner Issues (MOST CRITICAL):
1. **Complete Rewrite** - Instead of fixing bugs, generates entirely different code
2. **No Code Diff Analysis** - Doesn't identify WHAT is wrong, just tries a new approach
3. **No Incremental Fixes** - Doesn't preserve working parts, throws everything away
4. **Different Algorithm** - Example: Original used row slicing → Refiner uses color clustering
5. **Still Wrong** - New code is equally broken, just different bugs

---

## Improvement Suggestions

### A. Immediate Fix for Refiner (HIGHEST PRIORITY)

**Add to Refiner prompt:**
```
## CRITICAL DEBUGGING RULES

1. DO NOT rewrite the entire function - FIX THE SPECIFIC BUGS ONLY
2. Preserve the CORE ALGORITHM from the failed code
3. Identify the EXACT LINE(S) causing the problem
4. Make MINIMAL changes to fix those specific lines
5. Keep everything else UNCHANGED

## Debugging Process:
Step 1: Read the failed code carefully - what algorithm does it use?
Step 2: Look at the error messages - which specific lines fail?
Step 3: Compare expected vs actual outputs - what's the difference?
Step 4: Fix ONLY those specific issues - preserve everything else
Step 5: Verify your fix addresses the error without breaking working code

## Example of GOOD Debugging:

FAILED CODE:
```python
output = task_grid[2:-1, 1:-1]  # BUG: Wrong slice, should be [2:, 1:-1]
return output
```

FIXED CODE:
```python
output = task_grid[2:, 1:-1]  # Fixed: Changed slice to include all rows from index 2
return output
```

❌ BAD: Completely different algorithm using color clustering
✅ GOOD: Same algorithm, just fixed the slice indices
```

### B. Improve Programmer Precision

**Add to Programmer prompt:**
```
## Verification Steps (MANDATORY)

After writing your code, mentally walk through EACH training example:
1. For Example 1: Input shape X, my code would produce shape Y, expected shape Z
2. For Example 2: [same analysis]
3. If ANY example doesn't match, FIX YOUR CODE before returning it

Common mistakes to AVOID:
- Off-by-one errors in slicing (use inclusive ranges)
- Wrong assumption about grid dimensions
- Forgetting edge cases (empty arrays, single row/column)
- Using hardcoded values from one example

SELF-CHECK before returning code:
☐ Does my code handle varying input sizes?
☐ Does the output shape match expected for ALL examples?
☐ Did I test edge cases (empty, minimal size)?
☐ Are all numpy operations correct (not list operations)?
```

### C. Add Analyst "Ground Truth" Checks

**Add to Analyst prompt:**
```
## Pattern Validation

After describing your pattern, verify it against EACH example:
- Example 1: Does my pattern explain this transformation? [Yes/No]
- Example 2: Does my pattern explain this transformation? [Yes/No]
- Example 3: [etc.]

If ANY example doesn't fit your pattern, revise your pattern description.

## Common ARC Patterns (for reference):
- Grid transformations: rotation, reflection, transpose
- Extraction: crop/select specific regions based on color/pattern
- Filling: fill regions with colors based on rules
- Object detection: find shapes and transform them
- Pattern repetition: tile or stack patterns
- Conditional logic: if X then Y transformation
```

---

## Recommended Implementation Priority

1. **HIGHEST:** Fix Refiner prompt to debug instead of rewrite
   - Impact: Could enable actual fitness improvements across generations
   - Effort: Just prompt changes, no code
   - Time: 30 minutes

2. **HIGH:** Add verification steps to Programmer
   - Impact: Fewer wrong initial solutions
   - Effort: Prompt changes
   - Time: 20 minutes

3. **MEDIUM:** Add pattern validation to Analyst
   - Impact: Better initial pattern descriptions
   - Effort: Prompt changes
   - Time: 15 minutes

4. **LOW:** Increase token limits to 16k (if still getting truncation)
   - Impact: Allow more complex solutions
   - Effort: Config change
   - Time: 2 minutes

---

## Expected Outcomes

**With Refiner fix:**
- Generations should show: fitness 0.0 → 0.5 → 2.0 → 5.0 (incremental improvement)
- Instead of: fitness 0.0 → 0.0 → 0.0 → 0.0 (random broken code)

**With Programmer verification:**
- More first-generation solutions with fitness > 0
- Fewer "wrong shape" errors
- Still won't be perfect (ARC is hard!) but better baseline

**With Analyst validation:**
- More accurate pattern descriptions
- Higher confidence scores when actually confident
- Better guidance for Programmer
