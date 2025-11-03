# Cell 5: Test-Time Ensemble Agents (Full Implementation)

This file contains the complete Cell 5 code for the Kaggle notebook.

**Copy this entire code block into Cell 5 of your Kaggle notebook.**

---

## Complete Code for Cell 5

```python
# Cell 5: Test-Time Ensemble Agents (Exact Gemini Workflow Replication)

# IMPORTANT: These classes use EXACT prompts copied from:
# - src/arc_prometheus/cognitive_cells/multi_persona_analyst.py
# - src/arc_prometheus/cognitive_cells/multi_solution_programmer.py
# - src/arc_prometheus/cognitive_cells/synthesis_agent.py
#
# Temperatures match exactly: Analyst=1.0, Programmer=0.0, Synthesis=0.0


class OfflineMultiPersonaAnalyst:
    """
    Multi-Persona Analyst for offline inference with Code Gemma.

    Generates 4 diverse expert interpretations (exact Gemini workflow).
    Temperature: 1.0 (high diversity)
    """

    # Default 4-persona set (from multi_persona_analyst.py lines 50-75)
    DEFAULT_PERSONAS = {
        "persona_1": {
            "name": "Geometric Transformation Specialist",
            "emoji": "🔷",
            "focus": "Rotations (90°/180°/270°), reflections (horizontal/vertical), transpose, symmetry operations",
            "key_question": "Is this a spatial transformation of the grid itself?",
        },
        "persona_2": {
            "name": "Color Pattern Specialist",
            "emoji": "🎨",
            "focus": "Color changes, filling, replacement rules, color-based conditional logic",
            "key_question": "What color operations transform input to output?",
        },
        "persona_3": {
            "name": "Object Detection Specialist",
            "emoji": "🔍",
            "focus": "Shapes, connected regions, bounding boxes, object extraction/manipulation",
            "key_question": "Are there objects/regions to identify and manipulate?",
        },
        "persona_4": {
            "name": "Grid Structure Specialist",
            "emoji": "📐",
            "focus": "Cropping, slicing, tiling, partitioning, resizing, concatenation, grid dimensions",
            "key_question": "Is this about grid dimensions or regional selection?",
        },
    }

    def __init__(self, temperature: float = 1.0):
        """Initialize with exact Gemini temperature."""
        self.temperature = temperature
        self.personas = self.DEFAULT_PERSONAS

    def analyze_task(self, task_json: dict) -> list[dict]:
        """
        Analyze ARC task and return 4 diverse interpretations.

        Returns:
            List of 4 interpretation dicts with keys:
            - persona, pattern, observations, approach, confidence
        """
        # Format training examples (identical to multi_persona_analyst.py)
        train_examples = task_json["train"]
        examples_text = []

        for idx, example in enumerate(train_examples):
            input_grid = example["input"]
            output_grid = example["output"]

            input_str = format_grid(input_grid)
            output_str = format_grid(output_grid)

            examples_text.append(
                f"Example {idx + 1}:\n"
                f"Input ({len(input_grid)}x{len(input_grid[0])}):\n{input_str}\n\n"
                f"Output ({len(output_grid)}x{len(output_grid[0])}):\n{output_str}\n"
            )

        training_examples = "\n".join(examples_text)

        # Format persona instructions (identical to multi_persona_analyst.py)
        persona_instructions = []
        for _persona_id, persona_data in self.personas.items():
            persona_instructions.append(
                f"{persona_data['emoji']} {persona_data['name']}:\n"
                f"  Focus: {persona_data['focus']}\n"
                f"  Key Question: {persona_data['key_question']}"
            )

        personas_text = "\n\n".join(persona_instructions)

        # EXACT prompt from multi_persona_analyst.py lines 176-217
        prompt = f"""Analyze this ARC task from 4 DIFFERENT expert perspectives.

TRAINING EXAMPLES:
{training_examples}

YOUR TASK:
You are a team of 4 experts analyzing this puzzle. Each expert has a different specialization
and must provide their unique interpretation. Diversity is critical - all 4 perspectives
must be DIFFERENT.

THE 4 EXPERTS:
{personas_text}

INSTRUCTIONS:
1. Each expert analyzes the task from their specialized perspective
2. Each expert provides:
   - Pattern: One-sentence transformation rule (≤150 chars, be concise)
   - Observations: 1-3 key insights specific to their expertise (each ≤85 chars)
   - Approach: High-level implementation strategy (≤200 chars max, mention numpy operations)
   - Confidence: "high", "medium", or "low"
3. All 4 interpretations MUST be different - no duplicate patterns
4. Be extremely concise - stick to character limits strictly

CONCISENESS EXAMPLES (follow these patterns):
Good approach (158 chars):
  "Use np.rot90 to rotate grid 90° clockwise, then apply np.where to replace colors: blue→red. Check each training example to confirm rotation direction"

Bad approach (too verbose):
  "First we need to carefully analyze the grid structure and determine the appropriate rotation angle, then we should systematically examine each cell to identify which colors need to be changed and what the mapping should be..."

Good observation (72 chars):
  "Output is 90° rotation of input with color 1→3 substitution preserved"

Bad observation (too verbose):
  "The output grid appears to be a rotated version of the input grid, specifically rotated by 90 degrees, and additionally there seems to be some color transformations happening..."

IMPORTANT:
- Focus on the ABSTRACT RULE that works for ALL examples
- Be specific about what changes (colors, positions, shapes, sizes)
- Suggest numpy operations (np.rot90, np.flip, np.tile, np.where, etc.)
- Each expert's pattern must be unique from the others"""

        # Generate structured output (temperature=1.0 for diversity)
        result = generate_structured_json(
            prompt,
            MultiPersonaResponse,
            temperature=self.temperature,
            max_tokens=65536,
        )

        # Convert Pydantic models to dicts
        interpretations = []
        for interp in result.interpretations:
            interpretations.append(
                {
                    "persona": interp.persona,
                    "pattern": interp.pattern,
                    "observations": interp.observations,
                    "approach": interp.approach,
                    "confidence": interp.confidence,
                }
            )

        return interpretations


class OfflineMultiSolutionProgrammer:
    """
    Multi-Solution Programmer for offline inference with Code Gemma.

    Generates 4 solver implementations (exact Gemini workflow).
    Temperature: 0.0 (deterministic)
    """

    def __init__(self, temperature: float = 0.0):
        """Initialize with exact Gemini temperature."""
        self.temperature = temperature

    def generate_multi_solutions(
        self, task_json: dict, interpretations: list[dict]
    ) -> list[dict]:
        """
        Generate 4 solver implementations from 4 interpretations.

        Returns:
            List of solution dicts with keys:
            - interpretation_id, code, approach_summary
        """
        # Format training examples (identical to multi_solution_programmer.py)
        train_examples = task_json["train"]
        examples_text = []

        for idx, example in enumerate(train_examples):
            input_grid = example["input"]
            output_grid = example["output"]

            input_str = format_grid(input_grid)
            output_str = format_grid(output_grid)

            examples_text.append(
                f"Example {idx + 1}:\n"
                f"Input ({len(input_grid)}x{len(input_grid[0])}):\n{input_str}\n\n"
                f"Output ({len(output_grid)}x{len(output_grid[0])}):\n{output_str}\n"
            )

        training_examples = "\n".join(examples_text)

        # Format interpretations (identical to multi_solution_programmer.py)
        interpretation_instructions = []
        for idx, interp in enumerate(interpretations, 1):
            interpretation_instructions.append(
                f"Interpretation {idx} - {interp['persona']}:\n"
                f"  Pattern: {interp['pattern']}\n"
                f"  Approach: {interp['approach']}\n"
                f"  Confidence: {interp['confidence']}"
            )

        interpretations_text = "\n\n".join(interpretation_instructions)

        # EXACT prompt from multi_solution_programmer.py lines 149-206
        prompt = f"""Implement ALL 4 expert interpretations as complete Python solve() functions.

TRAINING EXAMPLES:
{training_examples}

EXPERT INTERPRETATIONS TO IMPLEMENT:
{interpretations_text}

YOUR TASK:
Implement each of the 4 interpretations above as a separate, complete solve() function.
Each implementation must:
1. Be a complete, runnable Python function
2. Include necessary imports (especially numpy)
3. Match the signature: def solve(task_grid: np.ndarray) -> np.ndarray:
4. Use ONLY numpy for array operations (no other libraries)
5. Follow the specific approach suggested by that interpretation
6. Be concise but complete

IMPORTANT CONSTRAINTS:
- Each solution's code must be self-contained and executable
- Include "import numpy as np" at the top if using numpy
- All 4 implementations must be DIFFERENT (different approaches/logic)
- Each approach_summary must be ≤200 characters max
- Link each solution to its interpretation_id (1-4)

CODE CONCISENESS REQUIREMENTS:
- Aim for ≤1500 characters per solution code (strict limit to prevent token overflow)
- NO comments whatsoever - code should be self-explanatory through clear variable names
- NO docstrings - function signature and context make purpose clear
- Use loops or helper functions instead of repetitive np.where/np.roll chains
- Avoid commented-out alternative approaches or exploratory notes

CRITICAL ERROR PREVENTION:
- NEVER import scipy or any library except numpy (import numpy as np)
- ALWAYS return np.ndarray type, NEVER return list (use np.array() to convert if needed)
- CHECK your syntax: no missing colons, no indentation errors, no unclosed brackets
- AVOID AttributeError: ensure you're calling methods on correct types (e.g., .shape on arrays not lists)
- TEST your logic: ensure index access doesn't go out of bounds

CONCISENESS EXAMPLES (follow these patterns):
Good approach_summary (134 chars):
  "Rotate input 90° clockwise using np.rot90(grid, k=-1), then replace color 1→3 with np.where(rotated == 1, 3, rotated). Return result."

Bad approach_summary (too verbose):
  "First apply a rotation transformation to the input grid by using the numpy rotation function, then carefully examine each cell to determine if any color changes are needed and apply those changes systematically..."

Good approach_summary (97 chars):
  "Use np.flip(grid, axis=0) for vertical flip, then np.where to fill color 0→5 in border regions"

Bad approach_summary (too verbose):
  "Start by flipping the grid vertically to mirror the rows, and then we need to identify the border cells and change their colors from background to the target color..."

OUTPUT FORMAT:
Provide a JSON array with 4 solutions, each containing:
- interpretation_id: Integer 1-4 (which interpretation it implements)
- code: Complete Python code string with solve() function
- approach_summary: Brief description (≤200 chars max)"""

        # Generate structured output (temperature=0.0 for deterministic)
        result = generate_structured_json(
            prompt,
            MultiSolutionResponse,
            temperature=self.temperature,
            max_tokens=65536,
        )

        # Convert Pydantic models to dicts
        solutions = []
        for sol in result.solutions:
            solutions.append(
                {
                    "interpretation_id": sol.interpretation_id,
                    "code": sol.code,
                    "approach_summary": sol.approach_summary,
                }
            )

        return solutions


class OfflineSynthesisAgent:
    """
    Synthesis Agent for offline inference with Code Gemma.

    Generates 5th solution via meta-learning (exact Gemini workflow).
    Temperature: 0.0 (deterministic)
    """

    def __init__(self, temperature: float = 0.0, timeout: int = 5):
        """Initialize with exact Gemini temperature."""
        self.temperature = temperature
        self.timeout = timeout

    def _calculate_solution_accuracies(
        self, task_json: dict, solutions: list[dict]
    ) -> list[dict]:
        """Calculate accuracy for each solution on training examples."""
        accuracies = []

        for solution in solutions:
            train_results = []
            for example in task_json["train"]:
                try:
                    input_grid = np.array(example["input"], dtype=np.int64)
                    expected = np.array(example["output"], dtype=np.int64)

                    success, result_grid, _ = execute_solver_safe(
                        solution["code"], input_grid, timeout=self.timeout
                    )

                    if (
                        success
                        and result_grid is not None
                        and result_grid.shape == expected.shape
                        and np.array_equal(result_grid, expected)
                    ):
                        train_results.append(True)
                    else:
                        train_results.append(False)

                except Exception:
                    train_results.append(False)

            train_correct = sum(train_results)
            train_total = len(train_results)
            train_accuracy = train_correct / train_total if train_total > 0 else 0.0

            accuracies.append(
                {
                    "train_accuracy": train_accuracy,
                    "train_correct": train_correct,
                    "train_total": train_total,
                }
            )

        return accuracies

    def synthesize_solution(
        self,
        task_json: dict,
        solutions: list[dict],
        interpretations: list[dict],
    ) -> dict:
        """
        Generate diverse 5th solution through meta-learning.

        Returns:
            Synthesis dict with keys:
            - code, approach_summary, successful_patterns, failed_patterns,
              synthesis_strategy, diversity_justification
        """
        # Calculate accuracies for all 4 solutions
        accuracies = self._calculate_solution_accuracies(task_json, solutions)

        # Format training examples (identical to synthesis_agent.py)
        train_examples = task_json["train"]
        examples_text = []

        for idx, example in enumerate(train_examples):
            input_grid = example["input"]
            output_grid = example["output"]

            input_str = format_grid(input_grid)
            output_str = format_grid(output_grid)

            examples_text.append(
                f"Example {idx + 1}:\n"
                f"Input ({len(input_grid)}x{len(input_grid[0])}):\n{input_str}\n\n"
                f"Output ({len(output_grid)}x{len(output_grid[0])}):\n{output_str}\n"
            )

        training_examples = "\n".join(examples_text)

        # Format solution summaries with accuracies (identical to synthesis_agent.py)
        solution_summaries = []
        for idx, (solution, accuracy, interp) in enumerate(
            zip(solutions, accuracies, interpretations), 1
        ):
            train_pct = accuracy["train_accuracy"] * 100
            status = "✓" if accuracy["train_accuracy"] > 0.5 else "✗"

            solution_summaries.append(
                f"{status} Solution {idx} (Interpretation: {interp['persona']})\n"
                f"  Train Accuracy: {train_pct:.0f}% ({accuracy['train_correct']}/{accuracy['train_total']})\n"
                f"  Approach: {solution['approach_summary']}\n"
                f"  Original Pattern: {interp['pattern']}\n"
            )

        solutions_summary = "\n".join(solution_summaries)

        # EXACT prompt from synthesis_agent.py lines 264-334
        prompt = f"""Analyze 4 previous solutions and create a DIVERSE 5th solution.

TRAINING EXAMPLES:
{training_examples}

PREVIOUS 4 SOLUTIONS (with accuracy on training examples):
{solutions_summary}

YOUR TASK:
You are a meta-learning expert analyzing these 4 attempts to solve the ARC task.

1. ANALYZE successful patterns:
   - What approaches worked well (high train accuracy)?
   - What operations/transformations led to correct outputs?
   - Extract 1-3 key successful patterns (≤80 chars each)

2. ANALYZE failed patterns:
   - What approaches failed (low train accuracy)?
   - What anti-patterns should be avoided?
   - Extract 1-3 key failed patterns (≤80 chars each)

3. SYNTHESIZE a 5th solution:
   - Learn from successful patterns
   - Avoid failed patterns
   - Use a DIFFERENT algorithm than all 4 previous attempts
   - Must be diverse in approach, not just different implementation

IMPORTANT CONSTRAINTS:
- The 5th solution MUST use a different algorithm/approach than all 4 previous
- Include "import numpy as np" at the top if using numpy
- Must follow signature: def solve(task_grid: np.ndarray) -> np.ndarray
- Use ONLY numpy for array operations (no other libraries)
- Code must be complete, runnable, and self-contained
- Provide clear diversity justification explaining why different
- Synthesis strategy should explain how you combined insights

CODE CONCISENESS REQUIREMENTS:
- Aim for ≤1500 characters for solution code (strict limit to prevent token overflow)
- NO comments whatsoever - code should be self-explanatory through clear variable names
- NO docstrings - function signature and context make purpose clear
- Use loops or helper functions instead of repetitive np.where/np.roll chains
- Avoid commented-out alternative approaches or exploratory notes

CRITICAL ERROR PREVENTION:
- NEVER import scipy or any library except numpy (import numpy as np)
- ALWAYS return np.ndarray type, NEVER return list (use np.array() to convert if needed)
- CHECK your syntax: no missing colons, no indentation errors, no unclosed brackets
- AVOID AttributeError: ensure you're calling methods on correct types (e.g., .shape on arrays not lists)
- TEST your logic: ensure index access doesn't go out of bounds

OUTPUT FORMAT:
Provide analysis and synthesis code in structured JSON format with:
- analysis.successful_patterns: List of 1-3 strings (each ≤85 chars)
- analysis.failed_patterns: List of 1-3 strings (each ≤85 chars)
- analysis.synthesis_strategy: String (≤250 chars max) explaining approach
- code: Complete Python code string with solve() function
- diversity_justification: String (≤200 chars max) why different from all 4

CONCISENESS EXAMPLES (follow these patterns):
Good synthesis_strategy (141 chars):
  "Combine border detection with flood fill from solutions 1 & 3, add rotation from 2, use different algorithm than all 4 (graph traversal)"

Bad synthesis_strategy (too verbose):
  "First we need to carefully identify the border regions of the input grid by examining each cell and determining if it's on the edge, then we should apply a flood fill algorithm similar to what was attempted in solution 1 but modified to work better..."

Good diversity_justification (84 chars):
  "Uses graph traversal instead of direct indexing; all 4 used array manipulation"

Bad diversity_justification (too verbose):
  "This solution is different from all the previous four solutions because it takes a completely different algorithmic approach by using graph traversal..." """

        # Generate structured output (temperature=0.0 for deterministic)
        result = generate_structured_json(
            prompt,
            SynthesisResponse,
            temperature=self.temperature,
            max_tokens=65536,
        )

        # Convert Pydantic model to dict
        synthesis = {
            "code": result.code,
            "approach_summary": result.analysis.synthesis_strategy[:100],
            "successful_patterns": result.analysis.successful_patterns,
            "failed_patterns": result.analysis.failed_patterns,
            "synthesis_strategy": result.analysis.synthesis_strategy,
            "diversity_justification": result.diversity_justification,
        }

        return synthesis


print("✅ Test-Time Ensemble agents ready (exact Gemini workflow)")
```

---

## Notes

This cell contains ~490 lines of code with 3 complete agent classes:

1. **OfflineMultiPersonaAnalyst** (~170 lines)
   - Temperature: 1.0 (high diversity)
   - Generates 4 diverse expert interpretations
   - Uses exact prompt from `multi_persona_analyst.py`

2. **OfflineMultiSolutionProgrammer** (~120 lines)
   - Temperature: 0.0 (deterministic)
   - Generates 4 solver implementations
   - Uses exact prompt from `multi_solution_programmer.py`

3. **OfflineSynthesisAgent** (~200 lines)
   - Temperature: 0.0 (deterministic)
   - Evaluates 4 solutions on training data
   - Generates 5th solution via meta-learning
   - Uses exact prompt from `synthesis_agent.py`

All prompts are copied verbatim from the Gemini agents to ensure exact workflow replication.
