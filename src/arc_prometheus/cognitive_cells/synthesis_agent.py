"""Synthesis Agent - Meta-learning from 5 solutions to create diverse 6th solution.

This agent analyzes 5 solutions and their performance to generate a 6th solution
that learns from successful patterns and avoids failed approaches.

Key features:
- Analyzes accuracy of all 5 solutions on training examples
- Extracts successful patterns from high-accuracy solutions
- Identifies anti-patterns from failed solutions
- Generates diverse 6th solution using different algorithm
- Temperature 0.5 for balanced creativity/consistency
- Structured JSON output with meta-learning analysis
"""

import logging
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from arc_prometheus.cognitive_cells.multi_persona_analyst import InterpretationResult
from arc_prometheus.cognitive_cells.multi_solution_programmer import SolutionResult
from arc_prometheus.utils.config import get_gemini_api_key
from arc_prometheus.utils.schemas import SYNTHESIS_SCHEMA, SynthesisResponse

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Meta-learning synthesis solution with analysis.

    Attributes:
        code: Complete Python code with solve() function
        approach_summary: Brief description of implementation (≤100 chars)
        successful_patterns: Patterns from successful solutions (1-3 items, each ≤80 chars)
        failed_patterns: Patterns from failed solutions (1-3 items, each ≤80 chars)
        synthesis_strategy: How to create diverse 6th solution (≤150 chars)
        diversity_justification: Why different from all 5 previous (≤100 chars)
    """

    code: str
    approach_summary: str
    successful_patterns: list[str]
    failed_patterns: list[str]
    synthesis_strategy: str
    diversity_justification: str


class SynthesisAgent:
    """Generates diverse 6th solution by meta-learning from 5 solutions.

    This agent analyzes the performance of 5 generated solutions and creates
    a 6th solution that learns from successful patterns and avoids failed
    approaches, while using a DIFFERENT algorithm than all previous attempts.

    Example:
        ```python
        analyst = MultiPersonaAnalyst()
        programmer = MultiSolutionProgrammer()
        synthesis = SynthesisAgent()

        task = load_task("data/arc-prize-2025/training/00576224.json")
        interpretations = analyst.analyze_task(task)
        solutions = programmer.generate_multi_solutions(task, interpretations)
        synthesis_result = synthesis.synthesize_solution(task, solutions, interpretations)

        # Returns SynthesisResult with meta-learning analysis
        print(f"Synthesis: {synthesis_result.approach_summary}")
        print(f"Successful patterns: {synthesis_result.successful_patterns}")
        print(f"Failed patterns: {synthesis_result.failed_patterns}")
        ```
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-thinking-exp",
        temperature: float = 0.5,
        use_cache: bool = True,
        timeout: int = 5,
        sandbox_mode: str = "multiprocess",
    ):
        """Initialize Synthesis Agent.

        Args:
            model_name: Gemini model to use (default: gemini-2.0-flash-thinking-exp)
            temperature: LLM temperature (default 0.5 for balanced creativity/consistency)
            use_cache: Whether to use LLM response caching (default True)
            timeout: Timeout for solution execution in seconds (default 5)
            sandbox_mode: Sandbox mode for fitness calculation (default "multiprocess")
        """
        self.model_name = model_name
        self.temperature = temperature
        self.use_cache = use_cache
        self.timeout = timeout
        self.sandbox_mode = sandbox_mode

        # Configure Gemini API
        api_key = get_gemini_api_key()
        genai.configure(api_key=api_key)

    def _format_grid(self, grid: list) -> str:
        """Format grid as readable text representation.

        Args:
            grid: 2D list of integers

        Returns:
            String representation of grid
        """
        rows = []
        for row in grid:
            rows.append(" ".join(str(cell) for cell in row))
        return "\n".join(rows)

    def _calculate_solution_accuracies(
        self, task_json: dict, solutions: list[SolutionResult]
    ) -> list[dict]:
        """Calculate accuracy for each solution on training examples.

        Args:
            task_json: ARC task dictionary with 'train' examples
            solutions: List of SolutionResult objects to evaluate

        Returns:
            List of dicts with accuracy info for each solution:
            - train_accuracy: float (0.0-1.0)
            - train_correct: int
            - train_total: int
            - execution_errors: list[str]
            - has_errors: bool
        """
        accuracies = []

        for idx, solution in enumerate(solutions):
            logger.info(f"Evaluating solution {idx + 1}/5...")

            # Calculate fitness using sandbox
            # Note: We pass task_json directly, not a file path
            # This requires creating a temporary task structure
            try:
                # Evaluate on training examples only
                train_results = []
                for example in task_json["train"]:
                    try:
                        # Execute solution on this example
                        from arc_prometheus.crucible.sandbox import (
                            MultiprocessSandbox,
                        )

                        sandbox = MultiprocessSandbox()
                        success, result_grid, error = sandbox.execute(
                            solution.code, example["input"], timeout=self.timeout
                        )

                        if success and result_grid is not None:
                            # Check if result matches expected output
                            import numpy as np

                            expected = np.array(example["output"], dtype=np.int64)
                            if result_grid.shape == expected.shape and np.array_equal(
                                result_grid, expected
                            ):
                                train_results.append(True)
                            else:
                                train_results.append(False)
                        else:
                            train_results.append(False)

                    except Exception as e:
                        logger.warning(f"Error evaluating solution {idx + 1}: {e}")
                        train_results.append(False)

                # Calculate accuracy
                train_correct = sum(train_results)
                train_total = len(train_results)
                train_accuracy = train_correct / train_total if train_total > 0 else 0.0

                accuracies.append(
                    {
                        "train_accuracy": train_accuracy,
                        "train_correct": train_correct,
                        "train_total": train_total,
                        "execution_errors": [],
                        "has_errors": train_accuracy == 0.0,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to evaluate solution {idx + 1}: {e}")
                accuracies.append(
                    {
                        "train_accuracy": 0.0,
                        "train_correct": 0,
                        "train_total": len(task_json["train"]),
                        "execution_errors": [str(e)],
                        "has_errors": True,
                    }
                )

        return accuracies

    def _create_prompt(
        self,
        task_json: dict,
        solutions: list[SolutionResult],
        accuracies: list[dict],
        interpretations: list[InterpretationResult],
    ) -> str:
        """Create synthesis prompt with meta-learning context.

        Args:
            task_json: ARC task dictionary with 'train' examples
            solutions: List of 5 SolutionResult objects
            accuracies: List of accuracy dicts for each solution
            interpretations: List of 5 InterpretationResult objects from Analyst

        Returns:
            Prompt string for LLM with synthesis instructions
        """
        # Format training examples
        train_examples = task_json["train"]
        examples_text = []

        for idx, example in enumerate(train_examples):
            input_grid = example["input"]
            output_grid = example["output"]

            input_str = self._format_grid(input_grid)
            output_str = self._format_grid(output_grid)

            examples_text.append(
                f"Example {idx + 1}:\n"
                f"Input ({len(input_grid)}x{len(input_grid[0])}):\n{input_str}\n\n"
                f"Output ({len(output_grid)}x{len(output_grid[0])}):\n{output_str}\n"
            )

        training_examples = "\n".join(examples_text)

        # Format solution summaries with accuracies
        solution_summaries = []
        for idx, (solution, accuracy, interp) in enumerate(
            zip(solutions, accuracies, interpretations, strict=True), 1
        ):
            train_pct = accuracy["train_accuracy"] * 100
            status = "✓" if accuracy["train_accuracy"] > 0.5 else "✗"

            # Show approach summary, not full code (save tokens)
            solution_summaries.append(
                f"{status} Solution {idx} (Interpretation: {interp.persona})\n"
                f"  Train Accuracy: {train_pct:.0f}% ({accuracy['train_correct']}/{accuracy['train_total']})\n"
                f"  Approach: {solution.approach_summary}\n"
                f"  Original Pattern: {interp.pattern}\n"
            )

        solutions_summary = "\n".join(solution_summaries)

        # Create prompt
        prompt = f"""Analyze 5 previous solutions and create a DIVERSE 6th solution.

TRAINING EXAMPLES:
{training_examples}

PREVIOUS 5 SOLUTIONS (with accuracy on training examples):
{solutions_summary}

YOUR TASK:
You are a meta-learning expert analyzing these 5 attempts to solve the ARC task.

1. ANALYZE successful patterns:
   - What approaches worked well (high train accuracy)?
   - What operations/transformations led to correct outputs?
   - Extract 1-3 key successful patterns (≤80 chars each)

2. ANALYZE failed patterns:
   - What approaches failed (low train accuracy)?
   - What anti-patterns should be avoided?
   - Extract 1-3 key failed patterns (≤80 chars each)

3. SYNTHESIZE a 6th solution:
   - Learn from successful patterns
   - Avoid failed patterns
   - Use a DIFFERENT algorithm than all 5 previous attempts
   - Must be diverse in approach, not just different implementation

IMPORTANT CONSTRAINTS:
- The 6th solution MUST use a different algorithm/approach than all 5 previous
- Include "import numpy as np" at the top if using numpy
- Must follow signature: def solve(task_grid: np.ndarray) -> np.ndarray
- Use ONLY numpy for array operations (no other libraries)
- Code must be complete, runnable, and self-contained
- Provide clear diversity justification (≤100 chars) explaining why different
- Synthesis strategy should explain how you combined insights (≤150 chars)

OUTPUT FORMAT:
Provide analysis and synthesis code in structured JSON format with:
- analysis.successful_patterns: List of 1-3 strings (each ≤85 chars)
- analysis.failed_patterns: List of 1-3 strings (each ≤85 chars)
- analysis.synthesis_strategy: String (≤250 chars max) explaining approach
- code: Complete Python code string with solve() function
- diversity_justification: String (≤200 chars max) why different from all 5

CONCISENESS EXAMPLES (follow these patterns):
Good synthesis_strategy (142 chars):
  "Combine border detection with flood fill from solutions 1 & 3, add rotation from 2, use different algorithm than all 5 (graph traversal)"

Bad synthesis_strategy (too verbose):
  "First we need to carefully identify the border regions of the input grid by examining each cell and determining if it's on the edge, then we should apply a flood fill algorithm similar to what was attempted in solution 1 but modified to work better..."

Good diversity_justification (85 chars):
  "Uses graph traversal instead of direct indexing; all 5 used array manipulation"

Bad diversity_justification (too verbose):
  "This solution is different from all the previous five solutions because it takes a completely different algorithmic approach by using graph traversal..."
"""

        return prompt

    def _validate_solution(self, code: str) -> tuple[bool, str]:
        """Validate that solution code is syntactically correct and has solve().

        Args:
            code: Python code string to validate

        Returns:
            Tuple of (is_valid, error_message)
            - If valid: (True, "")
            - If invalid: (False, "error description")
        """
        # Check for solve() function
        if "def solve(" not in code:
            return False, "Missing 'def solve(' function"

        # Check for numpy usage without import
        if "np." in code and "import numpy" not in code:
            return False, "Uses numpy (np.) without importing"

        # Check syntax
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        return True, ""

    def synthesize_solution(
        self,
        task_json: dict,
        solutions: list[SolutionResult],
        interpretations: list[InterpretationResult],
    ) -> SynthesisResult:
        """Generate diverse 6th solution through meta-learning.

        Args:
            task_json: ARC task dictionary with 'train' examples
            solutions: List of 5 SolutionResult objects from Programmer
            interpretations: List of 5 InterpretationResult objects from Analyst

        Returns:
            SynthesisResult with code and meta-learning analysis

        Raises:
            ValueError: If validation fails or wrong number of solutions/interpretations
        """
        if len(solutions) != 5:
            raise ValueError(f"Expected 5 solutions, got {len(solutions)}")

        if len(interpretations) != 5:
            raise ValueError(f"Expected 5 interpretations, got {len(interpretations)}")

        # Calculate accuracies for all 5 solutions
        logger.info("Calculating accuracies for 5 solutions...")
        accuracies = self._calculate_solution_accuracies(task_json, solutions)

        # Create prompt
        prompt = self._create_prompt(task_json, solutions, accuracies, interpretations)

        # Configure generation with structured output
        generation_config = GenerationConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=SYNTHESIS_SCHEMA,
        )

        # Check cache if enabled
        if self.use_cache:
            from ..utils.llm_cache import get_cache

            cache = get_cache()
            cached_response = cache.get(prompt, self.model_name, self.temperature)
            if cached_response is not None:
                # Parse cached response using Pydantic
                parsed_response = SynthesisResponse.model_validate_json(cached_response)
                return self._parse_response(parsed_response)

        # Call Gemini API
        logger.info("Calling Gemini API for synthesis...")
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt, generation_config=generation_config)

        # Store in cache if enabled
        if self.use_cache:
            from ..utils.llm_cache import get_cache

            cache = get_cache()
            cache.set(prompt, response.text, self.model_name, self.temperature)

        # Parse response using Pydantic (guaranteed valid JSON by schema)
        parsed_response = SynthesisResponse.model_validate_json(response.text)
        return self._parse_response(parsed_response)

    def _parse_response(self, result: SynthesisResponse) -> SynthesisResult:
        """Parse Pydantic response into SynthesisResult object.

        Args:
            result: SynthesisResponse Pydantic model

        Returns:
            SynthesisResult object

        Raises:
            ValueError: If code validation fails
        """
        # Extract fields from Pydantic model
        analysis = result.analysis
        code = result.code
        diversity_justification = result.diversity_justification

        # Validate code
        is_valid, error_msg = self._validate_solution(code)
        if not is_valid:
            raise ValueError(f"Synthesis solution validation failed: {error_msg}")

        # Create result object
        synthesis = SynthesisResult(
            code=code,
            approach_summary=analysis.synthesis_strategy[
                :100
            ],  # Use strategy as summary
            successful_patterns=analysis.successful_patterns,
            failed_patterns=analysis.failed_patterns,
            synthesis_strategy=analysis.synthesis_strategy,
            diversity_justification=diversity_justification,
        )

        logger.info(
            f"Synthesis complete: {len(synthesis.successful_patterns)} successful patterns, "
            f"{len(synthesis.failed_patterns)} failed patterns identified"
        )

        return synthesis
