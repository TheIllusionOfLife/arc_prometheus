"""Multi-Solution Programmer - Generates 5 diverse solver implementations.

This agent takes 5 expert interpretations and generates 5 complete solve()
function implementations in a single API call using Gemini structured output.

Key features:
- Single API call generates all 5 solutions
- Temperature 0.7 for moderate diversity
- Structured JSON output with validation
- Graceful handling of partial failures
- Links each solution to its interpretation via ID
"""

import logging
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from arc_prometheus.cognitive_cells.multi_persona_analyst import InterpretationResult
from arc_prometheus.utils.config import get_gemini_api_key
from arc_prometheus.utils.schemas import MULTI_SOLUTION_SCHEMA, MultiSolutionResponse

logger = logging.getLogger(__name__)


@dataclass
class SolutionResult:
    """Single solver implementation result.

    Attributes:
        interpretation_id: Which interpretation this implements (1-5)
        code: Complete Python code with solve() function
        approach_summary: Brief description of implementation (≤100 chars)
    """

    interpretation_id: int
    code: str
    approach_summary: str


class MultiSolutionProgrammer:
    """Generates 5 diverse solver implementations from 5 interpretations.

    This agent uses a single API call to generate 5 complete solve() function
    implementations, each corresponding to one of the expert interpretations
    from the Multi-Persona Analyst.

    Example:
        ```python
        analyst = MultiPersonaAnalyst()
        programmer = MultiSolutionProgrammer()

        task = load_task("data/arc-prize-2025/training/00576224.json")
        interpretations = analyst.analyze_task(task)
        solutions = programmer.generate_multi_solutions(task, interpretations)

        # Returns 3-5 SolutionResult objects (validated)
        for solution in solutions:
            print(f"Solution {solution.interpretation_id}: {solution.approach_summary}")
        ```
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-thinking-exp",
        temperature: float = 0.7,
        use_cache: bool = True,
    ):
        """Initialize Multi-Solution Programmer.

        Args:
            model_name: Gemini model to use (default: gemini-2.0-flash-thinking-exp)
            temperature: LLM temperature (default 0.7 for moderate diversity)
            use_cache: Whether to use LLM response caching (default True)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.use_cache = use_cache

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

    def _create_prompt(
        self, task_json: dict, interpretations: list[InterpretationResult]
    ) -> str:
        """Create multi-solution programming prompt.

        Args:
            task_json: ARC task dictionary with 'train' examples
            interpretations: List of 5 InterpretationResult objects from Analyst

        Returns:
            Prompt string for LLM with 5 interpretation implementations
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

        # Format interpretations
        interpretation_instructions = []
        for idx, interp in enumerate(interpretations, 1):
            interpretation_instructions.append(
                f"Interpretation {idx} - {interp.persona}:\n"
                f"  Pattern: {interp.pattern}\n"
                f"  Approach: {interp.approach}\n"
                f"  Confidence: {interp.confidence}"
            )

        interpretations_text = "\n\n".join(interpretation_instructions)

        # Create prompt
        prompt = f"""Implement ALL 5 expert interpretations as complete Python solve() functions.

TRAINING EXAMPLES:
{training_examples}

EXPERT INTERPRETATIONS TO IMPLEMENT:
{interpretations_text}

YOUR TASK:
Implement each of the 5 interpretations above as a separate, complete solve() function.
Each implementation must:
1. Be a complete, runnable Python function
2. Include necessary imports (especially numpy)
3. Match the signature: def solve(task_grid: np.ndarray) -> np.ndarray
4. Use ONLY numpy for array operations (no other libraries)
5. Follow the specific approach suggested by that interpretation
6. Be concise but complete

IMPORTANT CONSTRAINTS:
- Each solution's code must be self-contained and executable
- Include "import numpy as np" at the top if using numpy
- All 5 implementations must be DIFFERENT (different approaches/logic)
- Each approach_summary must be ≤100 characters
- Link each solution to its interpretation_id (1-5)

OUTPUT FORMAT:
Provide a JSON array with 5 solutions, each containing:
- interpretation_id: Integer 1-5 (which interpretation it implements)
- code: Complete Python code string with solve() function
- approach_summary: Brief description (≤100 chars)
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

    def generate_multi_solutions(
        self, task_json: dict, interpretations: list[InterpretationResult]
    ) -> list[SolutionResult]:
        """Generate 5 solver implementations from 5 interpretations.

        Args:
            task_json: ARC task dictionary with 'train' examples
            interpretations: List of 5 InterpretationResult objects

        Returns:
            List of 3-5 SolutionResult objects (only validated solutions)

        Raises:
            ValueError: If all 5 solutions fail validation or JSON schema violation
        """
        if len(interpretations) != 5:
            raise ValueError(f"Expected 5 interpretations, got {len(interpretations)}")

        # Create prompt
        prompt = self._create_prompt(task_json, interpretations)

        # Configure generation with structured output
        generation_config = GenerationConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=MULTI_SOLUTION_SCHEMA,
        )

        # Check cache if enabled
        if self.use_cache:
            from ..utils.llm_cache import get_cache

            cache = get_cache()
            cached_response = cache.get(prompt, self.model_name, self.temperature)
            if cached_response is not None:
                # Parse cached response using Pydantic
                parsed_response = MultiSolutionResponse.model_validate_json(
                    cached_response
                )
                return self._parse_and_validate_response(parsed_response)

        # Call Gemini API
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt, generation_config=generation_config)

        # Store in cache if enabled
        if self.use_cache:
            from ..utils.llm_cache import get_cache

            cache = get_cache()
            cache.set(prompt, response.text, self.model_name, self.temperature)

        # Parse response using Pydantic (guaranteed valid JSON by schema)
        parsed_response = MultiSolutionResponse.model_validate_json(response.text)
        return self._parse_and_validate_response(parsed_response)

    def _parse_and_validate_response(
        self, result: MultiSolutionResponse
    ) -> list[SolutionResult]:
        """Parse Pydantic response and validate each solution.

        Args:
            result: MultiSolutionResponse Pydantic model

        Returns:
            List of validated SolutionResult objects (3-5 solutions)

        Raises:
            ValueError: If all 5 solutions invalid or schema violation
        """
        solutions_data = result.solutions

        if len(solutions_data) != 5:
            raise ValueError(
                f"Expected 5 solutions in response, got {len(solutions_data)}"
            )

        # Parse and validate each solution
        valid_solutions = []
        invalid_count = 0
        error_messages = []

        for data in solutions_data:
            interp_id = data.interpretation_id
            code = data.code
            approach = data.approach_summary

            # Validate solution
            is_valid, error_msg = self._validate_solution(code)

            if is_valid:
                solution = SolutionResult(
                    interpretation_id=interp_id,
                    code=code,
                    approach_summary=approach,
                )
                valid_solutions.append(solution)
            else:
                invalid_count += 1
                error_messages.append(f"Solution {interp_id}: {error_msg}")
                logger.warning(
                    f"Solution {interp_id} failed validation: {error_msg}. Skipping."
                )

        # Check if we have at least some valid solutions
        if len(valid_solutions) == 0:
            # Include first 3 error messages for debugging
            error_summary = "; ".join(error_messages[:3])
            raise ValueError(
                f"All 5 solutions failed validation. Cannot proceed without any valid code. "
                f"Errors: {error_summary}"
            )

        if invalid_count > 0:
            logger.info(
                f"Returning {len(valid_solutions)}/5 valid solutions "
                f"({invalid_count} failed validation)"
            )

        return valid_solutions
