"""Multi-Persona Analyst - Generates 5 diverse interpretations per task.

This agent uses a single API call to generate 5 different expert perspectives
on an ARC task, promoting diversity and increasing generalization capability.

Key features:
- 5 expert personas with different specializations
- High temperature (1.0) for maximum diversity
- Structured JSON output with concise constraints
- Single API call for efficiency
"""

import logging
from dataclasses import dataclass, field

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from pydantic import ValidationError

from arc_prometheus.utils.config import get_gemini_api_key
from arc_prometheus.utils.schemas import MULTI_PERSONA_SCHEMA, MultiPersonaResponse

logger = logging.getLogger(__name__)


@dataclass
class InterpretationResult:
    """Single expert interpretation of an ARC task.

    Attributes:
        persona: Expert perspective name (e.g., "Geometric Transformation Specialist")
        pattern: One-sentence transformation rule description
        observations: List of 1-3 key insights (each ≤80 chars)
        approach: High-level implementation strategy (≤100 chars)
        confidence: How confident this interpretation is ("high", "medium", "low")
    """

    persona: str
    pattern: str
    observations: list[str] = field(default_factory=list)
    approach: str = ""
    confidence: str = ""


# Default 5-persona set (from plan appendix)
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
    "persona_5": {
        "name": "Logical Rules Specialist",
        "emoji": "⚡",
        "focus": "Conditional logic, counting, neighborhood analysis, cellular automata, pattern matching",
        "key_question": "Are there if-then rules based on positions, counts, or neighbors?",
    },
}


class MultiPersonaAnalyst:
    """Generates 5 diverse expert interpretations of ARC tasks.

    This agent uses multiple expert personas to analyze the same task from
    different perspectives, increasing the likelihood that at least one
    interpretation captures the correct transformation rule.

    Example:
        ```python
        analyst = MultiPersonaAnalyst()
        task = load_task("data/arc-prize-2025/training/00576224.json")
        interpretations = analyst.analyze_task(task)

        # Returns 5 InterpretationResult objects
        for interp in interpretations:
            print(f"{interp.persona}: {interp.pattern}")
            print(f"  Confidence: {interp.confidence}")
        ```
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-thinking-exp",
        temperature: float = 1.0,
        personas: dict | None = None,
        use_cache: bool = True,
    ):
        """Initialize Multi-Persona Analyst.

        Args:
            model_name: Gemini model to use (default: gemini-2.0-flash-thinking-exp)
            temperature: LLM temperature (default 1.0 for high diversity)
            personas: Custom persona definitions (default: DEFAULT_PERSONAS)
            use_cache: Whether to use LLM response caching (default True)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.personas = personas if personas is not None else DEFAULT_PERSONAS
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

    def _create_prompt(self, task_json: dict) -> str:
        """Create multi-persona analysis prompt.

        Args:
            task_json: ARC task dictionary with 'train' examples

        Returns:
            Prompt string for LLM with 5 persona instructions
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

        # Format persona instructions
        persona_instructions = []
        for _persona_id, persona_data in self.personas.items():
            persona_instructions.append(
                f"{persona_data['emoji']} {persona_data['name']}:\n"
                f"  Focus: {persona_data['focus']}\n"
                f"  Key Question: {persona_data['key_question']}"
            )

        personas_text = "\n\n".join(persona_instructions)

        # Create prompt
        prompt = f"""Analyze this ARC task from 5 DIFFERENT expert perspectives.

TRAINING EXAMPLES:
{training_examples}

YOUR TASK:
You are a team of 5 experts analyzing this puzzle. Each expert has a different specialization
and must provide their unique interpretation. Diversity is critical - all 5 perspectives
must be DIFFERENT.

THE 5 EXPERTS:
{personas_text}

INSTRUCTIONS:
1. Each expert analyzes the task from their specialized perspective
2. Each expert provides:
   - Pattern: One-sentence transformation rule (≤150 chars, be concise)
   - Observations: 1-3 key insights specific to their expertise (each ≤80 chars)
   - Approach: High-level implementation strategy (≤200 chars max, mention numpy operations)
   - Confidence: "high", "medium", or "low"
3. All 5 interpretations MUST be different - no duplicate patterns
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
- Each expert's pattern must be unique from the others
"""

        return prompt

    def analyze_task(self, task_json: dict) -> list[InterpretationResult]:
        """Analyze ARC task and return 5 diverse interpretations.

        Args:
            task_json: ARC task dictionary with 'train' examples

        Returns:
            List of 5 InterpretationResult objects (one per expert persona)

        Raises:
            ValueError: If API response is invalid or doesn't contain 5 interpretations
            TimeoutError: If API call times out
        """
        # Create prompt
        prompt = self._create_prompt(task_json)

        # Configure generation with structured output
        generation_config = GenerationConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=MULTI_PERSONA_SCHEMA,
        )

        # Check cache if enabled
        if self.use_cache:
            from ..utils.llm_cache import get_cache

            cache = get_cache()
            cached_response = cache.get(prompt, self.model_name, self.temperature)
            if cached_response is not None:
                try:
                    # Parse cached response using Pydantic
                    parsed_response = MultiPersonaResponse.model_validate_json(
                        cached_response
                    )
                    return self._parse_response(parsed_response)
                except ValidationError as e:
                    # Invalid cache entry (e.g., stale from schema migration)
                    logger.warning(
                        f"Invalid cache entry for MultiPersonaResponse, "
                        f"regenerating fresh response. Validation error: {e}"
                    )
                    # Fall through to regenerate fresh response
                    # (invalid entry will be overwritten when new response is cached)

        # Call Gemini API
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt, generation_config=generation_config)

        # Store in cache if enabled
        if self.use_cache:
            from ..utils.llm_cache import get_cache

            cache = get_cache()
            cache.set(prompt, response.text, self.model_name, self.temperature)

        # Parse response using Pydantic (guaranteed valid JSON by schema)
        parsed_response = MultiPersonaResponse.model_validate_json(response.text)
        return self._parse_response(parsed_response)

    def _parse_response(
        self, result: MultiPersonaResponse
    ) -> list[InterpretationResult]:
        """Parse Pydantic response into InterpretationResult objects.

        Args:
            result: MultiPersonaResponse Pydantic model

        Returns:
            List of 5 InterpretationResult objects

        Note:
            Length validation (exactly 5 items) is enforced by Pydantic model.
            No need for explicit check here - ValidationError raised before this method.
        """
        interpretations_data = result.interpretations

        interpretations = []
        for data in interpretations_data:
            interp = InterpretationResult(
                persona=data.persona,
                pattern=data.pattern,
                observations=data.observations,
                approach=data.approach,
                confidence=data.confidence,
            )
            interpretations.append(interp)

        return interpretations
