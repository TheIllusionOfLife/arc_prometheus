"""Analyst Agent - Pattern analysis and rule inference for ARC tasks.

The Analyst agent separates pattern understanding from code generation:
1. Analyzes input-output examples from ARC tasks
2. Infers abstract transformation rules in natural language
3. Identifies key observations (colors, shapes, positions, symmetries)
4. Suggests high-level implementation approaches
5. Assesses confidence in the analysis

This enables the Programmer agent to generate code from clear specifications
rather than raw examples, improving abstraction and reasoning.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import google.generativeai as genai

from arc_prometheus.utils.config import get_gemini_api_key


@dataclass
class AnalysisResult:
    """Structured result from Analyst's pattern analysis.

    Attributes:
        pattern_description: One-sentence natural language description of the transformation rule
        key_observations: List of important features noticed across examples
        suggested_approach: High-level implementation strategy (e.g., "use rotation", "fill pattern")
        confidence: How certain the analysis is ("high", "medium", "low")
    """

    pattern_description: str
    key_observations: list[str] = field(default_factory=list)
    suggested_approach: str = ""
    confidence: str = ""


def parse_analysis_response(response_text: str) -> AnalysisResult:
    """Parse LLM response into structured AnalysisResult.

    Expected format:
        PATTERN: [one sentence description]

        OBSERVATIONS:
        - [observation 1]
        - [observation 2]
        ...

        APPROACH: [implementation strategy]

        CONFIDENCE: [high/medium/low]

    Args:
        response_text: Raw text from LLM analysis

    Returns:
        AnalysisResult with parsed fields

    Note:
        - Handles case-insensitive keywords
        - Gracefully handles missing sections with empty defaults
        - Strips extra whitespace
        - Handles multi-line sections
    """
    # Normalize to uppercase for consistent parsing
    text = response_text.strip()

    # Extract pattern (everything after PATTERN: until next section or end)
    pattern_match = re.search(
        r"PATTERN:\s*(.+?)(?=\n\s*OBSERVATIONS:|\n\s*APPROACH:|\n\s*CONFIDENCE:|$)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    pattern = pattern_match.group(1).strip() if pattern_match else ""

    # Extract observations (bullet points after OBSERVATIONS:)
    observations = []
    obs_match = re.search(
        r"OBSERVATIONS:\s*(.+?)(?=\n\s*APPROACH:|\n\s*CONFIDENCE:|$)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if obs_match:
        obs_text = obs_match.group(1)
        # Find lines starting with - or bullet points
        for line in obs_text.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                # Remove the leading dash and whitespace
                observation = line[1:].strip()
                if observation:
                    observations.append(observation)

    # Extract approach (everything after APPROACH: until next section or end)
    approach_match = re.search(
        r"APPROACH:\s*(.+?)(?=\n\s*CONFIDENCE:|$)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    approach = approach_match.group(1).strip() if approach_match else ""

    # Extract confidence (should be single word: high/medium/low)
    confidence_match = re.search(
        r"CONFIDENCE:\s*(\w+)",
        text,
        re.IGNORECASE
    )
    confidence = confidence_match.group(1).lower() if confidence_match else ""

    return AnalysisResult(
        pattern_description=pattern,
        key_observations=observations,
        suggested_approach=approach,
        confidence=confidence
    )


class Analyst:
    """Analyzes ARC tasks to infer transformation patterns.

    The Analyst is a specialized LLM agent that examines input-output examples
    and produces natural language descriptions of the transformation rules.
    This abstracts pattern recognition from code generation.

    Example:
        ```python
        analyst = Analyst()
        task = load_task("data/arc-prize-2025/training/00576224.json")
        analysis = analyst.analyze_task(task)

        print(analysis.pattern_description)
        # "Fill entire grid with the single non-zero color"

        print(analysis.key_observations)
        # ["Input has one non-zero cell", "Output fills entire grid", ...]

        print(analysis.confidence)
        # "high"
        ```
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.3,
        use_cache: bool = True,
    ):
        """Initialize Analyst agent.

        Args:
            model_name: Gemini model to use for analysis
            temperature: LLM temperature (lower = more focused, default 0.3 for analysis)
            use_cache: Whether to use LLM response caching (default True)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.use_cache = use_cache

        # Configure Gemini API
        api_key = get_gemini_api_key()
        genai.configure(api_key=api_key)

    def _create_analysis_prompt(self, task_json: dict) -> str:
        """Create analysis prompt from ARC task.

        Args:
            task_json: ARC task dictionary with 'train' examples

        Returns:
            Prompt string for LLM analysis
        """
        # Format training examples as text
        train_examples = task_json["train"]
        examples_text = []

        for idx, example in enumerate(train_examples):
            input_grid = example["input"]
            output_grid = example["output"]

            # Format grids as readable text
            input_str = self._format_grid(input_grid)
            output_str = self._format_grid(output_grid)

            examples_text.append(
                f"Example {idx + 1}:\n"
                f"Input ({len(input_grid)}x{len(input_grid[0])}):\n{input_str}\n\n"
                f"Output ({len(output_grid)}x{len(output_grid[0])}):\n{output_str}\n"
            )

        training_examples = "\n".join(examples_text)

        # Create analysis prompt
        prompt = f"""Analyze this ARC (Abstraction and Reasoning Corpus) task and infer the transformation rule.

You will see {len(train_examples)} training example(s). Your task is to identify the pattern that transforms inputs into outputs.

{training_examples}

Your task:
1. Observe patterns across all training examples
2. Identify the core transformation rule that applies to ALL examples
3. Describe the rule in one clear sentence
4. Note key features you observe (colors, shapes, positions, symmetries, patterns)
5. Suggest a high-level implementation approach using numpy operations

Output format (REQUIRED - use these exact keywords):

PATTERN: [One clear sentence describing the transformation rule]

OBSERVATIONS:
- [Key observation 1]
- [Key observation 2]
- [Key observation 3]
...

APPROACH: [High-level implementation strategy using numpy operations]

CONFIDENCE: [high/medium/low - how confident are you in this analysis]

Important:
- Focus on the ABSTRACT RULE, not implementation details
- The rule must work for ALL training examples
- Be specific about what changes (colors, positions, shapes, sizes)
- Suggest numpy operations where applicable (np.rot90, np.flip, np.tile, etc.)
- Assess confidence based on pattern clarity and consistency across examples
"""

        return prompt

    def _format_grid(self, grid: list) -> str:
        """Format grid as readable text representation.

        Args:
            grid: 2D list of integers

        Returns:
            String representation of grid
        """
        rows = []
        for row in grid:
            # Format each row as space-separated integers
            rows.append(" ".join(str(cell) for cell in row))
        return "\n".join(rows)

    def analyze_task(self, task_json: dict) -> AnalysisResult:
        """Analyze ARC task and infer transformation pattern.

        Args:
            task_json: ARC task dictionary with 'train' examples

        Returns:
            AnalysisResult with pattern description, observations, approach, and confidence

        Raises:
            TimeoutError: If API call times out
            ValueError: If API response is invalid
        """
        # Create analysis prompt
        prompt = self._create_analysis_prompt(task_json)

        # Configure generation
        # Type as Any to satisfy mypy while maintaining runtime correctness
        generation_config: Any = {
            "temperature": self.temperature,
            "max_output_tokens": 2048,  # Allow detailed analysis
        }

        # Call Gemini API
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Parse response
        analysis = parse_analysis_response(response.text)

        return analysis
