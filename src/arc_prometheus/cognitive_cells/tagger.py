"""
Tagger Cognitive Cell - Technique Classification Agent

The Tagger agent analyzes successful solver code to identify techniques used
(rotation, fill, symmetry, etc.). This enables technique-based crossover in
Phase 3.4 where solvers with complementary techniques can be fused.

Key Features:
- Hybrid analysis: Static pattern matching + LLM semantic understanding
- Technique taxonomy: 12 predefined ARC-relevant techniques
- Structured output: Tags, confidence, and optional technique details
- Caching support: Reduces API costs for repeated code analysis
"""

import re
from dataclasses import dataclass, field

import google.generativeai as genai

from ..utils.config import get_gemini_api_key

# =============================================================================
# Technique Taxonomy
# =============================================================================

TECHNIQUE_TAXONOMY = [
    "rotation",  # Grid rotation (np.rot90)
    "flip",  # Grid flip/mirror (np.flip)
    "transpose",  # Matrix transpose (np.transpose, .T)
    "color_fill",  # Fill regions with colors
    "pattern_copy",  # Copy/replicate patterns
    "symmetry",  # Detect or create symmetry
    "grid_partition",  # Split grid into regions
    "object_detection",  # Identify distinct objects
    "counting",  # Count elements/objects (len, count)
    "conditional_logic",  # If/else branching logic
    "array_manipulation",  # General array operations
    "neighborhood_analysis",  # Analyze adjacent cells
]


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TaggingResult:
    """
    Result of technique tagging analysis.

    Attributes:
        tags: List of identified technique names (from TECHNIQUE_TAXONOMY)
        confidence: Overall confidence in tagging (high/medium/low)
        technique_details: Optional details about each technique usage
    """

    tags: list[str]
    confidence: str = "medium"
    technique_details: dict[str, str] = field(default_factory=dict)


# =============================================================================
# Tagger Agent
# =============================================================================


class Tagger:
    """
    Tagger agent classifies solver techniques for crossover selection.

    Combines static code analysis (pattern matching) with LLM semantic
    understanding to identify techniques used in ARC solver code.

    Example:
        >>> tagger = Tagger()
        >>> code = "def solve(grid): return np.rot90(grid)"
        >>> task_json = {"train": [...]}
        >>> result = tagger.tag_solver(code, task_json)
        >>> print(result.tags)  # ['rotation']
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.4,
        use_cache: bool = True,
    ):
        """
        Initialize Tagger agent.

        Args:
            model_name: Gemini model to use for semantic analysis
            temperature: LLM temperature (0.4 balances precision and coverage)
            use_cache: Whether to use LLM response caching
        """
        self.model_name = model_name
        self.temperature = temperature
        self.use_cache = use_cache

    def tag_solver(self, solver_code: str, task_json: dict) -> TaggingResult:
        """
        Analyze solver code and identify techniques used.

        Combines static pattern matching with LLM semantic analysis for
        comprehensive technique detection. Deduplicates results and filters
        against the technique taxonomy.

        Args:
            solver_code: Python solver code to analyze
            task_json: ARC task dict (provides context for LLM)

        Returns:
            TaggingResult with identified techniques, confidence, and details
        """
        # 1. Static code analysis
        static_tags = self._static_analysis(solver_code)

        # 2. LLM-based semantic analysis
        llm_tags = self._llm_analysis(solver_code, task_json)

        # 3. Combine and deduplicate
        all_tags = list(set(static_tags + llm_tags))

        # 4. Determine confidence
        confidence = self._assess_confidence(static_tags, llm_tags)

        return TaggingResult(tags=all_tags, confidence=confidence)

    def _static_analysis(self, code: str) -> list[str]:
        """
        Pattern matching for obvious techniques.

        Uses regex to detect technique-specific keywords and function calls
        in the solver code.

        Args:
            code: Solver code to analyze

        Returns:
            List of detected technique tags
        """
        tags = []

        # Rotation patterns
        if re.search(r"np\.rot90|\.rotate\(|rotate_", code, re.IGNORECASE):
            tags.append("rotation")

        # Flip patterns
        if re.search(r"np\.flip|\.flip\(|flip_", code, re.IGNORECASE):
            tags.append("flip")

        # Transpose patterns
        if re.search(r"np\.transpose|\.T\b", code):
            tags.append("transpose")

        # Color fill patterns
        if re.search(r"fill|flood", code, re.IGNORECASE):
            tags.append("color_fill")

        # Pattern copy patterns
        if re.search(r"\.copy\(|\.tile\(|np\.tile|repeat", code, re.IGNORECASE):
            tags.append("pattern_copy")

        # Symmetry patterns
        if re.search(r"symmetr|mirror", code, re.IGNORECASE):
            tags.append("symmetry")

        # Grid partition patterns
        if re.search(r"split|partition|chunk|slice", code, re.IGNORECASE):
            tags.append("grid_partition")

        # Object detection patterns
        if re.search(r"object|detect|find|locate|label", code, re.IGNORECASE):
            tags.append("object_detection")

        # Counting patterns
        if re.search(r"\blen\(|\.count\(|np\.count|np\.sum|\.sum\(", code):
            tags.append("counting")

        # Conditional logic patterns
        if re.search(r"\bif\b|\belse\b|\belif\b|np\.where|np\.any|np\.all", code):
            tags.append("conditional_logic")

        # Array manipulation patterns (broad category)
        if re.search(r"np\.|array|reshape|flatten|ravel", code, re.IGNORECASE):
            tags.append("array_manipulation")

        # Neighborhood analysis patterns
        if re.search(
            r"neighbor|adjacent|surrounding|convolve|kernel", code, re.IGNORECASE
        ):
            tags.append("neighborhood_analysis")

        return tags

    def _llm_analysis(self, code: str, task_json: dict) -> list[str]:
        """
        LLM-based semantic technique identification.

        Uses Gemini to identify techniques that static analysis might miss,
        such as implicit patterns or complex logic.

        Args:
            code: Solver code to analyze
            task_json: ARC task for context

        Returns:
            List of technique tags from LLM analysis
        """
        try:
            # Create tagging prompt
            prompt = self._create_tagging_prompt(code, task_json)

            # Check cache first
            if self.use_cache:
                from ..utils.llm_cache import get_cache

                cache = get_cache()
                cached_response = cache.get(prompt, self.model_name, self.temperature)
                if cached_response:
                    return self._parse_llm_response(cached_response)

            # Call LLM
            api_key = get_gemini_api_key()
            genai.configure(api_key=api_key)

            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": 512,  # Techniques list is short
                },
            )

            response = model.generate_content(prompt, request_options={"timeout": 30})

            # Cache the response
            if self.use_cache:
                cache.set(prompt, response.text, self.model_name, self.temperature)

            return self._parse_llm_response(response.text)

        except Exception as e:
            # Graceful degradation: return empty list on error
            print(f"LLM analysis error: {e}")
            return []

    def _create_tagging_prompt(self, code: str, task_json: dict) -> str:
        """
        Create prompt for LLM technique tagging.

        Args:
            code: Solver code to analyze
            task_json: ARC task for context

        Returns:
            Formatted prompt string
        """
        # Format task examples for context (limit to first 2 train examples)
        train_examples = task_json.get("train", [])[:2]
        task_context = ""
        for i, example in enumerate(train_examples):
            task_context += f"\nExample {i + 1}:\n"
            task_context += f"  Input shape: {len(example['input'])}x{len(example['input'][0]) if example['input'] else 0}\n"
            task_context += f"  Output shape: {len(example['output'])}x{len(example['output'][0]) if example['output'] else 0}\n"

        prompt = f"""Analyze this ARC solver code and identify the techniques used.

Available techniques (select from this list only):
{", ".join(TECHNIQUE_TAXONOMY)}

Solver code:
```python
{code}
```

Task context:{task_context}

Identify which techniques from the list above are used in this solver.
Focus on the core transformation logic, not just array operations.

Output format:
TECHNIQUES: technique1, technique2, technique3
CONFIDENCE: high/medium/low

Only list techniques you are confident are actually used.
"""  # noqa: S608 # nosec B608  # This is not SQL, it's an LLM prompt
        return prompt

    def _parse_llm_response(self, response_text: str) -> list[str]:
        """
        Parse LLM response to extract technique tags.

        Args:
            response_text: Raw LLM response

        Returns:
            List of valid technique tags
        """
        tags = []

        # Extract TECHNIQUES line
        match = re.search(r"TECHNIQUES:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE)
        if match:
            techniques_str = match.group(1).strip()
            # Split by comma and clean
            raw_tags = [t.strip() for t in techniques_str.split(",")]

            # Filter against taxonomy (only valid techniques)
            for tag in raw_tags:
                if tag.lower() in [t.lower() for t in TECHNIQUE_TAXONOMY]:
                    # Use canonical form from taxonomy
                    canonical_tag = next(
                        t for t in TECHNIQUE_TAXONOMY if t.lower() == tag.lower()
                    )
                    tags.append(canonical_tag)

        return tags

    def _assess_confidence(self, static_tags: list[str], llm_tags: list[str]) -> str:
        """
        Assess overall confidence in tagging results.

        Args:
            static_tags: Tags from static analysis
            llm_tags: Tags from LLM analysis

        Returns:
            Confidence level: "high", "medium", or "low"
        """
        # High confidence: Both analyses agree (overlap)
        overlap = len(set(static_tags) & set(llm_tags))
        total_tags = len(set(static_tags + llm_tags))

        if total_tags == 0:
            return "low"  # No techniques found

        if overlap >= 2 or (overlap >= 1 and total_tags <= 2):
            return "high"  # Strong agreement

        if total_tags >= 3 and overlap >= 1:
            return "medium"  # Some agreement

        return "medium"  # Default to medium for safety
