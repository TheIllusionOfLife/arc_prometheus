"""
Crossover Agent - Phase 3.4

LLM-based technique fusion for population-based evolution.
Combines successful solvers with complementary techniques.

Key Features:
- LLM-based compatibility assessment
- Technique fusion from 2+ parent solvers
- Semantic understanding of code complementarity
- Cached responses for efficiency
- Optional analyst specification integration
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import google.generativeai as genai

from ..utils.config import get_gemini_api_key

if TYPE_CHECKING:
    from ..evolutionary_engine.solver_library import SolverRecord


@dataclass
class CrossoverResult:
    """
    Result of crossover (technique fusion) operation.

    Attributes:
        fused_code: Python code combining parent techniques
        parent_ids: List of parent solver IDs
        parent_techniques: List of technique tags for each parent
        compatibility_assessment: LLM's reasoning about compatibility
        confidence: Overall confidence (high/medium/low)
    """

    fused_code: str
    parent_ids: list[str]
    parent_techniques: list[list[str]]
    compatibility_assessment: str
    confidence: str = "medium"


class Crossover:
    """
    Crossover agent for technique fusion.

    Combines solvers with complementary techniques using LLM-based
    semantic understanding. Enables genetic innovation beyond mutation.

    Example:
        >>> crossover = Crossover()
        >>> parents = library.get_diverse_solvers("task-001", num_solvers=2)
        >>> result = crossover.fuse_solvers(parents, task_json)
        >>> print(result.fused_code)  # Combined solver code
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.5,
        use_cache: bool = True,
    ):
        """
        Initialize Crossover agent.

        Args:
            model_name: Gemini model to use for fusion
            temperature: LLM temperature (0.5 balances creativity and precision)
            use_cache: Whether to use LLM response caching
        """
        self.model_name = model_name
        self.temperature = temperature
        self.use_cache = use_cache

        # Configure Gemini API once during initialization
        api_key = get_gemini_api_key()
        genai.configure(api_key=api_key)

    def fuse_solvers(
        self,
        parent_solvers: list[SolverRecord],
        task_json: dict,
        analyst_spec: Any = None,
    ) -> CrossoverResult:
        """
        Fuse parent solvers using LLM-based technique fusion.

        Args:
            parent_solvers: List of 2+ solver records to fuse
            task_json: ARC task dict (provides context for LLM)
            analyst_spec: Optional analyst specification for context

        Returns:
            CrossoverResult with fused code, parent info, and assessment

        Raises:
            ValueError: If fewer than 2 parent solvers provided
            Exception: If LLM API call fails
        """
        # 0. Validate minimum 2 parents
        if len(parent_solvers) < 2:
            raise ValueError(
                f"Crossover requires at least 2 parent solvers, got {len(parent_solvers)}"
            )

        # 1. Create fusion prompt
        prompt = self._create_fusion_prompt(parent_solvers, task_json, analyst_spec)

        # 2. Check cache first
        if self.use_cache:
            from ..utils.llm_cache import get_cache

            cache = get_cache()
            cached_response = cache.get(prompt, self.model_name, self.temperature)
            if cached_response:
                return self._parse_response(cached_response, parent_solvers)

        # 3. Call LLM (API already configured in __init__)
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": 2048,  # Fused code can be longer
            },
        )

        response = model.generate_content(prompt, request_options={"timeout": 60})

        # 4. Cache the response
        if self.use_cache:
            cache.set(prompt, response.text, self.model_name, self.temperature)

        # 5. Parse and return result
        return self._parse_response(response.text, parent_solvers)

    def _create_fusion_prompt(
        self,
        parent_solvers: list[SolverRecord],
        task_json: dict,
        analyst_spec: Any = None,
    ) -> str:
        """
        Create prompt for LLM technique fusion.

        Args:
            parent_solvers: List of parent solver records
            task_json: ARC task for context
            analyst_spec: Optional analyst specification

        Returns:
            Formatted prompt string
        """
        # Format task context (limit to first 2 train examples)
        train_examples = task_json.get("train", [])[:2]
        task_context = ""
        for i, example in enumerate(train_examples):
            input_grid = example["input"]
            output_grid = example["output"]

            # Robust grid shape calculation (handles empty grids/rows)
            input_rows = len(input_grid)
            input_cols = len(input_grid[0]) if input_grid and input_grid[0] else 0
            output_rows = len(output_grid)
            output_cols = len(output_grid[0]) if output_grid and output_grid[0] else 0

            task_context += f"\nExample {i + 1}:\n"
            task_context += f"  Input shape: {input_rows}x{input_cols}\n"
            task_context += f"  Output shape: {output_rows}x{output_cols}\n"

        # Format analyst specification if provided
        analyst_context = ""
        if analyst_spec is not None:
            analyst_context = f"""
ANALYST SPECIFICATION:
Pattern: {analyst_spec.pattern_description}
Key Observations: {", ".join(analyst_spec.key_observations)}
Suggested Approach: {analyst_spec.suggested_approach}
Confidence: {analyst_spec.confidence}
"""

        # Format parent solvers
        parent_sections = []
        for i, parent in enumerate(parent_solvers, 1):
            techniques_str = (
                ", ".join(parent.tags) if parent.tags else "none identified"
            )
            parent_section = f"""
PARENT SOLVER {i} (ID: {parent.solver_id}):
Fitness Score: {parent.fitness_score}
Techniques: {techniques_str}
Code:
```python
{parent.code_str}
```
"""
            parent_sections.append(parent_section)

        parent_text = "\n".join(parent_sections)

        # Construct full prompt
        prompt = f"""You are an expert at fusing ARC solver techniques to create innovative solutions.

TASK CONTEXT:{task_context}
{analyst_context}

PARENT SOLVERS:
{parent_text}

TASK: Fuse the best techniques from the parent solvers into a single, improved solver.

ANALYSIS:
1. Assess compatibility: Are these techniques complementary?
2. Identify strengths: What does each parent do well?
3. Design fusion: How can we combine techniques synergistically?

OUTPUT FORMAT:
COMPATIBILITY: <your compatibility assessment>
CONFIDENCE: <high/medium/low>

FUSED CODE:
```python
def solve(grid):
    # Your fused solver combining parent techniques
    ...
```

REQUIREMENTS:
- Must use the exact signature: def solve(grid: np.ndarray) -> np.ndarray
- Use only numpy for array operations
- Combine complementary techniques from parents
- Preserve the best aspects of each parent
- Output ONLY the solve() function (you may include helper functions if needed)

Begin your analysis and fusion:
"""  # noqa: S608 # nosec B608  # This is not SQL, it's an LLM prompt

        return prompt

    def _parse_fused_code(self, llm_response: str) -> str:
        """
        Parse fused code from LLM response.

        Extracts the solve() function from markdown blocks or raw code.

        Args:
            llm_response: Raw LLM response text

        Returns:
            Extracted Python code (solve function and helpers)
        """
        # Try to extract code from markdown block
        code_block_match = re.search(
            r"```python\s*(.*?)\s*```", llm_response, re.DOTALL | re.IGNORECASE
        )
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try to extract code from generic code block
        code_block_match = re.search(r"```\s*(.*?)\s*```", llm_response, re.DOTALL)
        if code_block_match:
            code = code_block_match.group(1).strip()
            # Remove language identifier if present
            if code.startswith("python\n"):
                code = code[7:]
            return code

        # If no markdown block, look for solve function directly
        solve_match = re.search(
            r"(def solve\([^)]*\):.*)", llm_response, re.DOTALL | re.IGNORECASE
        )
        if solve_match:
            return solve_match.group(1).strip()

        # Fallback: return entire response stripped
        return llm_response.strip()

    def _parse_assessment(self, llm_response: str) -> tuple[str, str]:
        """
        Parse compatibility assessment and confidence from LLM response.

        Args:
            llm_response: Raw LLM response text

        Returns:
            Tuple of (compatibility_assessment, confidence)
        """
        # Extract COMPATIBILITY line
        compatibility = "Compatible"  # Default
        compat_match = re.search(
            r"COMPATIBILITY:\s*(.+?)(?:\n|$)", llm_response, re.IGNORECASE
        )
        if compat_match:
            compatibility = compat_match.group(1).strip()

        # Extract CONFIDENCE line
        confidence = "medium"  # Default
        conf_match = re.search(
            r"CONFIDENCE:\s*(.+?)(?:\n|$)", llm_response, re.IGNORECASE
        )
        if conf_match:
            conf_str = conf_match.group(1).strip().lower()
            if "high" in conf_str:
                confidence = "high"
            elif "low" in conf_str:
                confidence = "low"
            else:
                confidence = "medium"

        return compatibility, confidence

    def _parse_response(
        self, llm_response: str, parent_solvers: list[SolverRecord]
    ) -> CrossoverResult:
        """
        Parse complete LLM response into CrossoverResult.

        Args:
            llm_response: Raw LLM response text
            parent_solvers: List of parent solvers (for metadata)

        Returns:
            CrossoverResult with parsed data
        """
        # Parse fused code
        fused_code = self._parse_fused_code(llm_response)

        # Parse assessment and confidence
        compatibility, confidence = self._parse_assessment(llm_response)

        # Extract parent metadata
        parent_ids = [p.solver_id for p in parent_solvers]
        parent_techniques = [p.tags for p in parent_solvers]

        return CrossoverResult(
            fused_code=fused_code,
            parent_ids=parent_ids,
            parent_techniques=parent_techniques,
            compatibility_assessment=compatibility,
            confidence=confidence,
        )
