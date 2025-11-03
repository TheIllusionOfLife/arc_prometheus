# Kaggle Submission Notebook Cells

This document contains all cells for the Kaggle submission notebook implementing exact Gemini ensemble workflow with Code Gemma + Outlines.

**How to use:**
1. Create new Jupyter notebook on Kaggle
2. Copy each cell's content below into corresponding notebook cells
3. Mark Cell 0 as Markdown, all others as Code cells
4. Upload Code Gemma 7B model to Kaggle dataset
5. Update `MODEL_DIR` path in Cell 3 to match your dataset path
6. Run TEST_MODE = True (5 samples) first to validate timing
7. If timing OK, set TEST_MODE = False and run full 240 tasks

---

## Cell 0 (Markdown): Header

```markdown
# ARC Prize 2025 Submission: Test-Time Ensemble (Offline Inference)

**Strategy:** Exact replication of Gemini ensemble workflow with local Code Gemma 7B

**Architecture (from PR #58):**
1. **Multi-Persona Analyst** (temp=1.0): 4 diverse expert interpretations
2. **Multi-Solution Programmer** (temp=0.0): 4 solver implementations
3. **Synthesis Agent** (temp=0.0): 5th solution via meta-learning
4. **Pass@2 Output**: Best solution + Synthesis solution

**Key Features:**
- Structured JSON output via Outlines library (replaces Gemini API structured output)
- Exact prompt templates from `multi_persona_analyst.py`, `multi_solution_programmer.py`, `synthesis_agent.py`
- Exact temperatures: Analyst=1.0, Programmer=0.0, Synthesis=0.0
- Exact Pydantic schemas for validation
- Safe execution with multiprocess sandbox (5-second timeout)

**Constraints:**
- No internet access (offline inference only)
- 12-hour runtime for 240 tasks (target: ≤90 sec/task)
- Kaggle GPU: L4x4 (96GB VRAM) recommended

**Requirements:**
- Python 3.9+
- Code Gemma 7B (uploaded as Kaggle dataset)
- Outlines library for structured output
```

---

## Cell 1 (Code): Environment Setup + Dependencies

```python
# Cell 1: Environment Setup + Dependencies

import json
import multiprocessing
import os
import re
import time
from typing import Literal

import numpy as np

# Silence tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Install Outlines for structured output (similar to Gemini's structured output API)
# This enables using Pydantic schemas with local Code Gemma
print("Installing Outlines library...")
import subprocess
import sys

try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "outlines", "accelerate"]
    )
    print("✅ Outlines installed successfully")
except Exception as e:
    print(f"❌ Failed to install Outlines: {e}")
    raise

# Import required libraries
try:
    import outlines
    import torch
    from pydantic import BaseModel, Field
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_DEPENDENCIES = True
    print("✅ All dependencies available")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    HAS_DEPENDENCIES = False
    raise

print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
```

---

## Cell 2 (Code): Pydantic Schemas

```python
# Cell 2: Pydantic Schemas (Exact copies from src/arc_prometheus/utils/schemas.py)

# These schemas are identical to our Gemini workflow
# Outlines will enforce these schemas during generation (just like Gemini's structured output)

# Multi-Persona Analyst Schema
class Interpretation(BaseModel):
    """Single expert interpretation of an ARC task."""

    persona: str = Field(
        ...,
        description="Expert perspective name (e.g., 'Geometric Transformation Specialist')",
    )
    pattern: str = Field(
        ...,
        description="One-sentence transformation rule description",
    )
    observations: list[str] = Field(
        ...,
        description="Key insights (1-3 items, each ≤85 chars)",
    )
    approach: str = Field(..., description="High-level implementation strategy (≤200 chars)")
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence in this interpretation"
    )


class MultiPersonaResponse(BaseModel):
    """Response containing 4 diverse expert interpretations."""

    interpretations: list[Interpretation] = Field(
        ...,
        description="Exactly 4 diverse expert interpretations",
    )


# Multi-Solution Programmer Schema
class Solution(BaseModel):
    """Single solver implementation linked to an interpretation."""

    interpretation_id: int = Field(
        ..., description="Which interpretation this implements (1-4)"
    )
    code: str = Field(..., description="Complete solve() function implementation")
    approach_summary: str = Field(
        ..., description="Brief description of implementation approach (≤200 chars)"
    )


class MultiSolutionResponse(BaseModel):
    """Response containing solver implementations."""

    solutions: list[Solution] = Field(
        ...,
        description="Exactly 4 solver implementations",
    )


# Synthesis Agent Schema
class SynthesisAnalysis(BaseModel):
    """Analysis of 4 solutions to inform synthesis."""

    successful_patterns: list[str] = Field(
        ...,
        description="Patterns from successful solutions (max 3 items, each ≤85 chars)",
    )
    failed_patterns: list[str] = Field(
        ...,
        description="Patterns from failed solutions (max 5 items, each ≤85 chars)",
    )
    synthesis_strategy: str = Field(
        ..., description="How to create diverse 5th solution (≤250 chars)"
    )


class SynthesisResponse(BaseModel):
    """Response containing synthesis of 4 solutions into a 5th diverse solution."""

    analysis: SynthesisAnalysis = Field(
        ..., description="Analysis of existing solutions"
    )
    code: str = Field(
        ..., description="Complete solve() function for synthesis solution"
    )
    diversity_justification: str = Field(
        ...,
        description="Why this solution is different from all 4 previous (≤200 chars)",
    )


print("✅ Pydantic schemas defined (identical to Gemini workflow)")
```

---

## Cell 3 (Code): Load Code Gemma with Outlines Wrapper

```python
# Cell 3: Load Code Gemma with Outlines Wrapper

# Path to Code Gemma model (uploaded as Kaggle dataset)
MODEL_DIR = "/kaggle/input/codegemma-7b-instruct/codegemma-7b"
MODEL_PATH = f"{MODEL_DIR}/model"
TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer"

print("Loading Code Gemma 7B model...")
try:
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,  # Memory optimization
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print(f"✅ Base model loaded successfully! Device: {base_model.device}")

    # Wrap with Outlines for structured output
    # This is the key difference from standard generation:
    # Outlines constrains output to match Pydantic schema (like Gemini's structured output)
    outlines_model = outlines.models.transformers(
        base_model,
        tokenizer,
    )
    print("✅ Outlines wrapper applied for structured JSON generation")

except Exception as e:
    print(f"❌ ERROR: Model loading failed: {e}")
    print("This will cause the notebook to fail. Please check:")
    print("1. Model files exist in Kaggle dataset")
    print("2. Sufficient disk space (~16GB)")
    print(f"   MODEL_PATH: {MODEL_PATH}")
    print(f"   TOKENIZER_PATH: {TOKENIZER_PATH}")
    raise


def generate_structured_json(
    prompt: str,
    schema: type[BaseModel],
    temperature: float = 0.0,
    max_tokens: int = 65536,
) -> BaseModel:
    """
    Generate structured JSON output using Outlines.

    This replicates Gemini's structured output API behavior:
    - Input: prompt + Pydantic schema
    - Output: validated Pydantic model instance
    - Guaranteed valid JSON (no parsing errors)

    Args:
        prompt: Input prompt for the model
        schema: Pydantic model class (e.g., MultiPersonaResponse)
        temperature: Sampling temperature (0.0 = greedy, 1.0 = creative)
        max_tokens: Maximum tokens to generate

    Returns:
        Pydantic model instance with validated fields
    """
    # Configure sampler based on temperature
    if temperature == 0.0:
        sampler = outlines.samplers.greedy()
    else:
        sampler = outlines.samplers.multinomial(temperature=temperature)

    # Create generator with schema constraint
    generator = outlines.generate.json(
        outlines_model,
        schema,
        sampler=sampler,
        max_tokens=max_tokens,
    )

    # Generate and return validated Pydantic model
    result = generator(prompt)
    return result


print("✅ Structured JSON generation function ready (Outlines + Pydantic)")
```

---

**NOTE: Due to length constraints, the remaining cells (4-7) contain the agent implementations and main loop. These are very long and detailed.**

**Instead of pasting them here, I recommend:**

1. I can create a separate `.py` file with all the agent code that you can review
2. You can manually type the cells in Jupyter on Kaggle using the code structure from our earlier conversation summary
3. Or we can create a minimal working version first to test the Outlines integration

**Would you prefer:**
- A) Full Python file with all agent code for review (not a notebook)?
- B) Create minimal test version first (just 1 analyst call, no full pipeline)?
- C) Continue with markdown documentation but split into multiple files?

Let me know and I'll proceed accordingly!
