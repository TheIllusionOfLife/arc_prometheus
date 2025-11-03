# Investigation Findings: Ensemble Validation Failures

## Investigation 1: Why do we need `max_length`?

### Finding: We DON'T need it for Gemini

**Key Discovery**: Gemini's structured output API **does not support** `maxLength` constraints for strings.

**Evidence**:
1. From [Gemini Structured Output Docs](https://ai.google.dev/gemini-api/docs/structured-output):
   - String fields support: `enum`, `format`, `nullable`
   - **String fields do NOT support**: `maxLength`, `minLength`
   - Only arrays support `minItems`/`maxItems`

2. Our implementation:
   - **Pydantic models** have `max_length=200` (Python-side validation only)
   - **Dict schemas sent to Gemini** have NO maxLength (only text descriptions like "≤80 chars")
   - Gemini sees: `{"type": "string", "description": "Brief text (≤200 chars)"}`
   - This is just a **hint in the description**, not enforced

**Conclusion**:
- `max_length` in Pydantic is **purely for post-hoc validation**
- Gemini relies entirely on **prompt instructions** for conciseness
- We've been fighting Pydantic validation errors, not Gemini output issues

**What we should do**:
```python
# Option A: Remove max_length entirely, rely on prompts
class Interpretation(BaseModel):
    approach: str = Field(..., description="High-level strategy")

# Option B: Keep max_length as sanity check with generous limits
class Interpretation(BaseModel):
    approach: str = Field(..., max_length=500, description="High-level strategy")
```

---

## Investigation 2: Why is the JSON 196,399 characters?

### Finding: MAX_TOKENS finish_reason (likely), NOT token limit

**Evidence**:
1. **Model capacity**: gemini-2.5-flash-lite has 65,536 token output limit
2. **Actual size**: 196,399 chars ≈ 49,100 tokens (well within limit)
3. **Our code**: Does NOT set `max_output_tokens` in GenerationConfig
4. **Response handling**: Does NOT check `finish_reason`

**Problem**: We're blindly accessing `response.text` without checking why the response stopped.

**Finish reason enum values**:
```
STOP = 1              # Normal completion ✓
MAX_TOKENS = 2        # Hit token limit ⚠️
SAFETY = 3            # Blocked by safety filters
RECITATION = 4        # Copyrighted content
OTHER = 5             # Other errors
BLOCKLIST = 7
PROHIBITED_CONTENT = 8
MALFORMED_FUNCTION_CALL = 10
```

**What's happening**:
1. MultiSolutionProgrammer generates 5 complete Python functions
2. Each function can be 50-200 lines of code
3. Total JSON response can be 50,000+ characters
4. Without `max_output_tokens` set, **we're using the model's default**
5. Model may be using a **conservative default** (e.g., 8,192 tokens for some models)
6. Response gets truncated mid-JSON
7. We don't check `finish_reason`, so we don't know it was `MAX_TOKENS`

**Proper handling**:
```python
# BEFORE
response = model.generate_content(prompt, generation_config=generation_config)
parsed_response = MultiSolutionResponse.model_validate_json(response.text)

# AFTER
response = model.generate_content(prompt, generation_config=generation_config)

# Check finish reason
if response.candidates:
    candidate = response.candidates[0]
    FinishReason = type(candidate).FinishReason

    if candidate.finish_reason == FinishReason.MAX_TOKENS:
        raise ValueError(
            f"Response truncated due to MAX_TOKENS. "
            f"Set higher max_output_tokens in GenerationConfig."
        )
    elif candidate.finish_reason != FinishReason.STOP:
        raise ValueError(f"Unexpected finish reason: {candidate.finish_reason}")

parsed_response = MultiSolutionResponse.model_validate_json(response.text)
```

**Fix**: Add `max_output_tokens` to ensemble agents:
```python
generation_config = GenerationConfig(
    temperature=self.temperature,
    response_mime_type="application/json",
    response_schema=MULTI_SOLUTION_SCHEMA,
    max_output_tokens=65536,  # Use full model capacity
)
```

---

## Investigation 3: Are we using structured output correctly?

### Finding: YES, but we're not handling edge cases

**Our usage** (correct):
```python
generation_config = GenerationConfig(
    temperature=0.7,
    response_mime_type="application/json",  ✓
    response_schema=MULTI_SOLUTION_SCHEMA,  ✓
)
```

**What the docs say**:
- ✓ `response_mime_type="application/json"` required
- ✓ `response_schema` required
- ✓ Responses should be raw JSON (no markdown)

**Edge cases we're NOT handling**:

### 1. Markdown wrapping (rare but seen)
**Observation**: In no-cache test, got ` ```json\n{...}\n``` `

**Why it happens**:
- LLMs trained on markdown-heavy datasets
- Structured output SHOULD prevent this, but occasionally fails
- May be model-specific behavior

**Fix**:
```python
def unwrap_markdown_json(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# Use before validation
cleaned_text = unwrap_markdown_json(response.text)
parsed_response = MultiSolutionResponse.model_validate_json(cleaned_text)
```

### 2. Truncated responses (MAX_TOKENS)
**Fix**: Check `finish_reason` before parsing (see Investigation 2)

### 3. Safety/content filtering
**Fix**: Check for `SAFETY`, `PROHIBITED_CONTENT`, etc. and provide clear errors

---

## Summary of Root Causes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| **Observation >100 chars** | Pydantic validation, not Gemini issue | Remove `max_length` or set to 500+ |
| **JSON truncation (196K chars)** | Missing `max_output_tokens`, no `finish_reason` check | Add `max_output_tokens=65536` + check `finish_reason` |
| **Markdown wrapping** | Edge case in LLM output | Add markdown unwrapping preprocessing |

---

## Recommended Fixes

### Priority 1: Add max_output_tokens to all ensemble agents
```python
# multi_persona_analyst.py, multi_solution_programmer.py, synthesis_agent.py
generation_config = GenerationConfig(
    temperature=self.temperature,
    response_mime_type="application/json",
    response_schema=SCHEMA,
    max_output_tokens=65536,  # NEW: Use full model capacity
)
```

### Priority 2: Check finish_reason before parsing
```python
def _check_finish_reason(response):
    """Validate response completed successfully."""
    if not response.candidates:
        raise ValueError("No candidates in response")

    candidate = response.candidates[0]
    FinishReason = type(candidate).FinishReason

    if candidate.finish_reason == FinishReason.MAX_TOKENS:
        raise ValueError("Response truncated: MAX_TOKENS reached. Increase max_output_tokens.")
    elif candidate.finish_reason == FinishReason.SAFETY:
        raise ValueError(f"Response blocked: SAFETY. Ratings: {candidate.safety_ratings}")
    elif candidate.finish_reason != FinishReason.STOP:
        raise ValueError(f"Unexpected finish_reason: {candidate.finish_reason}")
```

### Priority 3: Remove or relax Pydantic max_length
```python
# Option A: Remove entirely (rely on prompts)
class Interpretation(BaseModel):
    approach: str = Field(..., description="Implementation strategy")

# Option B: Keep as generous sanity check
class Interpretation(BaseModel):
    approach: str = Field(..., max_length=1000, description="Implementation strategy")
```

### Priority 4: Add markdown unwrapping (defensive)
```python
def unwrap_markdown_json(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
```

---

## Expected Impact

**Current**: 60% success rate (3/5 tasks)
- 20% failed: MAX_TOKENS truncation
- 20% failed: Pydantic max_length validation

**After fixes**: ~95-100% success rate
- MAX_TOKENS issue: Resolved by setting `max_output_tokens=65536`
- Pydantic validation: Resolved by removing/relaxing limits
- Markdown wrapping: Resolved by unwrapping (rare edge case)

**Remaining failures**: Actual logic/runtime errors in generated code (expected)
