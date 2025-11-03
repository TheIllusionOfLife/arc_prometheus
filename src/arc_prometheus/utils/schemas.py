"""Pydantic schemas for structured LLM outputs.

These schemas work with Gemini's structured output API via Pydantic model validation.

Benefits:
- Type enforcement with runtime validation
- Guaranteed valid JSON parsing
- ~70% token reduction vs free-form text
- Consistent output format every time
- IDE autocomplete and type hints

Reference: https://ai.google.dev/gemini-api/docs/structured-output

Note: google-generativeai library uses dict-based schemas for API calls,
but we validate responses with Pydantic models for type safety.
"""

from typing import Literal

from pydantic import BaseModel, Field


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
        min_length=1,
        max_length=3,
        description="Key insights (1-3 items)",
    )
    approach: str = Field(..., description="High-level implementation strategy")
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence in this interpretation"
    )


class MultiPersonaResponse(BaseModel):
    """Response containing 4 diverse expert interpretations."""

    interpretations: list[Interpretation] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Exactly 4 diverse expert interpretations",
    )


# Multi-Solution Programmer Schema
class Solution(BaseModel):
    """Single solver implementation linked to an interpretation."""

    interpretation_id: int = Field(
        ..., ge=1, le=4, description="Which interpretation this implements (1-4)"
    )
    code: str = Field(..., description="Complete solve() function implementation")
    approach_summary: str = Field(
        ..., description="Brief description of implementation approach"
    )


class MultiSolutionResponse(BaseModel):
    """Response containing solver implementations.

    Normally contains exactly 4 solutions, but can accept 1-4 when MAX_TOKENS
    truncates the response. The ensemble pipeline pads <4 solutions to 4 by
    duplicating the best solution.
    """

    solutions: list[Solution] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="1-4 solver implementations (target: 4)",
    )


# Synthesis Agent Schema
class SynthesisAnalysis(BaseModel):
    """Analysis of 4 solutions to inform synthesis."""

    successful_patterns: list[str] = Field(
        ...,
        max_length=3,
        description="Patterns from successful solutions (max 3 items)",
    )
    failed_patterns: list[str] = Field(
        ...,
        max_length=5,
        description="Patterns from failed solutions (max 5 items)",
    )
    synthesis_strategy: str = Field(
        ..., description="How to create diverse 5th solution"
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
        description="Why this solution is different from all 4 previous",
    )


# Dict-based schemas for Gemini API (without unsupported fields)
# These are used for response_schema parameter in generation_config
# Pydantic models above are used for parsing and validation
MULTI_PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "persona": {
                        "type": "string",
                        "description": "Expert perspective name",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "One-sentence transformation rule",
                    },
                    "observations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key insights (max 3, each ≤80 chars)",
                    },
                    "approach": {
                        "type": "string",
                        "description": "High-level implementation strategy",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Confidence level",
                    },
                },
                "required": [
                    "persona",
                    "pattern",
                    "observations",
                    "approach",
                    "confidence",
                ],
            },
            "description": "Exactly 4 diverse expert interpretations",
        }
    },
    "required": ["interpretations"],
}

MULTI_SOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "solutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "interpretation_id": {
                        "type": "integer",
                        "description": "Which interpretation this implements (1-4)",
                    },
                    "code": {
                        "type": "string",
                        "description": "Complete solve() function",
                    },
                    "approach_summary": {
                        "type": "string",
                        "description": "Brief implementation description",
                    },
                },
                "required": ["interpretation_id", "code", "approach_summary"],
            },
            "description": "Exactly 4 solver implementations",
        }
    },
    "required": ["solutions"],
}

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "object",
            "properties": {
                "successful_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Patterns from successful solutions",
                },
                "failed_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Patterns from failed solutions",
                },
                "synthesis_strategy": {
                    "type": "string",
                    "description": "How to synthesize diverse solution from 4 existing solutions",
                },
            },
            "required": [
                "successful_patterns",
                "failed_patterns",
                "synthesis_strategy",
            ],
        },
        "code": {
            "type": "string",
            "description": "Complete solve() function",
        },
        "diversity_justification": {
            "type": "string",
            "description": "Why this solution is different",
        },
    },
    "required": ["analysis", "code", "diversity_justification"],
}

__all__ = [
    "Interpretation",
    "MultiPersonaResponse",
    "Solution",
    "MultiSolutionResponse",
    "SynthesisAnalysis",
    "SynthesisResponse",
    # Dict schemas for Gemini API
    "MULTI_PERSONA_SCHEMA",
    "MULTI_SOLUTION_SCHEMA",
    "SYNTHESIS_SCHEMA",
]
