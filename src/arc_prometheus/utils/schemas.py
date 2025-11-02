"""JSON schemas for structured LLM outputs.

These schemas are used with Gemini's structured output API to ensure reliable,
type-safe, and concise responses from the cognitive cells.

Benefits:
- Type enforcement with maxLength constraints
- Guaranteed valid JSON parsing
- ~70% token reduction vs free-form text
- Consistent output format every time

Reference: https://ai.google.dev/gemini-api/docs/structured-output
"""

# Multi-Persona Analyst Schema
# Generates 5 diverse interpretations from different expert perspectives
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
                        "maxLength": 50,
                        "description": "Expert perspective name (e.g., 'Geometric Transformation Specialist')",
                    },
                    "pattern": {
                        "type": "string",
                        "maxLength": 150,
                        "description": "One-sentence transformation rule description",
                    },
                    "observations": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 80},
                        "maxItems": 3,
                        "description": "Key insights (max 3, each ≤80 chars)",
                    },
                    "approach": {
                        "type": "string",
                        "maxLength": 100,
                        "description": "High-level implementation strategy",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Confidence in this interpretation",
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
            "minItems": 5,
            "maxItems": 5,
            "description": "Exactly 5 diverse expert interpretations",
        }
    },
    "required": ["interpretations"],
}

# Multi-Solution Programmer Schema
# Generates 5 solver implementations linked to interpretations
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
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Which interpretation this implements (1-5)",
                    },
                    "code": {
                        "type": "string",
                        "description": "Complete solve() function implementation",
                    },
                    "approach_summary": {
                        "type": "string",
                        "maxLength": 100,
                        "description": "Brief description of implementation approach",
                    },
                },
                "required": ["interpretation_id", "code", "approach_summary"],
            },
            "minItems": 5,
            "maxItems": 5,
            "description": "Exactly 5 solver implementations",
        }
    },
    "required": ["solutions"],
}

# Synthesis Agent Schema
# Analyzes 5 solutions and creates a 6th diverse solution
SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "object",
            "properties": {
                "successful_patterns": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 80},
                    "maxItems": 3,
                    "description": "Patterns from successful solutions (≤80 chars each)",
                },
                "failed_patterns": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 80},
                    "maxItems": 3,
                    "description": "Patterns from failed solutions (≤80 chars each)",
                },
                "synthesis_strategy": {
                    "type": "string",
                    "maxLength": 150,
                    "description": "How to create diverse 6th solution",
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
            "description": "Complete solve() function for synthesis solution",
        },
        "diversity_justification": {
            "type": "string",
            "maxLength": 100,
            "description": "Why this solution is different from all 5 previous",
        },
    },
    "required": ["analysis", "code", "diversity_justification"],
}

# Export all schemas
__all__ = [
    "MULTI_PERSONA_SCHEMA",
    "MULTI_SOLUTION_SCHEMA",
    "SYNTHESIS_SCHEMA",
]
