"""Prompts module for Legal Policy Explainer."""

from .system_prompts import (
    LEGAL_EXPLAINER_SYSTEM_PROMPT,
    RESEARCHER_AGENT_PROMPT,
    EXPLAINER_AGENT_PROMPT,
    get_system_prompt,
    get_refusal_message,
    add_disclaimer,
)

__all__ = [
    "LEGAL_EXPLAINER_SYSTEM_PROMPT",
    "RESEARCHER_AGENT_PROMPT",
    "EXPLAINER_AGENT_PROMPT",
    "get_system_prompt",
    "get_refusal_message",
    "add_disclaimer",
]
