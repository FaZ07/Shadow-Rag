"""
Shadow Critique Layer: self-reflection on the generated answer.
Identifies weaknesses, hallucination risk, contradictions, and gaps.
"""
import logging
from typing import Dict

from rag.generator import LLMClient, SHADOW_PROMPT, parse_json_response

logger = logging.getLogger(__name__)

_FALLBACK = {
    "weaknesses": ["Shadow analysis unavailable."],
    "missing_information": [],
    "contradictions": [],
    "hallucination_risk": "unknown",
}


def shadow_critique(
    question: str,
    answer: str,
    context: str,
    llm: LLMClient,
) -> Dict:
    """
    Run the shadow critique pass on a generated answer.
    Returns dict with keys: weaknesses, missing_information, contradictions, hallucination_risk.
    """
    prompt = SHADOW_PROMPT.format(
        question=question,
        answer=answer,
        context=context,
    )

    try:
        raw = llm.generate(prompt)
        result = parse_json_response(raw)

        # Normalise: ensure all keys exist and lists are lists
        critique = {
            "weaknesses": _ensure_list(result.get("weaknesses", [])),
            "missing_information": _ensure_list(result.get("missing_information", [])),
            "contradictions": _ensure_list(result.get("contradictions", [])),
            "hallucination_risk": str(result.get("hallucination_risk", "unknown")).lower(),
        }

        # Clamp hallucination_risk to known values
        if critique["hallucination_risk"] not in ("low", "medium", "high"):
            critique["hallucination_risk"] = "unknown"

        logger.debug("Shadow critique done. Risk: %s", critique["hallucination_risk"])
        return critique

    except Exception as exc:
        logger.error("Shadow critique failed: %s", exc)
        return _FALLBACK


def _ensure_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        return [value]
    return []
