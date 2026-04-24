"""
Claim-Level Verification Layer.
Extracts atomic claims from the answer, then verifies each one independently.
"""
import logging
from typing import Dict, List

from rag.generator import (
    LLMClient,
    CLAIM_EXTRACTION_PROMPT,
    CLAIM_VERIFY_PROMPT,
    parse_json_response,
)
from rag.retriever import Retriever

logger = logging.getLogger(__name__)

VALID_STATUSES = {"VERIFIED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"}


def extract_claims(answer: str, llm: LLMClient) -> List[str]:
    """Break the answer into a list of atomic, verifiable factual claims."""
    prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
    try:
        raw = llm.generate(prompt)
        claims = parse_json_response(raw)
        if isinstance(claims, list):
            return [str(c).strip() for c in claims if str(c).strip()]
        logger.warning("Claim extraction returned non-list; wrapping: %s", type(claims))
        return []
    except Exception as exc:
        logger.error("Claim extraction failed: %s", exc)
        return []


def verify_claim(claim: str, evidence: str, llm: LLMClient) -> Dict:
    """Verify a single claim against provided evidence text."""
    prompt = CLAIM_VERIFY_PROMPT.format(claim=claim, evidence=evidence)
    try:
        raw = llm.generate(prompt)
        result = parse_json_response(raw)

        status = str(result.get("status", "NOT_SUPPORTED")).upper()
        if status not in VALID_STATUSES:
            status = "NOT_SUPPORTED"

        return {
            "claim": claim,
            "status": status,
            "justification": str(result.get("justification", "")).strip(),
            "evidence": str(result.get("evidence", "No relevant evidence found")).strip(),
        }
    except Exception as exc:
        logger.error("Claim verification failed for '%s': %s", claim[:60], exc)
        return {
            "claim": claim,
            "status": "NOT_SUPPORTED",
            "justification": "Verification error — could not parse LLM response.",
            "evidence": "No relevant evidence found",
        }


def verify_all_claims(
    answer: str,
    retriever: Retriever,
    question: str,
    llm: LLMClient,
    top_k: int = 4,
) -> List[Dict]:
    """
    Full claim-verification pipeline:
    1. Extract claims from the answer.
    2. For each claim, retrieve supporting evidence and verify.
    Returns a list of verification dicts.
    """
    claims = extract_claims(answer, llm)
    if not claims:
        logger.warning("No claims extracted from answer.")
        return []

    results = []
    for claim in claims:
        # Re-retrieve context specifically for this claim
        chunks = retriever.retrieve(claim, top_k=top_k)
        evidence = retriever.format_context(chunks)
        verification = verify_claim(claim, evidence, llm)
        # Attach which sources were consulted
        verification["sources_checked"] = list({c["source"] for c in chunks})
        results.append(verification)
        logger.debug("Claim verified: [%s] %s", verification["status"], claim[:60])

    return results


def compute_confidence_score(
    fact_check: List[Dict],
    shadow_analysis: Dict,
) -> int:
    """
    Compute 0–100 confidence score.
    Formula:
      - Base score from claim verification ratio
      - Deductions for hallucination risk and contradictions
    """
    if not fact_check:
        # No claims → rely on shadow analysis alone
        base = 50
    else:
        total = len(fact_check)
        verified = sum(1 for f in fact_check if f["status"] == "VERIFIED")
        partial = sum(1 for f in fact_check if f["status"] == "PARTIALLY_SUPPORTED")
        base = int(((verified + 0.5 * partial) / total) * 100)

    risk = shadow_analysis.get("hallucination_risk", "unknown")
    risk_penalty = {"low": 0, "medium": 15, "high": 30, "unknown": 10}.get(risk, 10)

    contradiction_penalty = min(len(shadow_analysis.get("contradictions", [])) * 5, 20)

    score = max(0, min(100, base - risk_penalty - contradiction_penalty))
    return score
