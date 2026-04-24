"""
LLM abstraction and answer generation.
Supports: groq (default, free) | gemini | huggingface
Set LLM_BACKEND env var to select backend.
"""
import json
import logging
import os
import re
import time
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

ANSWER_PROMPT = """\
You are a precise, factual research assistant.
Answer the question using ONLY the context provided below.

CONTEXT:
{context}

QUESTION: {question}

RULES:
- Use ONLY information from the context above.
- If the context is insufficient, say exactly: "The provided documents do not contain enough information to answer this question."
- Structure your answer clearly (use bullet points or short paragraphs).
- Never add information not present in the context.
- Cite source numbers like [1], [2] where relevant.

ANSWER:"""


SHADOW_PROMPT = """\
You are a ruthless critical analyst. Your task: find every flaw in this AI-generated answer.

QUESTION: {question}

GENERATED ANSWER:
{answer}

SOURCE CONTEXT USED:
{context}

Analyse strictly. Return ONLY valid JSON — no markdown, no extra text:
{{
  "weaknesses": ["..."],
  "missing_information": ["..."],
  "contradictions": ["..."],
  "hallucination_risk": "low" | "medium" | "high"
}}"""


CLAIM_EXTRACTION_PROMPT = """\
Extract every atomic, independently verifiable factual claim from this answer.

ANSWER:
{answer}

RULES:
- Each claim = one concrete, testable fact.
- No opinions, no compound statements, no filler.
- If no verifiable claims exist, return an empty array.

Return ONLY a JSON array of strings:
["claim 1", "claim 2"]"""


CLAIM_VERIFY_PROMPT = """\
You are a strict fact-checker. Verify the claim against the evidence. Be skeptical.

CLAIM: {claim}

EVIDENCE:
{evidence}

Return ONLY valid JSON — no markdown:
{{
  "status": "VERIFIED" | "PARTIALLY_SUPPORTED" | "NOT_SUPPORTED",
  "justification": "one concise sentence",
  "evidence": "direct quote from evidence, or 'No relevant evidence found'"
}}"""


# ── JSON parsing helper ───────────────────────────────────────────────────────

def parse_json_response(text: str) -> Union[Dict, List]:
    """Extract JSON from LLM response, stripping markdown code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse JSON from LLM response:\n{text[:300]}")


# ── LLM Client ────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper around supported LLM backends.
    Backend selected by LLM_BACKEND env var (default: 'groq').
    """

    def __init__(self):
        self.backend = os.getenv("LLM_BACKEND", "groq").lower()
        self._groq_client = None
        self._gemini_model = None
        self._hf_pipeline = None
        self._init()

    def _init(self):
        if self.backend == "groq":
            self._init_groq()
        elif self.backend == "gemini":
            self._init_gemini()
        elif self.backend == "huggingface":
            self._init_huggingface()
        else:
            raise ValueError(f"Unknown LLM_BACKEND='{self.backend}'. Use 'groq', 'gemini', or 'huggingface'.")

    def _init_groq(self):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Run: pip install groq")

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set.\n"
                "Get a free key at https://console.groq.com"
            )
        self._groq_client = Groq(api_key=api_key)
        self._groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        logger.info("LLM backend: Groq (%s)", self._groq_model)

    def _init_gemini(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-04-17")
        self._gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": 0.1, "max_output_tokens": 2048},
        )
        logger.info("LLM backend: Gemini (%s)", model_name)

    def _init_huggingface(self):
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("Run: pip install transformers accelerate")

        model_id = os.getenv("HF_MODEL", "google/flan-t5-large")
        self._hf_pipeline = hf_pipeline("text2text-generation", model=model_id, max_new_tokens=512)
        logger.info("LLM backend: HuggingFace (%s)", model_id)

    def generate(self, prompt: str, retries: int = 2) -> str:
        """Generate text with automatic retry on transient failures."""
        for attempt in range(retries + 1):
            try:
                if self.backend == "groq":
                    return self._groq_generate(prompt)
                elif self.backend == "gemini":
                    return self._gemini_model.generate_content(prompt).text.strip()
                else:
                    return self._hf_pipeline(prompt)[0]["generated_text"].strip()
            except Exception as exc:
                if attempt < retries:
                    wait = 2 ** attempt
                    logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %ds…",
                                   attempt + 1, retries + 1, exc, wait)
                    time.sleep(wait)
                else:
                    raise

    def _groq_generate(self, prompt: str) -> str:
        response = self._groq_client.chat.completions.create(
            model=self._groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()


# ── High-level generation functions ──────────────────────────────────────────

def generate_answer(question: str, context: str, llm: LLMClient) -> str:
    """Generate a grounded answer from retrieved context."""
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    return llm.generate(prompt)
