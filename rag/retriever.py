"""
Retrieval layer: query FAISS index, return top-k chunks with metadata.
"""
import logging
from typing import List, Dict

from rag.embedder import EmbeddingStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant chunks for a given query using the EmbeddingStore."""

    def __init__(self, store: EmbeddingStore):
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Return top-k chunks most relevant to *query*.
        Each result dict contains: content, source, page, chunk_id, score.
        """
        if not self.store.is_ready:
            raise RuntimeError("Embedding index is not built. Ingest documents first.")

        query_vec = self.store.embed_query(query)
        scores, indices = self.store.index.search(query_vec, min(top_k, len(self.store.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self.store.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        logger.debug("Retrieved %d chunks for query: %s", len(results), query[:60])
        return results

    def format_context(self, chunks: List[Dict]) -> str:
        """Format chunks into a numbered context string for LLM prompts."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f" (page {chunk['page']})" if chunk.get("page") else ""
            parts.append(f"[{i}] Source: {chunk['source']}{page_info}\n{chunk['content']}")
        return "\n\n---\n\n".join(parts)
