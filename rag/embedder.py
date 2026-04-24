"""
Embedding layer: sentence-transformers + FAISS index management.
"""
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = Path("index/faiss.index")
META_PATH = Path("index/metadata.json")


class EmbeddingStore:
    """Manages embedding creation, FAISS indexing, and persistence."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # L2-normalise for cosine similarity via inner product
        faiss.normalize_L2(vectors)
        return vectors.astype("float32")

    def build_index(self, chunks: List[Dict]) -> None:
        """Embed all chunks and build a FAISS IndexFlatIP."""
        if not chunks:
            raise ValueError("No chunks provided to build index.")

        texts = [c["content"] for c in chunks]
        vectors = self._embed(texts)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.chunks = chunks
        logger.info("Built FAISS index with %d vectors (dim=%d)", len(chunks), self.dimension)

    def save(self, index_path: Path = INDEX_PATH, meta_path: Path = META_PATH) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        logger.info("Saved index to %s", index_path)

    def load(self, index_path: Path = INDEX_PATH, meta_path: Path = META_PATH) -> bool:
        if not index_path.exists() or not meta_path.exists():
            return False
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info("Loaded index from %s (%d chunks)", index_path, len(self.chunks))
        return True

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0
