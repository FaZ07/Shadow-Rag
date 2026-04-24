"""
Document ingestion: load PDFs/TXTs, clean text, semantic chunking.
"""
import re
import logging
from pathlib import Path
from typing import List, Dict

import pdfplumber
import nltk

logger = logging.getLogger(__name__)


def _ensure_nltk():
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove non-printable characters except newlines
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    return text.strip()


def _load_pdf(path: Path) -> List[Dict]:
    chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                text = _clean_text(text)
                if text:
                    chunks.append({
                        "content": text,
                        "source": path.name,
                        "page": page_num,
                    })
    except Exception as exc:
        logger.error("Failed to load PDF %s: %s", path.name, exc)
    return chunks


def _load_txt(path: Path) -> List[Dict]:
    try:
        text = _clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if text:
            return [{"content": text, "source": path.name, "page": None}]
    except Exception as exc:
        logger.error("Failed to load TXT %s: %s", path.name, exc)
    return []


def load_documents(data_dir: str = "data") -> List[Dict]:
    """Load all PDF and TXT files from *data_dir*. Returns raw page-level dicts."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    documents: List[Dict] = []
    loaders = {".pdf": _load_pdf, ".txt": _load_txt}

    for file_path in sorted(data_path.iterdir()):
        loader = loaders.get(file_path.suffix.lower())
        if loader is None:
            continue
        docs = loader(file_path)
        documents.extend(docs)
        logger.info("Loaded %d pages from %s", len(docs), file_path.name)

    return documents


def semantic_chunk(
    documents: List[Dict],
    max_words: int = 400,
    overlap_sentences: int = 2,
) -> List[Dict]:
    """
    Split documents into sentence-aware chunks with overlap.
    Returns list of dicts: {content, source, page, chunk_id}
    """
    _ensure_nltk()
    from nltk.tokenize import sent_tokenize

    chunks: List[Dict] = []

    for doc in documents:
        sentences = sent_tokenize(doc["content"])
        current: List[str] = []
        current_words = 0
        chunk_idx = 0

        for sent in sentences:
            word_count = len(sent.split())

            if current_words + word_count > max_words and current:
                chunks.append({
                    "content": " ".join(current),
                    "source": doc["source"],
                    "page": doc.get("page"),
                    "chunk_id": f"{doc['source']}::{chunk_idx}",
                })
                chunk_idx += 1
                # Keep tail for overlap
                tail = current[-overlap_sentences:] if len(current) > overlap_sentences else current[:]
                current = tail + [sent]
                current_words = sum(len(s.split()) for s in current)
            else:
                current.append(sent)
                current_words += word_count

        if current:
            chunks.append({
                "content": " ".join(current),
                "source": doc["source"],
                "page": doc.get("page"),
                "chunk_id": f"{doc['source']}::{chunk_idx}",
            })

    logger.info("Produced %d chunks from %d document pages", len(chunks), len(documents))
    return chunks
