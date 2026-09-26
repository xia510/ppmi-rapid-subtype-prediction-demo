"""Local, page-aware literature indexing and retrieval for the research demo."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Protocol

import numpy as np


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
INDEX_METADATA_NAME = "metadata.json"
INDEX_VECTORS_NAME = "embeddings.npy"


class RAGError(RuntimeError):
    """Base class for expected local RAG failures."""


class EmptyCorpusError(RAGError):
    """Raised when no usable text can be indexed."""


class LiteratureIndexNotFoundError(RAGError):
    """Raised when a generated literature index is unavailable."""


class IndexDimensionError(RAGError):
    """Raised when the query and stored embeddings are incompatible."""


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one dense vector per input string."""


class SentenceTransformerEmbedder:
    """Lazy sentence-transformers adapter so importing the API stays lightweight."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RAGError(
                    "sentence-transformers is required to build or query the literature index."
                ) from error
            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise IndexDimensionError("Embeddings must be a two-dimensional matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def split_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[dict[str, Any]]:
    """Split page text while preserving document, title, source and page metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    step = chunk_size - overlap
    chunks: list[dict[str, Any]] = []
    for page in pages:
        text = re.sub(r"\s+", " ", str(page.get("text", ""))).strip()
        if not text:
            continue
        chunk_number = 0
        for start in range(0, len(text), step):
            piece = text[start : start + chunk_size].strip()
            if not piece:
                continue
            chunk_number += 1
            document_id = str(page["document_id"])
            page_number = int(page["page"])
            chunks.append(
                {
                    "source_id": f"{document_id}-p{page_number}-c{chunk_number}",
                    "document_id": document_id,
                    "title": str(page["title"]),
                    "source": str(page["source"]),
                    "page": page_number,
                    "text": piece,
                }
            )
            if start + chunk_size >= len(text):
                break
    return chunks


def extract_pdf_pages(source_dir: Path) -> list[dict[str, Any]]:
    """Extract text and one-based page numbers from every PDF in a directory."""
    try:
        import fitz
    except ImportError as error:
        raise RAGError("PyMuPDF is required to read literature PDF files.") from error

    source_dir = Path(source_dir)
    pages: list[dict[str, Any]] = []
    for pdf_path in sorted(source_dir.glob("*.pdf"), key=lambda item: item.name.lower()):
        with fitz.open(pdf_path) as document:
            title = str(document.metadata.get("title") or pdf_path.stem).strip()
            for page_index, page in enumerate(document, start=1):
                pages.append(
                    {
                        "document_id": pdf_path.stem,
                        "title": title,
                        "source": pdf_path.name,
                        "page": page_index,
                        "text": page.get_text("text"),
                    }
                )
    return pages


def build_literature_index_from_pages(
    pages: list[dict[str, Any]],
    index_dir: Path,
    embedder: Embedder,
    chunk_size: int = 800,
    overlap: int = 120,
) -> dict[str, Any]:
    """Build a persisted exact-search vector index from already extracted pages."""
    chunks = split_pages(pages, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise EmptyCorpusError("No usable text was found in the literature documents.")

    embeddings = _normalise_rows(embedder.encode([row["text"] for row in chunks]))
    if embeddings.shape[0] != len(chunks) or embeddings.shape[1] == 0:
        raise IndexDimensionError("The embedder returned an incompatible matrix.")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / INDEX_VECTORS_NAME, embeddings, allow_pickle=False)
    metadata = {
        "schema_version": 1,
        "embedding_model": str(embedder.model_name),
        "dimension": int(embeddings.shape[1]),
        "document_count": len({row["document_id"] for row in chunks}),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    (index_dir / INDEX_METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {key: metadata[key] for key in ("document_count", "chunk_count", "dimension")}


def build_literature_index(
    source_dir: Path,
    index_dir: Path,
    embedder: Embedder | None = None,
    chunk_size: int = 800,
    overlap: int = 120,
) -> dict[str, Any]:
    """Extract local PDFs and build the persisted literature index."""
    resolved_embedder = embedder or SentenceTransformerEmbedder()
    pages = extract_pdf_pages(Path(source_dir))
    return build_literature_index_from_pages(
        pages,
        Path(index_dir),
        resolved_embedder,
        chunk_size=chunk_size,
        overlap=overlap,
    )


class LiteratureIndex:
    """Loaded local vectors and metadata with exact cosine retrieval."""

    def __init__(self, embeddings: np.ndarray, chunks: list[dict], embedder: Embedder):
        self.embeddings = _normalise_rows(embeddings)
        self.chunks = chunks
        self.embedder = embedder

    @classmethod
    def load(cls, index_dir: Path, embedder: Embedder | None = None) -> "LiteratureIndex":
        index_dir = Path(index_dir)
        vectors_path = index_dir / INDEX_VECTORS_NAME
        metadata_path = index_dir / INDEX_METADATA_NAME
        if not vectors_path.exists() or not metadata_path.exists():
            raise LiteratureIndexNotFoundError(
                "Literature index is unavailable. Build it before starting RAG queries."
            )
        embeddings = np.load(vectors_path, allow_pickle=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = metadata.get("chunks", [])
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise IndexDimensionError("Stored vectors and metadata do not have matching rows.")
        resolved_embedder = embedder or SentenceTransformerEmbedder(
            str(metadata.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
        )
        return cls(embeddings, chunks, resolved_embedder)

    def search(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the highest cosine-similarity chunks for a non-empty question."""
        question = str(question).strip()
        if not question:
            raise ValueError("question must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        query = _normalise_rows(self.embedder.encode([question]))
        if query.shape[0] != 1 or query.shape[1] != self.embeddings.shape[1]:
            raise IndexDimensionError("Query embedding dimension does not match the index.")

        scores = self.embeddings @ query[0]
        limit = min(int(top_k), len(self.chunks))
        positions = np.argsort(-scores, kind="stable")[:limit]
        return [
            {**self.chunks[int(position)], "score": float(scores[int(position)])}
            for position in positions
        ]
