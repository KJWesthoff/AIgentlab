"""Semantic search over the corpus using local embeddings.

Splits the corpus into the same paragraphs as the keyword search, embeds
each one with a small local ONNX model (fastembed, no torch), and ranks
by cosine similarity. Embeddings are cached in a ``.vector-index.npz``
file next to the corpus, keyed by a content fingerprint, so paragraphs
are only re-embedded when the corpus or model changes.

The first run downloads the embedding model (~34 MB) to fastembed's
cache directory and needs network for that download only; everything
after that is local.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .corpus import load_chunks
from .definitions import Tool, ToolDefinition
from .registry import DEFAULT_CORPUS_DIR, EXCERPT_CHARS, SearchDocumentsInput

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CACHE_FILENAME = ".vector-index.npz"


class EmbeddingBackend(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastembedBackend:
    """Real backend; instantiating it loads (and on first use downloads)
    the ONNX model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        embed = getattr(self._model, "passage_embed", self._model.embed)
        return np.array(list(embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        embed = getattr(self._model, "query_embed", self._model.embed)
        return np.array(list(embed([text]))[0], dtype=np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class ParagraphIndex:
    def __init__(
        self,
        corpus_dir: Path,
        backend_factory=FastembedBackend,
    ) -> None:
        self._corpus_dir = corpus_dir
        self._cache_path = corpus_dir / CACHE_FILENAME
        self._backend_factory = backend_factory
        self._backend: EmbeddingBackend | None = None
        self._docs: list[str] = []
        self._texts: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._fingerprint: str | None = None

    def _get_backend(self) -> EmbeddingBackend:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    @staticmethod
    def _compute_fingerprint(paragraphs: list[tuple[str, str]]) -> str:
        digest = hashlib.sha256(EMBEDDING_MODEL.encode())
        for doc, text in paragraphs:
            digest.update(doc.encode())
            digest.update(b"\x00")
            digest.update(text.encode())
            digest.update(b"\x00")
        return digest.hexdigest()

    def _ensure_index(self) -> None:
        paragraphs = load_chunks(self._corpus_dir)
        fingerprint = self._compute_fingerprint(paragraphs)
        if self._fingerprint == fingerprint:
            return

        self._docs = [doc for doc, _ in paragraphs]
        self._texts = [text for _, text in paragraphs]

        if not paragraphs:
            self._embeddings = None
            self._fingerprint = fingerprint
            return

        cached = self._load_cache(fingerprint)
        if cached is not None:
            self._embeddings = cached
        else:
            self._embeddings = _normalize(
                self._get_backend().embed_passages(self._texts)
            )
            self._save_cache(fingerprint)
        self._fingerprint = fingerprint

    def _load_cache(self, fingerprint: str) -> np.ndarray | None:
        if not self._cache_path.exists():
            return None
        try:
            data = np.load(self._cache_path, allow_pickle=False)
            meta = json.loads(data["meta"].item())
        except Exception:
            return None
        if meta.get("fingerprint") != fingerprint:
            return None
        return data["embeddings"].astype(np.float32)

    def _save_cache(self, fingerprint: str) -> None:
        np.savez(
            self._cache_path,
            embeddings=self._embeddings,
            meta=np.array(json.dumps({"fingerprint": fingerprint})),
        )

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        self._ensure_index()
        if self._embeddings is None:
            return []

        query_vector = _normalize(self._get_backend().embed_query(query))
        similarities = self._embeddings @ query_vector
        top = np.argsort(similarities)[::-1][:limit]
        return [
            {
                "document": self._docs[i],
                "score": round(float(similarities[i]), 4),
                "excerpt": self._texts[i][:EXCERPT_CHARS],
            }
            for i in top
        ]


def build_vector_tools(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    backend_factory=FastembedBackend,
) -> dict[str, Tool]:
    index = ParagraphIndex(corpus_dir, backend_factory=backend_factory)

    def search_documents(query: str, limit: int = 5) -> dict[str, Any]:
        return {"query": query, "results": index.search(query, limit)}

    return {
        "search_documents": Tool(
            ToolDefinition(
                name="search_documents",
                description=(
                    "Semantically search the approved local document corpus. "
                    "Returns the most relevant paragraphs; phrasing does not "
                    "need to match the documents' wording exactly."
                ),
                risk="low",
                read_only=True,
                reads=["corpus"],
            ),
            SearchDocumentsInput,
            search_documents,
        )
    }
