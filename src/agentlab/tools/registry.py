"""Built-in tools.

``search_documents`` is a deliberately simple keyword search over local
markdown files — enough to exercise the tool-calling path end to end.
Replace it with BM25, a database or vector retrieval later without touching
the orchestrator.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .corpus import load_chunks
from .definitions import Tool, ToolDefinition

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[3] / "data" / "corpus"
EXCERPT_CHARS = 2000


class SearchDocumentsInput(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=20)


def make_search_documents(corpus_dir: Path = DEFAULT_CORPUS_DIR):
    def search_documents(query: str, limit: int = 5) -> dict[str, Any]:
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        scored = []

        # Rank chunks across all documents. Matching more *distinct*
        # query terms outranks repeating one common term many times.
        for document, chunk in load_chunks(corpus_dir):
            lowered = chunk.lower()
            distinct = sum(1 for term in terms if term in lowered)
            if distinct == 0:
                continue
            total = sum(lowered.count(term) for term in terms)
            scored.append((distinct, total, document, chunk))

        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        results = [
            {
                "document": document,
                "matched_terms": distinct,
                "score": total,
                "excerpt": chunk[:EXCERPT_CHARS],
            }
            for distinct, total, document, chunk in scored[:limit]
        ]
        return {"query": query, "results": results}

    return search_documents


def build_default_tools(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> dict[str, Tool]:
    return {
        "search_documents": Tool(
            ToolDefinition(
                name="search_documents",
                description=(
                    "Search the approved local document corpus. Returns "
                    "excerpts from matching documents."
                ),
                risk="low",
                read_only=True,
                reads=["corpus"],
                required_scope="read:corpus",
            ),
            SearchDocumentsInput,
            make_search_documents(corpus_dir),
        )
    }
