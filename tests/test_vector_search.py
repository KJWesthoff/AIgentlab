"""Offline tests for the vector search index.

Uses a deterministic bag-of-words embedding backend instead of the real
ONNX model, so the suite stays free of downloads and network. The real
FastembedBackend is exercised by live runs, not here.
"""

import re

import numpy as np

from agentlab.tools.corpus import load_chunks
from agentlab.tools.vector_search import (
    CACHE_FILENAME,
    ParagraphIndex,
    build_vector_tools,
)

VOCAB = [
    "asyncio",
    "coroutine",
    "concurrency",
    "dictionary",
    "lookup",
    "sorting",
]


class BagOfWordsBackend:
    """Deterministic embeddings: one axis per vocabulary word."""

    instances = 0

    def __init__(self) -> None:
        BagOfWordsBackend.instances += 1
        self.passage_calls = 0

    def _embed(self, text: str) -> np.ndarray:
        words = re.findall(r"\w+", text.lower())
        return np.array(
            [float(words.count(term)) for term in VOCAB], dtype=np.float32
        )

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.passage_calls += 1
        return np.stack([self._embed(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


def make_corpus(tmp_path):
    (tmp_path / "async.md").write_text(
        "# Async\n\nA coroutine runs under asyncio concurrency and is "
        "scheduled cooperatively on the event loop by the runtime.\n\n"
        "Concurrency with asyncio uses coroutine scheduling so that many "
        "operations can be in flight at the same time on one thread.\n"
    )
    (tmp_path / "dicts.md").write_text(
        "# Dicts\n\nA dictionary provides constant-time lookup by key and "
        "preserves the insertion order of its entries.\n"
    )
    return tmp_path


def test_ranks_semantically_relevant_paragraph_first(tmp_path):
    corpus = make_corpus(tmp_path)
    tools = build_vector_tools(corpus, backend_factory=BagOfWordsBackend)

    result = tools["search_documents"].execute(
        {"query": "dictionary lookup", "limit": 2}
    )

    assert result["results"][0]["document"] == "dicts.md"
    assert result["results"][0]["score"] > result["results"][1]["score"]


def test_chunks_carry_heading_context_without_markup(tmp_path):
    corpus = make_corpus(tmp_path)
    chunks = [text for _, text in load_chunks(corpus)]
    assert all(not text.startswith("#") for text in chunks)
    assert any(text.startswith("Async\n") for text in chunks)


def test_code_blocks_stay_attached_to_their_prose(tmp_path):
    (tmp_path / "doc.md").write_text(
        "# Sorting\n\nSort a list with the sorted builtin:\n\n"
        "```python\nsorted(items, key=len)\n```\n\n"
        "A separate closing remark about sorting stability that stands "
        "on its own as an independent paragraph of prose.\n"
    )
    chunks = [text for _, text in load_chunks(tmp_path)]
    assert len(chunks) == 2
    assert "sorted(items, key=len)" in chunks[0]
    assert chunks[0].startswith("Sorting\n")


def test_rst_underlined_headings_provide_context(tmp_path):
    (tmp_path / "doc.md").write_text(
        "Data Structures\n***************\n\n"
        "Lists support appending, extending, and slicing, and they are "
        "the workhorse sequence type for most programs.\n"
    )
    chunks = [text for _, text in load_chunks(tmp_path)]
    assert len(chunks) == 1
    assert chunks[0].startswith("Data Structures\n")
    assert "***" not in chunks[0]


def test_short_fragments_are_filtered_out(tmp_path):
    (tmp_path / "doc.md").write_text(
        "Previous topic\n\n"
        "A real paragraph that is comfortably long enough to clear the "
        "minimum chunk size and should therefore be kept in the index.\n"
    )
    chunks = [text for _, text in load_chunks(tmp_path)]
    assert len(chunks) == 1
    assert chunks[0].startswith("A real paragraph")


def test_recursive_glob_uses_relative_document_names(tmp_path):
    sub = tmp_path / "python-docs"
    sub.mkdir()
    (sub / "intro.md").write_text(
        "A paragraph about dictionary lookup that is long enough to "
        "survive the minimum chunk-size filter applied by the loader.\n"
    )
    docs = [doc for doc, _ in load_chunks(tmp_path)]
    assert docs == ["python-docs/intro.md"]


def test_embeddings_cached_across_index_instances(tmp_path):
    corpus = make_corpus(tmp_path)

    first = ParagraphIndex(corpus, backend_factory=BagOfWordsBackend)
    first.search("asyncio", limit=1)
    assert (corpus / CACHE_FILENAME).exists()

    second = ParagraphIndex(corpus, backend_factory=BagOfWordsBackend)
    backend = second._get_backend()
    second.search("asyncio", limit=1)
    assert backend.passage_calls == 0  # loaded from cache, not re-embedded


def test_cache_invalidated_when_corpus_changes(tmp_path):
    corpus = make_corpus(tmp_path)

    index = ParagraphIndex(corpus, backend_factory=BagOfWordsBackend)
    index.search("asyncio", limit=1)

    (corpus / "sorting.md").write_text(
        "# Sort\n\nSorting orders items into a stable, predictable "
        "sequence so that later processing can rely on the order.\n"
    )
    result = index.search("sorting", limit=1)

    assert result[0]["document"] == "sorting.md"


def test_empty_corpus_returns_no_results(tmp_path):
    tools = build_vector_tools(tmp_path, backend_factory=BagOfWordsBackend)
    result = tools["search_documents"].execute({"query": "anything"})
    assert result["results"] == []
