"""Corpus loading and chunking, shared by both search_documents variants.

Chunks are designed so that a retrieved excerpt is useful on its own:

- ``*.md`` files are found recursively, so a corpus can be organized in
  subfolders (document names are corpus-relative paths).
- Blocks split on blank lines, but never inside a fenced code block, and
  code blocks (fenced or indented) re-attach to the prose that precedes
  them — a code example keeps its explanation.
- Each chunk is prefixed with its most recent heading — markdown ``#``
  headings and RST underlined/overlined titles (as in the plain-text
  Python docs) are both recognized — so an excerpt carries its section
  context.
- Leading YAML frontmatter, word-free blocks (e.g. RST underlines), and
  fragments shorter than ``MIN_CHUNK_CHARS`` (navigation lines, orphaned
  signatures) are dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

MAX_CHUNK_CHARS = 2400
MIN_CHUNK_CHARS = 80

_RST_UNDERLINE = re.compile(r"""^([=\-~^"'`#*+.:_])\1{2,}\s*$""")


def _rst_heading(block: str) -> str | None:
    """Title text if the block is an RST heading (underlined, optionally
    also overlined), else None."""
    lines = block.splitlines()
    if (
        len(lines) == 2
        and _RST_UNDERLINE.match(lines[1])
        and not _RST_UNDERLINE.match(lines[0])
    ):
        return lines[0].strip()
    if (
        len(lines) == 3
        and _RST_UNDERLINE.match(lines[0])
        and _RST_UNDERLINE.match(lines[2])
    ):
        return lines[1].strip()
    return None


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[end + 4 :]
    return text


def _split_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def _attaches_to_previous(block: str, previous: str) -> bool:
    """Code blocks and continuation blocks belong with the prose above."""
    if block.startswith("```") or block[:1] in (" ", "\t"):
        return True
    return previous.rstrip().endswith(":")


def load_chunks(corpus_dir: Path) -> list[tuple[str, str]]:
    """(document, chunk) pairs for every ``*.md`` file under corpus_dir."""
    chunks: list[tuple[str, str]] = []

    for path in sorted(corpus_dir.rglob("*.md")):
        document = str(path.relative_to(corpus_dir))
        text = _strip_frontmatter(
            path.read_text(encoding="utf-8", errors="replace")
        )
        heading = ""
        pending: str | None = None

        def flush() -> None:
            nonlocal pending
            if pending is None:
                return
            if not re.search(r"\w", pending):
                pending = None
                return
            chunk = f"{heading}\n{pending}" if heading else pending
            if len(chunk) >= MIN_CHUNK_CHARS:
                chunks.append((document, chunk))
            pending = None

        for block in _split_blocks(text):
            if block.startswith("#"):
                flush()
                heading = block.lstrip("#").strip()
                continue
            rst_title = _rst_heading(block)
            if rst_title is not None:
                flush()
                heading = rst_title
                continue
            if (
                pending is not None
                and _attaches_to_previous(block, pending)
                and len(pending) + len(block) <= MAX_CHUNK_CHARS
            ):
                pending = f"{pending}\n\n{block}"
                continue
            flush()
            pending = block
        flush()

    return chunks
