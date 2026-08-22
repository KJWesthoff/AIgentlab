"""``save_report`` — the first tool here that changes state.

Everything else in this package reads. This one writes real files to
disk, which is what makes it the sink at the end of an attack path: the
permission graph's critical check looks for untrusted content reaching a
tool with ``read_only=False``, and until now no such tool existed.

Writes are confined to one root directory and the filename is validated
before anything is created. That is not theatre — the agent proposing
the filename may be acting on text an attacker wrote, so the path is
exactly the argument you cannot trust. Confinement bounds the blast
radius; it does not make the call safe, which is why the tool is still
gated on a human in ``policy.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .definitions import Tool, ToolDefinition

DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[3] / "data" / "reports"
MAX_REPORT_BYTES = 256_000


class SaveReportInput(BaseModel):
    filename: str = Field(
        description="Report file name, e.g. 'summary.md'. No directories."
    )
    content: str = Field(description="Markdown body to write.")


class UnsafePathError(ValueError):
    pass


def resolve_report_path(report_dir: Path, filename: str) -> Path:
    """Resolve a model-supplied filename inside ``report_dir``.

    Rejects anything that is not a plain name in the directory itself:
    absolute paths, parent traversal, nested directories and symlinks
    that would land outside. Checking the *resolved* path rather than the
    string is what makes traversal tricks fail rather than merely look
    suspicious.
    """
    name = filename.strip()
    if not name:
        raise UnsafePathError("Report filename is empty.")
    if name != Path(name).name:
        raise UnsafePathError(
            f"Report filename must be a bare file name, got {filename!r}."
        )
    if not name.endswith(".md"):
        raise UnsafePathError("Reports must be markdown (.md).")

    root = report_dir.resolve()
    candidate = (root / name).resolve()
    if candidate.parent != root:
        raise UnsafePathError(f"Refusing to write outside {root}.")
    return candidate


def make_save_report(report_dir: Path = DEFAULT_REPORT_DIR):
    def save_report(filename: str, content: str) -> dict[str, Any]:
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_REPORT_BYTES:
            raise ValueError(
                f"Report is {len(encoded)} bytes; the limit is "
                f"{MAX_REPORT_BYTES}."
            )

        path = resolve_report_path(report_dir, filename)
        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        return {
            "path": str(path),
            "bytes": len(encoded),
            "overwrote_existing": existed,
        }

    return save_report


def build_write_tools(report_dir: Path = DEFAULT_REPORT_DIR) -> dict[str, Tool]:
    return {
        "save_report": Tool(
            ToolDefinition(
                name="save_report",
                description=(
                    "Save the final report as a markdown file in the "
                    "approved reports directory. Writes to disk and "
                    "overwrites an existing file of the same name."
                ),
                risk="high",
                read_only=False,
                writes=["reports"],
                required_scope="write:reports",
            ),
            SaveReportInput,
            make_save_report(report_dir),
        )
    }
