"""Agent specifications and the typed artifacts agents exchange.

An agent is configuration — role, instructions, model profile, allowed
tools and a call budget — not a separate process or model. Agents exchange
validated artifacts, never an unlimited chat transcript.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    name: str
    description: str
    model_profile: str
    system_prompt: str
    allowed_tools: set[str] = Field(default_factory=set)
    required_capabilities: set[str] = Field(default_factory=set)
    max_calls: int = Field(default=4, ge=1)


def load_agents(path: Path | str) -> dict[str, AgentSpec]:
    data = yaml.safe_load(Path(path).read_text())
    agents: dict[str, AgentSpec] = {}

    for name, spec in (data.get("agents") or {}).items():
        agents[name] = AgentSpec.model_validate({"name": name, **spec})

    return agents


# --- Typed artifacts passed between workflow stages ---------------------


class EvidenceItem(BaseModel):
    claim: str
    source: str
    excerpt: str
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchResult(BaseModel):
    evidence: list[EvidenceItem]
    unanswered_questions: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    conclusions: list[str]
    contradictions: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewResult(BaseModel):
    approved: bool
    required_changes: list[str] = Field(default_factory=list)
    unsupported_statements: list[str] = Field(default_factory=list)
