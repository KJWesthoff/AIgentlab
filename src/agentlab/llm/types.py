"""Internal message and request types.

This is the language spoken between the orchestrator and every provider
adapter. It contains only concepts the application actually needs — no
provider-specific schema leaks past this module.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    # Only set on assistant messages that requested tool execution; adapters
    # translate this back into the provider's wire format.
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolSpecification(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class GenerationRequest(BaseModel):
    messages: list[Message]

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)

    tools: list[ToolSpecification] = Field(default_factory=list)
    required_output_schema: dict[str, Any] | None = None

    metadata: dict[str, str] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost: float | None = None


class GenerationResponse(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    provider: str
    requested_model: str
    resolved_model: str | None = None

    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)

    provider_request_id: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
