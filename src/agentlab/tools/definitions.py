"""Typed tool registry.

Tool arguments are validated against a Pydantic model before execution —
model-produced dictionaries are never passed anywhere unvalidated. Tools
retrieve any credentials from their own environment; the model never sees
secrets.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ..llm.types import ToolSpecification


class ToolDefinition(BaseModel):
    name: str
    description: str
    risk: str = "low"
    read_only: bool = True

    # Data sources this tool touches, by logical name. Declared here so the
    # permission graph can derive where untrusted content enters the system
    # instead of guessing from the tool's name.
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)


class Tool:
    def __init__(
        self,
        definition: ToolDefinition,
        input_model: type[BaseModel],
        function: Callable[..., Any],
    ) -> None:
        self.definition = definition
        self.input_model = input_model
        self.function = function

    def to_specification(self) -> ToolSpecification:
        return ToolSpecification(
            name=self.definition.name,
            description=self.definition.description,
            input_schema=self.input_model.model_json_schema(),
        )

    def execute(self, arguments: dict[str, Any]) -> Any:
        validated = self.input_model.model_validate(arguments)
        return self.function(**validated.model_dump())
