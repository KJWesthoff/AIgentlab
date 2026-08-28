"""Provider-neutral LLM interface.

The orchestration layer imports only ``LLMProvider`` — never OpenRouter,
OpenAI or Anthropic clients directly.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from .types import (
    GenerationRequest,
    GenerationResponse,
    Message,
    Role,
    Usage,
    sum_usage,
)

T = TypeVar("T", bound=BaseModel)

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> str:
    """Strip markdown fences and surrounding prose from a JSON reply."""
    match = _FENCE_PATTERN.search(text)
    if match:
        return match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]

    return text


class StructuredOutputError(RuntimeError):
    pass


@dataclass(frozen=True)
class StructuredGeneration(Generic[T]):
    """A validated artifact plus what producing it actually cost.

    Structured generation can take a second round-trip when the first
    reply fails schema validation. Returning the artifact alone would
    hide those tokens from the budget tracker and the live viewer, so
    the cost travels back with it: ``usage`` is the sum over every
    round-trip and ``calls`` is how many there were.
    """

    artifact: T
    usage: Usage
    resolved_model: str | None = None
    calls: int = 1


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        *,
        model: str,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate one model response."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release network resources. No-op by default."""

    async def generate_structured(
        self,
        *,
        model: str,
        request: GenerationRequest,
        response_type: type[T],
    ) -> StructuredGeneration[T]:
        """Generate a response validated against ``response_type``.

        On a validation failure the model gets exactly one retry that
        includes the validation errors, then the failure propagates.
        """
        request = request.model_copy(deep=True)
        request.required_output_schema = response_type.model_json_schema()

        response = await self.generate(model=model, request=request)

        if not response.text:
            raise StructuredOutputError("The model returned no structured text.")

        try:
            artifact = response_type.model_validate_json(extract_json(response.text))
        except (ValidationError, json.JSONDecodeError) as error:
            retry_request = request.model_copy(deep=True)
            retry_request.messages = [
                *request.messages,
                Message(role=Role.ASSISTANT, content=response.text),
                Message(
                    role=Role.USER,
                    content=(
                        "Your previous response was not valid for the required "
                        f"schema:\n{error}\n\n"
                        "Return only the corrected JSON object, nothing else."
                    ),
                ),
            ]

            retry_response = await self.generate(model=model, request=retry_request)
            if not retry_response.text:
                raise StructuredOutputError(
                    "The model returned no structured text on retry."
                ) from error

            try:
                retry_artifact = response_type.model_validate_json(
                    extract_json(retry_response.text)
                )
            except (ValidationError, json.JSONDecodeError) as retry_error:
                raise StructuredOutputError(
                    f"Structured output failed validation twice: {retry_error}"
                ) from retry_error

            return StructuredGeneration(
                artifact=retry_artifact,
                usage=sum_usage([response.usage, retry_response.usage]),
                resolved_model=retry_response.resolved_model,
                calls=2,
            )

        return StructuredGeneration(
            artifact=artifact,
            usage=response.usage,
            resolved_model=response.resolved_model,
            calls=1,
        )
