"""OpenRouter adapter.

OpenRouter exposes an OpenAI-compatible Chat Completions endpoint with
bearer-token auth. All OpenRouter-specific wire format lives here; nothing
above this module knows OpenRouter exists.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .interface import LLMProvider
from .types import (
    GenerationRequest,
    GenerationResponse,
    Message,
    Role,
    ToolCall,
    Usage,
)


class ProviderTemporaryError(RuntimeError):
    pass


# Constraint keywords that strict structured-output modes (Anthropic, OpenAI)
# reject. They are stripped from the wire schema and folded into the field
# description; Pydantic still enforces them when the response is parsed.
_UNSUPPORTED_SCHEMA_KEYWORDS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
)


def sanitize_schema(schema: Any) -> Any:
    """Make a Pydantic-emitted JSON schema acceptable to strict modes.

    Two transformations, applied recursively:
    - remove constraint keywords providers reject, folding them into the
      field description (Pydantic still enforces them at parse time);
    - on every object, set ``additionalProperties: false`` and mark all
      declared properties as required, which OpenAI strict mode demands.
    """
    if isinstance(schema, list):
        return [sanitize_schema(item) for item in schema]

    if not isinstance(schema, dict):
        return schema

    stripped = {
        key: schema[key] for key in _UNSUPPORTED_SCHEMA_KEYWORDS if key in schema
    }
    result = {
        key: sanitize_schema(value)
        for key, value in schema.items()
        if key not in _UNSUPPORTED_SCHEMA_KEYWORDS
    }

    if stripped:
        constraints = ", ".join(f"{k}={v}" for k, v in stripped.items())
        description = result.get("description", "")
        separator = " " if description else ""
        result["description"] = f"{description}{separator}(Constraints: {constraints})"

    if result.get("type") == "object" and "properties" in result:
        result["additionalProperties"] = False
        result["required"] = list(result["properties"].keys())

    return result


class OpenRouterProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        application_name: str = "agentlab",
        application_url: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": application_name,
        }

        if application_url:
            headers["HTTP-Referer"] = application_url

        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout_seconds,
        )

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type(
            (httpx.TimeoutException, ProviderTemporaryError)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self,
        *,
        model: str,
        request: GenerationRequest,
    ) -> GenerationResponse:
        payload = self._build_payload(model, request)

        response = await self._client.post("/chat/completions", json=payload)

        if response.status_code == 429 or response.status_code >= 500:
            raise ProviderTemporaryError(
                f"Temporary OpenRouter error: {response.status_code}"
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"OpenRouter request failed: "
                f"{response.status_code} {response.text}"
            ) from error

        return self._parse_response(requested_model=model, data=response.json())

    @staticmethod
    def _serialize_message(message: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.role.value,
            "content": message.content,
        }

        if message.name:
            payload["name"] = message.name

        if message.role is Role.TOOL and message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id

        if message.role is Role.ASSISTANT and message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
            if not message.content:
                payload["content"] = None

        return payload

    @classmethod
    def _build_payload(
        cls,
        model: str,
        request: GenerationRequest,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                cls._serialize_message(message) for message in request.messages
            ],
            "temperature": request.temperature,
            # Ask OpenRouter to report token usage and cost on every response
            # so the budget tracker can meter real spend.
            "usage": {"include": True},
        }

        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens

        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]

        if request.required_output_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": sanitize_schema(request.required_output_schema),
                },
            }

        return payload

    @staticmethod
    def _parse_response(
        *,
        requested_model: str,
        data: dict[str, Any],
    ) -> GenerationResponse:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Provider response contained no choices.")

        choice = choices[0]
        message = choice.get("message") or {}

        tool_calls: list[ToolCall] = []

        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            raw_arguments = function.get("arguments", "{}")

            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "Model returned invalid tool arguments."
                ) from error

            tool_calls.append(
                ToolCall(
                    id=call.get("id", ""),
                    name=function.get("name", ""),
                    arguments=arguments,
                )
            )

        usage_data = data.get("usage") or {}

        return GenerationResponse(
            text=message.get("content"),
            tool_calls=tool_calls,
            provider="openrouter",
            requested_model=requested_model,
            resolved_model=data.get("model"),
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                input_tokens=usage_data.get("prompt_tokens"),
                output_tokens=usage_data.get("completion_tokens"),
                total_tokens=usage_data.get("total_tokens"),
                estimated_cost=usage_data.get("cost"),
            ),
            provider_request_id=data.get("id"),
        )
