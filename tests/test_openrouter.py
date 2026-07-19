"""Payload building and response parsing — no network involved."""

import json

from agentlab.agents.definitions import AnalysisResult
from agentlab.llm.openrouter import OpenRouterProvider, sanitize_schema
from agentlab.llm.types import (
    GenerationRequest,
    Message,
    Role,
    ToolCall,
    ToolSpecification,
)


def test_build_payload_basic():
    request = GenerationRequest(
        messages=[
            Message(role=Role.SYSTEM, content="You are concise."),
            Message(role=Role.USER, content="Hello"),
        ],
        max_output_tokens=100,
    )

    payload = OpenRouterProvider._build_payload("vendor/model", request)

    assert payload["model"] == "vendor/model"
    assert payload["max_tokens"] == 100
    assert payload["messages"][0] == {"role": "system", "content": "You are concise."}
    assert "tools" not in payload
    assert "response_format" not in payload


def test_build_payload_tools_and_schema():
    request = GenerationRequest(
        messages=[Message(role=Role.USER, content="Search")],
        tools=[
            ToolSpecification(
                name="search_documents",
                description="Search docs.",
                input_schema={"type": "object"},
            )
        ],
        required_output_schema={"type": "object", "properties": {}},
    )

    payload = OpenRouterProvider._build_payload("vendor/model", request)

    assert payload["tools"][0]["function"]["name"] == "search_documents"
    assert payload["response_format"]["type"] == "json_schema"


def test_build_payload_tool_round_trip_messages():
    request = GenerationRequest(
        messages=[
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="search", arguments={"query": "x"})
                ],
            ),
            Message(
                role=Role.TOOL,
                content="result",
                name="search",
                tool_call_id="c1",
            ),
        ]
    )

    payload = OpenRouterProvider._build_payload("vendor/model", request)
    assistant, tool = payload["messages"]

    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["function"]["arguments"] == json.dumps(
        {"query": "x"}
    )
    assert tool["tool_call_id"] == "c1"


def test_sanitize_schema_strips_constraints_into_description():
    schema = {
        "type": "object",
        "properties": {
            "confidence": {
                "type": "number",
                "description": "How sure.",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "items": {
                "type": "array",
                "maxItems": 5,
                "items": {"type": "string", "pattern": "^x"},
            },
        },
    }

    result = sanitize_schema(schema)
    confidence = result["properties"]["confidence"]

    assert "minimum" not in confidence
    assert "maximum" not in confidence
    assert "minimum=0.0" in confidence["description"]
    assert "How sure." in confidence["description"]
    assert "maxItems" not in result["properties"]["items"]
    assert "pattern" not in result["properties"]["items"]["items"]

    # OpenAI strict mode requirements.
    assert result["additionalProperties"] is False
    assert set(result["required"]) == {"confidence", "items"}


def test_build_payload_sanitizes_real_artifact_schema():
    """Regression: Anthropic/OpenAI strict mode rejects minimum/maximum on
    numbers, which Pydantic emits for Field(ge=..., le=...)."""
    request = GenerationRequest(
        messages=[Message(role=Role.USER, content="Analyze")],
        required_output_schema=AnalysisResult.model_json_schema(),
    )

    payload = OpenRouterProvider._build_payload("vendor/model", request)
    schema = payload["response_format"]["json_schema"]["schema"]
    wire_schema = json.dumps(schema)

    assert '"minimum"' not in wire_schema
    assert '"maximum"' not in wire_schema

    # Every object level must satisfy OpenAI strict mode, including models
    # nested under $defs.
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"].keys())
    for definition in schema.get("$defs", {}).values():
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False
            assert set(definition["required"]) == set(
                definition["properties"].keys()
            )


def test_parse_response_with_tool_calls_and_usage():
    data = {
        "id": "gen-123",
        "model": "vendor/model-v2",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "search_documents",
                                "arguments": '{"query": "rag"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.0003,
        },
    }

    response = OpenRouterProvider._parse_response(
        requested_model="vendor/model", data=data
    )

    assert response.tool_calls[0].arguments == {"query": "rag"}
    assert response.resolved_model == "vendor/model-v2"
    assert response.usage.estimated_cost == 0.0003
    assert response.provider_request_id == "gen-123"
