"""Test-only scripted provider.

Lives in tests/, not in the agentlab package: it exists solely so the
orchestration tests can assert control flow (retries, policy denials,
budget accounting) deterministically. It never produces user-facing
results and is not importable from the installed package.
"""

from __future__ import annotations

from collections import deque

from agentlab.llm.interface import LLMProvider
from agentlab.llm.types import GenerationRequest, GenerationResponse


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[GenerationResponse]) -> None:
        self._responses = deque(responses)
        self.requests: list[GenerationRequest] = []

    async def generate(
        self,
        *,
        model: str,
        request: GenerationRequest,
    ) -> GenerationResponse:
        self.requests.append(request)

        if not self._responses:
            raise RuntimeError("Scripted response queue is empty.")

        return self._responses.popleft()


def scripted_text(text: str, **overrides) -> GenerationResponse:
    """Convenience constructor for scripted responses."""
    defaults = {
        "text": text,
        "provider": "scripted",
        "requested_model": "scripted/model",
        "resolved_model": "scripted/model",
        "finish_reason": "stop",
    }
    defaults.update(overrides)
    return GenerationResponse.model_validate(defaults)
