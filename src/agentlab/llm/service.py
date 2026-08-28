"""LLMService: routes a logical profile to a configured provider adapter.

Agent code calls this and never learns which vendor, model slug or auth
mechanism served the request.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from .interface import LLMProvider, StructuredGeneration
from .registry import ModelRegistry
from .types import GenerationRequest, GenerationResponse

T = TypeVar("T", bound=BaseModel)


class LLMService:
    def __init__(
        self,
        *,
        model_registry: ModelRegistry,
        providers: dict[str, LLMProvider],
    ) -> None:
        self._model_registry = model_registry
        self._providers = providers

    def _provider_for(self, provider_name: str) -> LLMProvider:
        try:
            return self._providers[provider_name]
        except KeyError as error:
            raise RuntimeError(
                f"Provider {provider_name!r} is not configured."
            ) from error

    async def generate(
        self,
        *,
        profile_name: str,
        request: GenerationRequest,
        required_capabilities: set[str] | None = None,
    ) -> GenerationResponse:
        profile = self._model_registry.resolve(profile_name, required_capabilities)
        provider = self._provider_for(profile.provider)
        return await provider.generate(model=profile.model, request=request)

    async def generate_structured(
        self,
        *,
        profile_name: str,
        request: GenerationRequest,
        response_type: type[T],
        required_capabilities: set[str] | None = None,
    ) -> StructuredGeneration[T]:
        required = {"structured_output"} | (required_capabilities or set())
        profile = self._model_registry.resolve(profile_name, required)
        provider = self._provider_for(profile.provider)
        return await provider.generate_structured(
            model=profile.model,
            request=request,
            response_type=response_type,
        )

    async def close(self) -> None:
        for provider in self._providers.values():
            await provider.close()
