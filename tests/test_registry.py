import pytest

from agentlab.llm.registry import ModelProfile, ModelRegistry


def make_registry() -> ModelRegistry:
    return ModelRegistry(
        {
            "researcher": ModelProfile(
                provider="openrouter",
                model="vendor/model-general",
                capabilities={"text", "tool_calling", "structured_output"},
            ),
            "economical": ModelProfile(
                provider="openrouter",
                model="vendor/model-small",
                capabilities={"text"},
            ),
        }
    )


def test_resolves_profile_with_capabilities():
    registry = make_registry()
    profile = registry.resolve("researcher", {"tool_calling"})
    assert profile.model == "vendor/model-general"


def test_unknown_profile_raises():
    registry = make_registry()
    with pytest.raises(ValueError, match="Unknown model profile"):
        registry.resolve("nonexistent")


def test_missing_capability_raises():
    registry = make_registry()
    with pytest.raises(ValueError, match="lacks capabilities"):
        registry.resolve("economical", {"tool_calling"})
