"""Model registry: resolves logical profiles into provider + model.

Agents request a logical profile ("researcher", "analyst"), never a
vendor-specific model slug. Swapping models is a config edit, not a code
change.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelLimits(BaseModel):
    maximum_cost_per_call_usd: float | None = None


class ModelProfile(BaseModel):
    provider: str
    model: str
    capabilities: set[str] = Field(default_factory=set)
    limits: ModelLimits = Field(default_factory=ModelLimits)


class ModelRegistry:
    def __init__(self, profiles: dict[str, ModelProfile]) -> None:
        self._profiles = profiles

    @classmethod
    def from_yaml(cls, path: Path | str) -> ModelRegistry:
        data = yaml.safe_load(Path(path).read_text())
        profiles = {
            name: ModelProfile.model_validate(profile)
            for name, profile in (data.get("profiles") or {}).items()
        }
        return cls(profiles)

    @property
    def profiles(self) -> dict[str, ModelProfile]:
        """Read-only view, for tooling that inspects the configuration."""
        return dict(self._profiles)

    def resolve(
        self,
        profile_name: str,
        required_capabilities: set[str] | None = None,
    ) -> ModelProfile:
        try:
            profile = self._profiles[profile_name]
        except KeyError as error:
            raise ValueError(f"Unknown model profile: {profile_name}") from error

        required = required_capabilities or set()
        missing = required - profile.capabilities

        if missing:
            raise ValueError(
                f"Profile {profile_name!r} lacks capabilities: {sorted(missing)}"
            )

        return profile
