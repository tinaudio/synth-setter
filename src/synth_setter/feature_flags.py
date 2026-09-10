"""Resolve Hydra feature-flag IDs and expose selected flags to the process.

.. code-block:: python

    resolved = FeatureFlagConfig.model_validate({"feature_flags": []})
    assert resolved.feature_flags == []
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

from omegaconf import ListConfig
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, StrictInt

__all__ = [
    "FeatureFlag",
    "FeatureFlagConfig",
    "ResolvedFeatureFlags",
    "apply_feature_flags",
]


@dataclass(frozen=True)
class FeatureFlag:
    """Describe one registered runtime feature flag.

    .. attribute :: number

        Stable integer selected in Hydra configuration.

    .. attribute :: name

        Full process environment-variable name.

    .. attribute :: description

        One-sentence behavior summary.
    """

    number: int
    name: str
    description: str


_REGISTERED_FEATURE_FLAGS: tuple[FeatureFlag, ...] = ()
_FEATURE_FLAGS: dict[int, FeatureFlag] = {
    feature_flag.number: feature_flag for feature_flag in _REGISTERED_FEATURE_FLAGS
}


def _resolve_feature_flags(value: object) -> object:
    if isinstance(value, ListConfig):
        value = list(value)
    if not isinstance(value, list):
        return value
    resolved: list[FeatureFlag] = []
    seen: set[int] = set()
    for number in value:
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError("feature flag numbers must be integers")
        if number in seen:
            raise ValueError(f"duplicate feature flag number: {number}")
        try:
            feature_flag = _FEATURE_FLAGS[number]
        except KeyError:
            raise ValueError(f"unknown feature flag number: {number}") from None
        seen.add(number)
        resolved.append(feature_flag)
    return resolved


def _feature_flag_numbers(feature_flags: list[FeatureFlag]) -> list[int]:
    return [feature_flag.number for feature_flag in feature_flags]


ResolvedFeatureFlags = Annotated[
    list[FeatureFlag],
    BeforeValidator(_resolve_feature_flags, json_schema_input_type=list[StrictInt]),
    PlainSerializer(_feature_flag_numbers, return_type=list[int]),
]


class FeatureFlagConfig(BaseModel):
    """Resolve configured integer IDs to registered feature-flag records.

    .. attribute :: model_config

        Pydantic model config sentinel.

    .. attribute :: feature_flags

        Registry records selected by the configured integer IDs.
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    feature_flags: ResolvedFeatureFlags = Field(default_factory=list)


def apply_feature_flags(cfg: Mapping[str, object]) -> FeatureFlagConfig:
    """Resolve ``cfg.feature_flags`` and set each selected environment variable.

    :param cfg: Hydra-compatible mapping containing an optional ``feature_flags`` list.
    :returns: Validated feature-flag metadata for the selected IDs.
    """
    resolved = FeatureFlagConfig.model_validate({"feature_flags": cfg.get("feature_flags", [])})
    for feature_flag in _REGISTERED_FEATURE_FLAGS:
        os.environ.pop(feature_flag.name, None)
    for feature_flag in resolved.feature_flags:
        os.environ[feature_flag.name] = "1"
    return resolved
