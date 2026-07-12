"""Stable parameter mapping for DawDreamer's plugin descriptions."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict
from typing import cast


def dawdreamer_parameter_key(name: str) -> str:
    """Convert a DawDreamer display name to the repository's parameter-key form.

    :param name: DawDreamer display name.
    :returns: Lowercase underscore-separated parameter key.
    """
    return "_".join(name.lower().replace("-", " ").split())


class DawDreamerParamRef(BaseModel):  # noqa: DOC601, DOC603 — Pydantic fields carry metadata.
    """Host index and metadata for one DawDreamer parameter."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    index: int
    name: str
    label: str
    category: str
    default_value: float
    value_strings: tuple[str, ...] = ()


class DawDreamerPluginMap(BaseModel):  # noqa: DOC601, DOC603 — Pydantic fields carry metadata.
    """Serializable parameter map produced from DawDreamer's introspector."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    plugin: str
    params: dict[str, DawDreamerParamRef]


def build_dawdreamer_map(plugin_path: Path, descriptions: list[dict[str, object]]) -> DawDreamerPluginMap:
    """Build a name-keyed map from ``PluginProcessor.get_parameters_description`` output.

    :param plugin_path: Loaded plugin bundle path.
    :param descriptions: DawDreamer's parameter description dictionaries.
    :returns: Strict, JSON-serializable parameter map.
    """
    params = {
        dawdreamer_parameter_key(str(description["name"])): DawDreamerParamRef(
            index=cast(int, description["index"]),
            name=str(description["name"]),
            label=str(description.get("label", "")),
            category=str(description.get("category", "unknown")),
            default_value=cast(float, description.get("defaultValue", 0.0)),
            value_strings=tuple(
                str(value) for value in cast(list[object], description.get("valueStrings", []))
            ),
        )
        for description in descriptions
    }
    return DawDreamerPluginMap(plugin=str(plugin_path), params=params)


def load_dawdreamer_map(path: Path) -> DawDreamerPluginMap:
    """Load a DawDreamer parameter map from JSON.

    :param path: JSON map path.
    :returns: Validated DawDreamer parameter map.
    """
    return DawDreamerPluginMap.model_validate_json(path.read_text())
