"""Validated cross-host parameter identities for one registered parameter spec."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from synth_setter.data.vst.clap_map import ClapParamRef, PluginFormatMap


class PedalboardParamRef(BaseModel):  # noqa: DOC601, DOC603
    """Pedalboard parameter identity."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    index: int
    name: str


class DawDreamerParamRef(BaseModel):  # noqa: DOC601, DOC603
    """DawDreamer parameter identity."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    index: int
    name: str


class BackendSnapshot(BaseModel):  # noqa: DOC601, DOC603
    """Plugin version and enumeration size observed in one host."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    plugin_version: str
    parameter_count: int


class ParamIdentity(BaseModel):  # noqa: DOC601, DOC603
    """One repository parameter's identities in every supported host."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    pedalboard: PedalboardParamRef
    clap: ClapParamRef | None
    dawdreamer: DawDreamerParamRef


class SynthParamMap(BaseModel):  # noqa: DOC601, DOC603
    """Immutable joint parameter map and the artifacts that establish its provenance."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    plugin: str
    param_spec_name: str
    preset_resource: str
    preset_sha256: str
    pedalboard: BackendSnapshot
    clap: BackendSnapshot
    dawdreamer: BackendSnapshot
    params: dict[str, ParamIdentity]

    @model_validator(mode="after")
    def _unique_host_indices(self) -> SynthParamMap:
        """Reject maps that alias two model parameters in either indexed host.

        :returns: This validated map.
        :raises ValueError: If Pedalboard or DawDreamer indices are duplicated.
        """
        for host in ("pedalboard", "dawdreamer"):
            indices = [getattr(identity, host).index for identity in self.params.values()]
            if len(indices) != len(set(indices)):
                raise ValueError(f"duplicate {host} parameter indices")
        return self

    def clap_projection(self) -> PluginFormatMap:
        """Return the legacy CLAP-only view used by capture CSV conversion.

        :returns: Complete CLAP projection.
        :raises ValueError: If any parameter lacks a CLAP identity.
        """
        missing = sorted(name for name, identity in self.params.items() if identity.clap is None)
        if missing:
            raise ValueError(f"parameters missing CLAP identities: {', '.join(missing)}")
        return PluginFormatMap(
            plugin=self.plugin,
            version=self.clap.plugin_version,
            params={name: identity.clap for name, identity in self.params.items() if identity.clap},
        )

    def dawdreamer_indices(self) -> dict[str, int]:
        """Return strict repository-name to DawDreamer-index dispatch.

        :returns: Immutable-map-derived dispatch dictionary.
        """
        return {name: identity.dawdreamer.index for name, identity in self.params.items()}


def load_param_map(path: Path) -> SynthParamMap:
    """Parse and validate a committed joint parameter map.

    :param path: JSON map path.
    :returns: Validated joint map.
    """
    return SynthParamMap.model_validate_json(path.read_text(encoding="utf-8"))
