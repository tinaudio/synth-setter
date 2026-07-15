"""Validated cross-host parameter identities for one registered parameter spec.

Example:
    >>> from synth_setter.resources import as_file, param_map
    >>> with as_file(param_map("surge_xt")) as path:
    ...     joint_map = load_param_map(path)
    >>> joint_map.dawdreamer_indices()["a_amp_eg_attack"]
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from synth_setter.data.vst.clap_map import ClapParamRef, PluginFormatMap
from synth_setter.param_spec_name import ValidatedParamSpecName


class PedalboardParamRef(BaseModel):  # noqa: DOC601, DOC603
    """Index and display name from Pedalboard's flushed post-preset enumeration."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    index: int
    name: str


class DawDreamerParamRef(BaseModel):  # noqa: DOC601, DOC603
    """Index and display name from DawDreamer's post-preset enumeration."""

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
    param_spec_name: ValidatedParamSpecName
    preset_resource: str
    preset_sha256: str
    pedalboard: BackendSnapshot
    clap: BackendSnapshot
    dawdreamer: BackendSnapshot
    params: dict[str, ParamIdentity]

    @model_validator(mode="after")
    def _unique_host_indices(self) -> SynthParamMap:
        """Reject maps that alias two model parameters in either indexed host.

        :returns: This map after indexed-host uniqueness validation.
        :raises ValueError: If Pedalboard or DawDreamer indices are duplicated.
        """
        for host in ("pedalboard", "dawdreamer"):
            indices = [getattr(identity, host).index for identity in self.params.values()]
            if len(indices) != len(set(indices)):
                raise ValueError(f"duplicate {host} parameter indices")
        return self

    def clap_projection(self) -> PluginFormatMap:
        """Return the legacy CLAP-only view used by capture CSV conversion.

        :returns: CLAP projection containing every mapped parameter.
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

        :returns: Repository parameter names mapped to DawDreamer host indices.
        """
        return {name: identity.dawdreamer.index for name, identity in self.params.items()}


def load_param_map(path: Path) -> SynthParamMap:
    """Parse and validate a committed joint parameter map.

    :param path: JSON map path.
    :returns: Strict joint map parsed from ``path``.
    """
    return SynthParamMap.model_validate_json(path.read_text(encoding="utf-8"))
