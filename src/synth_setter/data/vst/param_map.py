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


class SurgePyParamRef(BaseModel):  # noqa: DOC601, DOC603
    """Synth-side identifier and display name from SurgePy's loaded patch."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    synth_side_id: int
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
    surgepy: SurgePyParamRef | None = None


class SynthParamMap(BaseModel):  # noqa: DOC601, DOC603
    """Immutable joint parameter map and the artifacts that establish its provenance."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    plugin: str
    param_spec_name: ValidatedParamSpecName
    preset_resource: str
    preset_sha256: str
    pedalboard: BackendSnapshot
    clap: BackendSnapshot | None = None
    dawdreamer: BackendSnapshot
    surgepy_preset_resource: str | None = None
    surgepy_preset_sha256: str | None = None
    surgepy: BackendSnapshot | None = None
    params: dict[str, ParamIdentity]

    @model_validator(mode="after")
    def _unique_host_indices(self) -> SynthParamMap:
        """Reject incomplete SurgePy provenance and aliased host identities.

        :returns: This map after provenance and indexed-host uniqueness validation.
        :raises ValueError: If SurgePy provenance is partial or host indices are duplicated.
        """
        surgepy_provenance = (
            self.surgepy,
            self.surgepy_preset_resource,
            self.surgepy_preset_sha256,
        )
        if any(value is not None for value in surgepy_provenance) and not all(
            value is not None for value in surgepy_provenance
        ):
            raise ValueError("SurgePy snapshot and preset provenance must be provided together")
        identity_flags = [identity.surgepy is not None for identity in self.params.values()]
        if identity_flags and (any(identity_flags) != (self.surgepy is not None)):
            raise ValueError("SurgePy identities require snapshot and preset provenance")
        if any(identity_flags) and not all(identity_flags):
            raise ValueError("SurgePy identities must cover every mapped parameter")
        for host in ("pedalboard", "dawdreamer"):
            indices = [getattr(identity, host).index for identity in self.params.values()]
            if len(indices) != len(set(indices)):
                raise ValueError(f"duplicate {host} parameter indices")
        surgepy_ids = [
            identity.surgepy.synth_side_id
            for identity in self.params.values()
            if identity.surgepy is not None
        ]
        if len(surgepy_ids) != len(set(surgepy_ids)):
            raise ValueError("duplicate surgepy synth-side parameter IDs")
        return self

    def clap_projection(self) -> PluginFormatMap:
        """Return the legacy CLAP-only view used by capture CSV conversion.

        :returns: CLAP projection containing every mapped parameter.
        :raises ValueError: If the synth has no CLAP build or a parameter lacks its identity.
        """
        if self.clap is None:
            raise ValueError(f"{self.plugin!r} has no CLAP provenance")
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

    def surgepy_params(self) -> dict[str, SurgePyParamRef]:
        """Return the complete repository-name to SurgePy identity projection.

        :returns: Repository parameter names mapped to native Surge identities.
        :raises ValueError: If the map lacks SurgePy provenance or any identity.
        """
        if self.surgepy is None:
            raise ValueError("parameter map has no SurgePy snapshot")
        missing = sorted(name for name, identity in self.params.items() if identity.surgepy is None)
        if missing:
            raise ValueError(f"parameters missing SurgePy identities: {', '.join(missing)}")
        return {
            name: identity.surgepy
            for name, identity in self.params.items()
            if identity.surgepy is not None
        }


def load_param_map(path: Path) -> SynthParamMap:
    """Parse and validate a committed joint parameter map.

    :param path: JSON map path.
    :returns: Strict joint map parsed from ``path``.
    """
    return SynthParamMap.model_validate_json(path.read_text(encoding="utf-8"))
