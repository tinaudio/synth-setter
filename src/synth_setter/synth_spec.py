"""Single authoring point for each registered synth's identity.

A synth is identified by its ``ParamSpec``, rendering artifact and version, and
baseline preset. Those facts were previously restated across the two registry
dicts, the ``configs/render`` groups, the registration scaffolder, and the
packaged parameter maps, with nothing cross-checking them.

Interpreter-only, like ``param_spec_name`` and ``renderer_backend``: it holds no
``ParamSpec`` and imports no ``synth_setter.data.vst`` module, so
``pipeline.schemas.spec`` can depend on it without pulling pedalboard onto the
launcher's import path.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NewType

from pydantic import BaseModel, ConfigDict, model_validator

from synth_setter.param_spec_name import ParamSpecName, ValidatedParamSpecName
from synth_setter.renderer_backend import FAUST_REGISTRY_PREFIX, TORCHSYNTH_PLUGIN_NAME

if TYPE_CHECKING:
    from omegaconf import DictConfig

SynthName = NewType("SynthName", str)
type SynthFormat = Literal["faust", "pyfdn", "surgepy", "torchsynth", "vst3"]

_FAUST_SOURCE_SHA256 = {
    "faust_bright_organ": "a1bf9f6e45ebbf78dd11fc18603cda048a91a778af1ad79683339b1951813465",
    "faust_bubble": "731727e725ac0336a897c18df4e8b73f1e75c3d8add40a978efb1d95f88db23c",
    "faust_church_organ": "c753731f4053210d42757acb179010185e91d37fb56a8b45e093222be688b512",
    "faust_filter_osc": "6ad65d28d787f08a3fa66eb4de7d4091be8d2267ad1e9edc200618effbbe588c",
    "faust_kronecker_fdn": "b4fdabfaa3e2220bc8182f0f3a2ac1601baf6968731adfcfb5d795d6776918ba",
    "faust_shimmer_fdn": "30b485b40002bc721101c9c50ec4d84f1d96740db20df0273835d543f168a4df",
}


def validate_faust_registry_reference(reference: str, param_spec_name: str) -> ParamSpecName:
    """Validate and resolve an in-process Faust source reference.

    :param reference: Canonical ``registry://faust/<registered-source-name>`` URI.
    :param param_spec_name: Selected parameter-spec and source identity.
    :returns: Registered Faust source identity named by the reference.
    :raises ValueError: The URI is malformed, unknown, or mismatches ``param_spec_name``.
    """
    if not reference.startswith(FAUST_REGISTRY_PREFIX):
        raise ValueError(
            "Faust registry reference must be registry://faust/<registered-source-name>"
        )
    identity = reference.removeprefix(FAUST_REGISTRY_PREFIX)
    if not identity or any(character in identity for character in "/?#"):
        raise ValueError(
            "Faust registry reference must be registry://faust/<registered-source-name>"
        )
    if identity not in _FAUST_SOURCE_SHA256:
        raise ValueError(f"Faust source {identity!r} is not registered")
    if identity != param_spec_name:
        raise ValueError(
            f"Faust registry reference selects {identity!r} but "
            f"param_spec_name is {param_spec_name!r}"
        )
    return ParamSpecName(identity)


def _is_registry_reference(reference: object) -> bool:
    """Return whether an input declares the registry URI scheme.

    :param reference: Candidate synth artifact reference.
    :returns: Whether the value begins with the registry scheme, case-insensitively.
    """
    return isinstance(reference, str) and reference.casefold().startswith("registry:")


def _legacy_synth_format(plugin_path: str) -> SynthFormat:
    """Infer a non-Faust representation recorded before ``format`` was persisted.

    :param plugin_path: Historical plugin path or in-process backend sentinel.
    :returns: Representation implied by the historical path.
    """
    if plugin_path in {"pyfdn", "surgepy", "torchsynth"}:
        return plugin_path  # type: ignore[return-value]
    return "vst3"


class SynthSpec(BaseModel):  # noqa: DOC601, DOC603 — field semantics documented below.
    """One registered synth's param spec, rendering artifact, version, and preset.

    ``name`` and ``param_spec_name`` are separate so several rendering variants
    may share a single ``ParamSpec`` (the surgepy rows do).

    .. attribute :: name

        Registry key, doubling as the root ``configs/synth`` group name.

    .. attribute :: param_spec_name

        Key into the ``ParamSpec`` registry; several synths may share one.

    .. attribute :: plugin_path

        VST3 bundle path, in-process backend sentinel, or registered Faust source URI.

    .. attribute :: plugin_state_path

        Baseline preset applied before parameter override; ``""`` when the
        backend has no preset file.

    .. attribute :: synth_version

        Version of the synth source, plugin, or package.

    .. attribute :: source_sha256

        Checked-in source digest for Faust identities; absent for other formats.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    name: SynthName
    param_spec_name: ValidatedParamSpecName
    format: SynthFormat = "vst3"
    plugin_path: str
    plugin_state_path: str
    synth_version: str
    source_sha256: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_identity(cls, data: object) -> object:
        """Derive omitted formats from legacy sentinels or a Faust registry URI.

        :param data: Raw identity input.
        :returns: Input with its representation format filled when derivable.
        :raises ValueError: A Faust sentinel appears outside the exact legacy render pair.
        """
        if not isinstance(data, dict) or "format" in data:
            return data
        normalized = data.copy()
        plugin_path = normalized.get("plugin_path")
        if plugin_path == "faust":
            raise ValueError("plugin_path='faust' requires the legacy DawDreamer Faust contract")
        if not isinstance(plugin_path, str):
            return normalized
        if _is_registry_reference(plugin_path):
            param_spec_name = normalized.get("param_spec_name")
            if not isinstance(param_spec_name, str):
                return normalized
            validate_faust_registry_reference(plugin_path, param_spec_name)
            normalized["format"] = "faust"
            return normalized
        normalized["format"] = _legacy_synth_format(plugin_path)
        return normalized

    @model_validator(mode="after")
    def _version_must_not_be_blank(self) -> SynthSpec:
        """Reject an identity without a meaningful artifact version.

        :returns: This identity when the version contains non-whitespace text.
        :raises ValueError: The version is blank.
        """
        if not self.synth_version.strip():
            raise ValueError("synth_version must not be blank")
        return self

    @model_validator(mode="after")
    def _faust_identity_is_checked_in_source(self) -> SynthSpec:
        """Require a registered or legacy-pathless Faust source with its exact digest.

        :returns: This identity when its source provenance is coherent.
        :raises ValueError: Source provenance is present on another format or Faust provenance does
            not match a registered checked-in source.
        """
        if self.format != "faust":
            if self.plugin_path == "faust":
                raise ValueError("legacy Faust plugin sentinel requires format='faust'")
            if _is_registry_reference(self.plugin_path):
                validate_faust_registry_reference(self.plugin_path, self.param_spec_name)
                raise ValueError("a Faust registry reference requires format='faust'")
            if self.source_sha256 is not None:
                raise ValueError("source_sha256 is supported only for format='faust'")
            return self
        if self.plugin_path:
            validate_faust_registry_reference(self.plugin_path, self.param_spec_name)
        if self.plugin_state_path:
            raise ValueError("format='faust' does not accept plugin_state_path")
        expected = _FAUST_SOURCE_SHA256.get(self.param_spec_name)
        if expected is None or self.source_sha256 != expected:
            raise ValueError(
                f"format='faust' requires the registered source_sha256 for "
                f"param_spec_name={self.param_spec_name!r}"
            )
        return self

    @model_validator(mode="after")
    def _torchsynth_has_no_preset(self) -> SynthSpec:
        """Reject a preset path on the backend that renders without a plugin host.

        :returns: This identity, unchanged, when the pairing is coherent.
        :raises ValueError: The in-process backend was given a preset path.
        """
        if self.plugin_path == TORCHSYNTH_PLUGIN_NAME and self.plugin_state_path:
            raise ValueError(
                f"{TORCHSYNTH_PLUGIN_NAME} renders in-process and has no preset file, "
                f"but plugin_state_path is {self.plugin_state_path!r}"
            )
        return self


# One row per registered synth: name -> (param_spec_name, plugin_path, preset, version).
# Kept as literal tuples so ``synth-setter-introspect-plugin --register`` can
# compare and extend the table structurally without importing renderer dependencies.
_synth_rows: dict[str, tuple[str, str, str, str]] = {
    "cardinal": (
        "cardinal",
        "plugins/CardinalSynth.vst3",
        "presets/cardinal-base.vstpreset",
        "0.26.2",
    ),
    "faust_bright_organ": (
        "faust_bright_organ",
        "registry://faust/faust_bright_organ",
        "",
        "1",
    ),
    "faust_bubble": ("faust_bubble", "registry://faust/faust_bubble", "", "1"),
    "faust_church_organ": (
        "faust_church_organ",
        "registry://faust/faust_church_organ",
        "",
        "1",
    ),
    "faust_filter_osc": (
        "faust_filter_osc",
        "registry://faust/faust_filter_osc",
        "",
        "1",
    ),
    "faust_kronecker_fdn": (
        "faust_kronecker_fdn",
        "registry://faust/faust_kronecker_fdn",
        "",
        "1",
    ),
    "faust_shimmer_fdn": (
        "faust_shimmer_fdn",
        "registry://faust/faust_shimmer_fdn",
        "",
        "1.0",
    ),
    "surge_xt": ("surge_xt", "plugins/Surge XT.vst3", "presets/surge-base.vstpreset", "1.3.4"),
    "surge_simple": (
        "surge_simple",
        "plugins/Surge XT.vst3",
        "presets/surge-simple.vstpreset",
        "1.3.4",
    ),
    "surge_4": ("surge_4", "plugins/Surge XT.vst3", "presets/surge-mini.vstpreset", "1.3.4"),
    "surge_xt_surgepy": ("surge_xt", "surgepy", "presets/surge-base.fxp", "1.3.master.f7b97c68"),
    "surge_simple_surgepy": (
        "surge_simple",
        "surgepy",
        "presets/surge-simple.fxp",
        "1.3.master.f7b97c68",
    ),
    "surge_4_surgepy": (
        "surge_4",
        "surgepy",
        "presets/surge-mini.fxp",
        "1.3.master.f7b97c68",
    ),
    "obxf": ("obxf", "plugins/OB-Xf.vst3", "presets/obxf-base.vstpreset", "1.0.3"),
    "pyfdn_gotz_n8_mono_fixed_delays": (
        "pyfdn_gotz_n8_mono_fixed_delays",
        "pyfdn",
        "",
        "0.4.2",
    ),
    "pyfdn_gotz_n8_mono_fixed_delays_givens": (
        "pyfdn_gotz_n8_mono_fixed_delays_givens",
        "pyfdn",
        "",
        "0.4.2",
    ),
    "pyfdn_gotz_n8_mono_learned_delays": (
        "pyfdn_gotz_n8_mono_learned_delays",
        "pyfdn",
        "",
        "0.4.2",
    ),
    "pyfdn_gotz_n8_mono_learned_delays_givens": (
        "pyfdn_gotz_n8_mono_learned_delays_givens",
        "pyfdn",
        "",
        "0.4.2",
    ),
    "pyfdn_n8_mono_householder": (
        "pyfdn_n8_mono_householder",
        "pyfdn",
        "",
        "0.4.2",
    ),
    "pyfdn_n8_mono_householder_vector": (
        "pyfdn_n8_mono_householder_vector",
        "pyfdn",
        "",
        "0.4.2",
    ),
    "pyfdn_n8_mono_kronecker": (
        "pyfdn_n8_mono_kronecker",
        "pyfdn",
        "",
        "0.4.2",
    ),
    "pyfdn_pitchshift_n8_mono_householder": (
        "pyfdn_pitchshift_n8_mono_householder",
        "pyfdn",
        "",
        "0.4.2",
    ),
    "pyfdn_diffvox": ("pyfdn_diffvox", "pyfdn", "", "0.4.2"),
    "torchsynth_adsr": ("torchsynth_adsr", "torchsynth", "", "1.0.2"),
    "torchsynth_full": ("torchsynth_full", "torchsynth", "", "1.0.2"),
    "torchsynth_simple": ("torchsynth_simple", "torchsynth", "", "1.0.2"),
    "ultramaster_kr106": (
        "ultramaster_kr106",
        "plugins/Ultramaster KR-106.vst3",
        "presets/ultramaster_kr106-base.vstpreset",
        "2.5.13",
    ),
    "ultramaster_kr106_onehot": (
        "ultramaster_kr106_onehot",
        "plugins/Ultramaster KR-106.vst3",
        "presets/ultramaster_kr106-base.vstpreset",
        "2.5.13",
    ),
    "ultramaster_kr106_single_note": (
        "ultramaster_kr106_single_note",
        "plugins/Ultramaster KR-106.vst3",
        "presets/ultramaster_kr106_single_note-base.vstpreset",
        "2.5.13",
    ),
}

SYNTHS: Mapping[SynthName, SynthSpec] = MappingProxyType(
    {
        SynthName(name): SynthSpec(
            name=SynthName(name),
            param_spec_name=ParamSpecName(param_spec_name),
            format=(
                "faust" if name in _FAUST_SOURCE_SHA256 else _legacy_synth_format(plugin_path)
            ),
            plugin_path=plugin_path,
            plugin_state_path=preset,
            synth_version=synth_version,
            source_sha256=_FAUST_SOURCE_SHA256.get(name),
        )
        for name, (param_spec_name, plugin_path, preset, synth_version) in _synth_rows.items()
    }
)


def resolve_synth(name: SynthName) -> SynthSpec:
    """Resolve a registry key to its identity.

    :param name: Registry key.
    :returns: The registered identity, without copying it.
    :raises KeyError: If the name is not registered.
    """
    try:
        return SYNTHS[name]
    except KeyError:
        raise KeyError(name) from None


def validate_synth_identity(cfg: DictConfig) -> SynthSpec | None:
    """Fail fast when a composed root config contradicts the selected synth.

    Duck-typed so the module stays free of a runtime omegaconf import — the
    minimal-env CI install that runs ``validate_spec`` does not ship it. The
    ``name``, ``param_spec_name``, and ``format`` fields are pinned to the
    ``SYNTHS`` row; ``plugin_path``, ``plugin_state_path``, and ``synth_version``
    stay per-run overridable (relocated bundles, stub plugins in tests). A
    CLI-forced ``datamodule.param_spec_name`` literal that skews from the
    synth selection surfaces here instead of as a silent width mismatch; a
    malformed ``synth`` node propagates ``pydantic.ValidationError``.

    :param cfg: Root composed config (``train.yaml`` / ``eval.yaml`` shape).
    :returns: The composed synth identity, or ``None`` when no ``synth``
        group is composed (audio/legacy runs).
    :raises ValueError: The name is unregistered, the spec contradicts the
        registry row, or the datamodule's resolved ``param_spec_name``
        disagrees with the ``synth`` selection.
    """
    synth_node = cfg.get("synth")
    if synth_node is None:
        return None
    spec = SynthSpec.model_validate(dict(synth_node))
    row = SYNTHS.get(spec.name)
    if row is None:
        raise ValueError(f"synth {spec.name!r} is not registered in SYNTHS")
    if spec.param_spec_name != row.param_spec_name:
        raise ValueError(
            f"synth {spec.name!r} declares param_spec_name={spec.param_spec_name!r} "
            f"but the registry row says {row.param_spec_name!r}"
        )
    if spec.format != row.format:
        alternatives = [
            candidate.name
            for candidate in SYNTHS.values()
            if candidate.param_spec_name == spec.param_spec_name
            and candidate.format == spec.format
        ]
        suggestion = f"; select synth={alternatives[0]}" if alternatives else ""
        raise ValueError(
            f"synth {spec.name!r} declares format={spec.format!r} "
            f"but the registry row says {row.format!r}{suggestion}"
        )
    datamodule = cfg.get("datamodule")
    datamodule_spec = None if datamodule is None else datamodule.get("param_spec_name")
    if datamodule_spec is not None and str(datamodule_spec) != spec.param_spec_name:
        raise ValueError(
            f"datamodule.param_spec_name={datamodule_spec!r} disagrees with "
            f"synth={spec.name!r} (param_spec_name={spec.param_spec_name!r})"
        )
    return spec
