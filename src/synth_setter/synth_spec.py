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
from typing import TYPE_CHECKING, NewType

from pydantic import BaseModel, ConfigDict, model_validator

from synth_setter.param_spec_name import ParamSpecName, ValidatedParamSpecName
from synth_setter.renderer_backend import TORCHSYNTH_PLUGIN_NAME

if TYPE_CHECKING:
    from omegaconf import DictConfig

SynthName = NewType("SynthName", str)


class SynthSpec(BaseModel):  # noqa: DOC601, DOC603 — field semantics documented below.
    """One registered synth's param spec, rendering artifact, version, and preset.

    ``name`` and ``param_spec_name`` are separate so several preset variants may
    share a single ``ParamSpec``. Every shipped entry currently has them equal.

    .. attribute :: name

        Registry key, doubling as the ``configs/render`` group name.

    .. attribute :: param_spec_name

        Key into the ``ParamSpec`` registry; several synths may share one.

    .. attribute :: plugin_path

        VST3 bundle path, or the bare backend name for the in-process renderer.

    .. attribute :: plugin_state_path

        Baseline preset applied before parameter override; ``""`` when the
        backend has no preset file.

    .. attribute :: synth_version

        Version of the plugin, package, or runtime artifact implementing the synth.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    name: SynthName
    param_spec_name: ValidatedParamSpecName
    plugin_path: str
    plugin_state_path: str
    synth_version: str

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

    @classmethod
    def from_render_cfg(cls, render: DictConfig | None) -> SynthSpec | None:
        """Read synth identity out of a composed ``render`` group.

        Duck-typed so this module stays free of a runtime omegaconf import — the
        minimal-env CI install that runs ``validate_spec`` does not ship it.
        Flat render groups derive ``name`` from ``param_spec_name``; nested groups
        preserve their explicit identity.

        :param render: Composed ``render`` node, or ``None`` when unset.
        :returns: The identity the group declares, or ``None`` when it names no
            param spec (e.g. the generic ``render=vst`` scaffold).
        """
        if render is None:
            return None
        nested_synth = render.get("synth")
        if isinstance(nested_synth, cls):
            return nested_synth
        if nested_synth is not None:
            return cls.model_validate(dict(nested_synth))
        param_spec_name = render.get("param_spec_name")
        if param_spec_name is None:
            return None
        return cls(
            name=SynthName(str(param_spec_name)),
            param_spec_name=ParamSpecName(str(param_spec_name)),
            plugin_path=str(render.get("plugin_path") or ""),
            plugin_state_path=str(render.get("plugin_state_path") or ""),
            synth_version=str(render.get("synth_version") or ""),
        )


# One row per registered synth: name -> (param_spec_name, plugin_path, preset, version).
# Kept as literal tuples so ``synth-setter-introspect-plugin --register`` can
# compare and extend the table structurally without importing renderer dependencies.
_synth_rows: dict[str, tuple[str, str, str, str]] = {
    "faust_bright_organ": ("faust_bright_organ", "faust", "", "0.8.3"),
    "faust_bubble": ("faust_bubble", "faust", "", "0.8.3"),
    "faust_church_organ": ("faust_church_organ", "faust", "", "0.8.3"),
    "faust_filter_osc": ("faust_filter_osc", "faust", "", "0.8.3"),
    "surge_xt": ("surge_xt", "plugins/Surge XT.vst3", "presets/surge-base.vstpreset", "1.3.4"),
    "surge_simple": (
        "surge_simple",
        "plugins/Surge XT.vst3",
        "presets/surge-simple.vstpreset",
        "1.3.4",
    ),
    "surge_4": ("surge_4", "plugins/Surge XT.vst3", "presets/surge-mini.vstpreset", "1.3.4"),
    "obxf": ("obxf", "plugins/OB-Xf.vst3", "presets/obxf-base.vstpreset", "1.0.3"),
    "torchsynth_adsr": ("torchsynth_adsr", "torchsynth", "", "1.0.2"),
    "torchsynth_full": ("torchsynth_full", "torchsynth", "", "1.0.2"),
    "torchsynth_simple": ("torchsynth_simple", "torchsynth", "", "1.0.2"),
}

SYNTHS: Mapping[SynthName, SynthSpec] = MappingProxyType(
    {
        SynthName(name): SynthSpec(
            name=SynthName(name),
            param_spec_name=ParamSpecName(param_spec_name),
            plugin_path=plugin_path,
            plugin_state_path=preset,
            synth_version=synth_version,
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


class SynthIdentityConfig(BaseModel):  # noqa: DOC601, DOC603 — field semantics documented below.
    """Composed root ``synth`` group: an identity-only projection of one ``SYNTHS`` row.

    .. attribute :: name

        Registry key, doubling as the ``configs/synth`` group name.

    .. attribute :: param_spec_name

        Key into the ``ParamSpec`` registry; must match the registry row.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    name: SynthName
    param_spec_name: ValidatedParamSpecName

    @model_validator(mode="after")
    def _matches_registry_row(self) -> SynthIdentityConfig:
        """Reject an identity absent from, or contradicting, the ``SYNTHS`` table.

        :returns: This identity when it restates its registry row exactly.
        :raises ValueError: The name is unregistered or the spec disagrees.
        """
        spec = SYNTHS.get(self.name)
        if spec is None:
            raise ValueError(f"synth {self.name!r} is not registered in SYNTHS")
        if self.param_spec_name != spec.param_spec_name:
            raise ValueError(
                f"synth {self.name!r} declares param_spec_name={self.param_spec_name!r} "
                f"but the registry row says {spec.param_spec_name!r}"
            )
        return self


def validate_synth_identity(cfg: DictConfig) -> SynthSpec | None:
    """Fail fast when a composed root config contradicts the selected synth.

    Duck-typed like :meth:`SynthSpec.from_render_cfg` so the module stays free
    of a runtime omegaconf import. Reading config values resolves their
    interpolations, so a CLI-forced ``datamodule.param_spec_name`` literal that
    skews from the synth selection surfaces here instead of as a silent width
    mismatch; a malformed or registry-contradicting ``synth`` node propagates
    ``pydantic.ValidationError``, and a ``synth`` node whose interpolations
    dangle (e.g. ``dataset.yaml`` without a synth-bearing render group)
    propagates the interpolation error.

    :param cfg: Root composed config (``train.yaml`` / ``eval.yaml`` shape).
    :returns: The selected synth's registry row, or ``None`` when no ``synth``
        group is composed (audio/legacy runs).
    :raises ValueError: The datamodule's resolved ``param_spec_name`` disagrees
        with the ``synth`` selection.
    """
    synth_node = cfg.get("synth")
    if synth_node is None:
        return None
    identity = SynthIdentityConfig.model_validate(dict(synth_node))
    datamodule = cfg.get("datamodule")
    datamodule_spec = None if datamodule is None else datamodule.get("param_spec_name")
    if datamodule_spec is not None and str(datamodule_spec) != identity.param_spec_name:
        raise ValueError(
            f"datamodule.param_spec_name={datamodule_spec!r} disagrees with "
            f"synth={identity.name!r} (param_spec_name={identity.param_spec_name!r})"
        )
    return resolve_synth(identity.name)
