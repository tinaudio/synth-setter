"""Single authoring point for each registered synth's identity.

A synth is identified by three facts: which ``ParamSpec`` describes its
parameters, which plugin renders it, and which baseline preset that mapping was
captured against. Those facts were previously restated across the two registry
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
    """One registered synth's identity: param spec, plugin, and baseline preset.

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
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    name: SynthName
    param_spec_name: ValidatedParamSpecName
    plugin_path: str
    plugin_state_path: str

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
        )


# One row per registered synth: name -> (param_spec_name, plugin_path, preset).
# Kept as flat one-line literals because ``synth-setter-introspect-plugin --register``
# extends this dict by line anchor (``registration.synths_with_spec``); a nested
# constructor call would reflow under ruff-format and break that transform.
_synth_rows: dict[str, tuple[str, str, str]] = {
    "faust_bright_organ": ("faust_bright_organ", "faust", ""),
    "faust_bubble": ("faust_bubble", "faust", ""),
    "faust_church_organ": ("faust_church_organ", "faust", ""),
    "faust_filter_osc": ("faust_filter_osc", "faust", ""),
    "surge_xt": ("surge_xt", "plugins/Surge XT.vst3", "presets/surge-base.vstpreset"),
    "surge_simple": ("surge_simple", "plugins/Surge XT.vst3", "presets/surge-simple.vstpreset"),
    "surge_4": ("surge_4", "plugins/Surge XT.vst3", "presets/surge-mini.vstpreset"),
    "obxf": ("obxf", "plugins/OB-Xf.vst3", "presets/obxf-base.vstpreset"),
    "torchsynth_adsr": ("torchsynth_adsr", "torchsynth", ""),
    "torchsynth_full": ("torchsynth_full", "torchsynth", ""),
    "torchsynth_simple": ("torchsynth_simple", "torchsynth", ""),
}

SYNTHS: Mapping[SynthName, SynthSpec] = MappingProxyType(
    {
        SynthName(name): SynthSpec(
            name=SynthName(name),
            param_spec_name=ParamSpecName(param_spec_name),
            plugin_path=plugin_path,
            plugin_state_path=preset,
        )
        for name, (param_spec_name, plugin_path, preset) in _synth_rows.items()
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
