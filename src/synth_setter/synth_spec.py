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

        Reads the ``synth`` group the render configs compose, falling back to the
        pre-nesting flat keys so a hand-written or archived config still resolves.
        In the flat form ``name`` mirrors ``param_spec_name``, which those configs
        do not state separately.

        Duck-typed so this module stays free of a runtime omegaconf import — the
        minimal-env CI install that runs ``validate_spec`` does not ship it.

        :param render: Composed ``render`` node, or ``None`` when unset.
        :returns: The identity the group declares, or ``None`` when it names no
            synth (e.g. the generic ``render=vst`` scaffold).
        """
        if render is None:
            return None
        synth = render.get("synth")
        if synth is not None:
            return cls(
                name=SynthName(str(synth["name"])),
                param_spec_name=ParamSpecName(str(synth["param_spec_name"])),
                plugin_path=str(synth["plugin_path"]),
                plugin_state_path=str(synth["plugin_state_path"]),
            )
        param_spec_name = render.get("param_spec_name")
        if param_spec_name is None:
            return None
        return cls(
            name=SynthName(str(param_spec_name)),
            param_spec_name=ParamSpecName(str(param_spec_name)),
            plugin_path=str(render.get("plugin_path") or ""),
            plugin_state_path=str(render.get("plugin_state_path") or ""),
        )


def _synth(name: str, param_spec_name: str, plugin_path: str, preset: str) -> SynthSpec:
    """Build one table entry, keeping the literal rows below readable.

    :param name: Registry key and render group name.
    :param param_spec_name: Key into the ``ParamSpec`` registry.
    :param plugin_path: VST3 bundle path or bare backend name.
    :param preset: Baseline preset path; ``""`` for preset-less backends.
    :returns: The validated identity.
    """
    return SynthSpec(
        name=SynthName(name),
        param_spec_name=ParamSpecName(param_spec_name),
        plugin_path=plugin_path,
        plugin_state_path=preset,
    )


_SURGE_XT_PLUGIN = "plugins/Surge XT.vst3"

_synths: dict[SynthName, SynthSpec] = {
    SynthName("surge_xt"): _synth(
        "surge_xt", "surge_xt", _SURGE_XT_PLUGIN, "presets/surge-base.vstpreset"
    ),
    SynthName("surge_simple"): _synth(
        "surge_simple", "surge_simple", _SURGE_XT_PLUGIN, "presets/surge-simple.vstpreset"
    ),
    SynthName("surge_4"): _synth(
        "surge_4", "surge_4", _SURGE_XT_PLUGIN, "presets/surge-mini.vstpreset"
    ),
    SynthName("obxf"): _synth("obxf", "obxf", "plugins/OB-Xf.vst3", "presets/obxf-base.vstpreset"),
    SynthName("torchsynth_adsr"): _synth(
        "torchsynth_adsr", "torchsynth_adsr", TORCHSYNTH_PLUGIN_NAME, ""
    ),
    SynthName("torchsynth_full"): _synth(
        "torchsynth_full", "torchsynth_full", TORCHSYNTH_PLUGIN_NAME, ""
    ),
    SynthName("torchsynth_simple"): _synth(
        "torchsynth_simple", "torchsynth_simple", TORCHSYNTH_PLUGIN_NAME, ""
    ),
}

SYNTHS: Mapping[SynthName, SynthSpec] = MappingProxyType(_synths)


def resolve_synth(name: SynthName) -> SynthSpec:
    """Resolve a registry key to its identity.

    :param name: Registry key.
    :returns: The registered identity, without copying it.
    :raises KeyError: If the name is not registered.
    """
    try:
        return _synths[name]
    except KeyError:
        raise KeyError(name) from None
