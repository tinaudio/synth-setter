"""Python differentiable-synth backends hosted behind the VST plugin surface.

``TorchSynthPlugin`` (torchsynth ``Voice``) and ``SynthaxPlugin`` (synthax
``Voice``) duck-type the ``pedalboard.VST3Plugin`` render surface consumed by
``synth_setter.data.vst.core`` — ``version``, ``parameters[k].raw_value``,
``load_preset``, ``reset``, ``process``, ``show_editor`` — so the render
pipeline treats them exactly like a ``.vst3`` bundle. ``core.load_plugin``
dispatches here for the bare plugin-path names in ``PYTHON_SYNTH_NAMES``.

Both libraries store parameter values machine-normalized to [0, 1], matching
the ``raw_value`` convention, so raw values map 1:1 onto native parameters.
The synth voice renders its note from t=0; ``process`` shifts the buffer right
to honour the note-on time. MIDI velocity is accepted but ignored — neither
voice exposes a velocity control. Keyboard pitch/duration parameters are
driven by the MIDI events, not by ``parameters``.

Heavy imports (torch / torchsynth / jax / synthax) are deferred to first
render so importing this module stays cheap for VST-only workflows.
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import types
from collections.abc import Callable, Iterable, Sequence
from importlib.metadata import version as _dist_version
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

if TYPE_CHECKING:
    import jax
    from synthax.parameter import ModuleParameterRange
    from synthax.synth import Voice as SynthaxVoice
    from torchsynth.parameter import ModuleParameter
    from torchsynth.synth import Voice as TorchSynthVoice

    # The flax parameter tree: {"params": {module_key: {param: leaf}}}.
    SynthaxTree = dict[str, dict[str, dict[str, "jax.Array"]]]

PYTHON_SYNTH_NAMES = frozenset({"torchsynth", "synthax"})

_MIDI_STATUS_MASK = 0xF0
_MIDI_NOTE_ON_STATUS = 0x90
_MIDI_NOTE_OFF_STATUS = 0x80


def is_python_synth(plugin_path: str) -> bool:
    """Return whether ``plugin_path`` names a Python synth backend rather than a ``.vst3``.

    :param plugin_path: The ``RenderConfig.plugin_path`` value.
    :returns: True when the path is a bare name in ``PYTHON_SYNTH_NAMES``.
    """
    return plugin_path in PYTHON_SYNTH_NAMES


def python_synth_version(name: str) -> str:
    """Return the installed distribution version for a Python synth backend.

    :param name: A member of ``PYTHON_SYNTH_NAMES``.
    :returns: The installed package version, the backend's ``renderer_version``.
    """
    return _dist_version(name)


def load_python_synth(name: str) -> PythonSynthPlugin:
    """Instantiate the plugin adapter for a Python synth backend.

    :param name: A member of ``PYTHON_SYNTH_NAMES``.
    :returns: A fresh plugin instance duck-typing the VST3 render surface.
    :raises ValueError: ``name`` is not a known Python synth backend.
    """
    if name == "torchsynth":
        return TorchSynthPlugin()
    if name == "synthax":
        return SynthaxPlugin()
    raise ValueError(
        f"Unknown Python synth backend: {name!r} (known: {sorted(PYTHON_SYNTH_NAMES)})"
    )


@dataclasses.dataclass
class _PluginParameter:
    """Mutable ``raw_value`` holder matching pedalboard's parameter surface.

    .. attribute :: raw_value

       Machine-normalized parameter value in [0, 1].
    """

    raw_value: float


def _parse_note(
    midi_events: Iterable[tuple[Sequence[int], float]],
) -> tuple[int, float, float] | None:
    """Extract ``(pitch, note_start, note_end)`` from raw MIDI events.

    A ``note_on`` with velocity 0 counts as ``note_off`` (MIDI idiom). Without
    a matching ``note_off`` the note is treated as held to the end of the
    buffer (``note_end = inf``; callers clamp to the render duration).

    :param midi_events: ``(payload, time_seconds)`` pairs as produced by
        ``core.make_midi_events``.
    :returns: The first sounding note, or ``None`` for a flush call.
    """
    note_on: tuple[int, float] | None = None
    for payload, time in midi_events:
        status, pitch, velocity = payload[0] & _MIDI_STATUS_MASK, payload[1], payload[2]
        is_off = status == _MIDI_NOTE_OFF_STATUS or (
            status == _MIDI_NOTE_ON_STATUS and velocity == 0
        )
        if note_on is None and status == _MIDI_NOTE_ON_STATUS and velocity > 0:
            note_on = (pitch, time)
        elif note_on is not None and is_off and pitch == note_on[0]:
            return (note_on[0], note_on[1], time)
    if note_on is None:
        return None
    return (note_on[0], note_on[1], float("inf"))


class PythonSynthPlugin:
    """Shared VST3-surface adapter for Python synth voices.

    Subclasses set ``_dist_name`` and implement ``_initial_param_values``
    (native parameter enumeration) and ``_render_mono`` (one mono note buffer
    from the current ``raw_value`` state). Everything pedalboard-shaped —
    flush semantics, note timing, channel tiling — lives here.
    """

    _dist_name: str

    def __init__(self) -> None:
        self._parameters: dict[str, _PluginParameter] | None = None

    @property
    def version(self) -> str:
        """Report the backend's ``renderer_version``.

        :returns: The installed pip distribution version.
        """
        return python_synth_version(self._dist_name)

    @property
    def parameters(self) -> dict[str, _PluginParameter]:
        """Native synth parameters keyed by ``module.param`` flat name.

        :returns: Mapping of parameter name to ``raw_value`` holder in [0, 1].
        """
        if self._parameters is None:
            self._parameters = {
                name: _PluginParameter(raw_value=value)
                for name, value in self._initial_param_values()
            }
        return self._parameters

    def load_preset(self, preset_path: str) -> None:
        """Accept and ignore a preset path; Python synths have no preset files.

        :param preset_path: Ignored; render configs pass ``""``.
        """

    def reset(self) -> None:
        """Mirror the VST3 reset hook; renders are stateless so nothing to clear."""

    def show_editor(self, close_event: threading.Event) -> None:
        """Block until ``close_event`` is set, mirroring the VST3 editor contract.

        :param close_event: Signalled by the warm-up/cadence machinery to release the (virtual)
            editor.
        """
        close_event.wait()

    def process(
        self,
        midi_events: Iterable[tuple[Sequence[int], float]],
        duration_seconds: float,
        sample_rate: float,
        channels: int,
        block_size: int,
        tail: bool,
    ) -> np.ndarray:
        """Render audio for the buffer, matching ``pedalboard.VST3Plugin.process``.

        Empty ``midi_events`` (the pipeline's flush calls) return a zero-length
        buffer; a note renders from t=0 in the synth voice and is shifted right
        to its note-on time.

        :param midi_events: ``(payload, time_seconds)`` pairs; only the first
            sounding note is honoured.
        :param duration_seconds: Output length; ``num_samples = duration * sample_rate``.
        :param sample_rate: Output sample rate in Hz.
        :param channels: Channel count of the returned array (axis 0); the mono
            voice is tiled across channels.
        :param block_size: Accepted for signature parity; unused.
        :param tail: Accepted for signature parity; unused.
        :returns: ``(channels, num_samples)`` float32 for note renders, or
            ``(channels, 0)`` for flush calls.
        """
        note = _parse_note(midi_events)
        if note is None:
            return np.zeros((channels, 0), dtype=np.float32)
        pitch, note_start, note_end = note
        num_samples = int(duration_seconds * sample_rate)
        note_duration = min(note_end, duration_seconds) - note_start
        mono = self._render_mono(pitch, note_duration, duration_seconds, sample_rate)
        offset = int(note_start * sample_rate)
        out = np.zeros(num_samples, dtype=np.float32)
        length = min(len(mono), num_samples - offset)
        if length > 0:
            out[offset : offset + length] = mono[:length]
        return np.broadcast_to(out, (channels, num_samples)).copy()

    def _initial_param_values(self) -> Iterable[tuple[str, float]]:
        """Enumerate ``(name, raw_value)`` for every exposed native parameter.

        :returns: One entry per parameter, in the synth's native order.
        :raises NotImplementedError: Subclasses must implement.
        """
        raise NotImplementedError

    def _render_mono(
        self, pitch: int, note_duration: float, duration_seconds: float, sample_rate: float
    ) -> np.ndarray:
        """Render one mono note buffer from the current ``raw_value`` state.

        :param pitch: MIDI pitch driving the keyboard module.
        :param note_duration: Note-on to note-off span in seconds.
        :param duration_seconds: Voice buffer length in seconds.
        :param sample_rate: Render sample rate in Hz.
        :returns: Mono float32 audio of ``duration_seconds * sample_rate`` samples.
        :raises NotImplementedError: Subclasses must implement.
        """
        raise NotImplementedError


def _import_torchsynth() -> tuple[type, type]:
    """Import torchsynth, aliasing the pytorch-lightning 1.x path it expects.

    torchsynth 1.0.2 imports ``pytorch_lightning.core.lightning``, removed in
    pytorch-lightning 2.x; ``LightningModule`` (its only use) still exists at
    the top level, so a module alias restores compatibility.

    :returns: The ``(SynthConfig, Voice)`` classes.
    """
    try:
        import pytorch_lightning.core.lightning  # noqa: F401
    except ModuleNotFoundError:
        import pytorch_lightning

        shim = types.ModuleType("pytorch_lightning.core.lightning")
        shim.LightningModule = pytorch_lightning.LightningModule  # type: ignore[attr-defined]
        sys.modules["pytorch_lightning.core.lightning"] = shim
    from torchsynth.config import SynthConfig
    from torchsynth.synth import Voice

    return SynthConfig, Voice


# Default geometry for parameter enumeration before the first render request.
_DEFAULT_SAMPLE_RATE = 44100.0
_DEFAULT_BUFFER_SECONDS = 4.0


class TorchSynthPlugin(PythonSynthPlugin):
    """Torchsynth ``Voice`` behind the VST3 plugin surface.

    Parameter names are ``module.param`` (e.g. ``adsr_1.attack``); values are
    torchsynth's machine-normalized [0, 1] storage, assigned verbatim from
    ``raw_value``. The keyboard module's ``midi_f0``/``duration`` are excluded
    from ``parameters`` — the MIDI note drives them.
    """

    _dist_name = "torchsynth"

    def __init__(self) -> None:
        super().__init__()
        self._voice: TorchSynthVoice | None = None
        self._voice_key: tuple[float, float] | None = None
        # Rebuilt with the voice: name -> native ModuleParameter, and the
        # keyboard's (parameter, min, max) clamp triples.
        self._native: dict[tuple[str, str], ModuleParameter] = {}
        self._keyboard: dict[str, tuple[ModuleParameter, float, float]] = {}

    def _voice_for(self, sample_rate: float, duration_seconds: float) -> TorchSynthVoice:
        """Return a batch-1 ``Voice`` for the render geometry, rebuilding on change.

        :param sample_rate: Render sample rate in Hz.
        :param duration_seconds: Voice buffer length in seconds.
        :returns: A cached or fresh ``torchsynth.synth.Voice``.
        """
        key = (sample_rate, duration_seconds)
        voice = self._voice
        if voice is None or self._voice_key != key:
            synth_config_cls, voice_cls = _import_torchsynth()
            config = synth_config_cls(
                batch_size=1,
                sample_rate=int(sample_rate),
                buffer_size_seconds=duration_seconds,
                reproducible=False,
            )
            voice = voice_cls(synthconfig=config)
            self._voice = voice
            self._voice_key = key
            self._native = dict(voice.get_parameters())
            self._keyboard = {}
            for kb_name in ("midi_f0", "duration"):
                parameter = self._native[("keyboard", kb_name)]
                # parameter_range is set dynamically in ModuleParameter.__new__,
                # invisible to static analysis.
                kb_range = parameter.parameter_range  # type: ignore[attr-defined]
                self._keyboard[kb_name] = (
                    parameter,
                    float(kb_range.minimum),
                    float(kb_range.maximum),
                )
        return voice

    def _initial_param_values(self) -> Iterable[tuple[str, float]]:
        """Enumerate the voice's non-keyboard parameters with their current values.

        Reuses the live voice when one exists — rebuilding here at a different
        geometry would desync ``_native`` from the voice a render in progress
        is about to drive.

        :returns: ``(module.param, raw_value)`` pairs in torchsynth's order.
        """
        if self._voice is None:
            self._voice_for(_DEFAULT_SAMPLE_RATE, _DEFAULT_BUFFER_SECONDS)
        return [
            (f"{module}.{param}", float(parameter.data[0]))
            for (module, param), parameter in self._native.items()
            if module != "keyboard"
        ]

    def _render_mono(
        self, pitch: int, note_duration: float, duration_seconds: float, sample_rate: float
    ) -> np.ndarray:
        """Write raw values into the voice, drive the keyboard, and render.

        :param pitch: MIDI pitch driving the keyboard module.
        :param note_duration: Note-on to note-off span in seconds; clamped to
            torchsynth's keyboard duration range.
        :param duration_seconds: Voice buffer length in seconds.
        :param sample_rate: Render sample rate in Hz.
        :returns: Mono float32 audio of ``duration_seconds * sample_rate`` samples.
        """
        import torch

        voice = self._voice_for(sample_rate, duration_seconds)
        for name, holder in self.parameters.items():
            module, param = name.split(".", 1)
            self._native[(module, param)].data[0] = holder.raw_value
        for kb_name, human in (("midi_f0", float(pitch)), ("duration", note_duration)):
            parameter, low, high = self._keyboard[kb_name]
            # to_0to1 writes parameter.data in place; the return value is unused.
            parameter.to_0to1(torch.tensor([min(max(human, low), high)]))
        with torch.no_grad():
            audio = voice.output()
        return audio[0].cpu().numpy().astype(np.float32, copy=False)


class _LeafPlan(NamedTuple):
    """Per-leaf write plan for assembling the synthax parameter tree.

    .. attribute :: module_key

       Tree key including the ``modules_`` prefix.

    .. attribute :: param

       Parameter name within the module.

    .. attribute :: shape

       Leaf shape including the leading batch axis.

    .. attribute :: names

       Flat scalar parameter names, one per leaf element.
    """

    module_key: str
    param: str
    shape: tuple[int, ...]
    names: list[str]


def _leaf_names(module: str, param: str, shape: tuple[int, ...]) -> list[str]:
    """Flat scalar names for one synthax parameter leaf (batch axis dropped).

    :param module: Module tree key with the ``modules_`` prefix stripped.
    :param param: Parameter name within the module.
    :param shape: Leaf shape including the leading batch axis.
    :returns: One name per scalar element, ``module.param`` for scalars and
        ``module.param_i[_j...]`` for array elements.
    """
    if shape[1:] == ():
        return [f"{module}.{param}"]
    return [
        f"{module}.{param}_" + "_".join(str(i) for i in idx) for idx in np.ndindex(shape[1:])
    ]


def _iter_leaves(params: SynthaxTree) -> list[tuple[str, str, "jax.Array"]]:
    """List ``(module, param, leaf)`` for every non-keyboard parameter leaf.

    :param params: The flax parameter tree from ``Voice.init``.
    :returns: Module name (``modules_`` prefix stripped), parameter name,
        and the leaf array, one triple per leaf.
    """
    return [
        (module_key.removeprefix("modules_"), param, leaf)
        for module_key, module_params in params["params"].items()
        if module_key.removeprefix("modules_") != "keyboard"
        for param, leaf in module_params.items()
    ]


class SynthaxPlugin(PythonSynthPlugin):
    """Synthax ``Voice`` behind the VST3 plugin surface.

    Parameter names are ``module.param`` flat names with array-valued synthax
    parameters flattened to one scalar per element (``mixer.level_0``,
    ``mod_matrix.mod_2_1``). synthax stores parameter leaves machine-normalized
    to [0, 1], so ``raw_value`` maps verbatim. The keyboard module is excluded
    from ``parameters`` — the MIDI note drives it.
    """

    _dist_name = "synthax"

    def __init__(self) -> None:
        super().__init__()
        self._voice: SynthaxVoice | None = None
        self._apply: Callable[[SynthaxTree], jax.Array] | None = None
        self._init_params: SynthaxTree | None = None
        self._voice_key: tuple[float, float] | None = None
        # Rebuilt with the voice: per-leaf write plan and keyboard ranges.
        self._plan: list[_LeafPlan] = []
        self._kb_ranges: dict[str, ModuleParameterRange] = {}

    def _voice_for(
        self, sample_rate: float, duration_seconds: float
    ) -> Callable[[SynthaxTree], jax.Array]:
        """Return the jitted apply fn for the render geometry, rebuilding on change.

        :param sample_rate: Render sample rate in Hz.
        :param duration_seconds: Voice buffer length in seconds.
        :returns: ``jax.jit(voice.apply)`` for a batch-1 ``synthax.synth.Voice``.
        """
        key = (sample_rate, duration_seconds)
        apply_fn = self._apply
        if apply_fn is None or self._voice_key != key:
            import jax
            from synthax.config import SynthConfig
            from synthax.modules.keyboard import MonophonicKeyboard
            from synthax.parameter import ModuleParameterRange
            from synthax.synth import Voice

            config = SynthConfig(
                batch_size=1,
                sample_rate=int(sample_rate),
                buffer_size_seconds=duration_seconds,
            )
            voice = Voice(config=config)
            # flax returns a FrozenVariableDict; downstream only reads it as a
            # plain nested mapping of jax arrays.
            init_params = cast("SynthaxTree", voice.init(jax.random.PRNGKey(0)))
            apply_fn = jax.jit(voice.apply)
            self._voice = voice
            self._apply = apply_fn
            self._init_params = init_params
            self._voice_key = key
            self._plan = [
                _LeafPlan(
                    module_key=f"modules_{module}",
                    param=param,
                    shape=np.asarray(leaf).shape,
                    names=_leaf_names(module, param, np.asarray(leaf).shape),
                )
                for module, param, leaf in _iter_leaves(init_params)
            ]
            self._kb_ranges = {
                field.name: field.default
                for field in dataclasses.fields(MonophonicKeyboard)
                if isinstance(field.default, ModuleParameterRange)
            }
        return apply_fn

    def _initial_param_values(self) -> Iterable[tuple[str, float]]:
        """Enumerate flattened non-keyboard parameter leaves with their init values.

        Reuses the live voice when one exists — rebuilding here at a different
        geometry would replace the jitted apply a render in progress is about
        to call.

        :returns: ``(module.param, raw_value)`` pairs, arrays flattened per scalar.
        """
        if self._init_params is None:
            self._voice_for(_DEFAULT_SAMPLE_RATE, _DEFAULT_BUFFER_SECONDS)
        assert self._init_params is not None  # populated by _voice_for
        values: list[tuple[str, float]] = []
        for module, param, leaf in _iter_leaves(self._init_params):
            flat = np.asarray(leaf)[0].reshape(-1)
            values.extend(zip(_leaf_names(module, param, np.asarray(leaf).shape), flat.tolist()))
        return values

    def _render_mono(
        self, pitch: int, note_duration: float, duration_seconds: float, sample_rate: float
    ) -> np.ndarray:
        """Assemble the parameter tree from raw values, drive the keyboard, and render.

        :param pitch: MIDI pitch driving the keyboard module.
        :param note_duration: Note-on to note-off span in seconds; clamped to
            synthax's keyboard duration range.
        :param duration_seconds: Voice buffer length in seconds.
        :param sample_rate: Render sample rate in Hz.
        :returns: Mono float32 audio of ``duration_seconds * sample_rate`` samples.
        """
        import jax.numpy as jnp
        from synthax.parameter import to_0to1

        apply_fn = self._voice_for(sample_rate, duration_seconds)
        tree: SynthaxTree = {"params": {}}
        for plan in self._plan:
            flat = np.array(
                [self.parameters[name].raw_value for name in plan.names], dtype=np.float32
            )
            tree["params"].setdefault(plan.module_key, {})[plan.param] = jnp.asarray(
                flat.reshape(plan.shape)
            )
        keyboard = {}
        for kb_name, human in (("midi_f0", float(pitch)), ("duration", note_duration)):
            rng = self._kb_ranges[kb_name]
            clamped = min(max(human, float(np.asarray(rng.minimum))), float(np.asarray(rng.maximum)))
            keyboard[kb_name] = jnp.asarray(to_0to1(jnp.array([clamped], dtype=jnp.float32), rng))
        tree["params"]["modules_keyboard"] = keyboard
        audio = apply_fn(tree)
        return np.asarray(audio[0], dtype=np.float32)


def render_param_spec_module(name: str) -> str:
    """Render the checked-in ``<name>_param_spec.py`` module source.

    Every native parameter becomes a full-range ``ContinuousParameter`` (both
    libraries sample uniformly on machine-range [0, 1]); note params follow the
    repo convention pinned by ``introspect``'s defaults (pitch 48-72, note
    duration up to 4 s).

    :param name: A member of ``PYTHON_SYNTH_NAMES``.
    :returns: Python source for the spec module.
    """
    from synth_setter.data.vst.introspect import (
        DEFAULT_MAX_NOTE_DURATION_SECONDS,
        DEFAULT_PITCH_MAX,
        DEFAULT_PITCH_MIN,
    )

    plugin = load_python_synth(name)
    lines = [
        f'"""ParamSpec for {name} (v{plugin.version}), generated from live introspection.',
        "",
        f"Regenerate with ``python -m synth_setter.data.vst.python_synth`` after a {name}",
        "upgrade; ``tests/data/vst/test_python_synth_param_specs.py`` pins the sync.",
        '"""',
        "",
        "from synth_setter.data.vst.param_spec import (",
        "    ContinuousParameter,",
        "    DiscreteLiteralParameter,",
        "    NoteDurationParameter,",
        "    ParamSpec,",
        ")",
        "",
        f"{name.upper()}_PARAM_SPEC = ParamSpec(",
        "    [",
        *(
            f'        ContinuousParameter(name="{param_name}", min=0.0, max=1.0),'
            for param_name in plugin.parameters
        ),
        "    ],",
        "    [",
        f'        DiscreteLiteralParameter(name="pitch", min={DEFAULT_PITCH_MIN}, '
        f"max={DEFAULT_PITCH_MAX}),",
        "        NoteDurationParameter(",
        '            name="note_start_and_end",',
        f"            max_note_duration_seconds={DEFAULT_MAX_NOTE_DURATION_SECONDS},",
        "        ),",
        "    ],",
        ")",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path

    from loguru import logger

    for _name in sorted(PYTHON_SYNTH_NAMES):
        _target = Path(__file__).parent / f"{_name}_param_spec.py"
        _target.write_text(render_param_spec_module(_name))
        logger.info(f"wrote {_target}")
