"""Gradient-enabled TorchSynth rendering for the simulator-feedback spike (#2553).

Sibling of :func:`synth_setter.data.torchsynth_datamodule.render_torchsynth` that
keeps autograd connected from the normalized parameter batch to the rendered
audio. The production renderer copies values into ``ModuleParameter.data`` and
renders under ``torch.no_grad()``; here the parameter tensors are substituted
via ``torch.func.functional_call`` so the graph flows through
``Voice.output()``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch.func import functional_call

from synth_setter.data.torchsynth_datamodule import (
    _PARAM_CLAMP_EPS,
    _make_renderer,
)
from synth_setter.data.vst.torchsynth_param_spec import (
    INFERABLE_SPEC,
    KEYBOARD_DURATION_BOUNDS,
    NUM_PARAMS,
)
from synth_setter.metrics import complex_to_dbfs


class _VoiceOutputShim(torch.nn.Module):
    """Route ``functional_call`` to ``Voice.output()``.

    ``Voice.forward`` stacks ``p.data`` and honors ``synthconfig.no_grad``, both
    of which sever autograd; ``output()`` reads parameters through ``p()``.
    """

    def __init__(self, voice: torch.nn.Module) -> None:
        """Own the voice whose ``output()`` the shim exposes as ``forward``.

        :param voice: Live torchsynth voice to render.
        """
        super().__init__()
        self.voice = voice

    def forward(self) -> torch.Tensor:
        """Render the owned voice.

        :returns: Audio batch shaped ``(batch, signal_length)``.
        """
        return self.voice.output()


@contextlib.contextmanager
def _patched_module_p(voice: torch.nn.Module) -> Iterator[None]:
    """Make ``SynthModule.p`` tolerate plain-tensor parameter substitutes.

    ``functional_call`` swaps ``ModuleParameter`` entries for plain tensors,
    which lack ``.from_0to1``; the patch falls back to a side table of
    ``ModuleParameterRange`` objects captured before substitution.

    :param voice: Voice whose modules are patched for the duration.
    :yields: Control while the patch is active.
    :ytype: None
    """
    from torchsynth.module import SynthModule
    from torchsynth.parameter import ModuleParameter

    ranges = {}
    for module in voice.modules():
        torchparameters = getattr(module, "torchparameters", None)
        if torchparameters is None:
            continue
        for pid, parameter in torchparameters.items():
            ranges[(id(module), pid)] = parameter.parameter_range

    original_p = SynthModule.p

    def patched_p(self: SynthModule, parameter_id: str) -> torch.Tensor:
        value = self.torchparameters[parameter_id]
        if isinstance(value, ModuleParameter):
            return original_p(self, parameter_id)
        return ranges[(id(self), parameter_id)].from_0to1(value)

    SynthModule.p = patched_p
    try:
        yield
    finally:
        SynthModule.p = original_p


def render_torchsynth_grad(
    params: torch.Tensor,
    *,
    sample_rate: int,
    signal_length: int,
    midi_pitch: int,
    note_duration_seconds: float | None = None,
) -> torch.Tensor:
    """Render normalized TorchSynth parameters with autograd connected.

    :param params: Float32 parameter rows shaped ``(batch, NUM_PARAMS)`` in
        ``[0, 1]``; may carry ``requires_grad``. Values clamp strictly inside
        ``(0, 1)`` (clamped entries get zero gradient).
    :param sample_rate: Audio sample rate in Hz.
    :param signal_length: Number of output samples.
    :param midi_pitch: Fixed MIDI note rendered for every parameter row.
    :param note_duration_seconds: Note-on length before release; ``None`` holds
        the note for the whole buffer.
    :returns: Float32 audio shaped ``(batch, signal_length)``, differentiable
        w.r.t. ``params``.
    :raises ValueError: Wrong parameter width or an out-of-range note duration.
    """
    if params.shape[1] != NUM_PARAMS:
        raise ValueError(f"Expected {NUM_PARAMS} TorchSynth parameters, got {params.shape[1]}")
    duration = (
        note_duration_seconds if note_duration_seconds is not None else signal_length / sample_rate
    )
    minimum_duration, maximum_duration = KEYBOARD_DURATION_BOUNDS
    if not minimum_duration <= duration <= maximum_duration:
        raise ValueError(
            f"note duration {duration}s outside the keyboard's pinned range "
            f"[{minimum_duration}, {maximum_duration}]s"
        )
    renderer = _make_renderer(sample_rate, signal_length, len(params), str(params.device))
    voice = renderer.voice
    with renderer.lock:
        all_parameters = voice.get_parameters()
        for name, value in (("midi_f0", float(midi_pitch)), ("duration", duration)):
            keyboard = all_parameters[("keyboard", name)]
            keyboard.to_0to1(torch.full((len(params),), value, device=params.device))
        name_by_id = {
            id(parameter): f"voice.{name}" for name, parameter in voice.named_parameters()
        }
        clamped = params.clamp(_PARAM_CLAMP_EPS, 1 - _PARAM_CLAMP_EPS)
        overrides = {
            name_by_id[id(all_parameters[(spec.module, spec.name)])]: column
            for column, spec in zip(clamped.unbind(dim=1), INFERABLE_SPEC, strict=True)
        }
        shim = _VoiceOutputShim(voice)
        with _patched_module_p(voice):
            audio = functional_call(shim, overrides)
    return audio


def log_spectral_distance(
    pred_signal: torch.Tensor, target_signal: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Per-sample log-spectral distance, mirroring ``metrics.LogSpectralDistance``.

    :param pred_signal: Predicted audio shaped ``(batch, samples)``.
    :param target_signal: Target audio shaped ``(batch, samples)``.
    :param eps: Power-clamp floor before the dB conversion.
    :returns: Per-sample distance shaped ``(batch,)``.
    """
    pred_power = complex_to_dbfs(torch.fft.rfft(pred_signal, norm="forward"), eps)
    target_power = complex_to_dbfs(torch.fft.rfft(target_signal, norm="forward"), eps)
    return (pred_power - target_power).square().mean(dim=-1).sqrt()
