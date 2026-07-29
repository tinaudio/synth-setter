"""Gradient-enabled TorchSynth rendering for audio-domain training losses.

Parameter tensors are substituted through ``torch.func.functional_call`` so gradients flow from
rendered audio to normalized synthesizer parameters.

Typical usage:
    audio = render_torchsynth_grad(
        differentiable_decode(theta_hat),
        sample_rate=44_100,
        signal_length=176_400,
        midi_pitch=60,
    )
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    from torchsynth.parameter import ModuleParameter


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
        # nn.Module.__getattr__ types submodule access as Tensor; Voice.output() is a method.
        return self.voice.output()  # pyright: ignore[reportCallIssue]


def _as_substitute(column: torch.Tensor, original: ModuleParameter) -> ModuleParameter:
    """Dress a graph-connected column as the ``ModuleParameter`` it stands in for.

    ``as_subclass`` rebrands without copying, so stock ``SynthModule.p`` finds the
    ``.from_0to1`` it needs while autograd still reaches ``column``'s inputs.

    :param column: Normalized parameter column shaped ``(batch,)``.
    :param original: Voice parameter whose range and name the substitute adopts.
    :returns: Substitute carrying ``original``'s metadata and ``column``'s graph.
    """
    from torchsynth.parameter import ModuleParameter

    substitute = column.as_subclass(ModuleParameter)
    # ModuleParameter declares these in __new__, so pyright cannot see them on the class.
    substitute.parameter_range = original.parameter_range  # pyright: ignore[reportAttributeAccessIssue]
    substitute.parameter_name = original.parameter_name  # pyright: ignore[reportAttributeAccessIssue]
    # Unfrozen so a stray to_0to1 during the render raises no "Parameter is frozen".
    substitute.frozen = False  # pyright: ignore[reportAttributeAccessIssue]
    return substitute


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
        ``(0, 1)`` straight-through, so saturated entries keep their gradient.
    :param sample_rate: Audio sample rate in Hz.
    :param signal_length: Number of output samples.
    :param midi_pitch: Fixed MIDI note rendered for every parameter row.
    :param note_duration_seconds: Note-on length before release; ``None`` holds
        the note for the whole buffer.
    :returns: Float32 audio shaped ``(batch, signal_length)``, differentiable
        w.r.t. ``params``.
    :raises ValueError: Wrong parameter width or an out-of-range note duration.
    """
    validate_torchsynth_params(params)
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
    with renderer.lock, _aligned_noise(voice):
        all_parameters = voice.get_parameters()
        for name, value in (("midi_f0", float(midi_pitch)), ("duration", duration)):
            keyboard = all_parameters[("keyboard", name)]
            keyboard.to_0to1(torch.full((len(params),), value, device=params.device))
        name_by_id = {
            id(parameter): f"voice.{name}" for name, parameter in voice.named_parameters()
        }
        # Straight-through: the renderer only accepts the open interval, but a hard clamp
        # would zero gradient on saturated rows, stranding a diverged estimate out of range.
        in_range = params.clamp(_PARAM_CLAMP_EPS, 1 - _PARAM_CLAMP_EPS)
        clamped = params + (in_range - params).detach()
        overrides = {}
        for column, spec in zip(clamped.unbind(dim=1), INFERABLE_SPEC, strict=True):
            original = all_parameters[(spec.module, spec.name)]
            overrides[name_by_id[id(original)]] = _as_substitute(column, original)
        audio = functional_call(_VoiceOutputShim(voice), overrides)
    if not torch.isfinite(audio).all():
        # Mirrors render_torchsynth's output guard: one non-finite render would
        # otherwise write NaN into every weight on the next backward pass.
        raise ValueError("TorchSynth rendered non-finite audio")
    return audio


@contextlib.contextmanager
def _aligned_noise(voice: torch.nn.Module) -> Iterator[None]:
    """Give every batch row the noise realization the stored targets were rendered with.

    ``Noise`` pre-draws one seeded ``(batch, buffer)`` block, so row ``i`` gets chunk
    ``i`` — but :class:`TorchSynthDataset` renders targets one row at a time, always on
    chunk 0. Broadcasting chunk 0 here keeps the estimate/target comparison free of an
    irreducible noise-mismatch penalty. The original buffers are restored on exit.

    :param voice: Live torchsynth voice whose noise buffers are aligned.
    :yields: Control while the alignment is active.
    :ytype: None
    """
    from torchsynth.module import Noise

    noise_modules = [module for module in voice.modules() if isinstance(module, Noise)]
    saved = [cast(torch.Tensor, module.noise) for module in noise_modules]
    for module, original in zip(noise_modules, saved, strict=True):
        module.noise = original[0:1].expand_as(original)
    try:
        yield
    finally:
        for module, original in zip(noise_modules, saved, strict=True):
            module.noise = original


def validate_torchsynth_params(params: torch.Tensor) -> None:
    """Reject parameter batches the render would silently corrupt.

    :param params: Candidate parameter rows shaped ``(batch, NUM_PARAMS)``.
    :raises ValueError: Wrong parameter width or non-finite entries.
    """
    if params.shape[1] != NUM_PARAMS:
        raise ValueError(f"Expected {NUM_PARAMS} TorchSynth parameters, got {params.shape[1]}")
    if not torch.isfinite(params).all():
        # NaN/Inf survives the clamp and would propagate through the loss and gradients.
        raise ValueError("non-finite TorchSynth parameters; the model has diverged")


def differentiable_decode(theta: torch.Tensor) -> torch.Tensor:
    """Map model space ``[-1, 1]`` to renderable ``[eps, 1 - eps]``, keeping gradient.

    The forward clamp preserves valid renderer input while the straight-through
    backward pass lets saturated entries receive gradients, so an audio loss can
    pull them back into range where a plain ``clamp`` would zero them.

    :param theta: Params in model space shaped ``(batch, NUM_PARAMS)``.
    :returns: Params in torchsynth space, strictly inside ``(0, 1)``.
    """
    params01 = (theta + 1) / 2
    clamped = params01.clamp(_PARAM_CLAMP_EPS, 1 - _PARAM_CLAMP_EPS)
    return params01 + (clamped - params01).detach()
