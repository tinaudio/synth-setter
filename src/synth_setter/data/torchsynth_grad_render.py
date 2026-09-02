"""Gradient-enabled TorchSynth rendering for audio-domain training losses.

Parameter tensors are substituted through ``torch.func.functional_call`` so gradients flow from
rendered audio to normalized synthesizer parameters.

Typical usage:
    audio = render_torchsynth_grad(
        differentiable_decode(theta_hat),
        sample_rate=44_100,
        signal_length=176_400,
        render_batch_size=32,
    )
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from functools import partial
from typing import TYPE_CHECKING, cast

import torch
from torch.func import functional_call

from synth_setter.data.torchsynth_datamodule import (
    _PARAM_CLAMP_EPS,
    _delay_by_note_start,
    _make_renderer,
    _pad_to_render_size,
)
from synth_setter.data.vst.param_spec import require_note_params
from synth_setter.data.vst.torchsynth_param_spec import (
    INFERABLE_SPEC,
    TORCHSYNTH_FULL_PARAM_SPEC,
    note_on_duration,
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


# TorchSynth's own SynthConfig.eps, whose documented purpose is avoiding divide-by-zero.
_RAMP_GRAD_EPS: float = 1e-6
_TORCH_POW: Callable[..., torch.Tensor] = torch.pow


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
    render_batch_size: int,
) -> torch.Tensor:
    """Render ``torchsynth_full``-encoded rows with autograd connected.

    The differentiable twin of
    :func:`~synth_setter.data.torchsynth_datamodule.render_torchsynth` and bound by the
    same contract: a row's own note columns decide its pitch and note window, so an
    estimate and the target it is scored against cannot disagree on note conditioning.
    Gradient reaches the synth columns only — the note columns are read through
    ``detach()`` because pitch is a discrete category and the note window lands on ADSR
    segment boundaries via integer sample arithmetic, so neither carries usable gradient.

    :param params: Float32 rows shaped ``(batch, torchsynth_full encoded width)`` in
        ``[0, 1]``; may carry ``requires_grad``. Synth values clamp strictly inside
        ``(0, 1)`` straight-through, so saturated entries keep their gradient.
    :param sample_rate: Audio sample rate in Hz.
    :param signal_length: Number of output samples.
    :param render_batch_size: Fixed row count of the voice this render runs on; a
        shorter batch is padded up and sliced back, so a trailing partial batch
        neither allocates a second voice nor shifts the render.
    :returns: Float32 audio shaped ``(batch, signal_length)``, differentiable
        w.r.t. ``params``' synth columns.
    :raises ValueError: Wrong row width, a non-finite value, a batch exceeding
        ``render_batch_size``, or a non-finite render.
    """
    validate_torchsynth_params(params)
    rows = len(params)
    padded = _pad_to_render_size(params, render_batch_size)
    notes = [
        require_note_params(TORCHSYNTH_FULL_PARAM_SPEC.decode(row)[1])
        for row in padded.detach().clamp(0, 1).cpu().numpy()
    ]
    column = partial(torch.tensor, dtype=torch.float32, device=params.device)
    synth_params = padded[:, TORCHSYNTH_FULL_PARAM_SPEC.synth_columns]
    renderer = _make_renderer(sample_rate, signal_length, render_batch_size, str(params.device))
    voice = renderer.voice
    with renderer.lock, _aligned_noise(voice), finite_tensor_exponent_pow():
        all_parameters = voice.get_parameters()
        keyboard = (
            ("midi_f0", column([note["pitch"] for note in notes])),
            ("duration", column([note_on_duration(note["note_start_and_end"]) for note in notes])),
        )
        for name, value in keyboard:
            all_parameters[("keyboard", name)].to_0to1(value)
        name_by_id = {
            id(parameter): f"voice.{name}" for name, parameter in voice.named_parameters()
        }
        # Straight-through: the renderer only accepts the open interval, but a hard clamp
        # would zero gradient on saturated rows, stranding a diverged estimate out of range.
        in_range = synth_params.clamp(_PARAM_CLAMP_EPS, 1 - _PARAM_CLAMP_EPS)
        clamped = synth_params + (in_range - synth_params).detach()
        overrides = {}
        for values, spec in zip(clamped.unbind(dim=1), INFERABLE_SPEC, strict=True):
            original = all_parameters[(spec.module, spec.name)]
            overrides[name_by_id[id(original)]] = _as_substitute(values, original)
        audio = functional_call(_VoiceOutputShim(voice), overrides)[:rows]
    if not torch.isfinite(audio).all():
        # Mirrors render_torchsynth's output guard: one non-finite render would
        # otherwise write NaN into every weight on the next backward pass.
        raise ValueError("TorchSynth rendered non-finite audio")
    starts = column([note["note_start_and_end"][0] for note in notes[:rows]])
    return _delay_by_note_start(audio, starts, sample_rate)


def _finite_grad_pow(base: torch.Tensor, exponent: torch.Tensor | float) -> torch.Tensor:
    """Evaluate ``pow`` exactly while differentiating a base floored away from zero.

    ``d/dbase base**a`` is ``a * base**(a - 1)``, which is ``inf`` at ``base == 0`` for
    ``a < 1``. TorchSynth's ADSR reaches exactly that: its forward epsilon guard is
    destroyed by the ``1 - ramp`` inverse flip once the ramp saturates at one. The forward
    value is grafted back so stored renders stay bitwise reproducible.

    :param base: Pow base, possibly containing zeros.
    :param exponent: Pow exponent; only a tensor exponent reaches the singular derivative.
    :returns: ``base ** exponent`` with a finite gradient w.r.t. ``base``.
    """
    exact = _TORCH_POW(base, exponent)
    if not isinstance(exponent, torch.Tensor) or not base.requires_grad:
        return exact
    floored = _TORCH_POW(base.clamp_min(_RAMP_GRAD_EPS), exponent)
    return floored + (exact - floored).detach()


@contextlib.contextmanager
def finite_tensor_exponent_pow() -> Iterator[None]:
    """Swap ``torch.pow`` for a variant whose base gradient cannot reach ``inf``.

    Scoped to a single render rather than applied upstream because the defect is in
    ``torchsynth.module``'s ADSR ramp and its two sibling tensor-exponent pows, which this
    repo does not own.
    """
    torch.pow = _finite_grad_pow  # pyright: ignore[reportAttributeAccessIssue]
    try:
        yield
    finally:
        torch.pow = _TORCH_POW  # pyright: ignore[reportAttributeAccessIssue]


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

    :param params: Candidate encoded rows shaped ``(batch, encoded_width)``.
    :raises ValueError: Wrong row width or non-finite entries.
    """
    width = TORCHSYNTH_FULL_PARAM_SPEC.encoded_width
    if params.shape[1] != width:
        raise ValueError(f"Expected {width} encoded parameter columns, got {params.shape[1]}")
    if not torch.isfinite(params).all():
        # NaN/Inf survives the clamp and would propagate through the loss and gradients.
        raise ValueError("non-finite TorchSynth parameters; the model has diverged")


def differentiable_decode(theta: torch.Tensor) -> torch.Tensor:
    """Map model space ``[-1, 1]`` to renderable ``[eps, 1 - eps]``, keeping gradient.

    The forward clamp preserves valid renderer input while the straight-through
    backward pass lets saturated entries receive gradients, so an audio loss can
    pull them back into range where a plain ``clamp`` would zero them.

    :param theta: Encoded rows in model space shaped ``(batch, encoded_width)``.
    :returns: Params in torchsynth space, strictly inside ``(0, 1)``.
    """
    params01 = (theta + 1) / 2
    clamped = params01.clamp(_PARAM_CLAMP_EPS, 1 - _PARAM_CLAMP_EPS)
    return params01 + (clamped - params01).detach()
