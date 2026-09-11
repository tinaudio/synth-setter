"""Black-box rewards that score sampled parameter rows by re-rendering them.

Each reward declares ``target_key``, the batch column it prepares before candidate grouping:
``audio`` when the datamodule carries the target waveform, ``params`` when the target row must
be rendered first.
"""

from typing import Any, ClassVar

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from synth_setter.data.torchsynth_grad_render import (
    differentiable_decode,
    render_torchsynth_grad,
)
from synth_setter.data.vst.param_spec import (
    decode_model_output,
    require_note_params,
    require_scalar_synth_params,
    spec_quantize_model_output,
)
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst.renderers import AudioRenderer
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import anchor_render_preset, make_audio_renderer

_BATCH_SHAPE = "batch"
_BATCH_AUDIO_SHAPE = "batch samples"
_BATCH_CHANNEL_AUDIO_SHAPE = "batch channels samples"
_BATCH_PARAMS_SHAPE = "batch params"


class RenderedAudioReward(nn.Module):
    """Negative audio distance between a rendered torchsynth row and its target audio.

    .. attribute :: target_key

       Batch column holding the target: the stored waveform.
    """

    target_key: ClassVar[str] = "audio"

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        *,
        distance: nn.Module,
        sample_rate: int,
        signal_length: int,
        render_batch_size: int,
    ) -> None:
        """Configure the render geometry and the distance the reward negates.

        :param distance: Module mapping ``(rendered, target)`` to a per-row distance.
        :param sample_rate: Render sample rate in Hz.
        :param signal_length: Rendered samples per row; must match the target audio.
        :param render_batch_size: Rows the renderer's voice holds; wider batches render in
            chunks of this size.
        :raises ValueError: ``render_batch_size`` is not positive.
        """
        super().__init__()
        if render_batch_size <= 0:
            raise ValueError(f"render_batch_size must be positive, got {render_batch_size}")
        self.distance = distance
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.render_batch_size = render_batch_size

    @jaxtyped(typechecker=beartype)
    def prepare_target(
        self,
        target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE]
        | Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
    ) -> Float[Tensor, _BATCH_AUDIO_SHAPE] | Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE]:
        """Return stored target waveforms unchanged before candidate grouping.

        :param target_audio: Observed audio with one waveform per original row.
        :returns: The same target waveforms.
        """
        return target_audio

    @jaxtyped(typechecker=beartype)
    @torch.no_grad()
    def forward(
        self,
        theta: Float[Tensor, _BATCH_PARAMS_SHAPE],
        target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE]
        | Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
    ) -> Float[Tensor, _BATCH_SHAPE]:
        """Render every row and score it against its own target.

        :param theta: Sampled parameters in model space ``[-1, 1]``.
        :param target_audio: Observed audio shaped ``(batch, signal_length)`` or
            ``(batch, channels, signal_length)``.
        :returns: Per-row reward, higher for renders closer to the target.
        """
        # Reversed endpoint predictions remain degenerate until onset/bounded-duration timing (#2995).
        params = differentiable_decode(theta.detach())
        # The grad renderer, not the dataset one: it aligns every row to the noise chunk the
        # targets were rendered with, so a row's reward cannot depend on its batch position.
        rendered = torch.cat(
            [
                render_torchsynth_grad(
                    chunk,
                    sample_rate=self.sample_rate,
                    signal_length=self.signal_length,
                    render_batch_size=self.render_batch_size,
                )
                for chunk in params.split(self.render_batch_size)
            ]
        )
        return -self.distance(rendered, target_audio)


class SynthRenderedReward(nn.Module):
    """Negative audio distance between rendered samples and prepared target renders.

    Renders through whichever backend the render config names (surgepy in-process for Surge), so
    the reward needs no stored target audio. Each original target row renders once before candidate
    grouping; candidate rows still render independently.

    .. attribute :: target_key

       Batch column holding the target: the model-space parameter row.
    """

    target_key: ClassVar[str] = "params"

    @jaxtyped(typechecker=beartype)
    def __init__(self, *, render_config: RenderConfig, distance: nn.Module) -> None:
        """Bind the renderer identity and the distance the reward negates.

        :param render_config: Backend, geometry, and synth identity to render rows with.
        :param distance: Module mapping ``(rendered, target)`` to a per-row distance.
        """
        super().__init__()
        self.render_config = anchor_render_preset(render_config)
        self.distance = distance
        self.spec = resolve_param_spec(
            render_config.param_spec_name,
            render_config.note_timing_parameterization,
        )
        # Built on first use: the native engine is not picklable and must live in the
        # process that renders.
        self._renderer: AudioRenderer | None = None

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_cfg_nodes(cls, render: Any, synth: Any, distance: nn.Module) -> "SynthRenderedReward":
        """Build from the composed root ``render`` and ``synth`` nodes (Hydra ``_target_``).

        :param render: Composed ``render`` node.
        :param synth: Composed root ``synth`` node.
        :param distance: Module mapping ``(rendered, target)`` to a per-row distance.
        :returns: Reward rendering through the configured backend.
        """
        return cls(render_config=RenderConfig.from_cfg_nodes(render, synth), distance=distance)

    @jaxtyped(typechecker=beartype)
    def _render_rows(
        self, theta: Float[Tensor, _BATCH_PARAMS_SHAPE]
    ) -> Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE]:
        """Decode and render every model-space row without discarding channels.

        :param theta: Rows in model space; values outside ``[-1, 1]`` clamp into range.
        :returns: Float32 audio shaped ``(batch, channels, samples)`` on ``theta``'s device.
        """
        if self._renderer is None:
            self._renderer = make_audio_renderer(self.render_config)
        render = self.render_config
        rendered = []
        for row in theta.detach().float().cpu().numpy():
            quantized = spec_quantize_model_output(np.clip(row, -1.0, 1.0), self.spec)
            synth_values, note_values = decode_model_output(quantized, self.spec)
            note = require_note_params(note_values)
            start, end = _renderable_note_window(
                note["note_start_and_end"],
                signal_duration_seconds=render.signal_duration_seconds,
                sample_rate=render.sample_rate,
            )
            audio = self._renderer.render(
                require_scalar_synth_params(synth_values),
                int(note["pitch"]),
                render.velocity,
                (start, end),
            )
            rendered.append(torch.from_numpy(np.asarray(audio, dtype=np.float32)))
        return torch.stack(rendered).to(theta.device)

    @jaxtyped(typechecker=beartype)
    @torch.no_grad()
    def prepare_target(
        self, target_params: Float[Tensor, _BATCH_PARAMS_SHAPE]
    ) -> Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE]:
        """Render each original target parameter row once.

        :param target_params: Target parameters in model space before candidate grouping.
        :returns: One channel-preserving target waveform per original row.
        """
        return self._render_rows(target_params)

    @jaxtyped(typechecker=beartype)
    @torch.no_grad()
    def forward(
        self,
        theta: Float[Tensor, _BATCH_PARAMS_SHAPE],
        target_audio: Float[Tensor, _BATCH_AUDIO_SHAPE]
        | Float[Tensor, _BATCH_CHANNEL_AUDIO_SHAPE],
    ) -> Float[Tensor, _BATCH_SHAPE]:
        """Render sampled rows and score each against its prepared target waveform.

        :param theta: Sampled parameters in model space.
        :param target_audio: Prepared target waveforms, one per sampled row.
        :returns: Per-row reward, higher for renders closer to the target's render.
        """
        return -self.distance(self._render_rows(theta), target_audio)


@jaxtyped(typechecker=beartype)
def _renderable_note_window(
    note_window: tuple[float, float], *, signal_duration_seconds: float, sample_rate: int
) -> tuple[float, float]:
    """Clip a decoded note window into the renderer's accepted chronological range.

    :param note_window: Decoded note endpoints in seconds.
    :param signal_duration_seconds: Maximum renderable endpoint in seconds.
    :param sample_rate: Render sample rate in Hz.
    :returns: Endpoints with ``0 <= start < end <= signal_duration_seconds``.
    """
    start, end = sorted(float(value) for value in note_window)
    start = min(max(start, 0.0), signal_duration_seconds)
    end = min(max(end, 0.0), signal_duration_seconds)
    minimum = min(1.0 / sample_rate, signal_duration_seconds)
    if end - start < minimum:
        # A degenerate window still renders: pull the start back or push the end forward.
        start = max(0.0, end - minimum) if end >= minimum else 0.0
        end = start + minimum
    return start, end
