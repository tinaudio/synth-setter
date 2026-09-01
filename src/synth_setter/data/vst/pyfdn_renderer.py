"""Live pyFDN post-processing for shared offline audio renderers.

Typical usage wraps a configured synthesis backend before rendering::

    renderer = PyFDNEffectRenderer(inner=base_renderer, effect=effect_config)
    audio = renderer.render(params, 60, 100, (0.0, 1.0))
"""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np
from pyFDN import process_fdn
from pyFDN.build import FDNBuild
from pyFDN.preset import get_fdn_preset
from pyFDN.td import SOSBank
from pyFDN.train.build import build_set_decay

from synth_setter.data.vst.renderers import AudioRenderer, _validate_rendered_audio
from synth_setter.pipeline.schemas.spec import PyFDNEffectConfig

if TYPE_CHECKING:
    from pedalboard import VST3Plugin


class PyFDNEffectRenderer(AudioRenderer):
    """Apply a fixed feedback-delay-network effect after an inner render."""

    def __init__(self, *, inner: AudioRenderer, effect: PyFDNEffectConfig):
        """Load and validate the configured pyFDN build.

        :param inner: Synth renderer whose channel-first output is processed live.
        :param effect: Persisted effect identity and controls.
        :raises RuntimeError: Installed pyFDN does not match the persisted version.
        :raises ValueError: The preset geometry or sample rate is incompatible.
        """
        installed_version = version("pyFDN")
        if installed_version != effect.package_version:
            raise RuntimeError(
                f"pyFDN version mismatch: expected {effect.package_version}, "
                f"found {installed_version}"
            )
        build = build_set_decay(get_fdn_preset(effect.preset_name).build, effect.decay_seconds)
        if int(build.fs) != int(inner.sample_rate):
            raise ValueError(
                f"pyFDN preset sample rate {build.fs:g} != renderer sample rate "
                f"{inner.sample_rate:g}"
            )
        if build.B.shape[1] != 1 or build.C.shape[0] != 1 or build.D.shape != (1, 1):
            raise ValueError("pyFDN effect requires a mono-input, mono-output preset")
        self.inner = inner
        self.effect = effect
        self._build = build

    @property
    def plugin_path(self) -> str:
        return self.inner.plugin_path

    @property
    def sample_rate(self) -> float:
        return self.inner.sample_rate

    @property
    def channels(self) -> int:
        return self.inner.channels

    @property
    def signal_duration_seconds(self) -> float:
        return self.inner.signal_duration_seconds

    @property
    def plugin_state_path(self) -> str | None:
        return self.inner.plugin_state_path

    @property
    def editor_plugin(self) -> VST3Plugin | None:
        return self.inner.editor_plugin

    def render(
        self,
        params: dict[str, float],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Render dry audio and apply an independent fresh FDN per channel.

        :param params: Renderer-native synthesizer parameter values.
        :param midi_note: MIDI pitch of the note.
        :param velocity: MIDI velocity in the inclusive range ``[0, 127]``.
        :param note_start_and_end: Note start and end times in seconds.
        :param warmup: Whether the inner renderer performs editor warm-up.
        :returns: Effected channel-first contiguous ``float32`` audio.
        """
        dry = self.inner.render(
            params,
            midi_note,
            velocity,
            note_start_and_end,
            warmup=warmup,
        )
        wet = np.stack([self._process_channel(channel) for channel in dry])
        wet_mix = self.effect.wet_mix
        effected = np.asarray((1.0 - wet_mix) * dry + wet_mix * wet, dtype=np.float32)
        return _validate_rendered_audio(
            np.ascontiguousarray(effected),
            channels=self.channels,
            samples=int(self.sample_rate * self.signal_duration_seconds),
        )

    def _process_channel(self, channel: np.ndarray) -> np.ndarray:
        """Process one channel with fresh recursion and filter state.

        :param channel: One dry signal shaped ``(samples,)``.
        :returns: Wet mono signal with the same sample count.
        """
        build: FDNBuild = self._build
        post_delay = None if build.post_delay is None else SOSBank(build.post_delay)
        return np.asarray(
            process_fdn(
                channel,
                build.delays,
                build.A,
                build.B,
                build.C,
                build.D,
                post_delay=post_delay,
            )
        )
