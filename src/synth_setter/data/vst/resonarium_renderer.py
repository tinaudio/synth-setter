"""Isolated single-note rendering through the upstream Resonarium Python bindings."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Integral
from typing import Any

import numpy as np

from synth_setter.data.vst.param_spec import ParameterValue, require_scalar_synth_params
from synth_setter.data.vst.renderers import (
    AudioRenderer,
    _sample_index_at_or_after,
    _validate_rendered_audio,
)
from synth_setter.data.vst.resonarium_param_spec import RESONARIUM_CONTROLS, RESONARIUM_VERSION
from synth_setter.renderer_backend import RESONARIUM_PLUGIN_NAME

_BLOCK_SIZE = 512
_CONTROLS = {control.uid: control for control in RESONARIUM_CONTROLS}


@dataclass(kw_only=True)
class ResonariumRenderer(AudioRenderer):
    """Render each call on a fresh native processor; mono downmixes the native stereo output.

    .. attribute :: synth_version
       Required ``resonarium-native`` package version, pinned in the persisted synth identity.
    """

    synth_version: str = RESONARIUM_VERSION

    def __post_init__(self) -> None:
        if self.plugin_path != RESONARIUM_PLUGIN_NAME or self.plugin_state_path:
            raise ValueError(
                "Resonarium requires plugin_path='resonarium' and no plugin_state_path"
            )
        if self.channels not in (1, 2):
            raise ValueError("Resonarium supports one or two channels")
        if not math.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        if not math.isfinite(self.signal_duration_seconds) or self.signal_duration_seconds <= 0:
            raise ValueError("signal_duration_seconds must be finite and positive")
        installed = version("resonarium-native")
        if installed != RESONARIUM_VERSION or self.synth_version != RESONARIUM_VERSION:
            raise ValueError(
                f"Resonarium requires resonarium-native {RESONARIUM_VERSION}; "
                f"installed={installed}, configured={self.synth_version}"
            )

    def render(
        self,
        params: Mapping[str, ParameterValue],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Render host-normalized controls without clipping or per-patch normalization.

        :param params: Partial controls keyed by native uid, each a host-normalized [0, 1] value.
        :param midi_note: MIDI pitch in [0, 127].
        :param velocity: MIDI velocity in [0, 127]; zero produces silence.
        :param note_start_and_end: Ordered note endpoints within the output duration, in seconds;
            a window inside one frame gates for one sample, and a note-on past the last frame is
            silent.
        :param warmup: Unsupported editor warm-up request.
        :returns: Finite float32 audio shaped (channels, samples); non-finite native audio raises
            the shared ``NonFiniteAudioError`` that dataset sampling counts as a rejection.
        :raises ValueError: Parameters, MIDI values, timing, or warm-up are invalid.
        :raises KeyError: A control is not part of the Resonarium specification.
        """
        import resonarium

        if warmup:
            raise ValueError("Resonarium has no editor to warm up")
        values = require_scalar_synth_params(params)
        unknown = sorted(values.keys() - _CONTROLS.keys())
        if unknown:
            raise KeyError(f"unknown Resonarium parameters: {', '.join(unknown)}")
        native = {uid: _CONTROLS[uid].native_value(value) for uid, value in values.items()}
        for name, value in (("midi_note", midi_note), ("velocity", velocity)):
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 127:
                raise ValueError(f"MIDI {name} must be an integer in [0, 127]")
        start, end = note_start_and_end
        if not (math.isfinite(start) and math.isfinite(end)) or not (
            0 <= start <= end <= self.signal_duration_seconds
        ):
            raise ValueError("note endpoints must satisfy 0 <= start <= end <= duration")

        samples = int(self.sample_rate * self.signal_duration_seconds)
        start_sample = _sample_index_at_or_after(start, self.sample_rate)
        if start_sample >= samples or velocity == 0:
            return np.zeros((self.channels, samples), dtype=np.float32)
        end_sample = max(start_sample + 1, _sample_index_at_or_after(end, self.sample_rate))

        synth = resonarium.Resonarium(float(self.sample_rate), _BLOCK_SIZE)
        for uid, value in native.items():
            synth.set_param(uid, value)
        # Match a host that is already running the patch: without one processed block, newly
        # enabled effects ramp their controls from defaults at onset, unlike the VST3.
        synth.process_multi_block(synth.create_multi_block(1))
        synth.reset()
        note = _Note(int(midi_note), int(velocity), start_sample, end_sample)
        stereo = _render_note(synth, note, samples)
        audio = stereo if self.channels == 2 else stereo.mean(axis=0, keepdims=True)
        return _validate_rendered_audio(
            audio.astype(np.float32, copy=False), channels=self.channels, samples=samples
        )


@dataclass(frozen=True)
class _Note:
    """One note's MIDI values and sample-accurate frames.

    .. attribute :: pitch
       MIDI pitch.
    .. attribute :: velocity
       Positive MIDI velocity.
    .. attribute :: start
       Note-on frame.
    .. attribute :: end
       Note-off frame, after ``start``.
    """

    pitch: int
    velocity: int
    start: int
    end: int


def _render_note(synth: Any, note: _Note, samples: int) -> np.ndarray:
    """Render one note into a buffer, sending each MIDI event at its exact frame.

    The binding replays queued MIDI into every block of a ``process_multi_block`` call, so a block
    holding an event is processed alone.

    :param synth: Fresh, configured ``resonarium.Resonarium`` processor.
    :param note: Note to play; its note-on frame precedes ``samples``.
    :param samples: Output frame count; a note-off at or past it is not sent.
    :returns: Native stereo audio shaped (2, samples).
    """
    blocks = -(-samples // _BLOCK_SIZE)
    buffer = synth.create_multi_block(blocks)
    cursor = 0

    def advance_to(block: int) -> None:
        nonlocal cursor
        if block > cursor:
            synth.process_multi_block(buffer, cursor, block - cursor)
            cursor = block

    on_block, on_offset = divmod(note.start, _BLOCK_SIZE)
    advance_to(on_block)
    synth.play_note(0, note.pitch, note.velocity, on_offset)
    if note.end < samples:
        off_block, off_offset = divmod(note.end, _BLOCK_SIZE)
        if off_block > on_block:
            advance_to(on_block + 1)
            advance_to(off_block)
        synth.release_note(0, note.pitch, 0, off_offset)
    advance_to(cursor + 1)
    advance_to(blocks)
    return np.asarray(buffer)[:, :samples]
