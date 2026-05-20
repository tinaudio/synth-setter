"""Duck-typed VST3 plugin test double used in place of a real ``.vst3``."""

import threading
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

_MIDI_STATUS_MASK = 0xF0
_MIDI_NOTE_ON_STATUS = 0x90
_MIDI_MAX_VELOCITY = 127.0
_A4_MIDI_NOTE = 69
_A4_FREQUENCY_HZ = 440.0
_SEMITONES_PER_OCTAVE = 12.0
_DEFAULT_AMPLITUDE_SCALE = 0.5  # peak floor at velocity 127; keeps output within [-1, 1]


@dataclass
class _FakeParameter:
    raw_value: float = 0.0


class FakeVST3Plugin:
    """Stand-in for ``pedalboard.VST3Plugin`` that needs no ``.vst3`` and no X11.

    Mirrors the surface ``synth_setter.data.vst.core`` calls (``version``,
    ``parameters[k].raw_value``, ``show_editor``, ``load_preset``, ``reset``,
    ``process``). ``process`` emits a deterministic sine for any ``note_on``
    event and silence otherwise — enough to clear the loudness gate so the
    dataset pipeline writes a complete shard. ``block_size`` and ``tail`` are
    accepted to match the real plugin's signature but do not affect output.

    .. attribute :: version

       Fixed string the test-double reports for ``pedalboard.VST3Plugin.version``.
    """

    version = "fake-0.0.0"

    def __init__(self, plugin_path: str) -> None:
        """Construct the fake; ``plugin_path`` is recorded but never read from disk.

        :param plugin_path: Path string the production code passed; kept as metadata so callers
            asserting on it can read it back.
        """
        self.plugin_path = plugin_path
        self.parameters: defaultdict[str, _FakeParameter] = defaultdict(_FakeParameter)

    def show_editor(self, close_event: threading.Event) -> None:
        """Block until ``close_event`` is set, mirroring the real plugin's contract.

        :param close_event: Signalled by ``editor_held_open`` / ``warmup_plugin``
            to release the editor; ``wait`` returns the moment it is set.
        """
        close_event.wait()

    def load_preset(self, preset_path: str) -> None:
        """Accept the preset path; the fake has no preset state to apply.

        :param preset_path: Filesystem path the production code requested; ignored.
        """

    def reset(self) -> None:
        """Mirror the real plugin's reset hook; the fake holds no audio state."""

    def process(
        self,
        midi_events: Iterable[tuple[bytes, float]],
        duration_seconds: float,
        sample_rate: float,
        channels: int,
        block_size: int,
        tail: bool,
    ) -> np.ndarray:
        """Render deterministic audio matching the real plugin's output contract.

        Empty ``midi_events`` (the production flush calls) yield silence; any
        ``note_on`` yields a sine at the note's frequency, scaled by velocity.

        :param midi_events: Iterable of ``(payload_bytes, time_seconds)``; only
            the first ``note_on`` is honoured.
        :param duration_seconds: Output length; ``num_samples = duration * sample_rate``.
        :param sample_rate: Output sample rate in Hz.
        :param channels: Channel count of the returned ndarray (axis 0).
        :param block_size: Accepted to match the real plugin's signature; unused.
        :param tail: Accepted to match the real plugin's signature; unused.
        :returns: ``np.ndarray`` of shape ``(channels, num_samples)``, float32,
            with peak in ``[0, 1]`` and identical contents across repeated calls
            for the same inputs.
        """
        num_samples = int(duration_seconds * sample_rate)
        note = _first_note_on(midi_events)
        if note is None:
            return np.zeros((channels, num_samples), dtype=np.float32)

        pitch, velocity = note
        freq = _A4_FREQUENCY_HZ * 2.0 ** ((pitch - _A4_MIDI_NOTE) / _SEMITONES_PER_OCTAVE)
        amplitude = (velocity / _MIDI_MAX_VELOCITY) * _DEFAULT_AMPLITUDE_SCALE
        t = np.arange(num_samples, dtype=np.float32) / np.float32(sample_rate)
        wave = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        return np.broadcast_to(wave, (channels, num_samples)).copy()


def _first_note_on(midi_events: Iterable[tuple[bytes, float]]) -> tuple[int, int] | None:
    """Return ``(pitch, velocity)`` of the first ``note_on`` event, or ``None``.

    A ``note_on`` with velocity 0 is the MIDI idiom for ``note_off``; it does
    not count as a sounding event here.

    :param midi_events: Iterable of ``(payload_bytes, time_seconds)`` per the
        ``make_midi_events`` shape used by the production pipeline.
    :returns: ``(pitch, velocity)`` of the first true note-on event, or
        ``None`` if no sounding event is present.
    """
    for payload, _time in midi_events:
        status, pitch, velocity = payload[0], payload[1], payload[2]
        if (status & _MIDI_STATUS_MASK) == _MIDI_NOTE_ON_STATUS and velocity > 0:
            return pitch, velocity
    return None
