"""Tests for scripts/surge_xt_interactive.py prediction decoding helpers."""

import importlib
import queue
import threading
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from pedalboard.io import AudioFile

from src.data.vst import param_specs
from src.data.vst.param_spec import ParamSpec

SURGE_SIMPLE = "surge_simple"


@pytest.fixture(scope="module")
def surge_xt_interactive():
    """Import the script module lazily so collection doesn't fail on heavy imports."""
    return importlib.import_module("scripts.surge_xt_interactive")


@pytest.fixture(scope="module")
def simple_spec() -> ParamSpec:
    """The ``surge_simple`` ParamSpec used by the prediction-decoding tests."""
    return param_specs[SURGE_SIMPLE]


@pytest.fixture(scope="module")
def simple_spec_total_length(simple_spec: ParamSpec) -> int:
    """Total encoded row length (synth + note params) for the simple spec."""
    return simple_spec.synth_param_length + simple_spec.note_param_length


@pytest.fixture
def simple_pred_tensor(simple_spec_total_length: int) -> torch.Tensor:
    """A 2-row prediction tensor sized for the surge_simple spec.

    Row 0 cycles through ``[-1.0, 0.0, 1.0, 2.0]`` to exercise the
    ``(-1..1) -> (0..1)`` rescaling and clipping (``2.0`` is clipped to ``1``).
    Row 1 is all-zeros (decodes to mid-range values).
    """
    cycle = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    row_0 = np.tile(cycle, (simple_spec_total_length // 4) + 1)[:simple_spec_total_length]
    row_1 = np.zeros(simple_spec_total_length, dtype=np.float32)
    return torch.tensor(np.stack([row_0, row_1]), dtype=torch.float32)


def _write_param_array_h5(path: Path, rows: np.ndarray) -> None:
    """Write a 2D ``rows`` array to an h5 file under the ``param_array`` dataset."""
    with h5py.File(path, "w") as f:
        f.create_dataset("param_array", data=rows)


class TestDecodePredictionRow:
    """decode_prediction_row scales (-1..1) -> (0..1), clips, and decodes a row."""

    def test_returns_expected_keys_and_finite_floats(
        self,
        surge_xt_interactive,
        simple_pred_tensor: torch.Tensor,
        simple_spec: ParamSpec,
    ) -> None:
        """Decoded row contains every synth-param key with finite float values."""
        synth_params = surge_xt_interactive.decode_prediction_row(
            simple_pred_tensor, batch_idx=0, param_spec_name=SURGE_SIMPLE
        )

        expected_keys = {p.name for p in simple_spec.synth_params}
        assert set(synth_params.keys()) == expected_keys
        for name, value in synth_params.items():
            assert isinstance(value, float), f"{name} is {type(value).__name__}, expected float"
            assert np.isfinite(value), f"{name} = {value} is not finite"

    @pytest.mark.parametrize(
        "param_name, expected",
        [
            # col 0 = -1.0 -> rescaled 0.0 -> attack at spec min (0.0)
            ("a_amp_eg_attack", 0.0),
            # col 1 =  0.0 -> rescaled 0.5 -> decay at spec midpoint
            ("a_amp_eg_decay", 0.385),
            # col 2 =  1.0 -> rescaled 1.0 -> release at spec max
            ("a_amp_eg_release", 0.77),
            # col 3 =  2.0 -> clipped to 1.0 -> sustain at spec max
            ("a_amp_eg_sustain", 1.0),
        ],
    )
    def test_clips_and_rescales_per_column(
        self,
        surge_xt_interactive,
        simple_pred_tensor: torch.Tensor,
        param_name: str,
        expected: float,
    ) -> None:
        """Each column is rescaled from ``[-1, 1]`` to ``[0, 1]`` and clipped before decoding."""
        synth_params = surge_xt_interactive.decode_prediction_row(
            simple_pred_tensor, batch_idx=0, param_spec_name=SURGE_SIMPLE
        )
        assert synth_params[param_name] == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("bad_idx", [99, -1], ids=["above-range", "negative"])
    def test_out_of_range_idx_raises(
        self,
        surge_xt_interactive,
        simple_pred_tensor: torch.Tensor,
        bad_idx: int,
    ) -> None:
        """``batch_idx`` outside ``[0, batch_size)`` raises ``IndexError``."""
        with pytest.raises(IndexError):
            surge_xt_interactive.decode_prediction_row(
                simple_pred_tensor, batch_idx=bad_idx, param_spec_name=SURGE_SIMPLE
            )


class TestPredictionRefType:
    """PredictionRefType parses ``PATH:BATCH_IDX`` into a PredictionRef."""

    def test_parses_path_and_batch_idx(self, surge_xt_interactive) -> None:
        """A ``PATH:BATCH_IDX`` string parses into a ``PredictionRef`` with matching fields."""
        parser = surge_xt_interactive.PredictionRefType()

        ref = parser.convert("outputs/pred-0.pt:42", None, None)

        assert ref == surge_xt_interactive.PredictionRef(
            path=Path("outputs/pred-0.pt"), batch_idx=42
        )

    def test_splits_on_last_colon(self, surge_xt_interactive) -> None:
        """Absolute Windows-style paths still parse because rpartition uses the last ':'."""
        parser = surge_xt_interactive.PredictionRefType()

        ref = parser.convert(r"C:\models\pred-0.pt:7", None, None)

        assert ref.path == Path(r"C:\models\pred-0.pt")
        assert ref.batch_idx == 7

    @pytest.mark.parametrize(
        "value",
        ["pred-0.pt", "pred-0.pt:not-an-int", ":42", "pred-0.pt:"],
        ids=["missing-colon", "non-int-idx", "empty-path", "empty-idx"],
    )
    def test_rejects_invalid_uri(self, surge_xt_interactive, value: str) -> None:
        """Malformed prediction references raise ``click.BadParameter``."""
        parser = surge_xt_interactive.PredictionRefType()

        with pytest.raises(click.BadParameter):
            parser.convert(value, None, None)

    def test_rejects_negative_batch_idx(self, surge_xt_interactive) -> None:
        """Negative indices raise ``click.BadParameter`` to match ``decode_prediction_row``'s
        contract — h5py-style negative indexing would otherwise silently select the last row."""
        parser = surge_xt_interactive.PredictionRefType()

        with pytest.raises(click.BadParameter):
            parser.convert("pred-0.pt:-1", None, None)


class TestDatasetRefType:
    """DatasetRefType parses ``PATH:DATASET_IDX`` into a DatasetRef."""

    def test_parses_path_and_batch_idx(self, surge_xt_interactive) -> None:
        """A ``PATH:DATASET_IDX`` string parses into a ``DatasetRef`` with matching fields."""
        parser = surge_xt_interactive.DatasetRefType()

        ref = parser.convert("data/test.h5:3", None, None)

        assert ref == surge_xt_interactive.DatasetRef(path=Path("data/test.h5"), batch_idx=3)

    @pytest.mark.parametrize(
        "value",
        ["test.h5", "test.h5:not-an-int", ":0", "test.h5:"],
        ids=["missing-colon", "non-int-idx", "empty-path", "empty-idx"],
    )
    def test_rejects_invalid_uri(self, surge_xt_interactive, value: str) -> None:
        """Malformed dataset references raise ``click.BadParameter``."""
        parser = surge_xt_interactive.DatasetRefType()

        with pytest.raises(click.BadParameter):
            parser.convert(value, None, None)

    def test_rejects_negative_batch_idx(self, surge_xt_interactive) -> None:
        """Negative indices raise ``click.BadParameter`` — h5py's ``param_array[-1]`` would
        otherwise silently return the last row instead of failing."""
        parser = surge_xt_interactive.DatasetRefType()

        with pytest.raises(click.BadParameter):
            parser.convert("test.h5:-1", None, None)


class TestLoadPredictionSynthParams:
    """load_prediction_synth_params reads a .pt file row and decodes it."""

    def test_matches_decode_prediction_row_on_same_row(
        self,
        surge_xt_interactive,
        simple_pred_tensor: torch.Tensor,
        simple_spec: ParamSpec,
        tmp_path: Path,
    ) -> None:
        """Loading from disk and in-memory ``decode_prediction_row`` produce identical outputs."""
        pred_path = tmp_path / "pred-0.pt"
        torch.save(simple_pred_tensor, pred_path)
        ref = surge_xt_interactive.PredictionRef(path=pred_path, batch_idx=0)

        loaded = surge_xt_interactive.load_prediction_synth_params(
            ref, param_spec_name=SURGE_SIMPLE
        )

        direct = surge_xt_interactive.decode_prediction_row(
            simple_pred_tensor, batch_idx=0, param_spec_name=SURGE_SIMPLE
        )
        expected_keys = {p.name for p in simple_spec.synth_params}
        assert set(loaded.keys()) == expected_keys
        assert loaded == direct


class TestLoadDatasetSynthParams:
    """load_dataset_synth_params reads an h5 ``param_array`` row and decodes it."""

    def test_round_trip_returns_original_synth_params(
        self,
        surge_xt_interactive,
        simple_spec: ParamSpec,
        tmp_path: Path,
    ) -> None:
        """Encoding params, persisting to h5, and reloading recovers the original synth params."""
        synth_param_dict, note_param_dict = simple_spec.sample()
        encoded = simple_spec.encode(synth_param_dict, note_param_dict)
        h5_path = tmp_path / "test.h5"
        _write_param_array_h5(h5_path, encoded[None, :])
        ref = surge_xt_interactive.DatasetRef(path=h5_path, batch_idx=0)

        loaded = surge_xt_interactive.load_dataset_synth_params(ref, param_spec_name=SURGE_SIMPLE)

        for name, value in synth_param_dict.items():
            assert loaded[name] == pytest.approx(value, abs=1e-5)

    def test_selects_correct_row(
        self,
        surge_xt_interactive,
        simple_spec: ParamSpec,
        tmp_path: Path,
    ) -> None:
        """``batch_idx`` selects the matching row from a multi-row ``param_array``."""
        row_0_synth, row_0_note = simple_spec.sample()
        row_1_synth, row_1_note = simple_spec.sample()
        encoded = np.stack(
            [
                simple_spec.encode(row_0_synth, row_0_note),
                simple_spec.encode(row_1_synth, row_1_note),
            ]
        )
        h5_path = tmp_path / "test.h5"
        _write_param_array_h5(h5_path, encoded)
        ref = surge_xt_interactive.DatasetRef(path=h5_path, batch_idx=1)

        loaded = surge_xt_interactive.load_dataset_synth_params(ref, param_spec_name=SURGE_SIMPLE)

        for name, value in row_1_synth.items():
            assert loaded[name] == pytest.approx(value, abs=1e-5)

    def test_out_of_range_idx_raises(
        self,
        surge_xt_interactive,
        simple_spec: ParamSpec,
        tmp_path: Path,
    ) -> None:
        """A ``batch_idx`` past the end of ``param_array`` raises ``IndexError`` or
        ``ValueError``."""
        encoded = simple_spec.encode(*simple_spec.sample())
        h5_path = tmp_path / "test.h5"
        _write_param_array_h5(h5_path, encoded[None, :])
        ref = surge_xt_interactive.DatasetRef(path=h5_path, batch_idx=99)

        with pytest.raises((IndexError, ValueError)):
            surge_xt_interactive.load_dataset_synth_params(ref, param_spec_name=SURGE_SIMPLE)

    @pytest.mark.requires_vst
    @pytest.mark.slow
    def test_loads_row_from_surge_xt_smoke_fixture(
        self,
        surge_xt_interactive,
        surge_xt_smoke_datasets: Path,
        param_spec_name: str,
    ) -> None:
        """Loads row 0 from the real ``surge_xt_smoke_datasets`` test.h5.

        The decode spec must match the spec the fixture generated the dataset with —
        otherwise the decoder slices off the end of the row and ``.item()`` raises.
        """
        ref = surge_xt_interactive.DatasetRef(
            path=surge_xt_smoke_datasets / "test.h5", batch_idx=0
        )

        loaded = surge_xt_interactive.load_dataset_synth_params(
            ref, param_spec_name=param_spec_name
        )

        expected_keys = {p.name for p in param_specs[param_spec_name].synth_params}
        assert set(loaded.keys()) == expected_keys
        for name, value in loaded.items():
            assert isinstance(value, float), f"{name} is {type(value).__name__}, expected float"
            assert np.isfinite(value), f"{name} = {value} is not finite"


class _ConstantPlugin:
    """Stand-in plugin with a ``.process(...)`` method that returns constant audio.

    Duck-typed to satisfy ``play_audio_recorded``'s ``plugin.process(...)`` call —
    avoids loading a real VST3, which is unavailable in headless test runs.
    Stashes its last call's ``midi_messages`` argument for assertion in tests.
    """

    def __init__(self, sample_value: float) -> None:
        self.sample_value = sample_value
        self.process_call_count = 0
        self.last_midi_messages: list | None = None

    def process(
        self,
        midi_messages: list,
        duration_seconds: float,
        sample_rate: float,
        num_channels: int,
        buffer_size: int,
        reset: bool,
    ) -> np.ndarray:
        """Return a constant-valued ``(num_channels, duration * sample_rate)`` buffer."""
        del buffer_size, reset
        self.process_call_count += 1
        self.last_midi_messages = list(midi_messages)
        frames = int(duration_seconds * sample_rate)
        return np.full((num_channels, frames), self.sample_value, dtype=np.float32)


class TestPlayAudioRecorded:
    """play_audio_recorded renders a deterministic clip via a single plugin.process() call."""

    def test_writes_exact_duration_frames(self, surge_xt_interactive, tmp_path: Path) -> None:
        """The WAV's frame count is exactly ``DURATION * SAMPLE_RATE`` (one process call)."""
        plugin = _ConstantPlugin(sample_value=0.25)
        output_path = tmp_path / "session.wav"
        expected_frames = int(
            surge_xt_interactive.SESSION_RECORDING_DURATION_SECONDS
            * surge_xt_interactive.SAMPLE_RATE
        )

        surge_xt_interactive.play_audio_recorded(plugin, output_path)

        assert plugin.process_call_count == 1
        assert output_path.is_file()
        with AudioFile(str(output_path)) as f:
            audio = f.read(f.frames)
        assert audio.shape == (surge_xt_interactive.CHANNELS, expected_frames)
        np.testing.assert_allclose(audio, plugin.sample_value, atol=1e-3)

    def test_passes_expected_midi_events(self, surge_xt_interactive, tmp_path: Path) -> None:
        """plugin.process is called with note_on/off middle-C events at NOTE_START/END."""
        plugin = _ConstantPlugin(sample_value=0.0)

        surge_xt_interactive.play_audio_recorded(plugin, tmp_path / "events.wav")

        assert plugin.last_midi_messages is not None
        assert len(plugin.last_midi_messages) == 2
        (note_on_bytes, note_on_t), (note_off_bytes, note_off_t) = plugin.last_midi_messages

        assert note_on_t == pytest.approx(
            surge_xt_interactive.SESSION_RECORDING_NOTE_START_SECONDS
        )
        assert note_off_t == pytest.approx(surge_xt_interactive.SESSION_RECORDING_NOTE_END_SECONDS)

        # MIDI wire format: status byte (high nibble = type), note, velocity.
        # 0x90 = note_on (channel 0), 0x80 = note_off (channel 0).
        assert note_on_bytes[0] & 0xF0 == 0x90
        assert note_off_bytes[0] & 0xF0 == 0x80
        assert note_on_bytes[1] == surge_xt_interactive.SESSION_RECORDING_MIDI_NOTE
        assert note_off_bytes[1] == surge_xt_interactive.SESSION_RECORDING_MIDI_NOTE
        assert note_on_bytes[2] == surge_xt_interactive.SESSION_RECORDING_VELOCITY


class _FakeMidiMessage:
    """Minimal stand-in for a ``mido.Message`` — exposes ``type`` and ``bytes()``.

    ``bytes()`` returns ``list[int]`` to match real ``mido.Message.bytes()``.
    """

    def __init__(self, msg_type: str, payload: list[int] | None = None) -> None:
        self.type = msg_type
        self._payload = payload if payload is not None else [0x90, 0x3C, 0x40]

    def bytes(self) -> list[int]:
        return list(self._payload)

    def __repr__(self) -> str:
        return f"_FakeMidiMessage({self.type!r})"


class _FakeMidiPortHandle:
    """Mido-input replacement.

    ``poll()`` returns the next queued message or ``None`` when drained, mirroring
    ``mido.ports.IOPort.poll``. When the queue is exhausted, sets ``drain_event``
    (if provided) so the test can flip ``stop_event`` and let the listener exit
    deterministically — no time.sleep / wall-clock polling on the test side.
    """

    def __init__(
        self,
        messages: list[_FakeMidiMessage],
        drain_event: threading.Event | None = None,
    ) -> None:
        self._messages = list(messages)
        self._drain_event = drain_event

    def __enter__(self) -> "_FakeMidiPortHandle":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def poll(self) -> _FakeMidiMessage | None:
        if not self._messages:
            if self._drain_event is not None:
                self._drain_event.set()
            return None
        return self._messages.pop(0)


class TestMidiListener:
    """``midi_listener`` filters mido messages by type and forwards them to a queue."""

    def test_only_relevant_message_types_are_forwarded(self, surge_xt_interactive) -> None:
        """note_on/off, control_change, pitchwheel, aftertouch are queued; others are dropped."""
        forwarded = [
            _FakeMidiMessage("note_on", [0x90, 0x3C, 0x40]),
            _FakeMidiMessage("note_off", [0x80, 0x3C, 0x00]),
            _FakeMidiMessage("control_change", [0xB0, 0x07, 0x7F]),
            _FakeMidiMessage("pitchwheel", [0xE0, 0x00, 0x40]),
            _FakeMidiMessage("aftertouch", [0xD0, 0x40]),
        ]
        dropped = [
            _FakeMidiMessage("polytouch"),
            _FakeMidiMessage("sysex"),
            _FakeMidiMessage("clock"),
        ]
        # Interleave to make sure the filter doesn't depend on order.
        all_messages = [
            forwarded[0],
            dropped[0],
            forwarded[1],
            dropped[1],
            *forwarded[2:],
            dropped[2],
        ]

        drain_event = threading.Event()

        def fake_port_opener(_port_name: str) -> _FakeMidiPortHandle:
            return _FakeMidiPortHandle(all_messages, drain_event=drain_event)

        midi_queue: queue.Queue[tuple[list[int], float]] = queue.Queue()
        stop_event = threading.Event()
        # ``daemon=True`` so a hung listener can't block pytest shutdown if the
        # ``drain_event`` assertion below fails before we set ``stop_event``.
        listener_thread = threading.Thread(
            target=surge_xt_interactive.midi_listener,
            args=("fake-port", midi_queue, stop_event),
            kwargs={"port_opener": fake_port_opener},
            daemon=True,
        )
        listener_thread.start()
        # Belt-and-suspenders: even though the thread is now daemonic, signal shutdown
        # and join in a finally block so a failed drain assertion still cleans up.
        try:
            # ``_FakeMidiPortHandle`` flips ``drain_event`` once the queued list is empty,
            # so we know the listener has observed every message before we ask it to stop.
            assert drain_event.wait(timeout=2.0), "listener did not drain fake messages"
        finally:
            stop_event.set()
            listener_thread.join(timeout=2.0)
        assert not listener_thread.is_alive(), "listener did not exit on stop_event"

        drained: list[tuple[list[int], float]] = []
        while not midi_queue.empty():
            drained.append(midi_queue.get_nowait())

        assert drained == [(m.bytes(), 0.0) for m in forwarded]

    def test_stop_event_exits_listener_with_no_messages(self, surge_xt_interactive) -> None:
        """``stop_event`` set before any message arrives drains the listener cleanly."""

        class _IdlePort:
            def __enter__(self) -> "_IdlePort":
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def poll(self) -> None:
                return None

        midi_queue: queue.Queue[tuple[list[int], float]] = queue.Queue()
        stop_event = threading.Event()
        stop_event.set()

        surge_xt_interactive.midi_listener(
            "fake-port", midi_queue, stop_event, port_opener=lambda _port: _IdlePort()
        )

        assert midi_queue.empty()

    def test_open_input_failure_logs_and_exits_thread(self, surge_xt_interactive, caplog) -> None:
        """``midi_listener`` must not raise; failures are logged so the daemon exits cleanly."""

        def raising_port_opener(_port_name: str) -> object:
            raise OSError("device disconnected")

        midi_queue: queue.Queue[tuple[list[int], float]] = queue.Queue()
        stop_event = threading.Event()
        with caplog.at_level("ERROR"):
            surge_xt_interactive.midi_listener(
                "fake-port", midi_queue, stop_event, port_opener=raising_port_opener
            )

        assert midi_queue.empty()
        assert any("MIDI listener thread aborted" in rec.message for rec in caplog.records)


class _FakeStream:
    """Captures buffers written by ``play_audio``; an ``AudioStreamProtocol`` stand-in."""

    def __init__(self) -> None:
        self.writes: list[np.ndarray] = []

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def write(self, buffer: np.ndarray, _sample_rate: int) -> None:
        self.writes.append(np.asarray(buffer))


class _RecordingPlugin:
    """Stand-in for a ``VST3Plugin`` that records the ``messages`` arg to ``process()``.

    When ``stop_event`` and ``stop_after_n_calls`` are provided, the Nth ``process()``
    invocation flips ``stop_event`` so ``play_audio`` exits its loop deterministically.
    This replaces the late ``monkeypatch.setattr(plugin, "process", ...)`` re-bind that
    earlier versions of ``TestPlayAudioQueueDrain`` used to drive shutdown.
    """

    def __init__(
        self,
        channels: int,
        buffer_size: int,
        *,
        stop_event: threading.Event | None = None,
        stop_after_n_calls: int = 1,
    ) -> None:
        self._channels = channels
        self._buffer_size = buffer_size
        self._stop_event = stop_event
        self._stop_after_n_calls = stop_after_n_calls
        self.messages_per_call: list[list[tuple[list[int], float]]] = []

    def process(
        self,
        messages: list[tuple[list[int], float]],
        _duration_seconds: float,
        _sample_rate: int,
        _channels: int,
        _buffer_size: int,
        reset: bool,
    ) -> np.ndarray:
        assert reset is False
        self.messages_per_call.append(list(messages))
        if (
            self._stop_event is not None
            and len(self.messages_per_call) >= self._stop_after_n_calls
        ):
            self._stop_event.set()
        return np.zeros((self._channels, self._buffer_size), dtype=np.float32)


class TestPlayAudioQueueDrain:
    """``play_audio`` drains the MIDI queue into ``plugin.process`` once per buffer."""

    def test_queued_events_are_forwarded_to_plugin_process(self, surge_xt_interactive) -> None:
        """Tuples enqueued by the listener are drained and passed to ``plugin.process``."""
        stop_event = threading.Event()
        plugin = _RecordingPlugin(
            surge_xt_interactive.CHANNELS,
            surge_xt_interactive.BUFFER_SIZE,
            stop_event=stop_event,
            stop_after_n_calls=1,
        )
        stream = _FakeStream()

        midi_queue: queue.Queue[tuple[list[int], float]] = queue.Queue()
        midi_queue.put(([0x90, 0x3C, 0x40], 0.0))
        midi_queue.put(([0x80, 0x3C, 0x00], 0.0))

        surge_xt_interactive.play_audio(
            plugin, stop_event, midi_queue, audio_stream_factory=lambda: stream
        )

        assert plugin.messages_per_call == [[([0x90, 0x3C, 0x40], 0.0), ([0x80, 0x3C, 0x00], 0.0)]]
        assert stop_event.is_set()
        assert len(stream.writes) == 1

    def test_none_queue_passes_empty_messages_list(self, surge_xt_interactive) -> None:
        """When no MIDI port is configured, ``play_audio`` is invoked with ``midi_queue=None``."""
        stop_event = threading.Event()
        plugin = _RecordingPlugin(
            surge_xt_interactive.CHANNELS,
            surge_xt_interactive.BUFFER_SIZE,
            stop_event=stop_event,
            stop_after_n_calls=1,
        )
        stream = _FakeStream()

        surge_xt_interactive.play_audio(
            plugin, stop_event, None, audio_stream_factory=lambda: stream
        )

        assert plugin.messages_per_call == [[]]
        assert stop_event.is_set()

    def test_drain_is_capped_at_max_midi_events_per_buffer(self, surge_xt_interactive) -> None:
        """A queue larger than ``_MAX_MIDI_EVENTS_PER_BUFFER`` drains in chunks across buffers,
        preventing one realtime callback from stretching to process the full backlog."""
        cap = surge_xt_interactive._MAX_MIDI_EVENTS_PER_BUFFER
        stop_event = threading.Event()
        plugin = _RecordingPlugin(
            surge_xt_interactive.CHANNELS,
            surge_xt_interactive.BUFFER_SIZE,
            stop_event=stop_event,
            stop_after_n_calls=1,  # stop after the first buffer to assert the per-buffer cap
        )
        stream = _FakeStream()

        midi_queue: queue.Queue[tuple[list[int], float]] = queue.Queue()
        # Enqueue cap + 5 events; first buffer should drain ``cap``, leaving 5 in the queue.
        for _ in range(cap + 5):
            midi_queue.put(([0x90, 0x3C, 0x40], 0.0))

        surge_xt_interactive.play_audio(
            plugin, stop_event, midi_queue, audio_stream_factory=lambda: stream
        )

        assert len(plugin.messages_per_call) == 1
        assert len(plugin.messages_per_call[0]) == cap
        assert midi_queue.qsize() == 5


class TestResolveMidiPort:
    """``_resolve_midi_port`` maps the click flag value to a concrete port name."""

    def test_returns_first_available_when_requested_is_empty_string(
        self, surge_xt_interactive
    ) -> None:
        """Empty string means auto-pick: return ``available[0]``."""
        resolved = surge_xt_interactive._resolve_midi_port("", ["port-a", "port-b"])
        assert resolved == "port-a"

    def test_returns_requested_when_present_in_available(self, surge_xt_interactive) -> None:
        """A named port that exists in ``available`` is returned verbatim."""
        resolved = surge_xt_interactive._resolve_midi_port("port-b", ["port-a", "port-b"])
        assert resolved == "port-b"

    def test_raises_usage_error_when_requested_not_in_available(
        self, surge_xt_interactive
    ) -> None:
        """A named port absent from ``available`` raises ``click.UsageError``."""
        with pytest.raises(click.UsageError, match="port-z"):
            surge_xt_interactive._resolve_midi_port("port-z", ["port-a", "port-b"])

    def test_raises_usage_error_when_available_is_empty_and_auto(
        self, surge_xt_interactive
    ) -> None:
        """Auto-pick with no ports available raises ``click.UsageError``."""
        with pytest.raises(click.UsageError, match="no MIDI input"):
            surge_xt_interactive._resolve_midi_port("", [])

    def test_raises_usage_error_when_available_is_empty_and_named(
        self, surge_xt_interactive
    ) -> None:
        """Named port with no ports available raises ``click.UsageError``."""
        with pytest.raises(click.UsageError, match="no MIDI input"):
            surge_xt_interactive._resolve_midi_port("port-a", [])


class _FakeParam:
    """Stand-in for a pedalboard plugin parameter — exposes only ``raw_value``."""

    def __init__(self, raw_value: float) -> None:
        self.raw_value = raw_value


class _FakePlugin:
    """Stand-in plugin with a ``.parameters`` dict-like for drift-detection tests."""

    def __init__(self, params: dict[str, float]) -> None:
        self.parameters = {name: _FakeParam(value) for name, value in params.items()}


class _FakeSpec:
    """Stand-in ParamSpec exposing only the ``synth_param_names`` attribute."""

    def __init__(self, synth_param_names: list[str]) -> None:
        self.synth_param_names = synth_param_names


class TestValidateNoDrift:
    """``_validate_no_drift`` raises if a non-spec param drifted from its default."""

    def test_returns_none_when_all_non_spec_params_at_default(self, surge_xt_interactive) -> None:
        """No drift on any non-spec param → no exception."""
        plugin = _FakePlugin({"a_synth": 0.5, "fx_amount": 0.3, "global_volume": 0.7})
        spec = _FakeSpec(["a_synth"])
        defaults = {"a_synth": 0.5, "fx_amount": 0.3, "global_volume": 0.7}

        result = surge_xt_interactive._validate_no_drift(plugin, spec, defaults)

        assert result is None

    def test_raises_value_error_when_non_spec_param_drifted(self, surge_xt_interactive) -> None:
        """A non-spec param away from its default → ``ValueError`` naming the param."""
        plugin = _FakePlugin({"a_synth": 0.5, "fx_amount": 0.9})
        spec = _FakeSpec(["a_synth"])
        defaults = {"a_synth": 0.5, "fx_amount": 0.3}

        with pytest.raises(ValueError, match="fx_amount"):
            surge_xt_interactive._validate_no_drift(plugin, spec, defaults)

    def test_ignores_drift_on_spec_params(self, surge_xt_interactive) -> None:
        """Spec params are allowed to vary; only non-spec drift is flagged."""
        plugin = _FakePlugin({"a_synth": 0.99, "fx_amount": 0.3})
        spec = _FakeSpec(["a_synth"])
        defaults = {"a_synth": 0.5, "fx_amount": 0.3}

        result = surge_xt_interactive._validate_no_drift(plugin, spec, defaults)

        assert result is None

    def test_drift_within_tolerance_does_not_raise(self, surge_xt_interactive) -> None:
        """Tiny float deviation within abs_tol=1e-6 is treated as equal."""
        plugin = _FakePlugin({"fx_amount": 0.3 + 5e-7})
        spec = _FakeSpec([])
        defaults = {"fx_amount": 0.3}

        result = surge_xt_interactive._validate_no_drift(plugin, spec, defaults)

        assert result is None

    def test_drift_just_above_tolerance_raises(self, surge_xt_interactive) -> None:
        """Deviation just above abs_tol=1e-6 is flagged."""
        plugin = _FakePlugin({"fx_amount": 0.3 + 1e-3})
        spec = _FakeSpec([])
        defaults = {"fx_amount": 0.3}

        with pytest.raises(ValueError, match="fx_amount"):
            surge_xt_interactive._validate_no_drift(plugin, spec, defaults)


def _build_keyboard_loop_plugin(simple_spec: ParamSpec) -> tuple["_FakePlugin", dict[str, float]]:
    """Build a ``_FakePlugin`` carrying every ``surge_simple`` synth param + two non-spec params,
    plus the matching ``default_params`` dict that ``_validate_no_drift`` checks against.

    Returns ``(plugin, default_params)``.
    """
    spec_defaults = {name: 0.25 for name in simple_spec.synth_param_names}
    non_spec_defaults = {"fx_amount": 0.1, "global_volume": 0.7}
    plugin = _FakePlugin({**spec_defaults, **non_spec_defaults})
    return plugin, {**spec_defaults, **non_spec_defaults}


class TestKeyboardLoop:
    """``keyboard_loop`` reads keystrokes via the injectable ``keystroke_source`` and snapshots
    plugin params on ``p`` until ``q``, ``stop_event``, or source exhaustion."""

    def test_p_records_patch_q_quits(self, surge_xt_interactive, simple_spec: ParamSpec) -> None:
        """``["p", "q"]`` records exactly one patch with every spec synth-param key, then quits."""
        plugin, default_params = _build_keyboard_loop_plugin(simple_spec)
        stop_event = threading.Event()
        keystrokes = iter(["p", "q"])

        patches = surge_xt_interactive.keyboard_loop(
            plugin,
            stop_event,
            SURGE_SIMPLE,
            default_params,
            keystroke_source=keystrokes.__next__,
        )

        assert len(patches) == 1
        assert set(patches[0]) == set(simple_spec.synth_param_names)
        for name, value in patches[0].items():
            assert isinstance(value, float)
            assert np.isfinite(value), f"{name} = {value} is not finite"
        assert stop_event.is_set()

    def test_unknown_keys_are_ignored(self, surge_xt_interactive, simple_spec: ParamSpec) -> None:
        """Keystrokes outside ``{p, q}`` don't record patches, don't set ``stop_event``, and don't
        raise — the loop simply waits for the next keystroke."""
        plugin, default_params = _build_keyboard_loop_plugin(simple_spec)
        stop_event = threading.Event()
        keystrokes = iter(["x", "p", "z", "q"])

        patches = surge_xt_interactive.keyboard_loop(
            plugin,
            stop_event,
            SURGE_SIMPLE,
            default_params,
            keystroke_source=keystrokes.__next__,
        )

        assert len(patches) == 1, "x and z should be ignored — only p records"
        assert stop_event.is_set()

    def test_stop_event_set_externally_exits_without_consuming_source(
        self, surge_xt_interactive, simple_spec: ParamSpec
    ) -> None:
        """When ``stop_event`` is already set, the loop returns ``[]`` immediately — the keystroke
        source is not even polled."""
        plugin, default_params = _build_keyboard_loop_plugin(simple_spec)
        stop_event = threading.Event()
        stop_event.set()

        consumed: list[str] = []

        def watching_source() -> str:
            consumed.append("polled")
            return "p"

        patches = surge_xt_interactive.keyboard_loop(
            plugin,
            stop_event,
            SURGE_SIMPLE,
            default_params,
            keystroke_source=watching_source,
        )

        assert patches == []
        assert consumed == [], "loop must check stop_event before polling the source"

    def test_source_exhaustion_quits_gracefully_and_sets_stop_event(
        self, surge_xt_interactive, simple_spec: ParamSpec
    ) -> None:
        """A ``StopIteration`` from the source (no ``q`` pressed) cleanly returns recorded patches
        and sets ``stop_event`` so downstream threads notice the exit."""
        plugin, default_params = _build_keyboard_loop_plugin(simple_spec)
        stop_event = threading.Event()
        keystrokes = iter(["p"])  # No q — source will raise StopIteration on the second poll.

        patches = surge_xt_interactive.keyboard_loop(
            plugin,
            stop_event,
            SURGE_SIMPLE,
            default_params,
            keystroke_source=keystrokes.__next__,
        )

        assert len(patches) == 1
        assert stop_event.is_set()

    def test_drift_during_record_raises_valueerror_and_sets_stop_event(
        self, surge_xt_interactive, simple_spec: ParamSpec
    ) -> None:
        """If a non-spec param has drifted from its default, ``p`` triggers ``_validate_no_drift``
        which raises ``ValueError``; the loop sets ``stop_event`` and re-raises (so the
        orchestrator sees the failure instead of silently dropping it)."""
        plugin, default_params = _build_keyboard_loop_plugin(simple_spec)
        # Drift the non-spec ``fx_amount`` from its default to trip _validate_no_drift.
        plugin.parameters["fx_amount"].raw_value = 0.99
        stop_event = threading.Event()
        keystrokes = iter(["p"])

        with pytest.raises(ValueError, match="drifted"):
            surge_xt_interactive.keyboard_loop(
                plugin,
                stop_event,
                SURGE_SIMPLE,
                default_params,
                keystroke_source=keystrokes.__next__,
            )
        assert stop_event.is_set()


def _write_pred_files(
    output_dir: Path,
    num_samples: int,
    *,
    pred_tensor_factory=None,
) -> None:
    """Write the per-sample ``pred-{i}.pt`` / ``target-audio-{i}.pt`` / ``target-params-{i}.pt``
    files ``PredictionWriter`` would emit, populated with finite tensors by default.

    ``pred_tensor_factory`` lets a test override the ``pred-{i}.pt`` payload (e.g. to inject
    NaN/Inf); the target tensors are always finite stubs.
    """

    def _default_factory(_idx: int) -> torch.Tensor:
        return torch.zeros((1, 4), dtype=torch.float32)

    factory = pred_tensor_factory if pred_tensor_factory is not None else _default_factory
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_samples):
        torch.save(factory(i), output_dir / f"pred-{i}.pt")
        torch.save(torch.zeros(1, dtype=torch.float32), output_dir / f"target-audio-{i}.pt")
        torch.save(torch.zeros(1, dtype=torch.float32), output_dir / f"target-params-{i}.pt")


class TestExpectedPredictionFilenames:
    """``_expected_prediction_filenames`` enumerates ``PredictionWriter``'s output names."""

    def test_returns_three_names_per_sample(self, surge_xt_interactive) -> None:
        """For ``num_samples`` samples, three sorted filenames per sample are returned."""
        names = surge_xt_interactive._expected_prediction_filenames(num_samples=2)

        assert names == sorted(
            [
                "pred-0.pt",
                "pred-1.pt",
                "target-audio-0.pt",
                "target-audio-1.pt",
                "target-params-0.pt",
                "target-params-1.pt",
            ]
        )

    def test_zero_samples_returns_empty(self, surge_xt_interactive) -> None:
        """Zero samples returns an empty list."""
        assert surge_xt_interactive._expected_prediction_filenames(num_samples=0) == []


class _RecordingSubprocessRunner:
    """Test fake matching the ``SubprocessRunner`` shape from ``surge_xt_interactive``.

    Records every invocation's positional argv and keyword arguments so tests assert on
    real state instead of a ``MagicMock`` argv-spy. Used by every test that exercises a
    ``*_runner=`` seam introduced in the de-mock refactor (see issue #844).

    Returns ``0`` from ``__call__`` (a valid ``subprocess.check_call`` return; production
    callers ignore the value or read ``returncode`` from a ``CompletedProcess``-like object,
    which we don't simulate here — YAGNI).
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.kwargs_per_call: list[dict[str, object]] = []

    def __call__(self, args: list[str], **kwargs: object) -> int:
        self.calls.append(list(args))
        self.kwargs_per_call.append(dict(kwargs))
        return 0


class TestRunPredict:
    """``_run_predict`` builds the ``src/eval.py`` invocation with the right Hydra overrides."""

    def test_passes_d_out_override_and_absolute_paths(self, surge_xt_interactive) -> None:
        """``_run_predict`` overrides ``model.net.d_out`` with the encoded width of
        ``param_spec_name`` (otherwise the ``???`` sentinel in ``surge/test.yaml`` would error),
        and resolves all paths to absolute (otherwise Hydra's ``chdir`` would break relative
        refs)."""
        # Use relative paths so the test fails if .resolve() is dropped.
        ckpt = Path("relative/ckpt.ckpt")
        dataset_root = Path("relative/dataset")
        predict_file = Path("relative/dataset/predict.h5")
        predictions_dir = Path("relative/preds")

        runner = _RecordingSubprocessRunner()

        surge_xt_interactive._run_predict(
            ckpt,
            dataset_root,
            predict_file,
            predictions_dir,
            SURGE_SIMPLE,
            subprocess_runner=runner,
        )

        assert len(runner.calls) == 1, f"expected one invocation, got {runner.calls!r}"
        args = runner.calls[0]
        assert "experiment=surge/test" in args
        assert "mode=predict" in args
        # d_out must equal len(param_specs[SURGE_SIMPLE]) = synth+note width.
        expected_d_out = len(param_specs[SURGE_SIMPLE])
        assert f"model.net.d_out={expected_d_out}" in args
        # Every path-bearing override must be absolute.
        for prefix, original in (
            ("ckpt_path=", ckpt),
            ("data.predict_file=", predict_file),
            ("data.dataset_root=", dataset_root),
            ("callbacks.prediction_writer.output_dir=", predictions_dir),
        ):
            arg = next(a for a in args if a.startswith(prefix))
            value = arg.removeprefix(prefix)
            assert Path(value).is_absolute(), f"{prefix} should be absolute, got {value!r}"
            assert value == str(original.resolve())


class TestValidatePredictions:
    """``_validate_predictions`` checks expected files exist and tensors are finite."""

    def test_passes_on_complete_finite_outputs(self, surge_xt_interactive, tmp_path: Path) -> None:
        """Happy path: complete file set with finite predictions does not raise."""
        _write_pred_files(tmp_path, num_samples=2)

        surge_xt_interactive._validate_predictions(tmp_path, num_samples=2)

    def test_missing_file_raises_filenotfounderror(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """A missing per-sample file raises ``FileNotFoundError`` listing the missing entry."""
        _write_pred_files(tmp_path, num_samples=2)
        (tmp_path / "pred-1.pt").unlink()

        with pytest.raises(FileNotFoundError, match="pred-1.pt"):
            surge_xt_interactive._validate_predictions(tmp_path, num_samples=2)

    def test_extra_file_raises_filenotfounderror(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """An unexpected file in the directory raises ``FileNotFoundError`` (set mismatch)."""
        _write_pred_files(tmp_path, num_samples=1)
        torch.save(torch.zeros(1), tmp_path / "stray-extra.pt")

        with pytest.raises(FileNotFoundError, match="stray-extra.pt"):
            surge_xt_interactive._validate_predictions(tmp_path, num_samples=1)

    def test_nan_prediction_raises_valueerror(self, surge_xt_interactive, tmp_path: Path) -> None:
        """A NaN value in any ``pred-{i}.pt`` raises ``ValueError`` naming the offending file."""

        def factory(idx: int) -> torch.Tensor:
            if idx == 0:
                return torch.tensor([[float("nan")]], dtype=torch.float32)
            return torch.zeros((1, 1), dtype=torch.float32)

        _write_pred_files(tmp_path, num_samples=2, pred_tensor_factory=factory)

        with pytest.raises(ValueError, match="pred-0.pt"):
            surge_xt_interactive._validate_predictions(tmp_path, num_samples=2)


class TestValidateMetricsDf:
    """``_validate_metrics_df`` validates row count, expected columns, and finiteness."""

    def test_passes_on_matching_shape_and_finite(self, surge_xt_interactive) -> None:
        """Happy path: matching rows, expected columns, all-finite values does not raise."""
        df = pd.DataFrame({"mss": [0.1, 0.2], "extra": [1.0, 2.0]})
        spec = surge_xt_interactive._MetricsFileSpec(rows=2, columns=frozenset({"mss"}))

        surge_xt_interactive._validate_metrics_df(Path("metrics.csv"), df, spec)

    def test_wrong_rows_raises_valueerror(self, surge_xt_interactive) -> None:
        """Row count mismatch raises ``ValueError`` mentioning expected and actual."""
        df = pd.DataFrame({"mss": [0.1]})
        spec = surge_xt_interactive._MetricsFileSpec(rows=2, columns=frozenset({"mss"}))

        with pytest.raises(ValueError, match="expected 2 rows"):
            surge_xt_interactive._validate_metrics_df(Path("metrics.csv"), df, spec)

    def test_missing_column_raises_valueerror(self, surge_xt_interactive) -> None:
        """A missing expected column raises ``ValueError`` listing the missing column."""
        df = pd.DataFrame({"other": [0.1, 0.2]})
        spec = surge_xt_interactive._MetricsFileSpec(rows=2, columns=frozenset({"mss"}))

        with pytest.raises(ValueError, match="missing expected columns"):
            surge_xt_interactive._validate_metrics_df(Path("metrics.csv"), df, spec)

    def test_nan_in_expected_column_raises_valueerror(self, surge_xt_interactive) -> None:
        """A NaN in any expected column raises ``ValueError`` (NaN/Inf message)."""
        df = pd.DataFrame({"mss": [0.1, float("nan")]})
        spec = surge_xt_interactive._MetricsFileSpec(rows=2, columns=frozenset({"mss"}))

        with pytest.raises(ValueError, match="NaN/Inf"):
            surge_xt_interactive._validate_metrics_df(Path("metrics.csv"), df, spec)


class _RecordingEvalRunner:
    """Test fake matching the ``EvalRunner`` shape from ``surge_xt_interactive``.

    Records the positional args from each invocation and the call count so tests can assert
    on real state instead of patching the module-level ``eval_patches`` symbol. Reused
    across ``TestMaybeEvalCapturedPatches`` whenever a test needs to observe whether (and
    with what arguments) eval was invoked.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, Path, Path, str, str]] = []

    def __call__(
        self,
        num_samples: int,
        dataset_root_dir: Path,
        checkpoint_path: Path,
        param_spec_name: str,
        preset_path: str,
    ) -> None:
        self.calls.append(
            (num_samples, dataset_root_dir, checkpoint_path, param_spec_name, preset_path)
        )


class TestMaybeEvalCapturedPatches:
    """``_maybe_eval_captured_patches`` wires up the train.h5 -> sibling replication."""

    def test_no_checkpoint_skips_replication_and_eval(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """Without ``--checkpoint-path``, no sibling files are created and eval_patches is not
        invoked."""
        train_path = tmp_path / "train.h5"
        train_path.write_bytes(b"stub")
        runner = _RecordingEvalRunner()

        surge_xt_interactive._maybe_eval_captured_patches(
            patch_file_path=train_path,
            output_dataset_dir_path=tmp_path,
            num_patches=1,
            checkpoint_path=None,
            param_spec_name=SURGE_SIMPLE,
            preset_path="presets/surge-base.vstpreset",
            eval_runner=runner,
        )

        for sibling in ("test.h5", "val.h5", "predict.h5"):
            assert not (tmp_path / sibling).exists()
        assert runner.calls == []

    def test_replicates_train_h5_to_three_siblings(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """When ``--checkpoint-path`` is given, ``train.h5`` is copied to test/val/predict.h5 and
        ``param_spec_name`` / ``preset_path`` are forwarded verbatim to the eval runner."""
        train_path = tmp_path / "train.h5"
        train_path.write_bytes(b"train-content")
        ckpt_path = tmp_path / "model.ckpt"
        ckpt_path.write_bytes(b"ckpt")
        runner = _RecordingEvalRunner()

        surge_xt_interactive._maybe_eval_captured_patches(
            patch_file_path=train_path,
            output_dataset_dir_path=tmp_path,
            num_patches=3,
            checkpoint_path=ckpt_path,
            param_spec_name=SURGE_SIMPLE,
            preset_path="presets/surge-simple.vstpreset",
            eval_runner=runner,
        )

        for sibling in ("test.h5", "val.h5", "predict.h5"):
            assert (tmp_path / sibling).read_bytes() == b"train-content"
        assert runner.calls == [
            (3, tmp_path, ckpt_path, SURGE_SIMPLE, "presets/surge-simple.vstpreset")
        ]

    def test_failed_copy_rolls_back_partial_siblings(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """If a later ``shutil.copyfile`` raises ``OSError``, earlier siblings are removed and
        eval_runner is not invoked.

        Failure is triggered by a *real* OS error: ``val.h5`` is pre-created as a directory, so
        the second copy fails. The exact subclass varies by platform (``IsADirectoryError`` on
        POSIX, ``PermissionError`` on Windows); asserting on ``OSError`` matches the SUT's
        ``except OSError:`` contract.
        """
        train_path = tmp_path / "train.h5"
        train_path.write_bytes(b"train-content")
        ckpt_path = tmp_path / "model.ckpt"
        ckpt_path.write_bytes(b"ckpt")
        # Block the second sibling write with a real directory at its path. Order matches
        # ``_maybe_eval_captured_patches``: test.h5 → val.h5 → predict.h5.
        (tmp_path / "val.h5").mkdir()

        runner = _RecordingEvalRunner()

        with pytest.raises(OSError):
            surge_xt_interactive._maybe_eval_captured_patches(
                patch_file_path=train_path,
                output_dataset_dir_path=tmp_path,
                num_patches=1,
                checkpoint_path=ckpt_path,
                param_spec_name=SURGE_SIMPLE,
                preset_path="presets/surge-base.vstpreset",
                eval_runner=runner,
            )

        # First sibling was copied, second failed; rollback removes the first.
        assert not (tmp_path / "test.h5").exists(), "rollback should remove test.h5"
        # val.h5 still exists as the pre-created directory, but no file landed there.
        assert (tmp_path / "val.h5").is_dir()
        assert not (tmp_path / "predict.h5").exists()
        assert runner.calls == []


def _write_wav(path: Path, *, silent: bool, sample_rate: int = 44100) -> None:
    """Write a brief mono WAV at ``path``.

    ``silent=True`` writes zeros (peak == 0); otherwise a
    half-amplitude 440 Hz sine (peak ~0.5, well above ``SILENCE_PEAK_THRESHOLD``).

    Samples are shaped ``(num_frames, num_channels)`` to match the convention used by
    ``play_audio_recorded`` and ``predict_vst_audio.py`` (both pass ``output.T``).
    """
    duration_seconds = 0.05
    num_frames = int(sample_rate * duration_seconds)
    if silent:
        samples = np.zeros((num_frames, 1), dtype=np.float32)
    else:
        t = np.linspace(0, duration_seconds, num_frames, endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)[:, None]
    with AudioFile(str(path), "w", samplerate=sample_rate, num_channels=1) as f:
        f.write(samples.T)


def _populate_audio_dir(audio_dir: Path, num_samples: int, *, silent: bool = False) -> None:
    """Pre-create the ``sample_{i}`` subdirs that ``_render_predicted_audio`` validates after the
    subprocess returns, with a full set of per-sample artifacts."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_samples):
        sample_dir = audio_dir / f"sample_{i}"
        sample_dir.mkdir()
        _write_wav(sample_dir / "target.wav", silent=silent)
        _write_wav(sample_dir / "pred.wav", silent=silent)
        (sample_dir / "spec.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (sample_dir / "params.csv").write_text("name,value\n")


_RENDER_DEFAULT_PRESET = "presets/surge-base.vstpreset"


class TestBuildPredictVstAudioArgv:
    """``_build_predict_vst_audio_argv`` builds the argv list for ``predict_vst_audio.py``.

    Pure with respect to ``audio_dir`` / ``predictions_output_dir`` (no writes), and
    parameterised on ``platform`` + ``wrapper_path`` so each branch is tested without
    monkeypatching ``sys.platform`` or the module-level wrapper constant.
    """

    def test_linux_prepends_wrapper_to_argv(self, surge_xt_interactive, tmp_path: Path) -> None:
        """On Linux, the existing wrapper script is the first argv entry."""
        wrapper_path = tmp_path / "wrapper.sh"
        wrapper_path.write_text('#!/usr/bin/env bash\nexec "$@"\n')
        wrapper_path.chmod(0o755)

        argv = surge_xt_interactive._build_predict_vst_audio_argv(
            tmp_path / "preds",
            tmp_path / "audio",
            SURGE_SIMPLE,
            _RENDER_DEFAULT_PRESET,
            platform="linux",
            wrapper_path=wrapper_path,
        )

        assert argv[0] == str(wrapper_path)

    def test_linux_missing_wrapper_raises_filenotfounderror(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """On Linux, a missing wrapper path raises ``FileNotFoundError`` naming the path."""
        missing_wrapper = tmp_path / "definitely-does-not-exist.sh"

        with pytest.raises(FileNotFoundError, match="VST headless wrapper not found"):
            surge_xt_interactive._build_predict_vst_audio_argv(
                tmp_path / "preds",
                tmp_path / "audio",
                SURGE_SIMPLE,
                _RENDER_DEFAULT_PRESET,
                platform="linux",
                wrapper_path=missing_wrapper,
            )

    def test_non_linux_does_not_prepend_wrapper(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """On non-Linux platforms the wrapper is not prepended and a missing wrapper is fine."""
        missing_wrapper = tmp_path / "definitely-does-not-exist.sh"

        argv = surge_xt_interactive._build_predict_vst_audio_argv(
            tmp_path / "preds",
            tmp_path / "audio",
            SURGE_SIMPLE,
            _RENDER_DEFAULT_PRESET,
            platform="darwin",
            wrapper_path=missing_wrapper,
        )

        assert str(missing_wrapper) not in argv
        assert argv[0] == surge_xt_interactive.sys.executable

    def test_param_spec_and_preset_path_are_forwarded(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """``param_spec_name`` and ``preset_path`` follow their flag tokens in argv (otherwise
        ``predict_vst_audio.py`` would silently fall back to ``surge_xt`` / ``presets/surge-
        base.vstpreset`` and decode/render against a mismatched spec)."""
        argv = surge_xt_interactive._build_predict_vst_audio_argv(
            tmp_path / "preds",
            tmp_path / "audio",
            "custom-spec",
            "presets/custom.vstpreset",
            platform="darwin",
        )

        assert "--param_spec" in argv
        assert argv[argv.index("--param_spec") + 1] == "custom-spec"
        assert "--preset_path" in argv
        assert argv[argv.index("--preset_path") + 1] == "presets/custom.vstpreset"
        assert argv[-1] == "-t"

    def test_predictions_and_audio_dirs_appear_as_positional_args(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """The two positional argv entries match the source/destination paths the caller asked for
        (otherwise rendering would silently target the wrong directory)."""
        preds_dir = tmp_path / "preds"
        audio_dir = tmp_path / "audio"

        argv = surge_xt_interactive._build_predict_vst_audio_argv(
            preds_dir, audio_dir, SURGE_SIMPLE, _RENDER_DEFAULT_PRESET, platform="darwin"
        )

        assert str(preds_dir) in argv
        assert str(audio_dir) in argv
        # Positional args follow the predict_vst_audio.py entry; pred_dir before output_dir.
        assert argv.index(str(preds_dir)) < argv.index(str(audio_dir))


class TestValidateRenderedAudioDir:
    """``_validate_rendered_audio_dir`` checks the per-sample artifacts after rendering."""

    def test_returns_none_on_complete_non_silent_dir(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """All artifacts present and audible → returns ``None`` without raising."""
        audio_dir = tmp_path / "audio"
        _populate_audio_dir(audio_dir, num_samples=2)

        # Returns None (implicit) — no assertion needed beyond the absence of exceptions.
        surge_xt_interactive._validate_rendered_audio_dir(audio_dir, num_samples=2)

    def test_missing_artifact_raises_filenotfounderror(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """A missing per-sample artifact (``spec.png``) raises ``FileNotFoundError`` naming it."""
        audio_dir = tmp_path / "audio"
        _populate_audio_dir(audio_dir, num_samples=1)
        (audio_dir / "sample_0" / "spec.png").unlink()

        with pytest.raises(FileNotFoundError, match="spec.png"):
            surge_xt_interactive._validate_rendered_audio_dir(audio_dir, num_samples=1)

    def test_silent_wav_raises_valueerror(self, surge_xt_interactive, tmp_path: Path) -> None:
        """A silent rendered WAV raises ``ValueError`` naming the offending sample / file."""
        audio_dir = tmp_path / "audio"
        _populate_audio_dir(audio_dir, num_samples=1, silent=True)

        with pytest.raises(ValueError, match=r"sample_0/(target|pred)\.wav is silent"):
            surge_xt_interactive._validate_rendered_audio_dir(audio_dir, num_samples=1)

    def test_unexpected_sample_dirs_raises_filenotfounderror(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """Mismatched sample directory set (missing) raises ``FileNotFoundError``."""
        audio_dir = tmp_path / "audio"
        # Caller asks for num_samples=2 but only sample_0 exists.
        _populate_audio_dir(audio_dir, num_samples=1)

        with pytest.raises(FileNotFoundError, match="unexpected sample directories"):
            surge_xt_interactive._validate_rendered_audio_dir(audio_dir, num_samples=2)

    def test_extra_sample_dir_raises_filenotfounderror(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """An extra sample directory (e.g. ``sample_5`` for ``num_samples=2``) raises
        ``FileNotFoundError`` so a stale leftover doesn't silently pass validation."""
        audio_dir = tmp_path / "audio"
        _populate_audio_dir(audio_dir, num_samples=2)
        # Pollute the dir with an extra leftover.
        leftover = audio_dir / "sample_5"
        leftover.mkdir()
        _write_wav(leftover / "target.wav", silent=False)
        _write_wav(leftover / "pred.wav", silent=False)
        (leftover / "spec.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (leftover / "params.csv").write_text("name,value\n")

        with pytest.raises(FileNotFoundError, match="unexpected sample directories"):
            surge_xt_interactive._validate_rendered_audio_dir(audio_dir, num_samples=2)

    def test_num_samples_12_does_not_trip_lex_sort(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """Regression: ``num_samples=12`` must not raise just because ``sample_10`` sorts before
        ``sample_2`` in lexical order. Set comparison + index iteration keeps validation
        index-based regardless of directory iteration order."""
        audio_dir = tmp_path / "audio"
        _populate_audio_dir(audio_dir, num_samples=12)

        surge_xt_interactive._validate_rendered_audio_dir(audio_dir, num_samples=12)


class TestRenderPredictedAudioSubprocessIntegration:
    """``_render_predicted_audio`` orchestrator: argv build + subprocess invocation + validation.

    Verified with ``_RecordingSubprocessRunner`` (state-based, no monkeypatch). Real
    ``subprocess.run`` lifecycle is exercised by the e2e test below.
    """

    def test_runner_receives_argv_with_check_and_timeout(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """Orchestrator forwards a populated argv to the runner with ``check=True`` and a positive
        ``timeout`` — drift on either kwarg would silently turn fatal subprocess errors into
        successes."""
        audio_dir = tmp_path / "audio"
        _populate_audio_dir(audio_dir, num_samples=1)
        runner = _RecordingSubprocessRunner()

        surge_xt_interactive._render_predicted_audio(
            tmp_path / "preds",
            audio_dir,
            num_samples=1,
            param_spec_name=SURGE_SIMPLE,
            preset_path=_RENDER_DEFAULT_PRESET,
            subprocess_runner=runner,
        )

        assert len(runner.calls) == 1
        forwarded_argv = runner.calls[0]
        assert str(_RENDER_DEFAULT_PRESET) in forwarded_argv
        assert SURGE_SIMPLE in forwarded_argv
        last_kwargs = runner.kwargs_per_call[0]
        assert last_kwargs.get("check") is True
        timeout = last_kwargs.get("timeout")
        assert isinstance(timeout, int | float)
        assert timeout > 0


# --- E2E tests: real Surge XT plugin + real subprocess ----------------------------------------


@pytest.mark.requires_vst
@pytest.mark.slow
class TestPlayAudioRecordedE2E:
    """End-to-end: real Surge XT plugin renders a non-silent deterministic clip."""

    def test_play_audio_recorded_produces_non_silent_wav(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """``play_audio_recorded`` against the real Surge XT VST + ``surge-base.vstpreset`` writes
        a WAV with the expected frame count and audible peak amplitude."""
        from src.data.vst.core import load_plugin, load_preset

        plugin_path = "plugins/Surge XT.vst3"
        preset_path = "presets/surge-base.vstpreset"
        if not Path(plugin_path).exists():
            pytest.skip(f"Surge XT plugin not found at {plugin_path}")
        if not Path(preset_path).exists():
            pytest.skip(f"Surge XT base preset not found at {preset_path}")

        plugin = load_plugin(plugin_path)
        load_preset(plugin, preset_path)

        output_path = tmp_path / "session.wav"
        surge_xt_interactive.play_audio_recorded(plugin, output_path)

        assert output_path.is_file()
        with AudioFile(str(output_path)) as f:
            audio = f.read(f.frames)
        expected_frames = int(
            surge_xt_interactive.SESSION_RECORDING_DURATION_SECONDS
            * surge_xt_interactive.SAMPLE_RATE
        )
        assert audio.shape == (surge_xt_interactive.CHANNELS, expected_frames)
        peak = float(np.abs(audio).max())
        assert peak > surge_xt_interactive.SILENCE_PEAK_THRESHOLD, (
            f"recorded WAV is silent (peak={peak:.2e}); a real preset should produce audible output"
        )


def _write_synthetic_prediction_files(
    pred_dir: Path, num_samples: int, simple_spec: ParamSpec
) -> None:
    """Write synthetic ``pred-{i}.pt`` / ``target-params-{i}.pt`` / ``target-audio-{i}.pt`` so
    ``predict_vst_audio.py`` can render audio without a model checkpoint.

    Each pred row encodes mid-range params (zeros in the (-1..1) coordinate) so the rendered
    audio depends only on the preset and a deterministic note pattern.
    """
    pred_dir.mkdir(parents=True, exist_ok=True)
    total_length = simple_spec.synth_param_length + simple_spec.note_param_length
    for i in range(num_samples):
        pred_row = torch.zeros((1, total_length), dtype=torch.float32)
        torch.save(pred_row, pred_dir / f"pred-{i}.pt")
        torch.save(pred_row, pred_dir / f"target-params-{i}.pt")
        # target-audio-{i}.pt is loaded but only used when --rerender_target is on; the
        # contents don't affect the produced WAVs in the rerender path.
        torch.save(torch.zeros(1, dtype=torch.float32), pred_dir / f"target-audio-{i}.pt")


@pytest.mark.requires_vst
@pytest.mark.slow
class TestRenderPredictedAudioE2E:
    """End-to-end: real ``predict_vst_audio.py`` subprocess produces non-silent per-sample WAVs."""

    def test_render_predicted_audio_against_real_subprocess(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        simple_spec: ParamSpec,
    ) -> None:
        """``_render_predicted_audio`` invokes the real ``predict_vst_audio.py`` (no
        ``subprocess_runner`` override) and validates each ``sample_{i}`` directory against non-
        silent rendered WAVs."""
        plugin_path = "plugins/Surge XT.vst3"
        preset_path = "presets/surge-base.vstpreset"
        if not Path(plugin_path).exists():
            pytest.skip(f"Surge XT plugin not found at {plugin_path}")
        if not Path(preset_path).exists():
            pytest.skip(f"Surge XT base preset not found at {preset_path}")

        num_samples = 2
        pred_dir = tmp_path / "preds"
        audio_dir = tmp_path / "audio"
        _write_synthetic_prediction_files(pred_dir, num_samples, simple_spec)

        surge_xt_interactive._render_predicted_audio(
            pred_dir,
            audio_dir,
            num_samples,
            param_spec_name=SURGE_SIMPLE,
            preset_path=preset_path,
        )

        for i in range(num_samples):
            sample_dir = audio_dir / f"sample_{i}"
            for fname in ("target.wav", "pred.wav", "spec.png", "params.csv"):
                assert (sample_dir / fname).is_file()
            for wav_name in ("target.wav", "pred.wav"):
                with AudioFile(str(sample_dir / wav_name)) as f:
                    audio = f.read(f.frames)
                peak = float(np.abs(audio).max())
                assert peak > surge_xt_interactive.SILENCE_PEAK_THRESHOLD, (
                    f"{sample_dir.name}/{wav_name} is silent (peak={peak:.2e})"
                )


@pytest.mark.requires_vst
@pytest.mark.slow
class TestKeyboardLoopE2E:
    """End-to-end: real Surge XT plugin records a patch via deterministic keystrokes."""

    def test_p_q_against_real_plugin_records_one_patch(
        self, surge_xt_interactive, simple_spec: ParamSpec
    ) -> None:
        """``["p", "q"]`` against the real Surge XT + ``surge-base.vstpreset`` records one
        patch whose synth-param values are finite floats matching the post-preset-load state.
        """
        from src.data.vst.core import load_plugin, load_preset

        plugin_path = "plugins/Surge XT.vst3"
        preset_path = "presets/surge-base.vstpreset"
        if not Path(plugin_path).exists():
            pytest.skip(f"Surge XT plugin not found at {plugin_path}")
        if not Path(preset_path).exists():
            pytest.skip(f"Surge XT base preset not found at {preset_path}")

        plugin = load_plugin(plugin_path)
        load_preset(plugin, preset_path)

        # Snapshot post-preset-load defaults so _validate_no_drift has a known baseline.
        default_params = {
            name: plugin.parameters[name].raw_value  # type: ignore[attr-defined]
            for name in plugin.parameters.keys()  # type: ignore[attr-defined]
        }

        stop_event = threading.Event()
        keystrokes = iter(["p", "q"])

        patches = surge_xt_interactive.keyboard_loop(
            plugin,
            stop_event,
            SURGE_SIMPLE,
            default_params,
            keystroke_source=keystrokes.__next__,
        )

        assert len(patches) == 1
        patch = patches[0]
        assert set(patch) == set(simple_spec.synth_param_names)
        for name, value in patch.items():
            assert isinstance(value, float)
            assert np.isfinite(value), f"{name} = {value} is not finite"
        assert stop_event.is_set()
