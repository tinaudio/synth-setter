"""Tests for src/synth_setter/tools/vst_interactive.py prediction decoding helpers."""

import importlib
import os
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import NoReturn
from unittest import mock

import click
import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner
from pedalboard.io import AudioFile

from synth_setter.data.vst import param_specs
from synth_setter.data.vst.param_spec import ParamSpec
from tests.helpers.lance_fixtures import write_lance_shard

SURGE_SIMPLE = "surge_simple"


@pytest.fixture(scope="module")
def vst_interactive() -> ModuleType:
    """Import the tool module lazily so collection doesn't fail on heavy imports.

    :returns: Loaded VST interactive module.
    """
    return importlib.import_module("synth_setter.tools.vst_interactive")


@pytest.fixture(scope="module")
def simple_spec() -> ParamSpec:
    """Return the ``surge_simple`` ParamSpec used by the prediction-decoding tests.

    :returns: Registered simple ParamSpec.
    """
    return param_specs[SURGE_SIMPLE]


@pytest.fixture(scope="module")
def simple_spec_total_length(simple_spec: ParamSpec) -> int:
    """Total encoded row length (synth + note params) for the simple spec.

    :param simple_spec: Simple ParamSpec fixture used by the scenario.
    :returns: Combined encoded synth and note width.
    """
    return simple_spec.synth_param_length + simple_spec.note_param_length


@pytest.fixture
def simple_pred_tensor(simple_spec_total_length: int) -> torch.Tensor:
    """Build a 2-row prediction tensor sized for the surge_simple spec.

    Row 0 cycles through ``[-1.0, 0.0, 1.0, 2.0]`` to exercise the
    ``(-1..1) -> (0..1)`` rescaling and clipping (``2.0`` is clipped to ``1``).
    Row 1 is all-zeros (decodes to mid-range values).

    :param simple_spec_total_length: Encoded width of the simple ParamSpec.
    :returns: Two-row prediction tensor spanning rescale boundaries.
    """
    cycle = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    row_0 = np.tile(cycle, (simple_spec_total_length // 4) + 1)[:simple_spec_total_length]
    row_1 = np.zeros(simple_spec_total_length, dtype=np.float32)
    return torch.tensor(np.stack([row_0, row_1]), dtype=torch.float32)


def _write_param_array_lance(path: Path, rows: np.ndarray) -> None:
    """Write a 2D ``rows`` array to a Lance dataset under the ``param_array`` column.

    :param path: Destination ``.lance`` dataset directory.
    :param rows: 2D ``(num_rows, num_params)`` array written as the ``param_array`` column.
    """
    write_lance_shard(path, {"param_array": rows.astype(np.float32)})


class TestDecodePredictionRow:
    """decode_prediction_row scales (-1..1) -> (0..1), clips, and decodes a row."""

    def test_returns_expected_keys_and_finite_floats(
        self,
        vst_interactive: ModuleType,
        simple_pred_tensor: torch.Tensor,
        simple_spec: ParamSpec,
    ) -> None:
        """Decoded row contains every synth-param key with finite float values.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_pred_tensor: Prediction tensor fixture for the simple ParamSpec.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        """
        synth_params = vst_interactive.decode_prediction_row(
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
            ("a_amp_eg_attack", 0.0),
            ("a_amp_eg_decay", 0.385),
            ("a_amp_eg_release", 0.77),
            ("a_amp_eg_sustain", 1.0),
        ],
    )
    def test_clips_and_rescales_per_column(
        self,
        vst_interactive: ModuleType,
        simple_pred_tensor: torch.Tensor,
        param_name: str,
        expected: float,
    ) -> None:
        """Each column is rescaled from ``[-1, 1]`` to ``[0, 1]`` and clipped before decoding.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_pred_tensor: Prediction tensor fixture for the simple ParamSpec.
        :param param_name: ParamSpec field selected by the test case.
        :param expected: Expected decoded parameter value.
        """
        synth_params = vst_interactive.decode_prediction_row(
            simple_pred_tensor, batch_idx=0, param_spec_name=SURGE_SIMPLE
        )
        assert synth_params[param_name] == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("bad_idx", [99, -1], ids=["above-range", "negative"])
    def test_out_of_range_idx_raises(
        self,
        vst_interactive: ModuleType,
        simple_pred_tensor: torch.Tensor,
        bad_idx: int,
    ) -> None:
        """``batch_idx`` outside ``[0, batch_size)`` raises ``IndexError``.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_pred_tensor: Prediction tensor fixture for the simple ParamSpec.
        :param bad_idx: Out-of-range row index under test.
        """
        with pytest.raises(IndexError):
            vst_interactive.decode_prediction_row(
                simple_pred_tensor, batch_idx=bad_idx, param_spec_name=SURGE_SIMPLE
            )

    def test_wrong_width_when_decoded_raises_valueerror(
        self, vst_interactive: ModuleType, simple_spec_total_length: int
    ) -> None:
        """Reject rows encoded for a different ParamSpec.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec_total_length: Encoded width of the simple ParamSpec.
        """
        prediction = torch.zeros((1, simple_spec_total_length + 1))

        with pytest.raises(ValueError, match="prediction width"):
            vst_interactive.decode_prediction_row(prediction, 0, SURGE_SIMPLE)

    def test_non_finite_value_when_decoded_raises_valueerror(
        self, vst_interactive: ModuleType, simple_spec_total_length: int
    ) -> None:
        """Reject non-finite model output before parameter decoding.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec_total_length: Encoded width of the simple ParamSpec.
        """
        prediction = torch.zeros((1, simple_spec_total_length))
        prediction[0, 0] = torch.nan

        with pytest.raises(ValueError, match="non-finite"):
            vst_interactive.decode_prediction_row(prediction, 0, SURGE_SIMPLE)


class TestPredictionRefType:
    """PredictionRefType parses ``PATH:BATCH_IDX`` into a PredictionRef."""

    def test_parses_path_and_batch_idx(self, vst_interactive: ModuleType) -> None:
        """A ``PATH:BATCH_IDX`` string parses into a ``PredictionRef`` with matching fields.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        parser = vst_interactive.PredictionRefType()

        ref = parser.convert("outputs/pred-0.pt:42", None, None)

        assert ref == vst_interactive.PredictionRef(path=Path("outputs/pred-0.pt"), batch_idx=42)

    def test_splits_on_last_colon(self, vst_interactive: ModuleType) -> None:
        """Absolute Windows-style paths still parse because rpartition uses the last ':'.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        parser = vst_interactive.PredictionRefType()

        ref = parser.convert(r"C:\models\pred-0.pt:7", None, None)

        assert ref.path == Path(r"C:\models\pred-0.pt")
        assert ref.batch_idx == 7

    @pytest.mark.parametrize(
        "value",
        ["pred-0.pt", "pred-0.pt:not-an-int", ":42", "pred-0.pt:"],
        ids=["missing-colon", "non-int-idx", "empty-path", "empty-idx"],
    )
    def test_rejects_invalid_uri(self, vst_interactive: ModuleType, value: str) -> None:
        """Malformed prediction references raise ``click.BadParameter``.

        :param vst_interactive: Loaded VST interactive module under test.
        :param value: Malformed reference text under test.
        """
        parser = vst_interactive.PredictionRefType()

        with pytest.raises(click.BadParameter):
            parser.convert(value, None, None)

    def test_rejects_negative_batch_idx(self, vst_interactive: ModuleType) -> None:
        """Reject negative indices to match ``decode_prediction_row``'s contract.

        Negative indexing would otherwise silently select the last row.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        parser = vst_interactive.PredictionRefType()

        with pytest.raises(click.BadParameter):
            parser.convert("pred-0.pt:-1", None, None)


class TestDatasetRefType:
    """DatasetRefType parses ``PATH:DATASET_IDX`` into a DatasetRef."""

    def test_parses_path_and_batch_idx(self, vst_interactive: ModuleType) -> None:
        """A ``PATH:DATASET_IDX`` string parses into a ``DatasetRef`` with matching fields.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        parser = vst_interactive.DatasetRefType()

        ref = parser.convert("data/test.lance:3", None, None)

        assert ref == vst_interactive.DatasetRef(path=Path("data/test.lance"), batch_idx=3)

    @pytest.mark.parametrize(
        "value",
        ["test.lance", "test.lance:not-an-int", ":0", "test.lance:"],
        ids=["missing-colon", "non-int-idx", "empty-path", "empty-idx"],
    )
    def test_rejects_invalid_uri(self, vst_interactive: ModuleType, value: str) -> None:
        """Malformed dataset references raise ``click.BadParameter``.

        :param vst_interactive: Loaded VST interactive module under test.
        :param value: Malformed reference text under test.
        """
        parser = vst_interactive.DatasetRefType()

        with pytest.raises(click.BadParameter):
            parser.convert(value, None, None)

    def test_rejects_negative_batch_idx(self, vst_interactive: ModuleType) -> None:
        """Reject negative indices to avoid a silent ``param_array[-1]`` last-row fallback.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        parser = vst_interactive.DatasetRefType()

        with pytest.raises(click.BadParameter):
            parser.convert("test.lance:-1", None, None)


class TestLoadPredictionSynthParams:
    """load_prediction_synth_params reads a .pt file row and decodes it."""

    def test_matches_decode_prediction_row_on_same_row(
        self,
        vst_interactive: ModuleType,
        simple_pred_tensor: torch.Tensor,
        simple_spec: ParamSpec,
        tmp_path: Path,
    ) -> None:
        """Loading from disk and in-memory ``decode_prediction_row`` produce identical outputs.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_pred_tensor: Prediction tensor fixture for the simple ParamSpec.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        :param tmp_path: Per-test temporary directory.
        """
        pred_path = tmp_path / "pred-0.pt"
        torch.save(simple_pred_tensor, pred_path)
        ref = vst_interactive.PredictionRef(path=pred_path, batch_idx=0)

        loaded = vst_interactive.load_prediction_synth_params(ref, param_spec_name=SURGE_SIMPLE)

        direct = vst_interactive.decode_prediction_row(
            simple_pred_tensor, batch_idx=0, param_spec_name=SURGE_SIMPLE
        )
        expected_keys = {p.name for p in simple_spec.synth_params}
        assert set(loaded.keys()) == expected_keys
        assert loaded == direct


class TestLoadDatasetSynthParams:
    """load_dataset_synth_params reads a Lance ``param_array`` row and decodes it."""

    def test_round_trip_returns_original_synth_params(
        self,
        vst_interactive: ModuleType,
        simple_spec: ParamSpec,
        tmp_path: Path,
    ) -> None:
        """Encoding params, persisting to Lance, and reloading recovers the synth params.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        :param tmp_path: Per-test temporary directory.
        """
        synth_param_dict, note_param_dict = simple_spec.sample()
        encoded = simple_spec.encode(synth_param_dict, note_param_dict)
        lance_path = tmp_path / "test.lance"
        _write_param_array_lance(lance_path, encoded[None, :])
        ref = vst_interactive.DatasetRef(path=lance_path, batch_idx=0)

        loaded = vst_interactive.load_dataset_synth_params(ref, param_spec_name=SURGE_SIMPLE)

        for name, value in synth_param_dict.items():
            assert loaded[name] == pytest.approx(value, abs=1e-5)

    def test_selects_correct_row(
        self,
        vst_interactive: ModuleType,
        simple_spec: ParamSpec,
        tmp_path: Path,
    ) -> None:
        """``batch_idx`` selects the matching row from a multi-row ``param_array``.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        :param tmp_path: Per-test temporary directory.
        """
        row_0_synth, row_0_note = simple_spec.sample()
        row_1_synth, row_1_note = simple_spec.sample()
        encoded = np.stack(
            [
                simple_spec.encode(row_0_synth, row_0_note),
                simple_spec.encode(row_1_synth, row_1_note),
            ]
        )
        lance_path = tmp_path / "test.lance"
        _write_param_array_lance(lance_path, encoded)
        ref = vst_interactive.DatasetRef(path=lance_path, batch_idx=1)

        loaded = vst_interactive.load_dataset_synth_params(ref, param_spec_name=SURGE_SIMPLE)

        for name, value in row_1_synth.items():
            assert loaded[name] == pytest.approx(value, abs=1e-5)

    def test_out_of_range_idx_raises(
        self,
        vst_interactive: ModuleType,
        simple_spec: ParamSpec,
        tmp_path: Path,
    ) -> None:
        """Raise ``IndexError`` or ``ValueError`` when ``batch_idx`` exceeds ``param_array``.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        :param tmp_path: Per-test temporary directory.
        """
        encoded = simple_spec.encode(*simple_spec.sample())
        lance_path = tmp_path / "test.lance"
        _write_param_array_lance(lance_path, encoded[None, :])
        ref = vst_interactive.DatasetRef(path=lance_path, batch_idx=99)

        with pytest.raises((IndexError, ValueError)):
            vst_interactive.load_dataset_synth_params(ref, param_spec_name=SURGE_SIMPLE)

    def test_wrong_width_when_loaded_raises_valueerror(
        self,
        vst_interactive: ModuleType,
        simple_spec_total_length: int,
        tmp_path: Path,
    ) -> None:
        """Reject dataset rows encoded for another ParamSpec.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec_total_length: Encoded width of the simple ParamSpec.
        :param tmp_path: Per-test temporary directory.
        """
        lance_path = tmp_path / "wrong-width.lance"
        _write_param_array_lance(
            lance_path,
            np.zeros((1, simple_spec_total_length + 1), dtype=np.float32),
        )
        ref = vst_interactive.DatasetRef(path=lance_path, batch_idx=0)

        with pytest.raises(ValueError, match="dataset row width"):
            vst_interactive.load_dataset_synth_params(ref, param_spec_name=SURGE_SIMPLE)

    @pytest.mark.requires_vst
    @pytest.mark.slow
    def test_loads_row_from_surge_xt_smoke_fixture(
        self,
        vst_interactive: ModuleType,
        surge_xt_smoke_datasets: Path,
        param_spec_name: str,
    ) -> None:
        """Loads row 0 from the real ``surge_xt_smoke_datasets`` test.lance.

        The decode spec must match the spec the fixture generated the dataset with —
        otherwise the decoder slices off the end of the row and ``.item()`` raises.

        :param vst_interactive: Loaded VST interactive module under test.
        :param surge_xt_smoke_datasets: Root of the real Surge XT smoke dataset fixture.
        :param param_spec_name: ParamSpec registry key paired with the smoke dataset.
        """
        ref = vst_interactive.DatasetRef(path=surge_xt_smoke_datasets / "test.lance", batch_idx=0)

        loaded = vst_interactive.load_dataset_synth_params(ref, param_spec_name=param_spec_name)

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

    :param sample_value: Constant sample value returned by the fake plugin.
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
        """Return a constant-valued ``(num_channels, duration * sample_rate)`` buffer.

        :param midi_messages: Timestamped MIDI messages passed to the fake plugin.
        :param duration_seconds: Requested render duration in seconds.
        :param sample_rate: Audio sample rate in hertz.
        :param num_channels: Requested audio channel count.
        :param buffer_size: Audio buffer width used by the fake plugin.
        :param reset: Whether the fake plugin should reset before rendering.
        :returns: Constant audio buffer with the requested shape.
        """
        del buffer_size, reset
        self.process_call_count += 1
        self.last_midi_messages = list(midi_messages)
        frames = int(duration_seconds * sample_rate)
        return np.full((num_channels, frames), self.sample_value, dtype=np.float32)


class TestPlayAudioRecorded:
    """play_audio_recorded renders a deterministic clip via a single plugin.process() call."""

    def test_writes_exact_duration_frames(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """The WAV's frame count is exactly ``DURATION * SAMPLE_RATE`` (one process call).

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        plugin = _ConstantPlugin(sample_value=0.25)
        output_path = tmp_path / "session.wav"
        expected_frames = int(
            vst_interactive.SESSION_RECORDING_DURATION_SECONDS * vst_interactive.SAMPLE_RATE
        )

        vst_interactive.play_audio_recorded(plugin, output_path)

        assert plugin.process_call_count == 1
        assert output_path.is_file()
        with AudioFile(str(output_path)) as f:
            audio = f.read(f.frames)
        assert audio.shape == (vst_interactive.CHANNELS, expected_frames)
        np.testing.assert_allclose(audio, plugin.sample_value, atol=1e-3)

    def test_passes_expected_midi_events(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """plugin.process is called with note_on/off middle-C events at NOTE_START/END.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        plugin = _ConstantPlugin(sample_value=0.0)

        vst_interactive.play_audio_recorded(plugin, tmp_path / "events.wav")

        assert plugin.last_midi_messages is not None
        assert len(plugin.last_midi_messages) == 2
        (note_on_bytes, note_on_t), (note_off_bytes, note_off_t) = plugin.last_midi_messages

        assert note_on_t == pytest.approx(vst_interactive.SESSION_RECORDING_NOTE_START_SECONDS)
        assert note_off_t == pytest.approx(vst_interactive.SESSION_RECORDING_NOTE_END_SECONDS)

        # MIDI wire format: status byte (high nibble = type), note, velocity.
        # 0x90 = note_on (channel 0), 0x80 = note_off (channel 0).
        assert note_on_bytes[0] & 0xF0 == 0x90
        assert note_off_bytes[0] & 0xF0 == 0x80
        assert note_on_bytes[1] == vst_interactive.SESSION_RECORDING_MIDI_NOTE
        assert note_off_bytes[1] == vst_interactive.SESSION_RECORDING_MIDI_NOTE
        assert note_on_bytes[2] == vst_interactive.SESSION_RECORDING_VELOCITY

    @pytest.mark.parametrize(
        "sample_value, message",
        [
            (1.01, r"outside \[-1, 1\]"),
            (-1.01, r"outside \[-1, 1\]"),
            (float("nan"), "non-finite"),
        ],
        ids=["positive", "negative", "non-finite"],
    )
    def test_invalid_audio_raises_valueerror(
        self,
        vst_interactive: ModuleType,
        tmp_path: Path,
        sample_value: float,
        message: str,
    ) -> None:
        """Reject non-finite or out-of-range recording samples.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        :param sample_value: Invalid sample value returned by the plugin.
        :param message: Expected validation-error fragment.
        """
        output_path = tmp_path / "invalid.wav"

        with pytest.raises(ValueError, match=message):
            vst_interactive.play_audio_recorded(_ConstantPlugin(sample_value), output_path)

        assert not output_path.exists()


class _FakeMidiMessage:
    """Minimal stand-in for a ``mido.Message`` — exposes ``type`` and ``bytes()``.

    ``bytes()`` returns ``list[int]`` to match real ``mido.Message.bytes()``.

    :param msg_type: MIDI message type exposed by the fake.
    :param payload: Optional MIDI byte payload.
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

    :param messages: Fake MIDI messages returned by the test port.
    :param drain_event: Optional event set when fake MIDI messages are exhausted.
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

    def test_only_relevant_message_types_are_forwarded(self, vst_interactive: ModuleType) -> None:
        """note_on/off, control_change, pitchwheel, aftertouch are queued; others are dropped.

        :param vst_interactive: Loaded VST interactive module under test.
        """
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
            target=vst_interactive.midi_listener,
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

    def test_stop_event_exits_listener_with_no_messages(self, vst_interactive: ModuleType) -> None:
        """``stop_event`` set before any message arrives drains the listener cleanly.

        :param vst_interactive: Loaded VST interactive module under test.
        """

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

        vst_interactive.midi_listener(
            "fake-port", midi_queue, stop_event, port_opener=lambda _port: _IdlePort()
        )

        assert midi_queue.empty()

    def test_open_input_failure_logs_and_exits_thread(
        self, vst_interactive: ModuleType, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``midi_listener`` must not raise; failures are logged so the daemon exits cleanly.

        :param vst_interactive: Loaded VST interactive module under test.
        :param caplog: Pytest log-capture fixture.
        """

        def raising_port_opener(_port_name: str) -> object:
            raise OSError("device disconnected")

        midi_queue: queue.Queue[tuple[list[int], float]] = queue.Queue()
        stop_event = threading.Event()
        with caplog.at_level("ERROR"):
            vst_interactive.midi_listener(
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

    :param channels: Audio channel count used by the fake plugin.
    :param buffer_size: Audio buffer width used by the fake plugin.
    :param stop_event: Optional shutdown event controlled by the fake plugin.
    :param stop_after_n_calls: Optional process-call limit before signaling shutdown.
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

    def test_queued_events_are_forwarded_to_plugin_process(
        self, vst_interactive: ModuleType
    ) -> None:
        """Tuples enqueued by the listener are drained and passed to ``plugin.process``.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        stop_event = threading.Event()
        plugin = _RecordingPlugin(
            vst_interactive.CHANNELS,
            vst_interactive.BUFFER_SIZE,
            stop_event=stop_event,
            stop_after_n_calls=1,
        )
        stream = _FakeStream()

        midi_queue: queue.Queue[tuple[list[int], float]] = queue.Queue()
        midi_queue.put(([0x90, 0x3C, 0x40], 0.0))
        midi_queue.put(([0x80, 0x3C, 0x00], 0.0))

        vst_interactive.play_audio(
            plugin, stop_event, midi_queue, audio_stream_factory=lambda: stream
        )

        assert plugin.messages_per_call == [[([0x90, 0x3C, 0x40], 0.0), ([0x80, 0x3C, 0x00], 0.0)]]
        assert stop_event.is_set()
        assert len(stream.writes) == 1

    def test_none_queue_passes_empty_messages_list(self, vst_interactive: ModuleType) -> None:
        """When no MIDI port is configured, ``play_audio`` is invoked with ``midi_queue=None``.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        stop_event = threading.Event()
        plugin = _RecordingPlugin(
            vst_interactive.CHANNELS,
            vst_interactive.BUFFER_SIZE,
            stop_event=stop_event,
            stop_after_n_calls=1,
        )
        stream = _FakeStream()

        vst_interactive.play_audio(plugin, stop_event, None, audio_stream_factory=lambda: stream)

        assert plugin.messages_per_call == [[]]
        assert stop_event.is_set()

    def test_drain_is_capped_at_max_midi_events_per_buffer(
        self, vst_interactive: ModuleType
    ) -> None:
        """Drain queues larger than ``_MAX_MIDI_EVENTS_PER_BUFFER`` in chunks across buffers.

        Prevents one realtime callback from stretching to process the full backlog.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        cap = vst_interactive._MAX_MIDI_EVENTS_PER_BUFFER
        stop_event = threading.Event()
        plugin = _RecordingPlugin(
            vst_interactive.CHANNELS,
            vst_interactive.BUFFER_SIZE,
            stop_event=stop_event,
            stop_after_n_calls=1,  # stop after the first buffer to assert the per-buffer cap
        )
        stream = _FakeStream()

        midi_queue: queue.Queue[tuple[list[int], float]] = queue.Queue()
        # Enqueue cap + 5 events; first buffer should drain ``cap``, leaving 5 in the queue.
        for _ in range(cap + 5):
            midi_queue.put(([0x90, 0x3C, 0x40], 0.0))

        vst_interactive.play_audio(
            plugin, stop_event, midi_queue, audio_stream_factory=lambda: stream
        )

        assert len(plugin.messages_per_call) == 1
        assert len(plugin.messages_per_call[0]) == cap
        assert midi_queue.qsize() == 5


class TestResolveMidiPort:
    """``_resolve_midi_port`` maps the click flag value to a concrete port name."""

    def test_returns_first_available_when_requested_is_empty_string(
        self, vst_interactive: ModuleType
    ) -> None:
        """Empty string means auto-pick: return ``available[0]``.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        resolved = vst_interactive._resolve_midi_port("", ["port-a", "port-b"])
        assert resolved == "port-a"

    def test_returns_requested_when_present_in_available(
        self, vst_interactive: ModuleType
    ) -> None:
        """A named port that exists in ``available`` is returned verbatim.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        resolved = vst_interactive._resolve_midi_port("port-b", ["port-a", "port-b"])
        assert resolved == "port-b"

    def test_raises_usage_error_when_requested_not_in_available(
        self, vst_interactive: ModuleType
    ) -> None:
        """A named port absent from ``available`` raises ``click.UsageError``.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        with pytest.raises(click.UsageError, match="port-z"):
            vst_interactive._resolve_midi_port("port-z", ["port-a", "port-b"])

    def test_raises_usage_error_when_available_is_empty_and_auto(
        self, vst_interactive: ModuleType
    ) -> None:
        """Auto-pick with no ports available raises ``click.UsageError``.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        with pytest.raises(click.UsageError, match="no MIDI input"):
            vst_interactive._resolve_midi_port("", [])

    def test_raises_usage_error_when_available_is_empty_and_named(
        self, vst_interactive: ModuleType
    ) -> None:
        """Named port with no ports available raises ``click.UsageError``.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        with pytest.raises(click.UsageError, match="no MIDI input"):
            vst_interactive._resolve_midi_port("port-a", [])


class _FakeParam:
    """Stand-in for a pedalboard plugin parameter — exposes only ``raw_value``.

    :param raw_value: Raw parameter value exposed by the fake.
    """

    def __init__(self, raw_value: float) -> None:
        self.raw_value = raw_value


class _FakePlugin:
    """Stand-in plugin with a ``.parameters`` dict-like for drift-detection tests.

    :param params: Initial fake plugin parameter values.
    """

    def __init__(self, params: dict[str, float]) -> None:
        self.parameters = {name: _FakeParam(value) for name, value in params.items()}


class _FakeSpec:
    """Stand-in ParamSpec exposing only the ``synth_param_names`` attribute.

    :param synth_param_names: ParamSpec names accepted by the fake specification.
    """

    def __init__(self, synth_param_names: list[str]) -> None:
        self.synth_param_names = synth_param_names


class TestValidateNoDrift:
    """``_validate_no_drift`` raises if a non-spec param drifted from its default."""

    def test_returns_none_when_all_non_spec_params_at_default(
        self, vst_interactive: ModuleType
    ) -> None:
        """No drift on any non-spec param → no exception.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        plugin = _FakePlugin({"a_synth": 0.5, "fx_amount": 0.3, "global_volume": 0.7})
        spec = _FakeSpec(["a_synth"])
        defaults = {"a_synth": 0.5, "fx_amount": 0.3, "global_volume": 0.7}

        result = vst_interactive._validate_no_drift(plugin, spec, defaults)

        assert result is None

    def test_raises_value_error_when_non_spec_param_drifted(
        self, vst_interactive: ModuleType
    ) -> None:
        """A non-spec param away from its default → ``ValueError`` naming the param.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        plugin = _FakePlugin({"a_synth": 0.5, "fx_amount": 0.9})
        spec = _FakeSpec(["a_synth"])
        defaults = {"a_synth": 0.5, "fx_amount": 0.3}

        with pytest.raises(ValueError, match="fx_amount"):
            vst_interactive._validate_no_drift(plugin, spec, defaults)

    def test_ignores_drift_on_spec_params(self, vst_interactive: ModuleType) -> None:
        """Spec params are allowed to vary; only non-spec drift is flagged.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        plugin = _FakePlugin({"a_synth": 0.99, "fx_amount": 0.3})
        spec = _FakeSpec(["a_synth"])
        defaults = {"a_synth": 0.5, "fx_amount": 0.3}

        result = vst_interactive._validate_no_drift(plugin, spec, defaults)

        assert result is None

    def test_drift_within_tolerance_does_not_raise(self, vst_interactive: ModuleType) -> None:
        """Tiny float deviation within abs_tol=1e-6 is treated as equal.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        plugin = _FakePlugin({"fx_amount": 0.3 + 5e-7})
        spec = _FakeSpec([])
        defaults = {"fx_amount": 0.3}

        result = vst_interactive._validate_no_drift(plugin, spec, defaults)

        assert result is None

    def test_drift_just_above_tolerance_raises(self, vst_interactive: ModuleType) -> None:
        """Deviation just above abs_tol=1e-6 is flagged.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        plugin = _FakePlugin({"fx_amount": 0.3 + 1e-3})
        spec = _FakeSpec([])
        defaults = {"fx_amount": 0.3}

        with pytest.raises(ValueError, match="fx_amount"):
            vst_interactive._validate_no_drift(plugin, spec, defaults)


def _build_keyboard_loop_plugin(simple_spec: ParamSpec) -> tuple["_FakePlugin", dict[str, float]]:
    """Build a ``_FakePlugin`` and matching defaults for ``_validate_no_drift`` tests.

    Carries every ``surge_simple`` synth param plus two non-spec params, with the matching
    ``default_params`` dict that ``_validate_no_drift`` checks against.

    Returns ``(plugin, default_params)``.

    :param simple_spec: Simple ParamSpec fixture used by the scenario.
    :returns: Fake plugin state used by keyboard-loop tests.
    """
    spec_defaults = {name: 0.25 for name in simple_spec.synth_param_names}
    non_spec_defaults = {"fx_amount": 0.1, "global_volume": 0.7}
    plugin = _FakePlugin({**spec_defaults, **non_spec_defaults})
    return plugin, {**spec_defaults, **non_spec_defaults}


class TestKeyboardLoop:
    """Read keystrokes via ``keystroke_source`` and snapshot params on ``p`` until exit."""

    def test_p_records_patch_q_quits(
        self, vst_interactive: ModuleType, simple_spec: ParamSpec
    ) -> None:
        """``["p", "q"]`` records exactly one patch with every spec synth-param key, then quits.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        """
        plugin, default_params = _build_keyboard_loop_plugin(simple_spec)
        stop_event = threading.Event()
        keystrokes = iter(["p", "q"])

        patches = vst_interactive.keyboard_loop(
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

    def test_unknown_keys_are_ignored(
        self, vst_interactive: ModuleType, simple_spec: ParamSpec
    ) -> None:
        """Ignore keystrokes outside ``{p, q}``: no patch, no ``stop_event``, no raise.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        """
        plugin, default_params = _build_keyboard_loop_plugin(simple_spec)
        stop_event = threading.Event()
        keystrokes = iter(["x", "p", "z", "q"])

        patches = vst_interactive.keyboard_loop(
            plugin,
            stop_event,
            SURGE_SIMPLE,
            default_params,
            keystroke_source=keystrokes.__next__,
        )

        assert len(patches) == 1, "x and z should be ignored — only p records"
        assert stop_event.is_set()

    def test_stop_event_set_externally_exits_without_consuming_source(
        self, vst_interactive: ModuleType, simple_spec: ParamSpec
    ) -> None:
        """Return ``[]`` immediately without polling the source when ``stop_event`` is set.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        """
        plugin, default_params = _build_keyboard_loop_plugin(simple_spec)
        stop_event = threading.Event()
        stop_event.set()

        consumed: list[str] = []

        def watching_source() -> str:
            consumed.append("polled")
            return "p"

        patches = vst_interactive.keyboard_loop(
            plugin,
            stop_event,
            SURGE_SIMPLE,
            default_params,
            keystroke_source=watching_source,
        )

        assert patches == []
        assert consumed == [], "loop must check stop_event before polling the source"

    def test_source_exhaustion_quits_gracefully_and_sets_stop_event(
        self, vst_interactive: ModuleType, simple_spec: ParamSpec
    ) -> None:
        """Return recorded patches and set ``stop_event`` on source ``StopIteration``.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        """
        plugin, default_params = _build_keyboard_loop_plugin(simple_spec)
        stop_event = threading.Event()
        keystrokes = iter(["p"])  # No q — source will raise StopIteration on the second poll.

        patches = vst_interactive.keyboard_loop(
            plugin,
            stop_event,
            SURGE_SIMPLE,
            default_params,
            keystroke_source=keystrokes.__next__,
        )

        assert len(patches) == 1
        assert stop_event.is_set()

    def test_drift_during_record_raises_valueerror_and_sets_stop_event(
        self, vst_interactive: ModuleType, simple_spec: ParamSpec
    ) -> None:
        """Raise ``ValueError`` via ``_validate_no_drift`` when a non-spec param drifted.

        The loop sets ``stop_event`` and re-raises (so the orchestrator sees the failure
        instead of silently dropping it).

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        """
        plugin, default_params = _build_keyboard_loop_plugin(simple_spec)
        # Drift the non-spec ``fx_amount`` from its default to trip _validate_no_drift.
        plugin.parameters["fx_amount"].raw_value = 0.99
        stop_event = threading.Event()
        keystrokes = iter(["p"])

        with pytest.raises(ValueError, match="drifted"):
            vst_interactive.keyboard_loop(
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
    pred_tensor_factory: Callable[[int], torch.Tensor] | None = None,
) -> None:
    """Write the per-sample ``pred``/``target-audio``/``target-params`` files for a sample dir.

    Mirrors the files ``PredictionWriter`` emits, populated with finite tensors by default.
    ``pred_tensor_factory`` lets a test override the ``pred-{i}.pt`` payload (e.g. to inject
    NaN/Inf); the target tensors are always finite stubs.

    :param output_dir: Destination for generated prediction files.
    :param num_samples: Number of indexed samples to materialize.
    :param pred_tensor_factory: Factory producing each indexed prediction tensor.
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

    def test_returns_three_names_per_sample(self, vst_interactive: ModuleType) -> None:
        """For ``num_samples`` samples, three sorted filenames per sample are returned.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        names = vst_interactive._expected_prediction_filenames(num_samples=2)

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

    def test_zero_samples_returns_empty(self, vst_interactive: ModuleType) -> None:
        """Zero samples returns an empty list.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        assert vst_interactive._expected_prediction_filenames(num_samples=0) == []


class _RecordingSubprocessRunner:
    """Test fake matching the ``SubprocessRunner`` shape from ``vst_interactive``.

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
    """Tests for ``_run_predict``'s eval-CLI invocation and Hydra overrides."""

    def test_passes_param_spec_and_absolute_paths(self, vst_interactive: ModuleType) -> None:
        """No width override may reach the subprocess, and paths must be cwd-independent.

        ``model.net.d_out`` must be absent — width follows the spec selection — and
        relative inputs must arrive resolved because the subprocess may run elsewhere.

        :param vst_interactive: Lazily imported interactive module fixture.
        """
        # Use relative paths so the test fails if .resolve() is dropped.
        ckpt = Path("relative/ckpt.ckpt")
        dataset_root = Path("relative/dataset")
        predict_file = Path("relative/dataset/predict.lance")
        predictions_dir = Path("relative/preds")

        runner = _RecordingSubprocessRunner()

        vst_interactive._run_predict(
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
        assert f"datamodule.param_spec_name={SURGE_SIMPLE}" in args
        assert not any(arg.startswith("model.net.d_out=") for arg in args)
        # Every path-bearing override must be absolute.
        for prefix, original in (
            ("ckpt_path=", ckpt),
            ("datamodule.predict_file=", predict_file),
            ("datamodule.dataset_root=", dataset_root),
            ("callbacks.prediction_writer.output_dir=", predictions_dir),
        ):
            arg = next(a for a in args if a.startswith(prefix))
            value = arg.removeprefix(prefix)
            assert Path(value).is_absolute(), f"{prefix} should be absolute, got {value!r}"
            assert value == str(original.resolve())

    def test_explicit_experiment_when_provided_is_forwarded(
        self, vst_interactive: ModuleType
    ) -> None:
        """Forward the operator-selected Hydra experiment to prediction.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        runner = _RecordingSubprocessRunner()

        vst_interactive._run_predict(
            Path("model.ckpt"),
            Path("dataset"),
            Path("dataset/predict.lance"),
            Path("predictions"),
            SURGE_SIMPLE,
            experiment="vst/custom",
            subprocess_runner=runner,
        )

        assert "experiment=vst/custom" in runner.calls[0]
        assert "experiment=surge/test" not in runner.calls[0]


class TestValidatePredictions:
    """``_validate_predictions`` checks expected files exist and tensors are finite."""

    def test_passes_on_complete_finite_outputs(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Happy path: complete file set with finite predictions does not raise.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        _write_pred_files(tmp_path, num_samples=2)

        vst_interactive._validate_predictions(tmp_path, num_samples=2)

    def test_missing_file_raises_filenotfounderror(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """A missing per-sample file raises ``FileNotFoundError`` listing the missing entry.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        _write_pred_files(tmp_path, num_samples=2)
        (tmp_path / "pred-1.pt").unlink()

        with pytest.raises(FileNotFoundError, match="pred-1.pt"):
            vst_interactive._validate_predictions(tmp_path, num_samples=2)

    def test_extra_file_raises_filenotfounderror(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """An unexpected file in the directory raises ``FileNotFoundError`` (set mismatch).

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        _write_pred_files(tmp_path, num_samples=1)
        torch.save(torch.zeros(1), tmp_path / "stray-extra.pt")

        with pytest.raises(FileNotFoundError, match="stray-extra.pt"):
            vst_interactive._validate_predictions(tmp_path, num_samples=1)

    def test_nan_prediction_raises_valueerror(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """A NaN value in any ``pred-{i}.pt`` raises ``ValueError`` naming the offending file.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """

        def factory(idx: int) -> torch.Tensor:
            if idx == 0:
                return torch.tensor([[float("nan")]], dtype=torch.float32)
            return torch.zeros((1, 1), dtype=torch.float32)

        _write_pred_files(tmp_path, num_samples=2, pred_tensor_factory=factory)

        with pytest.raises(ValueError, match="pred-0.pt"):
            vst_interactive._validate_predictions(tmp_path, num_samples=2)


class TestValidateMetricsDf:
    """``_validate_metrics_df`` validates row count, expected columns, and finiteness."""

    def test_passes_on_matching_shape_and_finite(self, vst_interactive: ModuleType) -> None:
        """Happy path: matching rows, expected columns, all-finite values does not raise.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        df = pd.DataFrame({"mss": [0.1, 0.2], "extra": [1.0, 2.0]})
        spec = vst_interactive._MetricsFileSpec(rows=2, columns=frozenset({"mss"}))

        vst_interactive._validate_metrics_df(Path("metrics.csv"), df, spec)

    def test_wrong_rows_raises_valueerror(self, vst_interactive: ModuleType) -> None:
        """Row count mismatch raises ``ValueError`` mentioning expected and actual.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        df = pd.DataFrame({"mss": [0.1]})
        spec = vst_interactive._MetricsFileSpec(rows=2, columns=frozenset({"mss"}))

        with pytest.raises(ValueError, match="expected 2 rows"):
            vst_interactive._validate_metrics_df(Path("metrics.csv"), df, spec)

    def test_missing_column_raises_valueerror(self, vst_interactive: ModuleType) -> None:
        """A missing expected column raises ``ValueError`` listing the missing column.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        df = pd.DataFrame({"other": [0.1, 0.2]})
        spec = vst_interactive._MetricsFileSpec(rows=2, columns=frozenset({"mss"}))

        with pytest.raises(ValueError, match="missing expected columns"):
            vst_interactive._validate_metrics_df(Path("metrics.csv"), df, spec)

    def test_nan_in_expected_column_raises_valueerror(self, vst_interactive: ModuleType) -> None:
        """A NaN in any expected column raises ``ValueError`` (NaN/Inf message).

        :param vst_interactive: Loaded VST interactive module under test.
        """
        df = pd.DataFrame({"mss": [0.1, float("nan")]})
        spec = vst_interactive._MetricsFileSpec(rows=2, columns=frozenset({"mss"}))

        with pytest.raises(ValueError, match="NaN/Inf"):
            vst_interactive._validate_metrics_df(Path("metrics.csv"), df, spec)

    def test_nan_error_reports_offending_row_count_not_full_df(
        self, vst_interactive: ModuleType
    ) -> None:
        """Error message reports only the offending row count and rows, not the full DataFrame.

        Otherwise a 1000-row metrics.csv with one bad row dumps a thousand lines into the
        traceback.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        df = pd.DataFrame(
            {
                "mss": [0.1, float("nan"), 0.3, 0.4, 0.5],
                "wmfcc": [0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )
        spec = vst_interactive._MetricsFileSpec(rows=5, columns=frozenset({"mss", "wmfcc"}))

        with pytest.raises(ValueError) as exc_info:
            vst_interactive._validate_metrics_df(Path("metrics.csv"), df, spec)

        msg = str(exc_info.value)
        assert "1 of 5 rows" in msg
        # Finite rows must NOT appear in the message (i.e., the helper isn't dumping the full df).
        assert "0.5" not in msg
        assert "0.6" not in msg


class _RecordingEvalRunner:
    """Test fake matching the ``EvalRunner`` shape from ``vst_interactive``.

    Records the positional args from each invocation and the call count so tests can assert
    on real state instead of patching the module-level ``eval_patches`` symbol. Reused
    across ``TestMaybeEvalCapturedPatches`` whenever a test needs to observe whether (and
    with what arguments) eval was invoked.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, Path, Path, str, str, str]] = []

    def __call__(
        self,
        num_samples: int,
        *,
        dataset_root_dir: Path,
        checkpoint_path: Path,
        param_spec_name: str,
        plugin_state_path: str,
        experiment: str,
    ) -> None:
        self.calls.append(
            (
                num_samples,
                dataset_root_dir,
                checkpoint_path,
                param_spec_name,
                plugin_state_path,
                experiment,
            )
        )


def _write_stub_lance_dir(path: Path, content: bytes = b"train-content") -> None:
    """Create a stand-in ``.lance`` directory tree for copytree replication tests.

    :param path: Directory to create.
    :param content: Bytes written into a single ``data`` member so copies are content-checkable.
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / "data").write_bytes(content)


class TestMaybeEvalCapturedPatches:
    """``_maybe_eval_captured_patches`` wires up the train.lance -> sibling replication."""

    def test_no_checkpoint_skips_replication_and_eval(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Skip sibling-file creation and ``eval_patches`` when ``--checkpoint-path`` is absent.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        train_path = tmp_path / "train.lance"
        _write_stub_lance_dir(train_path, b"stub")
        runner = _RecordingEvalRunner()

        vst_interactive._maybe_eval_captured_patches(
            patch_file_path=train_path,
            output_dataset_dir_path=tmp_path,
            num_patches=1,
            checkpoint_path=None,
            param_spec_name=SURGE_SIMPLE,
            plugin_state_path="presets/surge-base.vstpreset",
            eval_runner=runner,
        )

        for sibling in ("test.lance", "val.lance", "predict.lance"):
            assert not (tmp_path / sibling).exists()
        assert runner.calls == []

    def test_replicates_train_lance_to_three_siblings(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Copy ``train.lance`` to test/val/predict.lance and forward args to the eval runner.

        :param vst_interactive: The lazily-imported tool module under test.
        :param tmp_path: Per-test tmp dir holding the source and replicated Lance dirs.
        """
        train_path = tmp_path / "train.lance"
        rows = np.arange(12, dtype=np.float32).reshape(3, 4)
        _write_param_array_lance(train_path, rows)
        ckpt_path = tmp_path / "model.ckpt"
        ckpt_path.write_bytes(b"ckpt")
        runner = _RecordingEvalRunner()

        vst_interactive._maybe_eval_captured_patches(
            patch_file_path=train_path,
            output_dataset_dir_path=tmp_path,
            num_patches=3,
            checkpoint_path=ckpt_path,
            param_spec_name=SURGE_SIMPLE,
            plugin_state_path="presets/surge-simple.vstpreset",
            experiment="vst/custom",
            eval_runner=runner,
        )

        import lance

        for sibling in ("test.lance", "val.lance", "predict.lance"):
            copied = lance.dataset(str(tmp_path / sibling)).to_table()["param_array"]
            np.testing.assert_array_equal(copied.combine_chunks().to_numpy_ndarray(), rows)
        assert runner.calls == [
            (
                3,
                tmp_path,
                ckpt_path,
                SURGE_SIMPLE,
                "presets/surge-simple.vstpreset",
                "vst/custom",
            )
        ]

    def test_failed_copy_rolls_back_partial_siblings(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Roll back earlier siblings on later ``shutil.copytree`` ``OSError``; skip eval.

        Failure is triggered by a *real* OS error: ``val.lance`` is pre-created as a directory, so
        the second ``copytree`` fails with ``FileExistsError``; asserting on ``OSError`` matches
        the SUT's ``except OSError:`` contract.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        train_path = tmp_path / "train.lance"
        _write_stub_lance_dir(train_path, b"train-content")
        ckpt_path = tmp_path / "model.ckpt"
        ckpt_path.write_bytes(b"ckpt")
        # Block the second sibling write with a pre-existing directory. Order matches
        # ``_maybe_eval_captured_patches``: test.lance → val.lance → predict.lance.
        (tmp_path / "val.lance").mkdir()

        runner = _RecordingEvalRunner()

        with pytest.raises(OSError):
            vst_interactive._maybe_eval_captured_patches(
                patch_file_path=train_path,
                output_dataset_dir_path=tmp_path,
                num_patches=1,
                checkpoint_path=ckpt_path,
                param_spec_name=SURGE_SIMPLE,
                plugin_state_path="presets/surge-base.vstpreset",
                eval_runner=runner,
            )

        # First sibling was copied, second failed; rollback removes the first.
        assert not (tmp_path / "test.lance").exists(), "rollback should remove test.lance"
        # val.lance still exists as the pre-created directory, but no copy landed inside it.
        assert (tmp_path / "val.lance").is_dir()
        assert not (tmp_path / "val.lance" / "data").exists()
        assert not (tmp_path / "predict.lance").exists()
        assert runner.calls == []

    def test_failed_copy_removes_current_partial_destination(
        self,
        vst_interactive: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Remove a new destination partially created by a failing copy.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        :param monkeypatch: Pytest fixture for inducing the partial-copy error.
        """
        train_path = tmp_path / "train.lance"
        _write_stub_lance_dir(train_path)
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_bytes(b"ckpt")
        real_copytree = vst_interactive.shutil.copytree

        def copytree_with_partial_failure(source: Path, destination: Path) -> Path:
            if destination.name == "val.lance":
                destination.mkdir()
                (destination / "partial").write_bytes(b"partial")
                raise OSError("simulated partial copy")
            return real_copytree(source, destination)

        monkeypatch.setattr(vst_interactive.shutil, "copytree", copytree_with_partial_failure)

        with pytest.raises(OSError, match="partial copy"):
            vst_interactive._maybe_eval_captured_patches(
                patch_file_path=train_path,
                output_dataset_dir_path=tmp_path,
                num_patches=1,
                checkpoint_path=checkpoint_path,
                param_spec_name=SURGE_SIMPLE,
                plugin_state_path="presets/surge-base.vstpreset",
                eval_runner=_RecordingEvalRunner(),
            )

        assert not (tmp_path / "test.lance").exists()
        assert not (tmp_path / "val.lance").exists()
        assert not (tmp_path / "predict.lance").exists()


def _write_wav(path: Path, *, silent: bool, sample_rate: int = 44100) -> None:
    """Write a brief mono WAV at ``path``.

    ``silent=True`` writes zeros (peak == 0); otherwise a
    half-amplitude 440 Hz sine (peak ~0.5, well above ``SILENCE_PEAK_THRESHOLD``).

    Samples are shaped ``(num_frames, num_channels)`` to match the convention used by
    ``play_audio_recorded`` and ``predict_vst_audio.py`` (both pass ``output.T``).

    :param path: Destination WAV path.
    :param silent: Whether generated audio should contain only zeros.
    :param sample_rate: Audio sample rate in hertz.
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
    """Pre-create ``sample_{i}`` subdirs with per-sample artifacts for render validation.

    :param audio_dir: Root containing indexed sample audio artifacts.
    :param num_samples: Number of indexed samples to materialize.
    :param silent: Whether generated audio should contain only zeros.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_samples):
        sample_dir = audio_dir / f"sample_{i}"
        sample_dir.mkdir()
        _write_wav(sample_dir / "target.wav", silent=silent)
        _write_wav(sample_dir / "pred.wav", silent=silent)
        (sample_dir / "spec.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (sample_dir / "params.csv").write_text("name,value\n")


class _MaterializingPipelineRunner:
    """Materialize each subprocess stage's outputs for an orchestration test.

    :param num_samples: Number of indexed artifacts each stage should emit.
    """

    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples
        self.calls: list[list[str]] = []

    def __call__(
        self,
        args: list[str],
        *,
        check: bool | None = None,
        timeout: int | None = None,
    ) -> None:
        """Record a command and create the artifacts its real subprocess owns.

        :param args: Subprocess argument vector.
        :param check: Whether the real subprocess would enforce a zero exit status.
        :param timeout: Real subprocess deadline in seconds.
        :raises AssertionError: The command targets an unknown module.
        """
        del check, timeout
        self.calls.append(args)
        module_index = args.index("-m") + 1
        module = args[module_index]
        if module == "synth_setter.cli.eval":
            output_arg = next(
                arg for arg in args if arg.startswith("callbacks.prediction_writer.output_dir=")
            )
            output_dir = Path(output_arg.partition("=")[2])
            for index in range(self.num_samples):
                torch.save(torch.zeros(1), output_dir / f"pred-{index}.pt")
                torch.save(torch.zeros(1), output_dir / f"target-audio-{index}.pt")
                torch.save(torch.zeros(1), output_dir / f"target-params-{index}.pt")
            return
        if module == "synth_setter.evaluation.predict_vst_audio":
            _populate_audio_dir(Path(args[module_index + 2]), self.num_samples)
            return
        if module == "synth_setter.evaluation.compute_audio_metrics":
            metrics_dir = Path(args[module_index + 2])
            pd.DataFrame(
                {
                    metric: np.full(self.num_samples, 0.5)
                    for metric in ("mss", "wmfcc", "sot", "rms")
                }
            ).to_csv(metrics_dir / "metrics.csv", index=False)
            pd.DataFrame({"mean": np.full(4, 0.5), "std": np.zeros(4)}).to_csv(
                metrics_dir / "aggregated_metrics.csv", index=False
            )
            return
        raise AssertionError(f"unexpected subprocess module: {module}")


class TestEvalPatches:
    """``eval_patches`` wires prediction, rendering, and metrics in order."""

    def test_materialized_pipeline_outputs_are_validated(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Exercise all stages through their real artifact validators.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_bytes(b"checkpoint")
        (tmp_path / "predict.lance").mkdir()
        runner = _MaterializingPipelineRunner(num_samples=2)

        vst_interactive.eval_patches(
            2,
            dataset_root_dir=tmp_path,
            checkpoint_path=checkpoint_path,
            param_spec_name=SURGE_SIMPLE,
            plugin_state_path="presets/surge-simple.vstpreset",
            experiment="vst/custom",
            subprocess_runner=runner,
        )

        modules = [args[args.index("-m") + 1] for args in runner.calls]
        assert modules == [
            "synth_setter.cli.eval",
            "synth_setter.evaluation.predict_vst_audio",
            "synth_setter.evaluation.compute_audio_metrics",
        ]
        assert "experiment=vst/custom" in runner.calls[0]
        assert "datamodule.param_spec_name=surge_simple" in runner.calls[0]
        assert "surge_simple" in runner.calls[1]
        assert "presets/surge-simple.vstpreset" in runner.calls[1]
        assert len(pd.read_csv(tmp_path / "metrics" / "metrics.csv")) == 2


_RENDER_DEFAULT_PRESET = "presets/surge-base.vstpreset"


class TestBuildPredictVstAudioArgv:
    """``_build_predict_vst_audio_argv`` builds the argv list for ``predict_vst_audio.py``.

    Pure with respect to ``audio_dir`` / ``predictions_output_dir`` (no writes), and
    parameterised on ``platform`` + ``wrapper_path`` so each branch is tested without
    monkeypatching ``sys.platform`` or the module-level wrapper constant.
    """

    def test_linux_prepends_wrapper_to_argv(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """On Linux, the existing wrapper script is the first argv entry.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        wrapper_path = tmp_path / "wrapper.sh"
        wrapper_path.write_text('#!/usr/bin/env bash\nexec "$@"\n')
        wrapper_path.chmod(0o755)

        argv = vst_interactive._build_predict_vst_audio_argv(
            tmp_path / "preds",
            tmp_path / "audio",
            SURGE_SIMPLE,
            _RENDER_DEFAULT_PRESET,
            platform="linux",
            wrapper_path=wrapper_path,
        )

        assert argv[0] == str(wrapper_path)

    def test_linux_missing_wrapper_raises_filenotfounderror(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """On Linux, a missing wrapper path raises ``FileNotFoundError`` naming the path.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        missing_wrapper = tmp_path / "definitely-does-not-exist.sh"

        with pytest.raises(FileNotFoundError, match="VST headless wrapper not found"):
            vst_interactive._build_predict_vst_audio_argv(
                tmp_path / "preds",
                tmp_path / "audio",
                SURGE_SIMPLE,
                _RENDER_DEFAULT_PRESET,
                platform="linux",
                wrapper_path=missing_wrapper,
            )

    def test_non_linux_does_not_prepend_wrapper(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """On non-Linux platforms the wrapper is not prepended and a missing wrapper is fine.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        missing_wrapper = tmp_path / "definitely-does-not-exist.sh"

        argv = vst_interactive._build_predict_vst_audio_argv(
            tmp_path / "preds",
            tmp_path / "audio",
            SURGE_SIMPLE,
            _RENDER_DEFAULT_PRESET,
            platform="darwin",
            wrapper_path=missing_wrapper,
        )

        assert str(missing_wrapper) not in argv
        assert argv[0] == vst_interactive.sys.executable

    def test_param_spec_and_plugin_state_path_are_forwarded(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """``param_spec_name`` and ``plugin_state_path`` follow their flag tokens in argv.

        Otherwise ``predict_vst_audio.py`` would silently fall back to ``surge_xt`` /
        ``presets/surge-base.vstpreset`` and decode/render against a mismatched spec.

        :param vst_interactive: The lazily-imported tool module under test.
        :param tmp_path: Per-test tmp dir for the predictions and audio output paths.
        """
        argv = vst_interactive._build_predict_vst_audio_argv(
            tmp_path / "preds",
            tmp_path / "audio",
            "custom-spec",
            "presets/custom.vstpreset",
            platform="darwin",
        )

        assert "--param_spec" in argv
        assert argv[argv.index("--param_spec") + 1] == "custom-spec"
        assert "--plugin_state_path" in argv
        assert argv[argv.index("--plugin_state_path") + 1] == "presets/custom.vstpreset"
        assert argv[-1] == "-t"

    def test_predictions_and_audio_dirs_appear_as_positional_args(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Match the two positional argv entries to the caller's source/destination paths.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        preds_dir = tmp_path / "preds"
        audio_dir = tmp_path / "audio"

        argv = vst_interactive._build_predict_vst_audio_argv(
            preds_dir, audio_dir, SURGE_SIMPLE, _RENDER_DEFAULT_PRESET, platform="darwin"
        )

        assert str(preds_dir) in argv
        assert str(audio_dir) in argv
        # Positional args follow the predict_vst_audio.py entry; pred_dir before output_dir.
        assert argv.index(str(preds_dir)) < argv.index(str(audio_dir))


class TestValidateRenderedAudioDir:
    """``_validate_rendered_audio_dir`` checks the per-sample artifacts after rendering."""

    def test_returns_none_on_complete_non_silent_dir(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """All artifacts present and audible → returns ``None`` without raising.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        audio_dir = tmp_path / "audio"
        _populate_audio_dir(audio_dir, num_samples=2)

        # Returns None (implicit) — no assertion needed beyond the absence of exceptions.
        vst_interactive._validate_rendered_audio_dir(audio_dir, num_samples=2)

    def test_missing_artifact_raises_filenotfounderror(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """A missing per-sample artifact (``spec.png``) raises ``FileNotFoundError`` naming it.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        audio_dir = tmp_path / "audio"
        _populate_audio_dir(audio_dir, num_samples=1)
        (audio_dir / "sample_0" / "spec.png").unlink()

        with pytest.raises(FileNotFoundError, match="spec.png"):
            vst_interactive._validate_rendered_audio_dir(audio_dir, num_samples=1)

    def test_silent_wav_raises_valueerror(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """A silent rendered WAV raises ``ValueError`` naming the offending sample / file.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        audio_dir = tmp_path / "audio"
        _populate_audio_dir(audio_dir, num_samples=1, silent=True)

        with pytest.raises(ValueError, match=r"sample_0/(target|pred)\.wav is silent"):
            vst_interactive._validate_rendered_audio_dir(audio_dir, num_samples=1)

    def test_unexpected_sample_dirs_raises_filenotfounderror(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Mismatched sample directory set (missing) raises ``FileNotFoundError``.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        audio_dir = tmp_path / "audio"
        # Caller asks for num_samples=2 but only sample_0 exists.
        _populate_audio_dir(audio_dir, num_samples=1)

        with pytest.raises(FileNotFoundError, match="unexpected sample directories"):
            vst_interactive._validate_rendered_audio_dir(audio_dir, num_samples=2)

    def test_extra_sample_dir_raises_filenotfounderror(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Raise ``FileNotFoundError`` on an extra sample dir to reject stale leftovers.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
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
            vst_interactive._validate_rendered_audio_dir(audio_dir, num_samples=2)

    def test_num_samples_12_does_not_trip_lex_sort(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Regression: ``num_samples=12`` must survive ``sample_10`` sorting before ``sample_2``.

        Set comparison + index iteration keeps validation index-based regardless of directory
        iteration order.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        audio_dir = tmp_path / "audio"
        _populate_audio_dir(audio_dir, num_samples=12)

        vst_interactive._validate_rendered_audio_dir(audio_dir, num_samples=12)


class TestRenderPredictedAudioSubprocessIntegration:
    """``_render_predicted_audio`` orchestrator: argv build + subprocess invocation + validation.

    Verified with ``_RecordingSubprocessRunner`` (state-based, no monkeypatch). Real
    ``subprocess.run`` lifecycle is exercised by the e2e test below.
    """

    def test_runner_receives_argv_with_check_and_timeout(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Forward argv to the runner with ``check=True`` and a positive ``timeout``.

        Drift on either kwarg would silently turn fatal subprocess errors into successes.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        audio_dir = tmp_path / "audio"
        _populate_audio_dir(audio_dir, num_samples=1)
        runner = _RecordingSubprocessRunner()

        vst_interactive._render_predicted_audio(
            tmp_path / "preds",
            audio_dir,
            num_samples=1,
            param_spec_name=SURGE_SIMPLE,
            plugin_state_path=_RENDER_DEFAULT_PRESET,
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


# These tests require real Surge XT plugin behavior rather than fakes.


@pytest.mark.requires_vst
@pytest.mark.slow
class TestPlayAudioRecordedE2E:
    """End-to-end: real Surge XT plugin renders a non-silent deterministic clip."""

    def test_play_audio_recorded_produces_non_silent_wav(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Render a WAV via real Surge XT + ``surge-simple.vstpreset`` with non-silent audio.

        Verifies ``play_audio_recorded`` writes a file with the expected frame count and an
        audible peak amplitude.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        from synth_setter.data.vst.core import load_plugin, load_preset

        plugin_path = "plugins/Surge XT.vst3"
        plugin_state_path = "presets/surge-simple.vstpreset"
        if not Path(plugin_path).exists():
            pytest.skip(f"Surge XT plugin not found at {plugin_path}")
        if not Path(plugin_state_path).exists():
            pytest.skip(f"Surge XT base preset not found at {plugin_state_path}")

        plugin = load_plugin(plugin_path)
        load_preset(plugin, plugin_state_path)
        # Production parity: post-load flush commits preset state — see render_params and main.
        vst_interactive._flush_plugin(plugin)

        output_path = tmp_path / "session.wav"
        vst_interactive.play_audio_recorded(plugin, output_path)

        assert output_path.is_file()
        with AudioFile(str(output_path)) as f:
            audio = f.read(f.frames)
        expected_frames = int(
            vst_interactive.SESSION_RECORDING_DURATION_SECONDS * vst_interactive.SAMPLE_RATE
        )
        assert audio.shape == (vst_interactive.CHANNELS, expected_frames)
        peak = float(np.abs(audio).max())
        assert peak > vst_interactive.SILENCE_PEAK_THRESHOLD, (
            f"recorded WAV is silent (peak={peak:.2e}); a real preset should produce audible output"
        )


def _write_synthetic_prediction_files(
    pred_dir: Path, num_samples: int, simple_spec: ParamSpec
) -> None:
    """Write synthetic per-sample pred/target tensors so ``predict_vst_audio.py`` can render.

    Renders without a model checkpoint. Each pred row encodes mid-range params (zeros in the
    (-1..1) coordinate) so the rendered audio depends only on the preset and a deterministic
    note pattern.

    :param pred_dir: Destination for synthetic prediction tensors.
    :param num_samples: Number of indexed samples to materialize.
    :param simple_spec: Simple ParamSpec fixture used by the scenario.
    """
    pred_dir.mkdir(parents=True, exist_ok=True)
    total_length = simple_spec.synth_param_length + simple_spec.note_param_length
    # ``predict_vst_audio.py`` loads ``target-audio-{i}.pt`` unconditionally and indexes
    # ``target_audio[j]`` for spectrogram generation even under ``-t``/``--rerender_target``,
    # so the saved tensor must be (batch, channels, frames) matching that script's CLI
    # defaults. Contents can be silent — only the post-render pred/target WAVs are checked
    # for non-silence.
    predict_default_channels = 2
    predict_default_frames = int(4.0 * 44100)
    synthetic_target_audio = torch.zeros(
        (1, predict_default_channels, predict_default_frames), dtype=torch.float32
    )
    for i in range(num_samples):
        pred_row = torch.zeros((1, total_length), dtype=torch.float32)
        torch.save(pred_row, pred_dir / f"pred-{i}.pt")
        torch.save(pred_row, pred_dir / f"target-params-{i}.pt")
        torch.save(synthetic_target_audio, pred_dir / f"target-audio-{i}.pt")


@pytest.mark.requires_vst
@pytest.mark.slow
class TestRenderPredictedAudioE2E:
    """End-to-end: real ``predict_vst_audio.py`` subprocess produces non-silent per-sample WAVs."""

    def test_render_predicted_audio_against_real_subprocess(
        self,
        vst_interactive: ModuleType,
        tmp_path: Path,
        simple_spec: ParamSpec,
    ) -> None:
        """Invoke the real ``predict_vst_audio.py`` and validate non-silent rendered WAVs.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        """
        plugin_path = "plugins/Surge XT.vst3"
        plugin_state_path = "presets/surge-simple.vstpreset"
        if not Path(plugin_path).exists():
            pytest.skip(f"Surge XT plugin not found at {plugin_path}")
        if not Path(plugin_state_path).exists():
            pytest.skip(f"Surge XT base preset not found at {plugin_state_path}")

        # Use one sample because subsequent identical-zero rows can render silent in Surge XT.
        num_samples = 1
        pred_dir = tmp_path / "preds"
        audio_dir = tmp_path / "audio"
        _write_synthetic_prediction_files(pred_dir, num_samples, simple_spec)

        vst_interactive._render_predicted_audio(
            pred_dir,
            audio_dir,
            num_samples,
            param_spec_name=SURGE_SIMPLE,
            plugin_state_path=plugin_state_path,
        )

        for i in range(num_samples):
            sample_dir = audio_dir / f"sample_{i}"
            for fname in ("target.wav", "pred.wav", "spec.png", "params.csv"):
                assert (sample_dir / fname).is_file()
            for wav_name in ("target.wav", "pred.wav"):
                with AudioFile(str(sample_dir / wav_name)) as f:
                    audio = f.read(f.frames)
                peak = float(np.abs(audio).max())
                assert peak > vst_interactive.SILENCE_PEAK_THRESHOLD, (
                    f"{sample_dir.name}/{wav_name} is silent (peak={peak:.2e})"
                )


@pytest.mark.requires_vst
@pytest.mark.slow
class TestKeyboardLoopE2E:
    """End-to-end: real Surge XT plugin records a patch via deterministic keystrokes."""

    def test_p_q_against_real_plugin_records_one_patch(
        self, vst_interactive: ModuleType, simple_spec: ParamSpec
    ) -> None:
        """Record a patch whose synth-param values match the post-preset-load Surge XT state.

        Drives the real Surge XT VST + ``surge-simple.vstpreset`` with ``["p", "q"]`` and asserts
        the recorded patch's synth-param values are finite floats matching the post-preset-load
        defaults.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        """
        from synth_setter.data.vst.core import load_plugin, load_preset

        plugin_path = "plugins/Surge XT.vst3"
        plugin_state_path = "presets/surge-simple.vstpreset"
        if not Path(plugin_path).exists():
            pytest.skip(f"Surge XT plugin not found at {plugin_path}")
        if not Path(plugin_state_path).exists():
            pytest.skip(f"Surge XT base preset not found at {plugin_state_path}")

        plugin = load_plugin(plugin_path)
        load_preset(plugin, plugin_state_path)
        # Match production: post-load flush commits the preset state so all spec params are
        # actually exposed (Surge XT hides oscillator-shape params until the active osc type
        # is committed). Same flush pattern used by ``render_params`` and ``main`` in the
        # production script.
        vst_interactive._flush_plugin(plugin)

        # Snapshot post-flush defaults so _validate_no_drift has a known baseline.
        default_params = {
            name: plugin.parameters[name].raw_value  # type: ignore[attr-defined]
            for name in plugin.parameters.keys()  # type: ignore[attr-defined]
        }

        stop_event = threading.Event()
        keystrokes = iter(["p", "q"])

        patches = vst_interactive.keyboard_loop(
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


class _StopBeforeGuiError(Exception):
    """Sentinel raised from a patched ``load_plugin``; carries the resolved plugin path."""


def _raise_stop(plugin_path: str) -> NoReturn:
    raise _StopBeforeGuiError(plugin_path)


class _CliPlugin:
    """Minimal VST3Plugin replacement for public-CLI branch tests."""

    def __init__(self) -> None:
        self.parameters: dict[str, object] = {}

    def show_editor(self, stop_event: threading.Event) -> None:
        """Close the fake editor immediately.

        :param stop_event: Shared shutdown signal.
        """
        stop_event.set()


def _invoke_cli_and_capture_params(
    vst_interactive: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
) -> dict[str, float]:
    """Invoke the Click command and return parameters sent to the plugin.

    :param vst_interactive: Loaded VST interactive module under test.
    :param monkeypatch: Pytest fixture replacing external plugin operations.
    :param args: CLI arguments after the command name.
    :returns: Parameter mapping passed to ``set_params``.
    """
    plugin = _CliPlugin()
    applied: list[dict[str, float]] = []
    monkeypatch.setattr(vst_interactive, "VST3Plugin", _CliPlugin)
    monkeypatch.setattr(vst_interactive, "load_plugin", lambda _path: plugin)
    monkeypatch.setattr(vst_interactive, "load_preset", lambda *_args: None)
    monkeypatch.setattr(vst_interactive, "_flush_plugin", lambda *_args: None)
    monkeypatch.setattr(
        vst_interactive, "set_params", lambda _plugin, params: applied.append(params)
    )
    monkeypatch.setattr(vst_interactive, "play_audio_recorded", lambda *_args: None)
    monkeypatch.setattr(vst_interactive, "keyboard_loop", lambda *_args: [])

    result = CliRunner().invoke(vst_interactive.main, args)

    assert result.exception is None, result.output
    assert len(applied) == 1
    return applied[0]


class TestMainParameterReference:
    """The public CLI applies decoded prediction and dataset references."""

    def test_pred_values_reach_plugin(
        self,
        vst_interactive: ModuleType,
        simple_spec_total_length: int,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Decode ``--pred`` and forward its synth values to ``set_params``.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec_total_length: Encoded width of the simple ParamSpec.
        :param tmp_path: Per-test temporary directory.
        :param monkeypatch: Pytest fixture replacing external plugin operations.
        """
        prediction = torch.zeros((1, simple_spec_total_length))
        prediction_path = tmp_path / "pred.pt"
        torch.save(prediction, prediction_path)

        applied = _invoke_cli_and_capture_params(
            vst_interactive,
            monkeypatch,
            [
                "--plugin-path",
                "fake.vst3",
                "--param-spec-name",
                SURGE_SIMPLE,
                "--pred",
                f"{prediction_path}:0",
                "--session-recording-path",
                str(tmp_path / "session.wav"),
            ],
        )

        assert applied == vst_interactive.decode_prediction_row(prediction, 0, SURGE_SIMPLE)

    def test_dataset_values_reach_plugin(
        self,
        vst_interactive: ModuleType,
        simple_spec: ParamSpec,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Decode ``--dataset-ref`` and forward its synth values to ``set_params``.

        :param vst_interactive: Loaded VST interactive module under test.
        :param simple_spec: Simple ParamSpec fixture used by the scenario.
        :param tmp_path: Per-test temporary directory.
        :param monkeypatch: Pytest fixture replacing external plugin operations.
        """
        synth_params, note_params = simple_spec.sample()
        dataset_path = tmp_path / "test.lance"
        _write_param_array_lance(
            dataset_path,
            simple_spec.encode(synth_params, note_params)[None, :],
        )

        applied = _invoke_cli_and_capture_params(
            vst_interactive,
            monkeypatch,
            [
                "--plugin-path",
                "fake.vst3",
                "--param-spec-name",
                SURGE_SIMPLE,
                "--dataset-ref",
                f"{dataset_path}:0",
                "--session-recording-path",
                str(tmp_path / "session.wav"),
            ],
        )

        assert applied == pytest.approx(synth_params, abs=1e-5)


class TestMainGuards:
    """Public CLI conflicts fail before plugin loading."""

    def test_pred_and_dataset_ref_when_combined_fail(self, vst_interactive: ModuleType) -> None:
        """Reject simultaneous parameter-row sources.

        :param vst_interactive: Loaded VST interactive module under test.
        """
        with mock.patch.object(vst_interactive, "load_plugin", _raise_stop):
            result = CliRunner().invoke(
                vst_interactive.main,
                [
                    "--param-spec-name",
                    SURGE_SIMPLE,
                    "--pred",
                    "pred.pt:0",
                    "--dataset-ref",
                    "test.lance:0",
                ],
            )

        assert result.exit_code == 2
        assert "mutually exclusive" in result.output

    def test_midi_and_recording_when_combined_fail(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Reject MIDI input during deterministic recording.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        with mock.patch.object(vst_interactive, "load_plugin", _raise_stop):
            result = CliRunner().invoke(
                vst_interactive.main,
                [
                    "--param-spec-name",
                    SURGE_SIMPLE,
                    "--midi-port",
                    "controller",
                    "--session-recording-path",
                    str(tmp_path / "session.wav"),
                ],
            )

        assert result.exit_code == 2
        assert "mutually exclusive" in result.output

    def test_existing_output_dataset_directory_fails(
        self, vst_interactive: ModuleType, tmp_path: Path
    ) -> None:
        """Reject an output directory that could be overwritten.

        :param vst_interactive: Loaded VST interactive module under test.
        :param tmp_path: Per-test temporary directory.
        """
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        with mock.patch.object(vst_interactive, "load_plugin", _raise_stop):
            result = CliRunner().invoke(
                vst_interactive.main,
                [
                    "--param-spec-name",
                    SURGE_SIMPLE,
                    "--output-dataset-dir-path",
                    str(output_dir),
                ],
            )

        assert result.exit_code == 2
        assert "already exists" in result.output


class TestMainPluginPathDefault:
    """``main``'s ``--plugin-path`` default flows through the env-aware registry resolver.

    ``load_plugin`` is patched to capture its argument and abort: once the guard
    clauses pass it is the first call that consumes the resolved ``--plugin-path``,
    so the captured path is exactly what the resolver produced.
    """

    def test_env_var_overrides_default_plugin_path(self) -> None:
        """``$SYNTH_SETTER_PLUGIN_PATH`` reaches ``load_plugin`` when no flag is passed."""
        sxi = importlib.import_module("synth_setter.tools.vst_interactive")
        with mock.patch.object(sxi, "load_plugin", _raise_stop):
            result = CliRunner().invoke(
                sxi.main,
                ["--param-spec-name", "surge_xt"],
                env={"SYNTH_SETTER_PLUGIN_PATH": "env-plugin.vst3"},
            )
        assert isinstance(result.exception, _StopBeforeGuiError)
        assert result.exception.args[0] == "env-plugin.vst3"

    def test_unset_env_falls_back_to_bundle(self) -> None:
        """With no override, ``--plugin-path`` resolves to the in-repo bundle default."""
        sxi = importlib.import_module("synth_setter.tools.vst_interactive")
        with mock.patch.dict(os.environ), mock.patch.object(sxi, "load_plugin", _raise_stop):
            os.environ.pop("SYNTH_SETTER_PLUGIN_PATH", None)
            result = CliRunner().invoke(sxi.main, ["--param-spec-name", "surge_xt"])
        assert isinstance(result.exception, _StopBeforeGuiError)
        assert result.exception.args[0] == "plugins/Surge XT.vst3"
