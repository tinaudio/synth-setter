"""Tests for scripts/surge_xt_interactive.py prediction decoding helpers."""

import importlib
from pathlib import Path

import click
import h5py
import numpy as np
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
    ) -> None:
        """Loads row 0 from the real ``surge_xt_smoke_datasets`` test.h5 via the surge_xt spec."""
        ref = surge_xt_interactive.DatasetRef(
            path=surge_xt_smoke_datasets / "test.h5", batch_idx=0
        )

        loaded = surge_xt_interactive.load_dataset_synth_params(ref, param_spec_name="surge_xt")

        expected_keys = {p.name for p in param_specs["surge_xt"].synth_params}
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
    """Minimal stand-in for a ``mido.Message`` — exposes ``type`` and prints recognizably."""

    def __init__(self, msg_type: str) -> None:
        self.type = msg_type

    def __repr__(self) -> str:
        return f"_FakeMidiMessage({self.type!r})"


class _FakeMidiPortHandle:
    """Iterable mido-input replacement; used as a context manager that yields fake messages."""

    def __init__(self, messages: list[_FakeMidiMessage]) -> None:
        self._messages = messages

    def __enter__(self) -> "_FakeMidiPortHandle":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def __iter__(self):
        return iter(self._messages)


class TestMidiListener:
    """``midi_listener`` filters mido messages by type and forwards them to a queue."""

    def test_only_relevant_message_types_are_forwarded(
        self, surge_xt_interactive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """note_on/off, control_change, pitchwheel, aftertouch are queued; others are dropped."""
        import queue as _queue

        forwarded = [
            _FakeMidiMessage("note_on"),
            _FakeMidiMessage("note_off"),
            _FakeMidiMessage("control_change"),
            _FakeMidiMessage("pitchwheel"),
            _FakeMidiMessage("aftertouch"),
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

        monkeypatch.setattr(
            surge_xt_interactive.mido,
            "open_input",
            lambda port_name: _FakeMidiPortHandle(all_messages),
        )

        midi_queue: _queue.Queue = _queue.Queue()
        surge_xt_interactive.midi_listener("fake-port", midi_queue)

        drained: list[_FakeMidiMessage] = []
        while not midi_queue.empty():
            drained.append(midi_queue.get_nowait())

        assert [m.type for m in drained] == [m.type for m in forwarded]


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
