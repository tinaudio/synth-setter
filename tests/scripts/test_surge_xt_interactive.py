"""Tests for scripts/surge_xt_interactive.py prediction decoding helpers."""

import importlib
import subprocess
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


class TestRunPredict:
    """``_run_predict`` builds the ``src/eval.py`` invocation with the right Hydra overrides."""

    def test_passes_d_out_override_and_absolute_paths(
        self,
        surge_xt_interactive,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``_run_predict`` overrides ``model.net.d_out`` with the encoded width of
        ``param_spec_name`` (otherwise the ``???`` sentinel in ``surge/test.yaml`` would error),
        and resolves all paths to absolute (otherwise Hydra's ``chdir`` would break relative
        refs)."""
        # Use relative paths so the test fails if .resolve() is dropped.
        ckpt = Path("relative/ckpt.ckpt")
        dataset_root = Path("relative/dataset")
        predict_file = Path("relative/dataset/predict.h5")
        predictions_dir = Path("relative/preds")

        captured: dict = {}

        def fake_check_call(args: list[str], **_kwargs: object) -> int:
            captured["args"] = list(args)
            return 0

        monkeypatch.setattr(surge_xt_interactive.subprocess, "check_call", fake_check_call)

        surge_xt_interactive._run_predict(
            ckpt, dataset_root, predict_file, predictions_dir, SURGE_SIMPLE
        )

        args = captured["args"]
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


class TestMaybeEvalCapturedPatches:
    """``_maybe_eval_captured_patches`` wires up the train.h5 -> sibling replication."""

    def test_no_checkpoint_skips_replication_and_eval(
        self, surge_xt_interactive, tmp_path: Path
    ) -> None:
        """Without ``--checkpoint-path``, no sibling files are created and eval_patches is not
        invoked."""
        train_path = tmp_path / "train.h5"
        train_path.write_bytes(b"stub")

        surge_xt_interactive._maybe_eval_captured_patches(
            patch_file_path=train_path,
            output_dataset_dir_path=tmp_path,
            num_patches=1,
            checkpoint_path=None,
            param_spec_name=SURGE_SIMPLE,
            preset_path="presets/surge-base.vstpreset",
        )

        for sibling in ("test.h5", "val.h5", "predict.h5"):
            assert not (tmp_path / sibling).exists()

    def test_replicates_train_h5_to_three_siblings(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``--checkpoint-path`` is given, ``train.h5`` is copied to test/val/predict.h5 and
        ``param_spec_name`` / ``preset_path`` are forwarded verbatim to ``eval_patches``."""
        train_path = tmp_path / "train.h5"
        train_path.write_bytes(b"train-content")
        ckpt_path = tmp_path / "model.ckpt"
        ckpt_path.write_bytes(b"ckpt")

        called_with: dict = {}

        def fake_eval_patches(
            num_samples: int,
            dataset_root_dir: Path,
            checkpoint_path: Path,
            param_spec_name: str,
            preset_path: str,
        ) -> None:
            called_with["num_samples"] = num_samples
            called_with["dataset_root_dir"] = dataset_root_dir
            called_with["checkpoint_path"] = checkpoint_path
            called_with["param_spec_name"] = param_spec_name
            called_with["preset_path"] = preset_path

        monkeypatch.setattr(surge_xt_interactive, "eval_patches", fake_eval_patches)

        surge_xt_interactive._maybe_eval_captured_patches(
            patch_file_path=train_path,
            output_dataset_dir_path=tmp_path,
            num_patches=3,
            checkpoint_path=ckpt_path,
            param_spec_name=SURGE_SIMPLE,
            preset_path="presets/surge-simple.vstpreset",
        )

        for sibling in ("test.h5", "val.h5", "predict.h5"):
            assert (tmp_path / sibling).read_bytes() == b"train-content"
        assert called_with == {
            "num_samples": 3,
            "dataset_root_dir": tmp_path,
            "checkpoint_path": ckpt_path,
            "param_spec_name": SURGE_SIMPLE,
            "preset_path": "presets/surge-simple.vstpreset",
        }

    def test_failed_copy_rolls_back_partial_siblings(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If a later ``shutil.copyfile`` fails, earlier siblings are removed and eval_patches is
        not invoked."""
        train_path = tmp_path / "train.h5"
        train_path.write_bytes(b"train-content")
        ckpt_path = tmp_path / "model.ckpt"
        ckpt_path.write_bytes(b"ckpt")

        original_copyfile = surge_xt_interactive.shutil.copyfile
        call_count = {"n": 0}

        def flaky_copyfile(src: Path, dst: Path) -> None:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise OSError("disk full")
            original_copyfile(src, dst)

        monkeypatch.setattr(surge_xt_interactive.shutil, "copyfile", flaky_copyfile)

        eval_called = {"n": 0}

        def fake_eval_patches(*_args, **_kwargs) -> None:
            eval_called["n"] += 1

        monkeypatch.setattr(surge_xt_interactive, "eval_patches", fake_eval_patches)

        with pytest.raises(OSError, match="disk full"):
            surge_xt_interactive._maybe_eval_captured_patches(
                patch_file_path=train_path,
                output_dataset_dir_path=tmp_path,
                num_patches=1,
                checkpoint_path=ckpt_path,
                param_spec_name=SURGE_SIMPLE,
                preset_path="presets/surge-base.vstpreset",
            )

        # First sibling was copied, second failed; rollback removes the first.
        assert not (tmp_path / "test.h5").exists()
        assert not (tmp_path / "val.h5").exists()
        assert not (tmp_path / "predict.h5").exists()
        assert eval_called["n"] == 0


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
        f.write(samples)


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


class TestRenderPredictedAudio:
    """``_render_predicted_audio`` runs ``predict_vst_audio.py`` and validates per-sample outputs.

    Each test mocks ``subprocess.run`` so no real VST subprocess executes; the post-subprocess
    walk over ``audio_dir`` is exercised against pre-created fixtures.
    """

    def test_happy_path_validates_per_sample_files(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Subprocess returns success and all per-sample artifacts are present and non-silent."""
        predictions_dir = tmp_path / "preds"
        predictions_dir.mkdir()
        audio_dir = tmp_path / "audio"
        num_samples = 2
        _populate_audio_dir(audio_dir, num_samples)

        def fake_run(_args: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(args=_args, returncode=0)

        monkeypatch.setattr(surge_xt_interactive.subprocess, "run", fake_run)

        surge_xt_interactive._render_predicted_audio(
            predictions_dir, audio_dir, num_samples, SURGE_SIMPLE, _RENDER_DEFAULT_PRESET
        )

    def test_subprocess_failure_raises_calledprocesserror(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-zero exit (``check=True``) re-raises ``CalledProcessError`` to the caller."""
        predictions_dir = tmp_path / "preds"
        predictions_dir.mkdir()
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
            raise subprocess.CalledProcessError(returncode=1, cmd=args)

        monkeypatch.setattr(surge_xt_interactive.subprocess, "run", fake_run)

        with pytest.raises(subprocess.CalledProcessError):
            surge_xt_interactive._render_predicted_audio(
                predictions_dir, audio_dir, 1, SURGE_SIMPLE, _RENDER_DEFAULT_PRESET
            )

    def test_subprocess_timeout_re_raises_timeoutexpired(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``TimeoutExpired`` from the subprocess is logged and re-raised."""
        predictions_dir = tmp_path / "preds"
        predictions_dir.mkdir()
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
            raise subprocess.TimeoutExpired(cmd=args, timeout=1.0)

        monkeypatch.setattr(surge_xt_interactive.subprocess, "run", fake_run)

        with pytest.raises(subprocess.TimeoutExpired):
            surge_xt_interactive._render_predicted_audio(
                predictions_dir, audio_dir, 1, SURGE_SIMPLE, _RENDER_DEFAULT_PRESET
            )

    def test_missing_per_sample_file_raises_filenotfounderror(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A missing per-sample artifact (``spec.png``) raises ``FileNotFoundError`` naming it."""
        predictions_dir = tmp_path / "preds"
        predictions_dir.mkdir()
        audio_dir = tmp_path / "audio"
        num_samples = 1
        _populate_audio_dir(audio_dir, num_samples)
        (audio_dir / "sample_0" / "spec.png").unlink()

        def fake_run(_args: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(args=_args, returncode=0)

        monkeypatch.setattr(surge_xt_interactive.subprocess, "run", fake_run)

        with pytest.raises(FileNotFoundError, match="spec.png"):
            surge_xt_interactive._render_predicted_audio(
                predictions_dir, audio_dir, num_samples, SURGE_SIMPLE, _RENDER_DEFAULT_PRESET
            )

    def test_silent_audio_raises_valueerror(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A silent rendered WAV (peak ``<= SILENCE_PEAK_THRESHOLD``) raises ``ValueError`` naming
        the offending sample / file."""
        predictions_dir = tmp_path / "preds"
        predictions_dir.mkdir()
        audio_dir = tmp_path / "audio"
        num_samples = 1
        _populate_audio_dir(audio_dir, num_samples, silent=True)

        def fake_run(_args: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(args=_args, returncode=0)

        monkeypatch.setattr(surge_xt_interactive.subprocess, "run", fake_run)

        with pytest.raises(ValueError, match=r"sample_0/(target|pred)\.wav is silent"):
            surge_xt_interactive._render_predicted_audio(
                predictions_dir, audio_dir, num_samples, SURGE_SIMPLE, _RENDER_DEFAULT_PRESET
            )

    def test_unexpected_sample_dirs_raises_filenotfounderror(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mismatched sample directory set (extra/missing) raises ``FileNotFoundError``."""
        predictions_dir = tmp_path / "preds"
        predictions_dir.mkdir()
        audio_dir = tmp_path / "audio"
        # Pre-create sample_0 only; the function expects sample_0 and sample_1 for num_samples=2.
        _populate_audio_dir(audio_dir, num_samples=1)

        def fake_run(_args: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(args=_args, returncode=0)

        monkeypatch.setattr(surge_xt_interactive.subprocess, "run", fake_run)

        with pytest.raises(FileNotFoundError, match="unexpected sample directories"):
            surge_xt_interactive._render_predicted_audio(
                predictions_dir,
                audio_dir,
                num_samples=2,
                param_spec_name=SURGE_SIMPLE,
                preset_path=_RENDER_DEFAULT_PRESET,
            )

    def test_linux_missing_wrapper_raises_filenotfounderror(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On Linux, a missing ``_VST_HEADLESS_WRAPPER`` raises ``FileNotFoundError`` before
        ``subprocess.run`` is ever invoked."""
        predictions_dir = tmp_path / "preds"
        predictions_dir.mkdir()
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        monkeypatch.setattr(surge_xt_interactive.sys, "platform", "linux")
        monkeypatch.setattr(
            surge_xt_interactive,
            "_VST_HEADLESS_WRAPPER",
            tmp_path / "definitely-does-not-exist.sh",
        )

        run_called = {"n": 0}

        def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
            run_called["n"] += 1
            return subprocess.CompletedProcess(args=[], returncode=0)

        monkeypatch.setattr(surge_xt_interactive.subprocess, "run", fake_run)

        with pytest.raises(FileNotFoundError, match="VST headless wrapper not found"):
            surge_xt_interactive._render_predicted_audio(
                predictions_dir, audio_dir, 1, SURGE_SIMPLE, _RENDER_DEFAULT_PRESET
            )
        assert run_called["n"] == 0

    def test_linux_prepends_wrapper_to_subprocess_args(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the wrapper exists on Linux, it is the first arg passed to ``subprocess.run``."""
        predictions_dir = tmp_path / "preds"
        predictions_dir.mkdir()
        audio_dir = tmp_path / "audio"
        num_samples = 1
        _populate_audio_dir(audio_dir, num_samples)

        wrapper_path = tmp_path / "wrapper.sh"
        wrapper_path.write_text('#!/usr/bin/env bash\nexec "$@"\n')
        wrapper_path.chmod(0o755)

        monkeypatch.setattr(surge_xt_interactive.sys, "platform", "linux")
        monkeypatch.setattr(surge_xt_interactive, "_VST_HEADLESS_WRAPPER", wrapper_path)

        captured: dict = {}

        def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
            captured["args"] = list(args)
            return subprocess.CompletedProcess(args=args, returncode=0)

        monkeypatch.setattr(surge_xt_interactive.subprocess, "run", fake_run)

        surge_xt_interactive._render_predicted_audio(
            predictions_dir, audio_dir, num_samples, SURGE_SIMPLE, _RENDER_DEFAULT_PRESET
        )

        assert captured["args"][0] == str(wrapper_path)

    def test_non_linux_does_not_prepend_wrapper(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On non-Linux platforms the wrapper is not prepended and its existence is not checked."""
        predictions_dir = tmp_path / "preds"
        predictions_dir.mkdir()
        audio_dir = tmp_path / "audio"
        num_samples = 1
        _populate_audio_dir(audio_dir, num_samples)

        # Wrapper does not exist; the precondition check must be skipped on darwin.
        monkeypatch.setattr(surge_xt_interactive.sys, "platform", "darwin")
        monkeypatch.setattr(
            surge_xt_interactive,
            "_VST_HEADLESS_WRAPPER",
            tmp_path / "definitely-does-not-exist.sh",
        )

        captured: dict = {}

        def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
            captured["args"] = list(args)
            return subprocess.CompletedProcess(args=args, returncode=0)

        monkeypatch.setattr(surge_xt_interactive.subprocess, "run", fake_run)

        surge_xt_interactive._render_predicted_audio(
            predictions_dir, audio_dir, num_samples, SURGE_SIMPLE, _RENDER_DEFAULT_PRESET
        )

        assert str(surge_xt_interactive._VST_HEADLESS_WRAPPER) not in captured["args"]
        # First arg should be the Python interpreter, not a wrapper.
        assert captured["args"][0] == surge_xt_interactive.sys.executable

    def test_param_spec_and_preset_path_are_forwarded_to_subprocess(
        self,
        surge_xt_interactive,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``param_spec_name`` and ``preset_path`` are passed verbatim to ``predict_vst_audio.py``
        (otherwise the script would default to ``surge_xt`` + ``presets/surge-base.vstpreset`` and
        decode/render against a mismatched spec)."""
        predictions_dir = tmp_path / "preds"
        predictions_dir.mkdir()
        audio_dir = tmp_path / "audio"
        num_samples = 1
        _populate_audio_dir(audio_dir, num_samples)

        monkeypatch.setattr(surge_xt_interactive.sys, "platform", "darwin")

        captured: dict = {}

        def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
            captured["args"] = list(args)
            return subprocess.CompletedProcess(args=args, returncode=0)

        monkeypatch.setattr(surge_xt_interactive.subprocess, "run", fake_run)

        surge_xt_interactive._render_predicted_audio(
            predictions_dir,
            audio_dir,
            num_samples,
            param_spec_name="custom-spec",
            preset_path="presets/custom.vstpreset",
        )

        args = captured["args"]
        assert "--param_spec" in args
        assert args[args.index("--param_spec") + 1] == "custom-spec"
        assert "--preset_path" in args
        assert args[args.index("--preset_path") + 1] == "presets/custom.vstpreset"
