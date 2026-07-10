"""Tests for ``synth-setter-predict-capture`` (sound-match bridge CLI, #1787).

The e2e tests drive real checkpoints of the real Lightning modules (tiny
hyperparameters, no mocks) through the real entrypoint and assert on the
produced ``pred-0.pt`` + ``params.csv`` bridge artifacts.
"""

import csv
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from lightning import LightningModule, Trainer
from pedalboard.io import AudioFile

from synth_setter.cli.predict_capture import (
    compute_capture_mel,
    decode_and_convert,
    detect_model_class,
    main,
    write_params_csv,
)
from synth_setter.data.audio_datamodule import AudioFolderDataset
from synth_setter.data.vst.clap_map import (
    ClapCsvRow,
    ClapParamRef,
    PluginFormatMap,
    load_clap_map,
)
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    ParamSpec,
)
from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    ASTWithProjectionHead,
    AudioSpectrogramTransformer,
    LearntProjection,
)
from synth_setter.models.surge_ff_module import VSTFeedForwardModule
from synth_setter.models.surge_flow_matching_module import VSTFlowMatchingModule
from synth_setter.resources import as_file, clap_map
from tests.data.vst._clap import SURGE_XT_MAPPED_PARAM_COUNT

# Encoded width of the surge_xt spec (297 synth + 3 note dims);
# test_pred_width_matches_registered_spec pins it to the live registry.
_SURGE_XT_PRED_WIDTH = 300


@pytest.fixture(scope="module")
def capture_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a bridge-contract capture: float32 stereo WAV, 4.0 s, non-44.1k host rate.

    :param tmp_path_factory: Pytest fixture providing a module-scoped directory.
    :returns: Path to the capture whose stem is the bridge uuid.
    """
    wav_dir = tmp_path_factory.mktemp("capture-sample-dir")
    path = wav_dir / "a3f9c2d4e5b60718.wav"
    sample_rate = 48000
    t = np.arange(int(4.0 * sample_rate)) / sample_rate
    tone = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    stereo = np.stack([tone, 0.5 * tone]).astype(np.float32)
    with AudioFile(str(path), "w", samplerate=float(sample_rate), num_channels=2) as f:
        f.write(stereo)
    return path


def _save_real_checkpoint(model: LightningModule, path: Path) -> Path:
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.strategy.connect(model)
    trainer.save_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def ff_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a real checkpoint of a tiny real feed-forward module.

    :param tmp_path_factory: Pytest fixture providing a module-scoped directory.
    :returns: Path to the saved Lightning checkpoint.
    """
    net = ASTWithProjectionHead(
        d_model=32,
        d_out=_SURGE_XT_PRED_WIDTH,
        n_heads=2,
        n_layers=1,
        patch_size=16,
        patch_stride=15,
        input_channels=2,
        # Upstream annotates spec_shape as tuple[int]; runtime accepts the pair.
        spec_shape=(128, 401),  # pyright: ignore[reportArgumentType]
    )
    model = VSTFeedForwardModule(
        net=net,
        # The modules take Hydra _partial_ optimizer factories despite the annotation.
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
    )
    return _save_real_checkpoint(model, tmp_path_factory.mktemp("ckpt") / "ff.ckpt")


@pytest.fixture(scope="module")
def simple_ff_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a real feed-forward checkpoint sized for the surge_simple spec.

    :param tmp_path_factory: Pytest fixture providing a module-scoped directory.
    :returns: Path to the saved Lightning checkpoint.
    """
    net = ASTWithProjectionHead(
        d_model=32,
        d_out=len(param_specs["surge_simple"]),
        n_heads=2,
        n_layers=1,
        patch_size=16,
        patch_stride=15,
        input_channels=2,
        # Upstream annotates spec_shape as tuple[int]; runtime accepts the pair.
        spec_shape=(128, 401),  # pyright: ignore[reportArgumentType]
    )
    model = VSTFeedForwardModule(
        net=net,
        # The modules take Hydra _partial_ optimizer factories despite the annotation.
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
    )
    return _save_real_checkpoint(model, tmp_path_factory.mktemp("ckpt") / "simple_ff.ckpt")


@pytest.fixture(scope="module")
def flow_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a real checkpoint of a tiny real flow-matching module.

    :param tmp_path_factory: Pytest fixture providing a module-scoped directory.
    :returns: Path to the saved Lightning checkpoint.
    """
    encoder = AudioSpectrogramTransformer(
        d_model=32,
        n_heads=2,
        n_layers=1,
        n_conditioning_outputs=2,
        patch_size=16,
        patch_stride=15,
        input_channels=2,
        # Upstream annotates spec_shape as tuple[int]; runtime accepts the pair.
        spec_shape=(128, 401),  # pyright: ignore[reportArgumentType]
    )
    vector_field = ApproxEquivTransformer(
        projection=LearntProjection(
            d_model=32,
            d_token=32,
            num_params=_SURGE_XT_PRED_WIDTH,
            num_tokens=8,
            initial_ffn=True,
            final_ffn=False,
        ),
        num_layers=1,
        d_model=32,
        conditioning_dim=32,
        num_heads=2,
        d_ff=32,
        num_tokens=8,
        learn_projection=True,
        time_encoding="sinusoidal",
        zero_init=False,
    )
    model = VSTFlowMatchingModule(
        encoder=encoder,
        vector_field=vector_field,
        # The modules take Hydra _partial_ optimizer factories despite the annotation.
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        num_params=_SURGE_XT_PRED_WIDTH,
        test_sample_steps=2,
        test_cfg_strength=1.0,
    )
    return _save_real_checkpoint(model, tmp_path_factory.mktemp("ckpt") / "flow.ckpt")


def _tiny_spec() -> ParamSpec:
    return ParamSpec(
        [
            ContinuousParameter(name="cutoff"),
            CategoricalParameter(
                name="mode",
                values=["Digital", "Analog"],
                raw_values=[0.25, 0.75],
                encoding="onehot",
            ),
        ],
        [
            DiscreteLiteralParameter(name="pitch", min=21, max=108),
            NoteDurationParameter(name="note_start_and_end", max_note_duration_seconds=4.0),
        ],
    )


def _tiny_map() -> PluginFormatMap:
    return PluginFormatMap(
        plugin="Surge XT",
        version="1.3.4",
        params={
            "cutoff": ClapParamRef(
                clap_param_id=42,
                clap_name="Cutoff",
                clap_module_name="/Filter/",
                min_value=-10.0,
                max_value=10.0,
                is_stepped=False,
            ),
            "mode": ClapParamRef(
                clap_param_id=7,
                clap_name="Mode",
                clap_module_name="/Global/",
                min_value=0.0,
                max_value=1.0,
                is_stepped=True,
            ),
        },
    )


class TestDetectModelClass:
    """State-dict-based model-class detection backing the --model-class default."""

    # slow: the checkpoint fixtures spin up real Lightning Trainers.
    @pytest.mark.slow
    def test_ff_checkpoint_detects_ff(self, ff_checkpoint: Path):
        """A real feed-forward checkpoint is recognized by its net.* prefix.

        :param ff_checkpoint: Real feed-forward checkpoint.
        """
        assert detect_model_class(ff_checkpoint) == "ff"

    @pytest.mark.slow
    def test_flow_checkpoint_detects_flow(self, flow_checkpoint: Path):
        """A real flow checkpoint is recognized by its encoder.*/vector_field.* prefixes.

        :param flow_checkpoint: Real flow-matching checkpoint.
        """
        assert detect_model_class(flow_checkpoint) == "flow"

    def test_unrecognized_state_dict_raises_naming_the_prefixes(self, tmp_path: Path):
        """A checkpoint matching neither layout errors with the found prefixes.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        alien = tmp_path / "alien.ckpt"
        torch.save({"state_dict": {"decoder.weight": torch.zeros(1)}}, alien)

        with pytest.raises(ValueError, match="decoder"):
            detect_model_class(alien)


class TestPackagedMapResolution:
    """Per-spec packaged map lookup backing the CLI's --map default."""

    @pytest.mark.parametrize("spec_name", ["surge_xt", "surge_simple", "surge_4"])
    def test_every_mapped_spec_resolves_to_a_loadable_map(self, spec_name: str):
        """Each packaged map loads and covers exactly its spec's synth params.

        :param spec_name: Registry key under test.
        """
        with as_file(clap_map(spec_name)) as path:
            format_map = load_clap_map(path)

        assert set(format_map.params) == {p.name for p in param_specs[spec_name].synth_params}

    def test_unmapped_spec_raises_file_not_found(self):
        """A spec without a packaged map is a hard error, not a silent fallback."""
        with pytest.raises(FileNotFoundError, match="obxf"):
            clap_map("obxf")


class TestDecodeAndConvert:
    """Decode+convert unit tests over a hand-written spec and map."""

    def test_fixture_tensor_decodes_to_exact_native_rows(self):
        """A fixture tensor decodes to exact native rows (continuous lerp + stepped index)."""
        # Widths: cutoff 1, mode onehot 2, pitch 1, note duration 2 -> 6.
        # Model range [-1, 1]: 0.0 rescales to 0.5; mode logits pick index 1.
        prediction = torch.tensor([[0.0, -1.0, 1.0, 0.2, 0.2, 0.2]])

        rows = decode_and_convert(prediction, _tiny_spec(), _tiny_map())

        assert [(r.pb_name, r.clap_param_id, r.clap_value) for r in rows] == [
            ("cutoff", 42, pytest.approx(0.0)),
            ("mode", 7, 1.0),
        ]

    def test_note_params_are_discarded(self):
        """Only synth params appear in the converted rows."""
        prediction = torch.tensor([[0.0, 1.0, -1.0, 0.9, 0.9, 0.9]])

        rows = decode_and_convert(prediction, _tiny_spec(), _tiny_map())

        assert {r.pb_name for r in rows} == {"cutoff", "mode"}

    def test_out_of_range_prediction_values_clip_to_native_bounds(self):
        """Predictions outside [-1, 1] clip to both native bounds instead of overshooting."""
        high = decode_and_convert(
            torch.tensor([[7.5, 1.0, -1.0, 0.0, 0.0, 0.0]]), _tiny_spec(), _tiny_map()
        )
        low = decode_and_convert(
            torch.tensor([[-7.5, 1.0, -1.0, 0.0, 0.0, 0.0]]), _tiny_spec(), _tiny_map()
        )

        assert high[0].clap_value == pytest.approx(10.0)
        assert low[0].clap_value == pytest.approx(-10.0)

    def test_decoded_param_missing_from_map_raises_value_error(self):
        """A decoded param absent from the map is a hard error, not a silent skip."""
        lonely_map = PluginFormatMap(
            plugin="Surge XT", version="1.3.4", params={"cutoff": _tiny_map().params["cutoff"]}
        )
        prediction = torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="mode"):
            decode_and_convert(prediction, _tiny_spec(), lonely_map)


class TestWriteParamsCsv:
    """CSV schema and atomicity tests."""

    def test_rows_serialize_to_exact_schema_text(self, tmp_path: Path):
        """Rows serialize to the exact cross-repo CSV schema.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        rows = decode_and_convert(
            torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]]), _tiny_spec(), _tiny_map()
        )
        dest = tmp_path / "params.csv"

        write_params_csv(rows, dest)

        assert dest.read_text() == (
            "pb_name,clap_name,clap_module_name,clap_param_id,clap_value\n"
            "cutoff,Cutoff,/Filter/,42,0\n"
            "mode,Mode,/Global/,7,1\n"
        )

    def test_fractional_values_serialize_float32_faithfully(self, tmp_path: Path):
        """The %.9g format pins float32 fidelity without trailing noise digits.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        third = ClapCsvRow(
            pb_name="cutoff",
            clap_name="Cutoff",
            clap_module_name="/Filter/",
            clap_param_id=42,
            clap_value=float(np.float32(1.0 / 3.0)),
        )
        dest = tmp_path / "params.csv"

        write_params_csv([third], dest)

        assert dest.read_text().splitlines()[1] == "cutoff,Cutoff,/Filter/,42,0.333333343"

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_value_raises_instead_of_writing_it(self, bad: float, tmp_path: Path):
        """Any non-finite clap_value is a hard error; params.csv is never written.

        Rows are built directly: the decode path clips ±Inf into range, so only
        the writer's own guard covers rows from other producers.

        :param bad: The non-finite value under test.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        row = ClapCsvRow(
            pb_name="cutoff",
            clap_name="Cutoff",
            clap_module_name="/Filter/",
            clap_param_id=42,
            clap_value=bad,
        )
        dest = tmp_path / "params.csv"

        with pytest.raises(ValueError, match="cutoff"):
            write_params_csv([row], dest)
        assert not dest.exists()
        assert not (tmp_path / "params.csv.tmp").exists()

    def test_pred_width_matches_registered_spec(self):
        """The e2e fixture width tracks the live surge_xt registry entry."""
        assert len(param_specs["surge_xt"]) == _SURGE_XT_PRED_WIDTH

    def test_replace_failure_cleans_up_the_tmp_file(self, tmp_path: Path):
        """A failure at the final rename removes the staged .tmp before re-raising.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        dest = tmp_path / "params.csv"
        dest.mkdir()

        with pytest.raises(OSError):
            write_params_csv([], dest)
        assert not (tmp_path / "params.csv.tmp").exists()

    def test_no_tmp_file_survives_a_successful_write(self, tmp_path: Path):
        """The .tmp staging file is renamed away on success.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        dest = tmp_path / "params.csv"

        write_params_csv([], dest)

        assert dest.exists()
        assert not (tmp_path / "params.csv.tmp").exists()


class TestComputeCaptureMel:
    """Preprocessing parity with the training data path."""

    def test_silent_capture_yields_finite_mel(self, tmp_path: Path):
        """An all-zero capture (silence) still produces a finite mel.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        path = tmp_path / "0000000000000000.wav"
        silence = np.zeros((2, 4 * 48000), dtype=np.float32)
        with AudioFile(str(path), "w", samplerate=48000.0, num_channels=2) as f:
            f.write(silence)

        mel = compute_capture_mel(path)

        assert torch.isfinite(mel).all()

    def test_stats_file_applies_training_normalization(self, capture_wav: Path, tmp_path: Path):
        """With --stats-file semantics, the mel is z-scored exactly like training.

        :param capture_wav: Fixture capture WAV.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        stats_path = tmp_path / "stats.npz"
        np.savez(stats_path, mean=np.float64(-40.0), std=np.float64(20.0))
        raw = compute_capture_mel(capture_wav)

        normalized = compute_capture_mel(capture_wav, stats_file=stats_path)

        assert torch.allclose(normalized, (raw - (-40.0)) / 20.0)

    def test_explicit_files_bypass_the_folder_glob(self, capture_wav: Path, tmp_path: Path):
        """An explicit file list yields only that capture, even from a foreign root.

        :param capture_wav: Fixture capture WAV.
        :param tmp_path: Pytest fixture providing a fresh (empty) directory.
        """
        dataset = AudioFolderDataset(root=str(tmp_path), files=[capture_wav])

        assert dataset.files == [capture_wav]
        assert dataset[0]["mel_spec"].shape == (2, 128, 401)

    def test_mel_matches_audio_folder_dataset_for_same_file(self, capture_wav: Path):
        """The CLI mel equals AudioFolderDataset's mel for the same file.

        :param capture_wav: Fixture capture WAV.
        """
        expected = AudioFolderDataset(root=str(capture_wav.parent))
        expected_item = expected[expected.files.index(capture_wav)]

        mel = compute_capture_mel(capture_wav)

        assert mel.shape == (2, 128, 401)
        assert torch.equal(mel, expected_item["mel_spec"])


@pytest.mark.slow
class TestPredictCaptureEndToEnd:
    """End-to-end runs with real checkpoints of the real Lightning modules."""

    @pytest.fixture(autouse=True)
    def _run_from_tmp(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep the CLI's repo-relative defaults (--log-dir) out of the repo tree.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture for scoped environment changes.
        """
        monkeypatch.chdir(tmp_path)

    def test_contract_invocation_ff_writes_bridge_artifacts(
        self, capture_wav: Path, ff_checkpoint: Path, tmp_path: Path
    ):
        """The exact cross-repo invocation (python -m, real subprocess) produces the artifacts.

        No ``--model-class`` is passed — the C++ side sends only the wav and
        the prediction dir (#1787), so this also exercises autodetection.

        :param capture_wav: Fixture capture WAV.
        :param ff_checkpoint: Real feed-forward checkpoint.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        prediction_dir = tmp_path / "param-prediction-dir"

        result = subprocess.run(  # noqa: S603 — argv is a fixed list of test-owned paths
            [
                sys.executable,
                "-m",
                "synth_setter.cli.predict_capture",
                str(capture_wav),
                "--prediction-dir",
                str(prediction_dir),
                "--checkpoint",
                str(ff_checkpoint),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, result.stderr
        uuid_dir = prediction_dir / "a3f9c2d4e5b60718"
        prediction = torch.load(uuid_dir / "pred-0.pt", map_location="cpu", weights_only=True)
        assert prediction.shape == (1, _SURGE_XT_PRED_WIDTH)
        assert not (uuid_dir / "params.csv.tmp").exists()
        _assert_valid_bridge_csv(uuid_dir / "params.csv")

    def test_contract_invocation_flow_writes_bridge_artifacts(
        self, capture_wav: Path, flow_checkpoint: Path, tmp_path: Path
    ):
        """The contract invocation also holds for a flow checkpoint (real subprocess).

        :param capture_wav: Fixture capture WAV.
        :param flow_checkpoint: Real flow-matching checkpoint.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        prediction_dir = tmp_path / "param-prediction-dir"

        result = subprocess.run(  # noqa: S603 — argv is a fixed list of test-owned paths
            [
                sys.executable,
                "-m",
                "synth_setter.cli.predict_capture",
                str(capture_wav),
                "--prediction-dir",
                str(prediction_dir),
                "--checkpoint",
                str(flow_checkpoint),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, result.stderr
        assert "detected model class flow" in result.stderr
        _assert_valid_bridge_csv(prediction_dir / "a3f9c2d4e5b60718" / "params.csv")

    def test_console_script_is_installed_and_callable(self):
        """The synth-setter-predict-capture console script resolves and prints usage."""
        script = Path(sys.executable).parent / "synth-setter-predict-capture"

        result = subprocess.run(  # noqa: S603 — argv is a fixed list of test-owned paths
            [str(script), "--help"], capture_output=True, text=True, timeout=60
        )

        assert result.returncode == 0, result.stderr
        assert "Predict Surge params" in result.stdout

    def test_mono_low_rate_capture_is_resampled_and_upmixed(
        self, ff_checkpoint: Path, tmp_path: Path
    ):
        """A mono 22 050 Hz capture (host rate not guaranteed by #1787) still converts.

        :param ff_checkpoint: Real feed-forward checkpoint.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        wav_dir = tmp_path / "capture-sample-dir"
        wav_dir.mkdir()
        wav_path = wav_dir / "feedbeefcafe0123.wav"
        sample_rate = 22050
        t = np.arange(int(4.0 * sample_rate)) / sample_rate
        mono = (0.4 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)[np.newaxis, :]
        with AudioFile(str(wav_path), "w", samplerate=float(sample_rate), num_channels=1) as f:
            f.write(mono)
        prediction_dir = tmp_path / "param-prediction-dir"

        result = CliRunner().invoke(
            main,
            [
                str(wav_path),
                "--prediction-dir",
                str(prediction_dir),
                "--checkpoint",
                str(ff_checkpoint),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        _assert_valid_bridge_csv(prediction_dir / "feedbeefcafe0123" / "params.csv")

    def test_default_model_class_flow_writes_bridge_artifacts(
        self, capture_wav: Path, flow_checkpoint: Path, tmp_path: Path
    ):
        """The default --model-class flow path produces the artifacts.

        :param capture_wav: Fixture capture WAV.
        :param flow_checkpoint: Real flow-matching checkpoint.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        prediction_dir = tmp_path / "param-prediction-dir"

        result = CliRunner().invoke(
            main,
            [
                str(capture_wav),
                "--prediction-dir",
                str(prediction_dir),
                "--checkpoint",
                str(flow_checkpoint),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        uuid_dir = prediction_dir / "a3f9c2d4e5b60718"
        prediction = torch.load(uuid_dir / "pred-0.pt", map_location="cpu", weights_only=True)
        assert prediction.shape == (1, _SURGE_XT_PRED_WIDTH)
        _assert_valid_bridge_csv(uuid_dir / "params.csv")

    def test_spec_name_resolves_the_matching_packaged_map(
        self, capture_wav: Path, simple_ff_checkpoint: Path, tmp_path: Path
    ):
        """--param-spec-name alone selects that spec's packaged map end-to-end.

        :param capture_wav: Fixture capture WAV.
        :param simple_ff_checkpoint: Real feed-forward checkpoint sized for surge_simple.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        prediction_dir = tmp_path / "param-prediction-dir"

        result = CliRunner().invoke(
            main,
            [
                str(capture_wav),
                "--prediction-dir",
                str(prediction_dir),
                "--checkpoint",
                str(simple_ff_checkpoint),
                "--model-class",
                "ff",
                "--param-spec-name",
                "surge_simple",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        csv_path = prediction_dir / "a3f9c2d4e5b60718" / "params.csv"
        rows = list(csv.DictReader(csv_path.read_text().splitlines()))
        assert len(rows) == len(param_specs["surge_simple"].synth_params)

    def test_successful_run_writes_milestones_to_the_uuid_log(
        self, capture_wav: Path, ff_checkpoint: Path, tmp_path: Path
    ):
        """A run appends its milestones to <log-dir>/<uuid>.log.

        :param capture_wav: Fixture capture WAV.
        :param ff_checkpoint: Real feed-forward checkpoint.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        prediction_dir = tmp_path / "param-prediction-dir"
        log_dir = tmp_path / "bridge-logs"

        result = CliRunner().invoke(
            main,
            [
                str(capture_wav),
                "--prediction-dir",
                str(prediction_dir),
                "--checkpoint",
                str(ff_checkpoint),
                "--log-dir",
                str(log_dir),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        log_text = (log_dir / "a3f9c2d4e5b60718.log").read_text()
        assert "detected model class ff" in log_text
        assert "wrote" in log_text and "params.csv" in log_text

    def test_failed_run_logs_the_traceback_before_nonzero_exit(
        self, capture_wav: Path, ff_checkpoint: Path, tmp_path: Path
    ):
        """Any exception lands in the uuid log with a traceback; exit stays nonzero.

        :param capture_wav: Fixture capture WAV.
        :param ff_checkpoint: Real feed-forward checkpoint.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        prediction_dir = tmp_path / "param-prediction-dir"
        log_dir = tmp_path / "bridge-logs"
        crippled_map = tmp_path / "crippled_map.json"
        crippled_map.write_text(_tiny_map().model_dump_json())

        result = CliRunner().invoke(
            main,
            [
                str(capture_wav),
                "--prediction-dir",
                str(prediction_dir),
                "--checkpoint",
                str(ff_checkpoint),
                "--map",
                str(crippled_map),
                "--log-dir",
                str(log_dir),
            ],
        )

        assert result.exit_code != 0
        log_text = (log_dir / "a3f9c2d4e5b60718.log").read_text()
        assert "Traceback" in log_text
        assert "ValueError" in log_text

    def test_repeated_runs_append_without_duplicate_lines(
        self, capture_wav: Path, ff_checkpoint: Path, tmp_path: Path
    ):
        """Two in-process runs append to one log; no handler leaks double lines.

        :param capture_wav: Fixture capture WAV.
        :param ff_checkpoint: Real feed-forward checkpoint.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        log_dir = tmp_path / "bridge-logs"
        args = [
            str(capture_wav),
            "--prediction-dir",
            str(tmp_path / "param-prediction-dir"),
            "--checkpoint",
            str(ff_checkpoint),
            "--log-dir",
            str(log_dir),
        ]

        first = CliRunner().invoke(main, args, catch_exceptions=False)
        second = CliRunner().invoke(main, args, catch_exceptions=False)

        assert first.exit_code == 0 and second.exit_code == 0
        lines = (log_dir / "a3f9c2d4e5b60718.log").read_text().splitlines()
        detections = [line for line in lines if "detected model class ff" in line]
        assert len(detections) == 2

    def test_identical_inputs_produce_byte_identical_outputs(
        self, capture_wav: Path, ff_checkpoint: Path, tmp_path: Path
    ):
        """Two runs over the same capture and checkpoint emit identical bridge files.

        :param capture_wav: Fixture capture WAV.
        :param ff_checkpoint: Real feed-forward checkpoint (deterministic forward).
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        outputs = []
        for run_dir in ("run-a", "run-b"):
            result = CliRunner().invoke(
                main,
                [
                    str(capture_wav),
                    "--prediction-dir",
                    str(tmp_path / run_dir),
                    "--checkpoint",
                    str(ff_checkpoint),
                    "--model-class",
                    "ff",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            outputs.append((tmp_path / run_dir / "a3f9c2d4e5b60718" / "params.csv").read_bytes())

        assert outputs[0] == outputs[1]

    def test_retry_failure_removes_the_previous_runs_params_csv(
        self, capture_wav: Path, ff_checkpoint: Path, tmp_path: Path
    ):
        """A failed re-run of the same uuid never leaves the prior success visible.

        :param capture_wav: Fixture capture WAV.
        :param ff_checkpoint: Real feed-forward checkpoint.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        prediction_dir = tmp_path / "param-prediction-dir"
        ok = CliRunner().invoke(
            main,
            [
                str(capture_wav),
                "--prediction-dir",
                str(prediction_dir),
                "--checkpoint",
                str(ff_checkpoint),
                "--model-class",
                "ff",
            ],
            catch_exceptions=False,
        )
        params_csv = prediction_dir / "a3f9c2d4e5b60718" / "params.csv"
        assert ok.exit_code == 0 and params_csv.exists()

        crippled_map = tmp_path / "crippled_map.json"
        crippled_map.write_text(_tiny_map().model_dump_json())
        retry = CliRunner().invoke(
            main,
            [
                str(capture_wav),
                "--prediction-dir",
                str(prediction_dir),
                "--checkpoint",
                str(ff_checkpoint),
                "--model-class",
                "ff",
                "--map",
                str(crippled_map),
            ],
        )

        assert retry.exit_code != 0
        assert not params_csv.exists()

    def test_conversion_failure_exits_nonzero_and_writes_no_params_csv(
        self, capture_wav: Path, ff_checkpoint: Path, tmp_path: Path
    ):
        """Absence of params.csv is the failure signal when conversion fails.

        :param capture_wav: Fixture capture WAV.
        :param ff_checkpoint: Real feed-forward checkpoint.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        prediction_dir = tmp_path / "param-prediction-dir"
        crippled_map = tmp_path / "crippled_map.json"
        crippled_map.write_text(_tiny_map().model_dump_json())

        result = CliRunner().invoke(
            main,
            [
                str(capture_wav),
                "--prediction-dir",
                str(prediction_dir),
                "--checkpoint",
                str(ff_checkpoint),
                "--model-class",
                "ff",
                "--map",
                str(crippled_map),
            ],
        )

        assert result.exit_code != 0
        assert not (prediction_dir / "a3f9c2d4e5b60718" / "params.csv").exists()


def _assert_valid_bridge_csv(csv_path: Path) -> None:
    """Assert the produced ``params.csv`` honors the cross-repo contract.

    :param csv_path: The ``params.csv`` a CLI run produced.
    """
    with as_file(clap_map("surge_xt")) as map_path:
        committed_map = load_clap_map(map_path)

    text = csv_path.read_text()
    rows = list(csv.DictReader(text.splitlines()))

    assert text.splitlines()[0] == "pb_name,clap_name,clap_module_name,clap_param_id,clap_value"
    assert len(rows) == len(committed_map.params) == SURGE_XT_MAPPED_PARAM_COUNT
    for row in rows:
        ref = committed_map.params[row["pb_name"]]
        assert row["clap_name"] == ref.clap_name
        assert row["clap_param_id"] == str(ref.clap_param_id)
        assert ref.min_value <= float(row["clap_value"]) <= ref.max_value
