"""Behavior tests for the Surge sketch-render CLI."""

import csv
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from pedalboard.io import AudioFile

from synth_setter.cli._cfg_strength import CfgStrengths
from synth_setter.cli.clap import (
    PreparedAudioInputs,
    RenderedPatch,
    _fit_audio_to_model_grid,
    _load_model_audio,
    _predict_patch,
    _render_patch,
    _run_request,
    _run_under_headless_wrapper,
    _validate_stats,
    main,
    prepare_audio_inputs,
    upload_output_artifacts,
    validate_checkpoint_compatibility,
    write_output_artifacts,
    write_run_manifest,
)
from synth_setter.data.vst.shapes import make_spectrogram
from synth_setter.features.sketch_controls import NUM_SKETCH_CONTROLS, SKETCH_PITCH_SLICE
from synth_setter.resources import surge_simple_preset
from synth_setter.synth_spec import SYNTHS, SynthName


def test_help_exposes_audio_and_cfg_options() -> None:
    """Help names the audio inputs and independent guidance overrides."""
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--guide_audio FILE" in result.output
    assert "--ref_audio FILE" in result.output
    assert "--content-cfg-strength FLOAT" in result.output
    assert "--sketch-cfg-strength FLOAT" in result.output
    assert "Guide audio supplying sketch controls." in result.output
    assert "Reference audio supplying mel/timbre" in result.output
    assert "conditioning." in result.output
    assert ":param" not in result.output


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--content-cfg-strength", "-1"),
        ("--content-cfg-strength", "nan"),
        ("--content-cfg-strength", "inf"),
        ("--sketch-cfg-strength", "-1"),
        ("--sketch-cfg-strength", "nan"),
        ("--sketch-cfg-strength", "inf"),
    ],
)
def test_cli_cfg_strength_invalid_value_rejected(option: str, value: str, tmp_path: Path) -> None:
    """Guidance overrides reject negative and non-finite values.

    :param option: Guidance option under test.
    :param value: Invalid CLI value.
    :param tmp_path: Holds validated audio path fixtures.
    """
    guide = tmp_path / "guide.wav"
    reference = tmp_path / "reference.wav"
    guide.write_bytes(b"validated path")
    reference.write_bytes(b"validated path")

    result = CliRunner().invoke(
        main,
        ["--guide_audio", str(guide), "--ref_audio", str(reference), option, value],
    )

    assert result.exit_code != 0
    assert "finite and greater than or equal to zero" in result.output


def test_cli_cfg_strength_zero_forwarded_to_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Zero remains a valid explicit override for both guidance branches.

    :param monkeypatch: Captures the request boundary.
    :param tmp_path: Holds validated audio path fixtures.
    """
    guide = tmp_path / "guide.wav"
    reference = tmp_path / "reference.wav"
    guide.write_bytes(b"validated path")
    reference.write_bytes(b"validated path")
    received: list[CfgStrengths[float | None]] = []

    def capture_request(
        _guide: Path,
        _reference: Path,
        strengths: CfgStrengths[float | None],
    ) -> tuple[Path, str]:
        received.append(strengths)
        return tmp_path / "output", "r2://result"

    monkeypatch.setenv("SYNTH_SETTER_CLAP_HEADLESS", "1")
    monkeypatch.setattr("synth_setter.cli.clap._run_request", capture_request)

    result = CliRunner().invoke(
        main,
        [
            "--guide_audio",
            str(guide),
            "--ref_audio",
            str(reference),
            "--content-cfg-strength",
            "0",
            "--sketch-cfg-strength",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert received == [CfgStrengths(content=0.0, sketch=0.0)]


def test_installed_console_script_exposes_public_help() -> None:
    """The installed synth-setter-clap command resolves to the Click entrypoint."""
    script = Path(sys.executable).parent / "synth-setter-clap"

    result = subprocess.run(  # noqa: S603 — fixed venv console entrypoint
        [str(script), "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "--guide_audio FILE" in result.stdout
    assert "--ref_audio FILE" in result.stdout


@pytest.mark.parametrize(("duration_seconds", "expected_last_sample"), [(2, 0.0), (5, 0.25)])
def test_load_model_audio_short_and_long_inputs_fit_four_second_grid(
    duration_seconds: int, expected_last_sample: float, tmp_path: Path
) -> None:
    """Pad short clips and trim long clips without shifting their onset.

    :param duration_seconds: Source duration exercising one grid boundary.
    :param expected_last_sample: Value after padding or truncation.
    :param tmp_path: Holds the source WAV.
    """
    sample_rate = 44100
    source = np.full((1, duration_seconds * sample_rate), 0.25, dtype=np.float32)
    source_path = tmp_path / "source.wav"
    with AudioFile(str(source_path), "w", sample_rate, 1) as audio_file:
        audio_file.write(source)

    prepared = _load_model_audio(source_path)

    assert prepared.shape == (2, 176400)
    assert torch.all(prepared[:, 0] > 0.2)
    assert prepared[0, -1].item() == pytest.approx(expected_last_sample, abs=1e-5)


@pytest.mark.parametrize("invalid_sample", [float("nan"), float("inf"), 1.01, -1.01])
def test_fit_audio_to_model_grid_invalid_sample_raises(invalid_sample: float) -> None:
    """Reject non-finite or out-of-range input before feature extraction.

    :param invalid_sample: Value outside the model waveform contract.
    """
    audio = np.zeros((1, 176400), dtype=np.float32)
    audio[0, 0] = invalid_sample

    with pytest.raises(ValueError, match=r"finite.*\[-1, 1\]"):
        _fit_audio_to_model_grid(audio)


def test_prepare_audio_inputs_normalizes_real_wavs_and_extracts_features(tmp_path: Path) -> None:
    """Real audio reaches the training mel path and guide-control extractor.

    :param tmp_path: Holds the source WAVs and mel statistics.
    """
    sample_rate = 48000
    time = np.arange(4 * sample_rate, dtype=np.float32) / sample_rate
    guide_mono = (0.8 * np.sin(2 * np.pi * 220.0 * time))[None]
    ref_stereo = np.stack(
        [0.4 * np.sin(2 * np.pi * 330.0 * time), 0.2 * np.sin(2 * np.pi * 330.0 * time)]
    ).astype(np.float32)
    guide_path = tmp_path / "guide-source.wav"
    ref_path = tmp_path / "ref-source.wav"
    with AudioFile(str(guide_path), "w", sample_rate, 1) as audio_file:
        audio_file.write(guide_mono)
    with AudioFile(str(ref_path), "w", sample_rate, 2) as audio_file:
        audio_file.write(ref_stereo)
    stats_path = tmp_path / "stats.npz"
    np.savez(stats_path, mean=np.float32(-40.0), std=np.float32(20.0))

    prepared = prepare_audio_inputs(guide_path, ref_path, stats_path)
    raw_ref_mel = torch.from_numpy(make_spectrogram(prepared.ref_audio.numpy(), 44100.0))

    assert prepared.guide_audio.shape == (2, 176400)
    assert prepared.ref_audio.shape == (2, 176400)
    assert prepared.ref_mel.shape == (2, 128, 401)
    assert prepared.sketch_controls.shape == (NUM_SKETCH_CONTROLS, 401)
    assert torch.any(prepared.guide_audio[:, 1:2205] != 0)
    assert prepared.guide_audio.abs().max().item() == pytest.approx(0.8, abs=1e-4)
    assert torch.equal(prepared.guide_audio[0], prepared.guide_audio[1])
    assert torch.isfinite(prepared.ref_mel).all()
    assert torch.allclose(prepared.ref_mel, (raw_ref_mel + 40.0) / 20.0)
    assert torch.isfinite(prepared.sketch_controls).all()
    nonzero_pitch = prepared.sketch_controls[SKETCH_PITCH_SLICE]
    assert torch.all(nonzero_pitch[nonzero_pitch > 0] >= 0.1)


def test_write_output_artifacts_wrong_render_shape_raises(tmp_path: Path) -> None:
    """Reject a finite render that does not match the published duration.

    :param tmp_path: Holds the rejected output directory.
    """
    prepared = PreparedAudioInputs(
        guide_audio=torch.zeros(2, 176400),
        ref_audio=torch.zeros(2, 176400),
        ref_mel=torch.zeros(2, 128, 401),
        sketch_controls=torch.zeros(NUM_SKETCH_CONTROLS, 401),
    )

    with pytest.raises(ValueError, match=r"shape.*\(2, 176400\)"):
        write_output_artifacts(
            tmp_path / "output",
            prepared,
            RenderedPatch(
                audio=np.zeros((2, 88200), dtype=np.float32),
                synth_params={"filter_1_cutoff": 0.5},
                note_params={"pitch": 60, "note_start_and_end": (0.05, 1.95)},
                effective_note_window=(0.05, 1.95),
            ),
        )


@pytest.mark.parametrize("invalid_sample", [float("nan"), float("inf"), 1.01, -1.01])
def test_write_output_artifacts_invalid_rendered_audio_raises(
    invalid_sample: float, tmp_path: Path
) -> None:
    """Reject non-finite or out-of-range renders before WAV serialization.

    :param invalid_sample: Value outside the publishable waveform contract.
    :param tmp_path: Holds the rejected output directory.
    """
    rendered = np.zeros((2, 176400), dtype=np.float32)
    rendered[0, 0] = invalid_sample
    prepared = PreparedAudioInputs(
        guide_audio=torch.zeros(2, 176400),
        ref_audio=torch.zeros(2, 176400),
        ref_mel=torch.zeros(2, 128, 401),
        sketch_controls=torch.zeros(NUM_SKETCH_CONTROLS, 401),
    )

    with pytest.raises(ValueError, match=r"finite.*\[-1, 1\]"):
        write_output_artifacts(
            tmp_path / "output",
            prepared,
            RenderedPatch(
                audio=rendered,
                synth_params={"filter_1_cutoff": 0.5},
                note_params={"pitch": 60, "note_start_and_end": (0.05, 3.95)},
                effective_note_window=(0.05, 3.95),
            ),
        )


def test_output_artifacts_round_trip_through_real_rclone(
    fake_r2_remote: Path, tmp_path: Path
) -> None:
    """Normalized audio and patch params remain local and land at the requested prefix.

    :param fake_r2_remote: Local filesystem backing the real rclone transport.
    :param tmp_path: Holds the retained local output directory.
    """
    output_dir = tmp_path / "local-output"
    prepared = PreparedAudioInputs(
        guide_audio=torch.full((2, 176400), 0.125),
        ref_audio=torch.full((2, 176400), -0.25),
        ref_mel=torch.zeros(2, 128, 401),
        sketch_controls=torch.zeros(NUM_SKETCH_CONTROLS, 401),
    )
    pred_audio = np.full((2, 176400), 0.0625, dtype=np.float32)

    write_output_artifacts(
        output_dir,
        prepared,
        RenderedPatch(
            audio=pred_audio,
            synth_params={"filter_1_cutoff": 0.5},
            note_params={"pitch": 60, "note_start_and_end": (3.95, 0.05)},
            effective_note_window=(0.05, 3.95),
        ),
    )
    destination = "r2://intermediate-data/eval/synth-setter-clap/test-run"
    write_run_manifest(output_dir, destination, CfgStrengths(content=2.0, sketch=3.0))
    upload_output_artifacts(output_dir, destination)

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "guide.wav",
        "manifest.json",
        "params.csv",
        "pred.wav",
        "ref.wav",
    ]
    remote = fake_r2_remote / "intermediate-data" / "eval" / "synth-setter-clap" / "test-run"
    assert sorted(path.name for path in remote.iterdir()) == [
        "guide.wav",
        "manifest.json",
        "params.csv",
        "pred.wav",
        "ref.wav",
    ]
    with AudioFile(str(remote / "pred.wav"), "r") as audio_file:
        round_tripped = audio_file.read(audio_file.frames)
    assert round_tripped.shape == (2, 176400)
    assert np.isfinite(round_tripped).all()
    assert np.abs(round_tripped).max() > 0.05
    with (remote / "params.csv").open(newline="", encoding="utf-8") as stream:
        rows = {row[""]: row for row in csv.DictReader(stream)}
    assert rows["note_start_and_end"]["pred"] == "(3.95, 0.05)"
    assert rows["note_start_and_end"]["pred_effective"] == "(0.05, 3.95)"
    manifest = json.loads((remote / "manifest.json").read_text())
    assert manifest["run_id"] == "local-output"
    assert manifest["r2_uri"] == destination
    assert manifest["checkpoint"]["sha256"]
    assert manifest["stats"]["sha256"]
    assert manifest["git_sha"]
    assert manifest["content_cfg_strength"] == 2.0
    assert manifest["sketch_cfg_strength"] == 3.0


def _compatible_checkpoint() -> dict[str, object]:
    return {
        "hyper_parameters": {
            "conditioning": "mel",
            "num_params": 92,
            "param_spec": None,
            "sketch_controls": {
                "column": "sketch",
                "num_frames": 401,
                "num_ctrl_tokens": 32,
                "pitch_zero_threshold": 0.1,
            },
        },
        "state_dict": {
            "encoder.patch_embed.projection.weight": torch.empty(512, 2, 16, 16),
            "sketch_tokens.positional_encoding": torch.empty(1, 32, 512),
            "sketch_tokens.projections.loudness.weight": torch.empty(512, 1),
            "sketch_tokens.projections.centroid.weight": torch.empty(512, 1),
            "sketch_tokens.projections.pitch.weight": torch.empty(512, 384),
            "vector_field.projection._assignment": torch.empty(128, 92),
        },
    }


def test_validate_stats_expected_arrays_returns_without_error(tmp_path: Path) -> None:
    """Finite positive training statistics satisfy the model input contract.

    :param tmp_path: Holds the candidate statistics artifact.
    """
    stats_path = tmp_path / "stats.npz"
    np.savez(
        stats_path,
        mean=np.zeros((2, 128, 401), dtype=np.float32),
        std=np.ones((2, 128, 401), dtype=np.float32),
    )

    _validate_stats(stats_path)


def test_validate_stats_missing_std_raises(tmp_path: Path) -> None:
    """A statistics artifact missing std fails before inference.

    :param tmp_path: Holds the candidate statistics artifact.
    """
    stats_path = tmp_path / "stats.npz"
    np.savez(stats_path, mean=np.zeros((2, 128, 401)))

    with pytest.raises(KeyError, match="std"):
        _validate_stats(stats_path)


def test_validate_stats_wrong_shape_raises(tmp_path: Path) -> None:
    """Statistics for another model input shape fail before inference.

    :param tmp_path: Holds the candidate statistics artifact.
    """
    stats_path = tmp_path / "stats.npz"
    np.savez(
        stats_path,
        mean=np.zeros((1, 128, 401)),
        std=np.ones((1, 128, 401)),
    )

    with pytest.raises(ValueError, match="shape"):
        _validate_stats(stats_path)


def test_validate_stats_nonpositive_std_raises(tmp_path: Path) -> None:
    """Nonpositive standard deviations fail before inference.

    :param tmp_path: Holds the candidate statistics artifact.
    """
    stats_path = tmp_path / "stats.npz"
    np.savez(
        stats_path,
        mean=np.zeros((2, 128, 401)),
        std=np.zeros((2, 128, 401)),
    )

    with pytest.raises(ValueError, match="positive"):
        _validate_stats(stats_path)


def test_predict_patch_finite_prediction_decodes_real_param_spec() -> None:
    """A finite model-space prediction reaches the real Surge decoder."""
    prepared = PreparedAudioInputs(
        guide_audio=torch.zeros(2, 176400),
        ref_audio=torch.zeros(2, 176400),
        ref_mel=torch.zeros(2, 128, 401),
        sketch_controls=torch.zeros(NUM_SKETCH_CONTROLS, 401),
    )
    model = Mock()
    model.device = torch.device("cpu")
    model.hparams = {"test_cfg_strength": 4.0, "test_sketch_cfg_strength": None}
    model.predict_step.return_value = (torch.zeros(1, 92), None)

    synth_params, note_params, strengths = _predict_patch(
        prepared,
        model,
        CfgStrengths(content=None, sketch=None),
    )

    assert len(synth_params) > 80
    assert 0 <= note_params["pitch"] <= 127
    assert len(note_params["note_start_and_end"]) == 2
    assert strengths == CfgStrengths(content=4.0, sketch=4.0)


def test_predict_patch_zero_cfg_strengths_override_checkpoint() -> None:
    """Explicit zero disables both guidance branches without falling back."""
    prepared = PreparedAudioInputs(
        guide_audio=torch.zeros(2, 176400),
        ref_audio=torch.zeros(2, 176400),
        ref_mel=torch.zeros(2, 128, 401),
        sketch_controls=torch.zeros(NUM_SKETCH_CONTROLS, 401),
    )
    model = Mock()
    model.device = torch.device("cpu")
    model.hparams = {"test_cfg_strength": 4.0, "test_sketch_cfg_strength": 6.0}
    model.predict_step.return_value = (torch.zeros(1, 92), None)

    _, _, strengths = _predict_patch(
        prepared,
        model,
        CfgStrengths(content=0.0, sketch=0.0),
    )

    assert strengths == CfgStrengths(content=0.0, sketch=0.0)
    assert model.hparams["test_cfg_strength"] == 0.0
    assert model.hparams["test_sketch_cfg_strength"] == 0.0


def test_predict_patch_saved_cfg_strengths_preserved() -> None:
    """Omitted overrides preserve distinct checkpoint guidance strengths."""
    prepared = PreparedAudioInputs(
        guide_audio=torch.zeros(2, 176400),
        ref_audio=torch.zeros(2, 176400),
        ref_mel=torch.zeros(2, 128, 401),
        sketch_controls=torch.zeros(NUM_SKETCH_CONTROLS, 401),
    )
    model = Mock()
    model.device = torch.device("cpu")
    model.hparams = {"test_cfg_strength": 2.5, "test_sketch_cfg_strength": 6.5}
    model.predict_step.return_value = (torch.zeros(1, 92), None)

    _, _, strengths = _predict_patch(
        prepared,
        model,
        CfgStrengths(content=None, sketch=None),
    )

    assert strengths == CfgStrengths(content=2.5, sketch=6.5)
    assert model.hparams["test_cfg_strength"] == 2.5
    assert model.hparams["test_sketch_cfg_strength"] == 6.5


@pytest.mark.parametrize("legacy_sketch_strength", [None, pytest.param("missing", id="missing")])
def test_predict_patch_legacy_sketch_strength_uses_effective_content(
    legacy_sketch_strength: float | str | None,
) -> None:
    """A legacy absent sketch scale follows the effective content override.

    :param legacy_sketch_strength: Legacy missing or null checkpoint representation.
    """
    prepared = PreparedAudioInputs(
        guide_audio=torch.zeros(2, 176400),
        ref_audio=torch.zeros(2, 176400),
        ref_mel=torch.zeros(2, 128, 401),
        sketch_controls=torch.zeros(NUM_SKETCH_CONTROLS, 401),
    )
    model = Mock()
    model.device = torch.device("cpu")
    model.hparams = {"test_cfg_strength": 4.0}
    if legacy_sketch_strength is None:
        model.hparams["test_sketch_cfg_strength"] = None
    model.predict_step.return_value = (torch.zeros(1, 92), None)

    _, _, strengths = _predict_patch(
        prepared,
        model,
        CfgStrengths(content=1.5, sketch=None),
    )

    assert strengths == CfgStrengths(content=1.5, sketch=1.5)
    assert model.hparams["test_cfg_strength"] == 1.5
    assert model.hparams["test_sketch_cfg_strength"] == 1.5


def test_predict_patch_wrong_shape_raises() -> None:
    """A checkpoint prediction with the wrong output width fails before decoding."""
    prepared = PreparedAudioInputs(
        guide_audio=torch.zeros(2, 176400),
        ref_audio=torch.zeros(2, 176400),
        ref_mel=torch.zeros(2, 128, 401),
        sketch_controls=torch.zeros(NUM_SKETCH_CONTROLS, 401),
    )
    model = Mock()
    model.device = torch.device("cpu")
    model.hparams = {"test_cfg_strength": 4.0, "test_sketch_cfg_strength": None}
    model.predict_step.return_value = (torch.zeros(1, 91), None)

    with pytest.raises(ValueError, match="shape"):
        _predict_patch(prepared, model, CfgStrengths(content=None, sketch=None))


def test_render_patch_descending_note_interval_reaches_renderer_ordered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Independently decoded note endpoints reach Surge in ascending order.

    :param monkeypatch: Replaces the installed VST and renderer boundaries.
    :param tmp_path: Supplies a non-system plugin fixture path.
    """
    synth_version = SYNTHS[SynthName("surge_simple")].synth_version
    plugin_path = str(tmp_path / "Surge.vst3")
    monkeypatch.setattr("synth_setter.cli.clap.default_plugin_path", lambda: plugin_path)
    monkeypatch.setattr("synth_setter.cli.clap.extract_renderer_version", lambda _: synth_version)
    renderer = Mock()
    renderer.render.return_value = np.zeros((2, 176400), dtype=np.float32)
    monkeypatch.setattr("synth_setter.cli.clap.make_audio_renderer", lambda _: renderer)

    patch = _render_patch(
        {"filter_1_cutoff": 0.5},
        {"pitch": 60, "note_start_and_end": (3.2, 0.8)},
    )

    assert patch.audio.shape == (2, 176400)
    assert patch.note_params["note_start_and_end"] == (3.2, 0.8)
    assert patch.effective_note_window == (0.8, 3.2)
    assert renderer.render.call_args.args[3] == (0.8, 3.2)


def test_render_patch_note_interval_outside_signal_clipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Decoded endpoints remain raw while rendering uses the clipped interval.

    :param monkeypatch: Replaces the installed VST and renderer boundaries.
    :param tmp_path: Supplies a non-system plugin fixture path.
    """
    synth_version = SYNTHS[SynthName("surge_simple")].synth_version
    plugin_path = str(tmp_path / "Surge.vst3")
    monkeypatch.setattr("synth_setter.cli.clap.default_plugin_path", lambda: plugin_path)
    monkeypatch.setattr("synth_setter.cli.clap.extract_renderer_version", lambda _: synth_version)
    renderer = Mock()
    renderer.render.return_value = np.zeros((2, 176400), dtype=np.float32)
    monkeypatch.setattr("synth_setter.cli.clap.make_audio_renderer", lambda _: renderer)

    patch = _render_patch(
        {"filter_1_cutoff": 0.5},
        {"pitch": 60, "note_start_and_end": (-1.0, 5.0)},
    )

    assert patch.note_params["note_start_and_end"] == (-1.0, 5.0)
    assert patch.effective_note_window == (0.0, 4.0)
    assert renderer.render.call_args.args[3] == (0.0, 4.0)


def test_render_patch_degenerate_note_interval_expands_one_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Equal decoded endpoints render through a one-sample effective interval.

    :param monkeypatch: Replaces the installed VST and renderer boundaries.
    :param tmp_path: Supplies a non-system plugin fixture path.
    """
    synth_version = SYNTHS[SynthName("surge_simple")].synth_version
    plugin_path = str(tmp_path / "Surge.vst3")
    monkeypatch.setattr("synth_setter.cli.clap.default_plugin_path", lambda: plugin_path)
    monkeypatch.setattr("synth_setter.cli.clap.extract_renderer_version", lambda _: synth_version)
    renderer = Mock()
    renderer.render.return_value = np.zeros((2, 176400), dtype=np.float32)
    monkeypatch.setattr("synth_setter.cli.clap.make_audio_renderer", lambda _: renderer)

    patch = _render_patch(
        {"filter_1_cutoff": 0.5},
        {"pitch": 60, "note_start_and_end": (2.0, 2.0)},
    )

    assert patch.note_params["note_start_and_end"] == (2.0, 2.0)
    assert patch.effective_note_window == pytest.approx((2.0, 2.0 + 1.0 / 44100))
    assert renderer.render.call_args.args[3] == pytest.approx((2.0, 2.0 + 1.0 / 44100))


def test_headless_wrapper_nonexecutable_script_runs_through_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A package-extracted wrapper does not need executable mode bits.

    :param monkeypatch: Routes package-resource lookup to the fixture script.
    :param tmp_path: Holds the non-executable wrapper and audio path fixtures.
    """
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
    wrapper.chmod(0o644)
    monkeypatch.setattr("synth_setter.cli.clap.vst_headless_wrapper", lambda: wrapper)
    monkeypatch.setattr("synth_setter.cli.clap.as_file", nullcontext)

    _run_under_headless_wrapper(
        tmp_path / "guide.wav",
        tmp_path / "reference.wav",
        CfgStrengths(content=None, sketch=None),
    )


def test_headless_wrapper_cfg_strengths_forwarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both guidance overrides survive Linux headless re-entry.

    :param monkeypatch: Captures the subprocess command.
    :param tmp_path: Holds audio path fixtures.
    """
    monkeypatch.setattr("synth_setter.cli.clap.vst_headless_wrapper", lambda: tmp_path / "wrapper")
    monkeypatch.setattr("synth_setter.cli.clap.as_file", nullcontext)
    run = Mock()
    monkeypatch.setattr("synth_setter.cli.clap.subprocess.run", run)

    _run_under_headless_wrapper(
        tmp_path / "guide.wav",
        tmp_path / "reference.wav",
        CfgStrengths(content=2.0, sketch=3.0),
    )

    args = run.call_args.args[0]
    assert args[-4:] == ["--content-cfg-strength", "2.0", "--sketch-cfg-strength", "3.0"]


def test_packaged_surge_simple_preset_is_nonempty() -> None:
    """The preset consumed by the renderer is shipped with package data."""
    preset = surge_simple_preset()

    assert preset.is_file()
    assert preset.read_bytes().startswith(b"VST3")


def test_run_request_real_preprocessing_and_rclone_publish_artifacts(
    fake_r2_remote: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One request retains and uploads consumable artifacts around model boundaries.

    :param fake_r2_remote: Local filesystem backing the real rclone transport.
    :param monkeypatch: Replaces heavyweight checkpoint and VST boundaries.
    :param tmp_path: Holds real inputs, statistics, and retained outputs.
    """
    sample_rate = 44100
    time = np.arange(4 * sample_rate, dtype=np.float32) / sample_rate
    guide_path = tmp_path / "guide.wav"
    ref_path = tmp_path / "ref.wav"
    with AudioFile(str(guide_path), "w", sample_rate, 1) as audio_file:
        audio_file.write((0.5 * np.sin(2 * np.pi * 220.0 * time))[None])
    with AudioFile(str(ref_path), "w", sample_rate, 2) as audio_file:
        audio_file.write(
            np.stack(
                [
                    0.4 * np.sin(2 * np.pi * 330.0 * time),
                    0.2 * np.sin(2 * np.pi * 330.0 * time),
                ]
            ).astype(np.float32)
        )
    stats_path = tmp_path / "stats.npz"
    np.savez(
        stats_path,
        mean=np.zeros((2, 128, 401), dtype=np.float32),
        std=np.ones((2, 128, 401), dtype=np.float32),
    )
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_bytes(b"checkpoint boundary fixture")

    output_root = tmp_path / "retained"
    upload_root = "r2://intermediate-data/custom-sketch-renders"
    monkeypatch.setenv("SYNTH_SETTER_SKETCH_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("SYNTH_SETTER_SKETCH_UPLOAD_PREFIX", upload_root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("synth_setter.cli.clap.r2_io.ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(
        "synth_setter.cli.clap.cache_r2_file",
        lambda uri, _namespace, _digest: (
            stats_path if uri.endswith("stats.npz") else checkpoint_path
        ),
    )
    model = Mock()
    monkeypatch.setattr("synth_setter.cli.clap._load_model", lambda *_: model)
    monkeypatch.setattr(
        "synth_setter.cli.clap._predict_patch",
        lambda *_: (
            {"filter_1_cutoff": 0.5},
            {"pitch": 60, "note_start_and_end": (0.05, 3.95)},
            CfgStrengths(content=2.0, sketch=3.0),
        ),
    )
    monkeypatch.setattr(
        "synth_setter.cli.clap._render_patch",
        lambda synth_params, note_params: RenderedPatch(
            audio=np.full((2, 176400), 0.0625, dtype=np.float32),
            synth_params=synth_params,
            note_params=note_params,
            effective_note_window=note_params["note_start_and_end"],
        ),
    )

    output_dir, destination = _run_request(
        guide_path,
        ref_path,
        CfgStrengths(content=2.0, sketch=3.0),
    )

    assert output_dir.parent == output_root
    assert output_dir.is_dir()
    assert destination.startswith(f"{upload_root}/")
    remote = fake_r2_remote / destination.removeprefix("r2://")
    with AudioFile(str(remote / "pred.wav"), "r") as audio_file:
        rendered = audio_file.read(audio_file.frames)
    assert rendered.shape == (2, 176400)
    assert np.isfinite(rendered).all()
    assert np.abs(rendered).max() > 0.05
    manifest = json.loads((remote / "manifest.json").read_text())
    assert manifest["content_cfg_strength"] == 2.0
    assert manifest["sketch_cfg_strength"] == 3.0


def test_validate_checkpoint_compatibility_accepts_pinned_legacy_sketch_metadata() -> None:
    """Accept ``num_ctrl_tokens`` when validating sketch metadata."""
    sketch_config = validate_checkpoint_compatibility(_compatible_checkpoint())

    assert sketch_config == {
        "column": "sketch",
        "num_frames": 401,
        "num_control_tokens": 32,
        "pitch_zero_threshold": 0.1,
    }


def test_validate_checkpoint_compatibility_ambiguous_token_count_raises() -> None:
    """Conflicting legacy and current sketch widths cannot be normalized silently."""
    checkpoint = _compatible_checkpoint()
    checkpoint["hyper_parameters"]["sketch_controls"]["num_control_tokens"] = 64  # type: ignore[index]

    with pytest.raises(ValueError, match="token"):
        validate_checkpoint_compatibility(checkpoint)


def test_validate_checkpoint_compatibility_conflicting_legacy_token_count_raises() -> None:
    """Reject a conflicting legacy token count even when the current value is valid."""
    checkpoint = _compatible_checkpoint()
    checkpoint["hyper_parameters"]["sketch_controls"]["num_ctrl_tokens"] = 64  # type: ignore[index]
    checkpoint["hyper_parameters"]["sketch_controls"]["num_control_tokens"] = 32  # type: ignore[index]

    with pytest.raises(ValueError, match="token"):
        validate_checkpoint_compatibility(checkpoint)


def test_validate_checkpoint_compatibility_wrong_param_spec_raises() -> None:
    """An explicitly foreign synth identity fails before model construction."""
    checkpoint = _compatible_checkpoint()
    checkpoint["hyper_parameters"]["param_spec"] = "surge_xt"  # type: ignore[index]

    with pytest.raises(ValueError, match="surge_simple"):
        validate_checkpoint_compatibility(checkpoint)


def test_validate_checkpoint_compatibility_without_sketch_raises() -> None:
    """A base-arm checkpoint cannot silently ignore the guide audio."""
    checkpoint = _compatible_checkpoint()
    checkpoint["hyper_parameters"]["sketch_controls"] = None  # type: ignore[index]

    with pytest.raises(ValueError, match="sketch"):
        validate_checkpoint_compatibility(checkpoint)


def test_validate_checkpoint_compatibility_wrong_output_width_raises() -> None:
    """A checkpoint targeting another encoded parameter width fails early."""
    checkpoint = _compatible_checkpoint()
    checkpoint["hyper_parameters"]["num_params"] = 300  # type: ignore[index]

    with pytest.raises(ValueError, match="92"):
        validate_checkpoint_compatibility(checkpoint)


def test_validate_checkpoint_compatibility_wrong_feature_shape_raises() -> None:
    """A checkpoint trained for another mel channel shape fails early."""
    checkpoint = _compatible_checkpoint()
    checkpoint["state_dict"]["encoder.patch_embed.projection.weight"] = torch.empty(  # type: ignore[index]
        512, 1, 16, 16
    )

    with pytest.raises(ValueError, match="encoder.patch_embed"):
        validate_checkpoint_compatibility(checkpoint)
