"""Behavior tests for the Surge sketch-render CLI."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from pedalboard.io import AudioFile

from synth_setter.cli.clap import (
    PreparedAudioInputs,
    RenderedPatch,
    _load_model_audio,
    main,
    prepare_audio_inputs,
    upload_output_artifacts,
    validate_checkpoint_compatibility,
    write_output_artifacts,
    write_run_manifest,
)
from synth_setter.data.vst.generate_vst_dataset import make_spectrogram
from synth_setter.features.sketch_controls import NUM_SKETCH_CONTROLS, SKETCH_PITCH_SLICE


def test_help_exposes_two_required_audio_options() -> None:
    """Help names the guide and reference inputs required by the public command."""
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--guide_audio FILE" in result.output
    assert "--ref_audio FILE" in result.output
    assert "Guide audio supplying sketch controls." in result.output
    assert "Reference audio supplying mel/timbre conditioning." in result.output
    assert ":param" not in result.output


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
            note_params={"pitch": 60, "note_start_and_end": (0.05, 3.95)},
        ),
    )
    destination = "r2://intermediate-data/eval/synth-setter-clap/test-run"
    write_run_manifest(output_dir, destination)
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
    assert "filter_1_cutoff" in (remote / "params.csv").read_text()
    manifest = json.loads((remote / "manifest.json").read_text())
    assert manifest["run_id"] == "local-output"
    assert manifest["r2_uri"] == destination
    assert manifest["checkpoint"]["sha256"]
    assert manifest["stats"]["sha256"]


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
