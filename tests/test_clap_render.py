"""Behavior tests for the text-to-Surge CLAP rendering CLI."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from pedalboard.io import AudioFile

from synth_setter.cli import clap_render
from synth_setter.cli._cfg_strength import CfgStrengths
from synth_setter.cli.clap_render import (
    compare_embeddings,
    main,
    resolve_inverse_checkpoint,
    summarize_cosine_distances,
    write_summary_csv,
)
from synth_setter.pipeline import r2_io

_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
_RETAINED_FILENAMES = ("guide.wav", "manifest.json", "params.csv", "pred.wav", "ref.wav")


def _write_retained_run(output_dir: Path, destination: str, run_id: str | None = None) -> None:
    """Write the five retained artifacts needed by upload recovery.

    :param output_dir: Retained run directory to populate.
    :param destination: Manifest R2 destination.
    :param run_id: Manifest run identifier, defaulting to the directory name.
    """
    output_dir.mkdir(parents=True)
    for filename in _RETAINED_FILENAMES:
        (output_dir / filename).write_bytes(f"retained {filename}".encode())
    manifest = {"run_id": run_id or output_dir.name, "r2_uri": destination}
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_cli_retry_upload_real_rclone_publishes_retained_artifacts_without_inference(
    fake_r2_remote: Path,
) -> None:
    """The installed command uploads an existing run through real rclone only.

    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    """
    output_dir = fake_r2_remote / "retained" / "run-1"
    destination = "r2://intermediate-data/eval/synth-setter-clap/run-1"
    _write_retained_run(output_dir, destination)

    executable = Path(sys.executable).with_name("synth-setter-clap")
    result = subprocess.run(  # noqa: S603 — fixed installed entrypoint and fixture path.
        [str(executable), "--retry-upload", str(output_dir)],
        cwd=fake_r2_remote,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == destination
    remote = fake_r2_remote / destination.removeprefix("r2://")
    assert sorted(path.name for path in remote.iterdir()) == sorted(_RETAINED_FILENAMES)
    assert (remote / "pred.wav").read_bytes() == b"retained pred.wav"


def test_cli_retry_upload_non_r2_manifest_rejected_before_upload(
    fake_r2_remote: Path,
) -> None:
    """A retained manifest cannot redirect recovery to a non-R2 destination.

    :param fake_r2_remote: Local remote asserted to remain empty.
    """
    output_dir = fake_r2_remote / "retained" / "run-1"
    _write_retained_run(output_dir, "file:///tmp/not-r2")

    result = CliRunner().invoke(main, ["--retry-upload", str(output_dir)])

    assert result.exit_code != 0
    assert "manifest r2_uri must use r2://" in result.output
    assert not (fake_r2_remote / "intermediate-data").exists()


def test_cli_retry_upload_mismatched_run_id_rejected_before_upload(
    fake_r2_remote: Path,
) -> None:
    """A manifest from another retained run cannot choose the upload destination.

    :param fake_r2_remote: Local remote asserted to remain empty.
    """
    output_dir = fake_r2_remote / "retained" / "run-1"
    destination = "r2://intermediate-data/eval/synth-setter-clap/run-1"
    _write_retained_run(output_dir, destination, run_id="run-2")

    result = CliRunner().invoke(main, ["--retry-upload", str(output_dir)])

    assert result.exit_code != 0
    assert "manifest run_id must match output directory name" in result.output
    assert not (fake_r2_remote / "intermediate-data").exists()


def test_cli_retry_upload_missing_expected_file_rejected_before_upload(
    fake_r2_remote: Path,
) -> None:
    """Recovery requires the complete retained artifact set before upload.

    :param fake_r2_remote: Local remote asserted to remain empty.
    """
    output_dir = fake_r2_remote / "retained" / "run-1"
    destination = "r2://intermediate-data/eval/synth-setter-clap/run-1"
    _write_retained_run(output_dir, destination)
    (output_dir / "params.csv").unlink()

    result = CliRunner().invoke(main, ["--retry-upload", str(output_dir)])

    assert result.exit_code != 0
    assert "missing retained artifact: params.csv" in result.output
    assert not (fake_r2_remote / "intermediate-data").exists()


def test_cli_retry_upload_unexpected_file_rejected_before_upload(
    fake_r2_remote: Path,
) -> None:
    """Recovery never uploads files outside the retained artifact contract.

    :param fake_r2_remote: Local remote asserted to remain empty.
    """
    output_dir = fake_r2_remote / "retained" / "run-1"
    destination = "r2://intermediate-data/eval/synth-setter-clap/run-1"
    _write_retained_run(output_dir, destination)
    (output_dir / ".env").write_text("SECRET=do-not-upload", encoding="utf-8")

    result = CliRunner().invoke(main, ["--retry-upload", str(output_dir)])

    assert result.exit_code != 0
    assert "unexpected retained artifact: .env" in result.output
    assert not (fake_r2_remote / "intermediate-data").exists()


def test_cli_retry_upload_invalid_manifest_encoding_rejected_before_upload(
    fake_r2_remote: Path,
) -> None:
    """A corrupted manifest reports a concise CLI error before upload.

    :param fake_r2_remote: Local remote asserted to remain empty.
    """
    output_dir = fake_r2_remote / "retained" / "run-1"
    destination = "r2://intermediate-data/eval/synth-setter-clap/run-1"
    _write_retained_run(output_dir, destination)
    (output_dir / "manifest.json").write_bytes(b"\xff")

    result = CliRunner().invoke(main, ["--retry-upload", str(output_dir)])

    assert result.exit_code != 0
    assert "could not read manifest.json" in result.output
    assert not (fake_r2_remote / "intermediate-data").exists()


def test_cli_retry_upload_text_prompt_rejected_before_upload(tmp_path: Path) -> None:
    """Upload recovery cannot be combined with a text inference request.

    :param tmp_path: Holds a valid retained run.
    """
    output_dir = tmp_path / "run-1"
    _write_retained_run(output_dir, "r2://bucket/run-1")

    result = CliRunner().invoke(main, ["prompt", "--retry-upload", str(output_dir)])

    assert result.exit_code != 0
    assert "TEXT_PROMPT cannot be combined with --retry-upload" in result.output


def test_cli_retry_upload_audio_options_rejected_before_upload(tmp_path: Path) -> None:
    """Upload recovery cannot be combined with guide/reference inference inputs.

    :param tmp_path: Holds retained and audio path fixtures.
    """
    output_dir = tmp_path / "run-1"
    _write_retained_run(output_dir, "r2://bucket/run-1")
    guide = tmp_path / "guide-input.wav"
    reference = tmp_path / "ref-input.wav"
    guide.write_bytes(b"path fixture")
    reference.write_bytes(b"path fixture")

    result = CliRunner().invoke(
        main,
        [
            "--retry-upload",
            str(output_dir),
            "--guide_audio",
            str(guide),
            "--ref_audio",
            str(reference),
        ],
    )

    assert result.exit_code != 0
    assert "--guide_audio cannot be combined with --retry-upload" in result.output


def test_cli_retry_upload_cfg_option_rejected_before_upload(tmp_path: Path) -> None:
    """Upload recovery cannot be combined with guidance overrides.

    :param tmp_path: Holds a valid retained run.
    """
    output_dir = tmp_path / "run-1"
    _write_retained_run(output_dir, "r2://bucket/run-1")

    result = CliRunner().invoke(
        main,
        ["--retry-upload", str(output_dir), "--sketch-cfg-strength", "0"],
    )

    assert result.exit_code != 0
    assert "--sketch-cfg-strength cannot be combined with --retry-upload" in result.output


def test_cli_retry_upload_text_only_option_rejected_before_upload(tmp_path: Path) -> None:
    """Upload recovery cannot be combined with text-render configuration.

    :param tmp_path: Holds a valid retained run.
    """
    output_dir = tmp_path / "run-1"
    _write_retained_run(output_dir, "r2://bucket/run-1")

    result = CliRunner().invoke(
        main,
        ["--retry-upload", str(output_dir), "--checkpoint", "model.ckpt"],
    )

    assert result.exit_code != 0
    assert "--checkpoint cannot be combined with --retry-upload" in result.output


def test_cli_retry_upload_upload_flag_rejected_before_upload(tmp_path: Path) -> None:
    """Upload recovery cannot be combined with text-mode upload controls.

    :param tmp_path: Holds a valid retained run.
    """
    output_dir = tmp_path / "run-1"
    _write_retained_run(output_dir, "r2://bucket/run-1")

    result = CliRunner().invoke(main, ["--retry-upload", str(output_dir), "--no-upload"])

    assert result.exit_code != 0
    assert "--upload/--no-upload cannot be combined with --retry-upload" in result.output


def test_cli_one_audio_conditioning_input_exits_before_inference(tmp_path: Path) -> None:
    """Guide and reference audio must be supplied as one mode.

    :param tmp_path: Holds the argument-validation fixture.
    """
    guide = tmp_path / "guide.wav"
    guide.write_bytes(b"not decoded before argument validation")

    result = CliRunner().invoke(main, ["--guide_audio", str(guide)])

    assert result.exit_code != 0
    assert "must be provided together" in result.output


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--checkpoint", "checkpoint.ckpt"),
        ("--clap-checkpoint", "clap.pt"),
        ("--output", "output.wav"),
        ("--upload-uri", "r2://intermediate-data/test.wav"),
        ("--device", "cpu"),
        ("--seed", "4"),
    ],
)
def test_cli_audio_mode_text_option_exits_before_inference(
    option: str, value: str, tmp_path: Path
) -> None:
    """Text-render options cannot be accepted and then ignored in audio mode.

    :param option: Text-render option rejected by audio mode.
    :param value: Value belonging to the rejected option.
    :param tmp_path: Holds the argument-validation fixtures.
    """
    guide = tmp_path / "guide.wav"
    reference = tmp_path / "reference.wav"
    guide.write_bytes(b"not decoded before argument validation")
    reference.write_bytes(b"not decoded before argument validation")
    result = CliRunner().invoke(
        main,
        ["--guide_audio", str(guide), "--ref_audio", str(reference), option, value],
    )

    assert result.exit_code != 0
    assert f"{option} is not supported with guide/reference audio" in result.output


def test_cli_audio_mode_no_upload_exits_before_inference(tmp_path: Path) -> None:
    """Audio mode rejects the text renderer's no-upload behavior.

    :param tmp_path: Holds the argument-validation fixtures.
    """
    guide = tmp_path / "guide.wav"
    reference = tmp_path / "reference.wav"
    guide.write_bytes(b"not decoded before argument validation")
    reference.write_bytes(b"not decoded before argument validation")

    result = CliRunner().invoke(
        main,
        ["--guide_audio", str(guide), "--ref_audio", str(reference), "--no-upload"],
    )

    assert result.exit_code != 0
    assert "--no-upload is not supported with guide/reference audio" in result.output


def test_cli_audio_mode_dispatches_cfg_strengths_to_sketch_renderer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The installed public command routes both audio inputs to sketch rendering.

    :param monkeypatch: Replaces the expensive request after Click routing.
    :param tmp_path: Holds argument-validation fixtures.
    """
    guide = tmp_path / "guide.wav"
    reference = tmp_path / "reference.wav"
    guide.write_bytes(b"validated path")
    reference.write_bytes(b"validated path")
    destination = "r2://intermediate-data/eval/synth-setter-clap/test-run"
    received: list[CfgStrengths[float | None]] = []

    def capture_request(
        _guide: Path,
        _reference: Path,
        strengths: CfgStrengths[float | None],
    ) -> tuple[Path, str]:
        received.append(strengths)
        return tmp_path / "output", destination

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
            "2",
            "--sketch-cfg-strength",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == destination
    assert received == [CfgStrengths(content=2.0, sketch=3.0)]


def test_cli_text_mode_sketch_cfg_strength_rejected_before_inference() -> None:
    """Text conditioning cannot accept a sketch-only guidance override."""
    result = CliRunner().invoke(main, ["soft bell", "--sketch-cfg-strength", "2"])

    assert result.exit_code != 0
    assert "--sketch-cfg-strength is only supported with guide/reference audio" in result.output


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_cli_text_content_cfg_strength_invalid_value_rejected(value: str) -> None:
    """Text guidance rejects negative and non-finite values.

    :param value: Invalid CLI value.
    """
    result = CliRunner().invoke(main, ["soft bell", "--content-cfg-strength", value])

    assert result.exit_code != 0
    assert "finite and greater than or equal to zero" in result.output


def test_cli_text_and_audio_modes_together_exit_before_inference(tmp_path: Path) -> None:
    """Text and audio conditioning modes are mutually exclusive.

    :param tmp_path: Holds the argument-validation fixtures.
    """
    guide = tmp_path / "guide.wav"
    reference = tmp_path / "reference.wav"
    guide.write_bytes(b"not decoded before argument validation")
    reference.write_bytes(b"not decoded before argument validation")

    result = CliRunner().invoke(
        main,
        ["prompt", "--guide_audio", str(guide), "--ref_audio", str(reference)],
    )

    assert result.exit_code != 0
    assert "cannot be combined" in result.output


def test_cli_whitespace_prompt_exits_before_creating_output() -> None:
    """A blank text condition fails clearly instead of running model inference."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(main, ["   "])

        assert result.exit_code != 0
        assert "prompt must contain text" in result.output
        assert not Path("outputs").exists()


def test_resolve_inverse_checkpoint_local_path_returns_same_file(tmp_path: Path) -> None:
    """A local checkpoint override is used without copying it into the cache.

    :param tmp_path: Temporary checkpoint directory.
    """
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"local checkpoint")

    resolved = resolve_inverse_checkpoint(str(checkpoint))

    assert resolved == checkpoint


def test_resolve_inverse_checkpoint_r2_uri_materializes_cached_bytes(
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An R2 checkpoint is downloaded through real rclone into the shared cache.

    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param monkeypatch: Environment override fixture.
    """
    source = fake_r2_remote / "models" / "inverse" / "model.ckpt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"r2 checkpoint")
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_r2_remote / "cache"))

    resolved = resolve_inverse_checkpoint("r2://models/inverse/model.ckpt")

    assert resolved.read_bytes() == b"r2 checkpoint"
    assert resolved.is_relative_to(fake_r2_remote / "cache" / "synth-setter")


def test_resolve_inverse_checkpoint_wrong_digest_rejects_download(
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mutated R2 checkpoint never publishes into the shared model cache.

    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param monkeypatch: Environment override fixture.
    """
    source = fake_r2_remote / "models" / "inverse" / "changed.ckpt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"changed checkpoint")
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_r2_remote / "cache"))

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        resolve_inverse_checkpoint(
            "r2://models/inverse/changed.ckpt",
            expected_sha256="0" * 64,
        )


def test_compare_embeddings_identical_vectors_returns_zero_cosine_distance() -> None:
    """Identical normalized vectors have similarity one and distance zero."""
    comparison = compare_embeddings(
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    assert comparison.cosine_similarity == pytest.approx(1.0)
    assert comparison.cosine_distance == pytest.approx(0.0)
    assert comparison.text_embedding_norm == pytest.approx(1.0)
    assert comparison.audio_embedding_norm == pytest.approx(1.0)


def test_compare_embeddings_orthogonal_vectors_returns_unit_cosine_distance() -> None:
    """Orthogonal vectors have similarity zero and distance one."""
    comparison = compare_embeddings(
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 2.0]], dtype=np.float32),
    )

    assert comparison.cosine_similarity == pytest.approx(0.0)
    assert comparison.cosine_distance == pytest.approx(1.0)


def test_compare_embeddings_zero_vector_raises() -> None:
    """A degenerate embedding cannot produce a meaningful cosine distance."""
    with pytest.raises(ValueError, match="non-zero"):
        compare_embeddings(
            np.array([[0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
        )


def test_summarize_cosine_distances_reports_population_statistics() -> None:
    """Aggregate output includes fixed descriptive statistics over all distances."""
    summary = summarize_cosine_distances([0.1, 0.2, 0.3, 0.4])

    assert summary == pytest.approx(
        {
            "count": 4,
            "mean": 0.25,
            "std_population": 0.11180339887498948,
            "min": 0.1,
            "p25": 0.175,
            "median": 0.25,
            "p75": 0.325,
            "max": 0.4,
        }
    )


def test_write_summary_csv_persists_named_statistics(tmp_path: Path) -> None:
    """Aggregate statistics remain machine-readable after CSV persistence.

    :param tmp_path: Temporary summary destination.
    """
    output = tmp_path / "aggregate.csv"

    write_summary_csv(output, {"count": 2, "mean": 0.25})

    with output.open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [{"count": "2", "mean": "0.25"}]


def test_predict_patch_omitted_content_strength_preserves_checkpoint_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Text prediction retains the checkpoint content guidance when omitted.

    :param monkeypatch: Replaces checkpoint loading with a lightweight model.
    :param tmp_path: Supplies a checkpoint path fixture.
    """
    model = Mock()
    model.hparams = {
        "conditioning": {"column": "clap", "input_shape": [512]},
        "num_params": 92,
        "sketch_controls": None,
        "test_cfg_strength": 5.5,
        "test_sketch_cfg_strength": None,
    }
    model.to.return_value = model
    model.eval.return_value = model
    model.predict_step.return_value = (torch.zeros(1, 92), None)
    monkeypatch.setattr(
        clap_render.VSTFlowMatchingModule,
        "load_from_checkpoint",
        lambda *_args, **_kwargs: model,
    )

    _, effective_strengths = clap_render._predict_patch(
        torch.zeros(1, 512),
        tmp_path / "model.ckpt",
        clap_render._load_settings().render,
        torch.device("cpu"),
        0,
        CfgStrengths(content=None, sketch=None),
    )

    assert effective_strengths == CfgStrengths(content=5.5, sketch=5.5)
    assert model.hparams["test_cfg_strength"] == 5.5
    assert model.hparams["test_sketch_cfg_strength"] == 5.5


def test_render_wav_descending_predicted_note_coordinates_reaches_renderer_sorted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Descending inverse output is projected onto the sampler's note-window contract.

    :param tmp_path: Temporary WAV destination.
    :param monkeypatch: Renderer-boundary replacement fixture.
    """
    render_config = clap_render._load_settings().render
    spec = clap_render.param_specs[render_config.param_spec_name]
    synth_params, _ = spec.sample(np.random.default_rng(0))
    encoded = spec.encode(
        synth_params,
        {"pitch": 60, "note_start_and_end": (3.2, 0.8)},
    )
    prediction = torch.from_numpy(spec.encoded_to_model(encoded)).unsqueeze(0)

    class StrictRenderer:
        note_window: tuple[float, float] | None = None

        def render(
            self,
            params: dict[str, float],
            midi_note: int,
            velocity: int,
            note_start_and_end: tuple[float, float],
        ) -> np.ndarray:
            del params, midi_note, velocity
            start, end = note_start_and_end
            if not 0.0 <= start < end <= render_config.signal_duration_seconds:
                raise ValueError("note times must satisfy 0 <= start < end <= signal duration")
            self.note_window = note_start_and_end
            samples = int(render_config.sample_rate * render_config.signal_duration_seconds)
            return np.zeros((render_config.channels, samples), dtype=np.float32)

    renderer = StrictRenderer()
    monkeypatch.setattr(clap_render, "make_audio_renderer", lambda _: renderer)

    output = tmp_path / "descending-note.wav"
    clap_render._render_wav(prediction, render_config, output)

    assert renderer.note_window == pytest.approx((0.8, 3.2))
    assert output.is_file()


def test_predict_patch_zero_content_cfg_strength_overrides_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit zero reaches text inference without checkpoint fallback.

    :param tmp_path: Holds the checkpoint path fixture.
    :param monkeypatch: Replaces checkpoint loading and model validation.
    """
    model = Mock()
    model.hparams = {"test_cfg_strength": 4.0, "test_sketch_cfg_strength": None}
    model.to.return_value = model
    model.eval.return_value = model
    model.predict_step.return_value = (torch.zeros(1, 92), None)
    monkeypatch.setattr(
        clap_render.VSTFlowMatchingModule,
        "load_from_checkpoint",
        lambda *_args, **_kwargs: model,
    )
    monkeypatch.setattr(clap_render, "_validate_inverse_model", lambda *_: None)

    _, effective_strengths = clap_render._predict_patch(
        torch.zeros(1, 512),
        tmp_path / "model.ckpt",
        clap_render._load_settings().render,
        torch.device("cpu"),
        0,
        CfgStrengths(content=0.0, sketch=None),
    )

    assert effective_strengths == CfgStrengths(content=0.0, sketch=0.0)
    assert model.hparams["test_cfg_strength"] == 0.0


def test_cli_local_no_upload_writes_prompt_audio_comparison_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local CLI shell persists cosine metrics without requiring R2.

    :param tmp_path: Temporary checkpoints and output paths.
    :param monkeypatch: Boundary patch fixture for heavyweight model and renderer calls.
    """
    inverse = tmp_path / "model.ckpt"
    inverse.write_bytes(b"checkpoint")
    clap_checkpoint = tmp_path / "clap"
    clap_checkpoint.mkdir()
    output = tmp_path / "render.wav"
    monkeypatch.setattr(
        clap_render,
        "_encode_text",
        lambda *_: torch.tensor([[1.0, 0.0]]),
    )
    monkeypatch.setattr(
        clap_render,
        "_predict_patch",
        lambda *_: (
            torch.zeros(1, 92),
            CfgStrengths(content=0.0, sketch=0.0),
        ),
    )
    monkeypatch.setattr(
        clap_render,
        "_render_wav",
        lambda *_: np.zeros((2, 176400), dtype=np.float32),
    )
    monkeypatch.setattr(
        clap_render,
        "_encode_audio",
        lambda *_: np.array([[0.0, 1.0]], dtype=np.float32),
    )

    result = CliRunner().invoke(
        main,
        [
            "soft bell",
            "--checkpoint",
            str(inverse),
            "--clap-checkpoint",
            str(clap_checkpoint),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--content-cfg-strength",
            "0",
            "--no-upload",
        ],
    )

    assert result.exit_code == 0, result.output
    with output.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["prompt"] == "soft bell"
    assert float(row["content_cfg_strength"]) == 0.0
    assert float(row["cosine_distance"]) == pytest.approx(1.0)
    assert row["wav_r2_uri"] == ""
    assert row["csv_r2_uri"] == ""


def test_console_script_is_installed_and_callable() -> None:
    """The documented executable is installed by the package entrypoint."""
    executable = Path(sys.executable).with_name("synth-setter-clap")

    result = subprocess.run(  # noqa: S603 — fixed package entrypoint.
        [str(executable), "--help"],
        cwd=_CHECKOUT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "TEXT_PROMPT" in result.stdout
    assert "--retry-upload DIRECTORY" in result.stdout
    assert 'synth-setter-clap "frog croak"' in result.stdout


@pytest.mark.slow
@pytest.mark.r2
@pytest.mark.integration_r2
@pytest.mark.requires_surgepy
def test_cli_prompt_defaults_render_nonsilent_surge_wav_and_upload_to_r2() -> None:
    """Default model and render settings produce and upload a usable Surge WAV."""
    executable = Path(sys.executable).with_name("synth-setter-clap")
    run_token = uuid4().hex
    output = _CHECKOUT_ROOT / "outputs" / "clap" / f"e2e-{run_token}.wav"
    upload_uri = f"r2://experiments/clap-renders/e2e/{run_token}/frog-croak.wav"
    metrics = output.with_suffix(".csv")
    metrics_uri = upload_uri.removesuffix(".wav") + ".csv"
    downloaded = output.with_name(f"e2e-{run_token}-downloaded.wav")
    downloaded_metrics = output.with_name(f"e2e-{run_token}-downloaded.csv")
    try:
        result = subprocess.run(  # noqa: S603 — fixed package entrypoint and test values.
            [
                str(executable),
                "frog croak",
                "--output",
                str(output),
                "--upload-uri",
                upload_uri,
            ],
            cwd=_CHECKOUT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}\nstdout:\n{result.stdout[-2000:]}"
            f"\nstderr:\n{result.stderr[-2000:]}"
        )
        assert f"Local WAV: {output}" in result.stdout
        assert f"Local CSV: {metrics}" in result.stdout
        assert f"R2 WAV: {upload_uri}" in result.stdout
        assert f"R2 CSV: {metrics_uri}" in result.stdout

        r2_io.download_to_path(upload_uri, downloaded)
        with AudioFile(str(downloaded)) as audio_file:
            audio = audio_file.read(audio_file.frames)
            assert audio_file.samplerate == 44100
            assert audio_file.num_channels == 2
        assert audio.shape == (2, 176400)
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 1e-4

        r2_io.download_to_path(metrics_uri, downloaded_metrics)
        with downloaded_metrics.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert rows[0]["prompt"] == "frog croak"
        assert 0.0 <= float(rows[0]["cosine_distance"]) <= 2.0
        assert rows[0]["wav_r2_uri"] == upload_uri
    finally:
        output.unlink(missing_ok=True)
        metrics.unlink(missing_ok=True)
        downloaded.unlink(missing_ok=True)
        downloaded_metrics.unlink(missing_ok=True)
        r2_io.purge_prefix("experiments", f"clap-renders/e2e/{run_token}/")
