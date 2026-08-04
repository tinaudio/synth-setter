"""Behavior tests for Stable Audio text-to-Surge rendering."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from pedalboard.io import AudioFile
from safetensors.torch import load_file, save_file

from synth_setter.cli import stable_audio_render
from synth_setter.cli.stable_audio_render import (
    generate_same_latent,
    load_profile,
    main,
    validate_same_latent,
)
from synth_setter.cli.surge_render import validate_rendered_audio
from synth_setter.pipeline import r2_io
from tests.helpers.run_if import RunIf

_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
_CLI_HELP_TIMEOUT_SECONDS = 120


class _LatentModel:
    """Record generation inputs while returning a fixed latent.

    .. attribute :: model_config

        Source model sample budget consumed by the production generator.
    """

    model_config = {"sample_size": 5_292_032}

    def __init__(self, latent: torch.Tensor) -> None:
        """Store the latent returned by every generation request.

        :param latent: Fixed generated value.
        """
        self.latent = latent
        self.arguments: dict[str, object] = {}

    def generate(self, **kwargs: object) -> torch.Tensor:
        """Return the configured latent after recording generation arguments.

        :param **kwargs: Stable Audio generation arguments.
        :returns: Fixed latent supplied at construction.
        """
        self.arguments = kwargs
        return self.latent


def test_localize_prompt_conditioner_points_to_pinned_snapshot(tmp_path: Path) -> None:
    """Nested T5Gemma hydration uses the same immutable local snapshot.

    :param tmp_path: Pinned snapshot path.
    """
    model_config: dict[str, object] = {
        "model": {
            "conditioning": {
                "configs": [
                    {"id": "prompt", "config": {"repo_id": "mutable-upstream"}},
                    {"id": "seconds_total", "config": {}},
                ]
            }
        }
    }

    stable_audio_render._localize_prompt_conditioner(model_config, tmp_path)

    model = model_config["model"]
    assert isinstance(model, dict)
    conditioning = model["conditioning"]
    assert isinstance(conditioning, dict)
    configs = conditioning["configs"]
    assert isinstance(configs, list)
    prompt = configs[0]
    assert isinstance(prompt, dict)
    prompt_config = prompt["config"]
    assert isinstance(prompt_config, dict)
    assert prompt_config["repo_id"] == str(tmp_path)


def test_localize_prompt_conditioner_missing_model_raises() -> None:
    """A malformed pinned config fails before heavyweight model construction."""
    with pytest.raises(ValueError, match="no model mapping"):
        stable_audio_render._localize_prompt_conditioner({}, Path("/snapshot"))


def test_checkpoint_target_key_accepts_upstream_segment_removal() -> None:
    """The streaming loader preserves upstream's compatible key remapping."""
    assert (
        stable_audio_render._checkpoint_target_key("model.transformer.weight", {"model.weight"})
        == "model.weight"
    )
    assert stable_audio_render._checkpoint_target_key("missing.weight", {"model.weight"}) is None


def test_load_cuda_diffusion_streaming_loads_real_safetensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming publication fills and freezes a real tiny model state.

    :param tmp_path: Temporary safetensors checkpoint path.
    :param monkeypatch: Upstream model-factory boundary replacement fixture.
    """
    checkpoint = tmp_path / "model.safetensors"
    save_file({"weight": torch.tensor([[3.0]], dtype=torch.float32)}, checkpoint)
    monkeypatch.setattr(
        "stable_audio_3.factory.create_diffusion_cond_from_config",
        lambda _config: torch.nn.Linear(1, 1, bias=False),
    )

    model = stable_audio_render._load_cuda_diffusion_streaming({}, checkpoint, torch.device("cpu"))

    assert isinstance(model, torch.nn.Linear)
    assert model.weight.dtype == torch.float16
    assert model.weight.item() == pytest.approx(3.0)
    assert not model.training
    assert not model.weight.requires_grad


def test_load_profile_small_selects_same_s_inverse() -> None:
    """The small text model and SAME-S inverse checkpoint form one profile."""
    profile = load_profile("small")

    assert profile.model_name == "small-music"
    assert profile.conditioning.column == "same_s"
    assert profile.conditioning.input_shape == (256, 44)


def test_load_profile_medium_selects_same_l_inverse() -> None:
    """The medium text model and SAME-L inverse checkpoint form one profile."""
    profile = load_profile("medium")

    assert profile.model_name == "medium"
    assert profile.conditioning.column == "same_l"
    assert profile.conditioning.input_shape == (256, 44)


def test_generate_same_latent_uses_fixed_four_second_geometry() -> None:
    """Generation requests the aligned model sample budget without duration padding."""
    model = _LatentModel(torch.ones(1, 256, 44, dtype=torch.float16))

    latent = generate_same_latent(
        model,
        "warm analog pad",
        seed=17,
        generation=stable_audio_render.load_generation_settings(),
    )

    assert latent.shape == (1, 256, 44)
    assert latent.dtype == torch.float32
    assert latent.device.type == "cpu"
    assert model.arguments == {
        "prompt": "warm analog pad",
        "duration": 4.0,
        "steps": 8,
        "cfg_scale": 1.0,
        "batch_size": 1,
        "sample_size": 5_292_032,
        "duration_padding_sec": 0.0,
        "return_latents": True,
        "seed": 17,
    }


def test_validate_same_latent_wrong_frame_count_raises() -> None:
    """A latent that cannot match the trained inverse sequence is rejected."""
    with pytest.raises(ValueError, match=r"expected \(1, 256, 44\)"):
        validate_same_latent(torch.ones(1, 256, 43))


def test_validate_same_latent_nonfinite_value_raises() -> None:
    """Non-finite diffusion output never reaches inverse inference."""
    latent = torch.ones(1, 256, 44)
    latent[0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        validate_same_latent(latent)


def test_cli_whitespace_prompt_exits_before_model_loading() -> None:
    """A blank prompt fails before hydrating either heavyweight model."""
    result = CliRunner().invoke(main, ["   "])

    assert result.exit_code != 0
    assert "prompt must contain text" in result.output


def test_cli_local_small_writes_render_provenance_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI persists the model, latent contract, and local artifact identity.

    :param tmp_path: Temporary checkpoint and artifact directory.
    :param monkeypatch: Heavyweight model and renderer boundary patches.
    """
    inverse = tmp_path / "model.ckpt"
    inverse.write_bytes(b"checkpoint")
    output = tmp_path / "render.wav"
    latent_path = output.with_suffix(".safetensors")
    latent = torch.ones(1, 256, 44)
    monkeypatch.setattr(stable_audio_render, "load_stable_audio_model", lambda *_: object())
    monkeypatch.setattr(stable_audio_render, "generate_same_latent", lambda *_args, **_kw: latent)
    monkeypatch.setattr(
        stable_audio_render,
        "resolve_inverse_checkpoint",
        lambda *_args, **_kw: inverse,
    )
    monkeypatch.setattr(
        stable_audio_render,
        "predict_patch",
        lambda *_args, **_kw: torch.zeros(1, 92),
    )
    monkeypatch.setattr(
        stable_audio_render,
        "render_wav",
        lambda *_: np.zeros((2, 176400), dtype=np.float32),
    )

    result = CliRunner().invoke(
        main,
        [
            "soft bell",
            "--model",
            "small",
            "--checkpoint",
            str(inverse),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    with output.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["prompt"] == "soft bell"
    assert row["model"] == "small"
    assert row["stable_audio_model"] == "small-music"
    assert row["conditioning"] == "same_s"
    assert row["latent_shape"] == "1x256x44"
    assert float(row["latent_norm"]) == pytest.approx(106.131996)
    assert row["duration_seconds"] == "4.0"
    assert row["diffusion_steps"] == "8"
    assert row["cfg_scale"] == "1.0"
    assert row["inverse_checkpoint_sha256"] == (
        "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    )
    assert row["wav_r2_uri"] == ""
    assert row["latent_r2_uri"] == ""
    assert row["csv_r2_uri"] == ""
    assert load_file(latent_path)["latent"].equal(latent)


def test_cli_existing_latent_resumes_without_source_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed post-generation run can reuse its validated latent artifact.

    :param tmp_path: Temporary checkpoint and artifact directory.
    :param monkeypatch: Heavyweight inverse and renderer boundary patches.
    """
    inverse = tmp_path / "model.ckpt"
    inverse.write_bytes(b"checkpoint")
    output = tmp_path / "render.wav"
    stable_audio_render.write_latent_artifact(
        output.with_suffix(".safetensors"),
        torch.ones(1, 256, 44),
        stable_audio_render.latent_identity("soft bell", load_profile("small"), seed=0),
    )
    monkeypatch.setattr(
        stable_audio_render,
        "load_stable_audio_model",
        lambda *_: pytest.fail("source model should not load"),
    )
    monkeypatch.setattr(
        stable_audio_render,
        "resolve_inverse_checkpoint",
        lambda *_args, **_kw: inverse,
    )
    monkeypatch.setattr(
        stable_audio_render,
        "predict_patch",
        lambda *_args, **_kw: torch.zeros(1, 92),
    )
    monkeypatch.setattr(
        stable_audio_render,
        "render_wav",
        lambda *_: np.zeros((2, 176400), dtype=np.float32),
    )

    result = CliRunner().invoke(
        main,
        [
            "soft bell",
            "--checkpoint",
            str(inverse),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Reusing SAME latent" in result.output


def test_load_latent_artifact_different_prompt_raises(tmp_path: Path) -> None:
    """A resumable latent cannot silently cross prompt identities.

    :param tmp_path: Temporary latent artifact directory.
    """
    path = tmp_path / "latent.safetensors"
    profile = load_profile("small")
    stable_audio_render.write_latent_artifact(
        path,
        torch.ones(1, 256, 44),
        stable_audio_render.latent_identity("soft bell", profile, seed=0),
    )

    with pytest.raises(ValueError, match="identity does not match"):
        stable_audio_render.load_latent_artifact(
            path,
            stable_audio_render.latent_identity("bright bell", profile, seed=0),
        )


def test_load_latent_artifact_without_identity_raises(tmp_path: Path) -> None:
    """An unidentifiable tensor cannot be treated as resumable pipeline state.

    :param tmp_path: Temporary latent artifact directory.
    """
    path = tmp_path / "latent.safetensors"
    save_file({"latent": torch.ones(1, 256, 44)}, path)

    with pytest.raises(ValueError, match="no identity metadata"):
        stable_audio_render.load_latent_artifact(
            path,
            stable_audio_render.latent_identity("soft bell", load_profile("small"), seed=0),
        )


def test_cli_upload_uri_without_upload_rejected_before_model_loading() -> None:
    """An upload destination requires explicit remote-side-effect opt-in."""
    result = CliRunner().invoke(main, ["soft bell", "--upload-uri", "r2://bucket/render.wav"])

    assert result.exit_code != 0
    assert "cannot be combined with --no-upload" in result.output


def test_cli_non_r2_upload_uri_rejected_before_model_loading() -> None:
    """Upload destinations cannot escape the configured object-store scheme."""
    result = CliRunner().invoke(
        main,
        ["soft bell", "--upload", "--upload-uri", "file:///tmp/render.wav"],
    )

    assert result.exit_code != 0
    assert "must use r2://" in result.output


def test_cli_upload_writes_wav_latent_and_provenance_to_r2(
    tmp_path: Path,
    fake_r2_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit upload publishes all three artifacts through real rclone.

    :param tmp_path: Temporary checkpoint and local output directory.
    :param fake_r2_remote: Local filesystem backing the real rclone remote.
    :param monkeypatch: Heavyweight model, inverse, and renderer boundary patches.
    """
    inverse = tmp_path / "model.ckpt"
    inverse.write_bytes(b"checkpoint")
    output = tmp_path / "render.wav"
    monkeypatch.setattr(stable_audio_render, "load_stable_audio_model", lambda *_: object())
    monkeypatch.setattr(
        stable_audio_render,
        "generate_same_latent",
        lambda *_args, **_kwargs: torch.ones(1, 256, 44),
    )
    monkeypatch.setattr(
        stable_audio_render,
        "resolve_inverse_checkpoint",
        lambda *_args, **_kwargs: inverse,
    )
    monkeypatch.setattr(
        stable_audio_render,
        "predict_patch",
        lambda *_args, **_kwargs: torch.zeros(1, 92),
    )

    def render_to_path(_prediction: object, _render: object, path: Path) -> np.ndarray:
        """Materialize a bounded stand-in at the real publication boundary.

        :param _prediction: Ignored inverse prediction.
        :param _render: Ignored render configuration.
        :param path: WAV destination consumed by real rclone.
        :returns: Finite bounded waveform.
        """
        path.write_bytes(b"wav")
        return np.zeros((2, 176400), dtype=np.float32)

    monkeypatch.setattr(stable_audio_render, "render_wav", render_to_path)

    result = CliRunner().invoke(
        main,
        [
            "soft bell",
            "--checkpoint",
            str(inverse),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--upload",
            "--upload-uri",
            "r2://experiments/sao-test/render.wav",
        ],
    )

    assert result.exit_code == 0, result.output
    remote = fake_r2_remote / "experiments" / "sao-test"
    assert (remote / "render.wav").read_bytes() == b"wav"
    assert load_file(remote / "render.safetensors")["latent"].shape == (1, 256, 44)
    with (remote / "render.csv").open(newline="", encoding="utf-8") as stream:
        assert next(csv.DictReader(stream))["latent_r2_uri"].endswith("render.safetensors")


def test_cli_non_wav_output_rejected_before_model_loading(tmp_path: Path) -> None:
    """WAV and provenance artifacts cannot alias through a CSV output suffix.

    :param tmp_path: Temporary invalid output path.
    """
    result = CliRunner().invoke(
        main,
        ["soft bell", "--output", str(tmp_path / "render.csv"), "--no-upload"],
    )

    assert result.exit_code != 0
    assert "--output must end in .wav" in result.output


def test_validate_rendered_audio_out_of_range_raises() -> None:
    """A corrupted Surge render is rejected before WAV persistence."""
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        validate_rendered_audio(np.array([[0.0, 1.01], [0.0, 0.0]], dtype=np.float32))


def test_console_script_is_installed_and_callable() -> None:
    """The package installs the documented Stable Audio command."""
    executable = Path(sys.executable).with_name("synth-setter-sao")

    result = subprocess.run(  # noqa: S603 — fixed package entrypoint.
        [str(executable), "--help"],
        cwd=_CHECKOUT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=_CLI_HELP_TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, result.stderr
    assert "TEXT_PROMPT" in result.stdout
    assert "--model [small|medium]" in result.stdout


@pytest.mark.slow
@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.network
@pytest.mark.r2
@pytest.mark.integration_r2
@pytest.mark.requires_surgepy
@pytest.mark.parametrize(
    ("model_name", "conditioning"),
    [("small", "same_s"), ("medium", "same_l")],
)
def test_cli_prompt_model_renders_nonsilent_surge_wav_and_uploads_to_r2(
    model_name: str,
    conditioning: str,
) -> None:
    """Each Stable Audio profile drives its real inverse and Surge consumer.

    :param model_name: Small or Medium CLI profile.
    :param conditioning: SAME column expected in persisted provenance.
    """
    executable = Path(sys.executable).with_name("synth-setter-sao")
    run_token = uuid4().hex
    output = _CHECKOUT_ROOT / "outputs" / "stable-audio" / f"e2e-{model_name}-{run_token}.wav"
    upload_uri = f"r2://experiments/stable-audio-renders/e2e/{run_token}/{model_name}-warm-pad.wav"
    latent = output.with_suffix(".safetensors")
    metrics = output.with_suffix(".csv")
    lock = output.with_suffix(".lock")
    latent_uri = upload_uri.removesuffix(".wav") + ".safetensors"
    metrics_uri = upload_uri.removesuffix(".wav") + ".csv"
    downloaded = output.with_name(f"e2e-{model_name}-{run_token}-downloaded.wav")
    downloaded_latent = output.with_name(f"e2e-{model_name}-{run_token}-downloaded.safetensors")
    downloaded_metrics = output.with_name(f"e2e-{model_name}-{run_token}-downloaded.csv")
    try:
        result = subprocess.run(  # noqa: S603 — fixed package entrypoint and test values.
            [
                str(executable),
                "warm analog synthesizer pad",
                "--model",
                model_name,
                "--output",
                str(output),
                "--upload-uri",
                upload_uri,
                "--upload",
            ],
            cwd=_CHECKOUT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}\nstdout:\n{result.stdout[-2000:]}"
            f"\nstderr:\n{result.stderr[-2000:]}"
        )

        r2_io.download_to_path(upload_uri, downloaded)
        with AudioFile(str(downloaded)) as audio_file:
            audio = audio_file.read(audio_file.frames)
            assert audio_file.samplerate == 44100
            assert audio_file.num_channels == 2
        assert audio.shape == (2, 176400)
        assert np.isfinite(audio).all()
        assert np.min(audio) >= -1.0
        assert np.max(audio) <= 1.0
        assert np.max(np.abs(audio)) > 1e-4

        r2_io.download_to_path(latent_uri, downloaded_latent)
        consumed_latent = load_file(downloaded_latent)["latent"]
        assert consumed_latent.shape == (1, 256, 44)
        assert torch.isfinite(consumed_latent).all()

        r2_io.download_to_path(metrics_uri, downloaded_metrics)
        with downloaded_metrics.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert rows[0]["model"] == model_name
        assert rows[0]["conditioning"] == conditioning
        assert rows[0]["latent_shape"] == "1x256x44"
        assert float(rows[0]["latent_norm"]) > 0.0
        assert rows[0]["wav_r2_uri"] == upload_uri
        assert rows[0]["latent_r2_uri"] == latent_uri
    finally:
        output.unlink(missing_ok=True)
        latent.unlink(missing_ok=True)
        metrics.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
        downloaded.unlink(missing_ok=True)
        downloaded_latent.unlink(missing_ok=True)
        downloaded_metrics.unlink(missing_ok=True)
        r2_io.purge_prefix("experiments", f"stable-audio-renders/e2e/{run_token}/")
