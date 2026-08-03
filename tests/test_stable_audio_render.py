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

from synth_setter.cli import stable_audio_render
from synth_setter.cli.stable_audio_render import (
    generate_same_latent,
    load_profile,
    main,
    validate_same_latent,
)
from synth_setter.pipeline import r2_io
from tests.helpers.run_if import RunIf

_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]


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

    latent = generate_same_latent(model, "warm analog pad", seed=17)

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
            "--no-upload",
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
    assert row["wav_r2_uri"] == ""
    assert row["csv_r2_uri"] == ""


def test_console_script_is_installed_and_callable() -> None:
    """The package installs the documented Stable Audio command."""
    executable = Path(sys.executable).with_name("synth-setter-sao")

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
    metrics = output.with_suffix(".csv")
    metrics_uri = upload_uri.removesuffix(".wav") + ".csv"
    downloaded = output.with_name(f"e2e-{model_name}-{run_token}-downloaded.wav")
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
        assert np.max(np.abs(audio)) > 1e-4

        r2_io.download_to_path(metrics_uri, downloaded_metrics)
        with downloaded_metrics.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert rows[0]["model"] == model_name
        assert rows[0]["conditioning"] == conditioning
        assert rows[0]["latent_shape"] == "1x256x44"
        assert float(rows[0]["latent_norm"]) > 0.0
        assert rows[0]["wav_r2_uri"] == upload_uri
    finally:
        output.unlink(missing_ok=True)
        metrics.unlink(missing_ok=True)
        downloaded.unlink(missing_ok=True)
        downloaded_metrics.unlink(missing_ok=True)
        r2_io.purge_prefix("experiments", f"stable-audio-renders/e2e/{run_token}/")
