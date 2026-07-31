"""Behavior tests for the text-to-Surge CLAP rendering CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from click.testing import CliRunner
from pedalboard.io import AudioFile

from synth_setter.cli.clap_render import main, resolve_inverse_checkpoint
from synth_setter.pipeline import r2_io

_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]


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
    downloaded = output.with_name(f"e2e-{run_token}-downloaded.wav")
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
        assert f"R2 WAV: {upload_uri}" in result.stdout

        r2_io.download_to_path(upload_uri, downloaded)
        with AudioFile(str(downloaded)) as audio_file:
            audio = audio_file.read(audio_file.frames)
            assert audio_file.samplerate == 44100
            assert audio_file.num_channels == 2
        assert audio.shape == (2, 176400)
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 1e-4
    finally:
        output.unlink(missing_ok=True)
        downloaded.unlink(missing_ok=True)
        r2_io.purge_prefix("experiments", f"clap-renders/e2e/{run_token}/")
