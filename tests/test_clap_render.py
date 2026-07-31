"""Behavior tests for the text-to-Surge CLAP rendering CLI."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from click.testing import CliRunner
from pedalboard.io import AudioFile

from synth_setter.cli.clap_render import (
    compare_embeddings,
    main,
    resolve_inverse_checkpoint,
    summarize_cosine_distances,
)
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
