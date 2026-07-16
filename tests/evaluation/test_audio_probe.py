"""Unit tests for :mod:`synth_setter.evaluation.audio_probe`.

Covers the argv builder and sample counter in isolation, and drives
``run_audio_probe`` with a fake subprocess runner that writes a real
aggregated-metrics CSV plus a real rclone upload against ``fake_r2_remote`` —
so URI construction, stage ordering, the upload exclusion, and the returned
metric dict are all validated without a VST. The full render chain is covered
by ``test_train.py::test_train_surge_xt_val_audio_probe_renders_scores_and_uploads``.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import NoReturn

import pytest
import torch

from synth_setter.evaluation import audio_probe
from synth_setter.evaluation.audio_probe import (
    ProbeRenderSettings,
    _render_argv,
    _staged_sample_count,
    run_audio_probe,
)

_SETTINGS = ProbeRenderSettings(
    param_spec_name="surge_4",
    plugin_state_path="presets/surge-mini.vstpreset",
    plugin_path="plugins/Surge XT.vst3",
    sample_rate=8000.0,
    channels=2,
    velocity=100,
    signal_duration_seconds=0.1,
)


def _stage(probe_dir: Path, rows: int = 3) -> None:
    """Write the pred/target-params tensors ``ValAudioProbe`` stages.

    :param probe_dir: Probe directory to create ``predictions/`` under.
    :param rows: Row count of both staged tensors.
    """
    predictions = probe_dir / "predictions"
    predictions.mkdir(parents=True)
    torch.save(torch.zeros(rows, 4), predictions / "pred-0.pt")
    torch.save(torch.zeros(rows, 4), predictions / "target-params-0.pt")


def test_staged_sample_count_returns_pred_row_count(tmp_path: Path) -> None:
    """The sample count is the staged prediction tensor's batch dimension.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    _stage(tmp_path, rows=3)

    assert _staged_sample_count(tmp_path) == 3


def test_render_argv_forwards_settings_and_rerenders_target(tmp_path: Path) -> None:
    """The argv names both probe subdirs, every render field, and --rerender_target.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    with ExitStack() as stack:
        argv = _render_argv(tmp_path, _SETTINGS, stack)

    assert str(tmp_path / "predictions") in argv
    assert str(tmp_path / "audio") in argv
    assert "--rerender_target" in argv
    for flag, value in (
        ("--param_spec", "surge_4"),
        ("--plugin_state_path", "presets/surge-mini.vstpreset"),
        ("--plugin_path", "plugins/Surge XT.vst3"),
        ("--sample_rate", "8000.0"),
        ("--channels", "2"),
        ("--velocity", "100"),
        ("--signal_duration_seconds", "0.1"),
    ):
        assert argv[argv.index(flag) + 1] == value


def test_render_argv_omits_unset_optional_fields(tmp_path: Path) -> None:
    """``None`` render fields stay off the argv so the CLI's defaults apply.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    settings = ProbeRenderSettings(param_spec_name="surge_4", plugin_state_path="p.vstpreset")

    with ExitStack() as stack:
        argv = _render_argv(tmp_path, settings, stack)

    for flag in ("--plugin_path", "--sample_rate", "--channels", "--velocity"):
        assert flag not in argv


@pytest.mark.skipif(sys.platform != "linux", reason="wrapper is prepended on Linux only")
def test_render_argv_prepends_headless_wrapper_on_linux(tmp_path: Path) -> None:
    """On Linux the Xvfb wrapper precedes the interpreter so the VST3 gets a display.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    with ExitStack() as stack:
        argv = _render_argv(tmp_path, _SETTINGS, stack)

        assert argv[0].endswith("run-linux-vst-headless.sh")
        assert argv[1] == sys.executable


def _fake_probe_subprocesses(
    probe_dir: Path, calls: list[tuple[list[str], dict[str, object]]], stderr: str | None = None
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Return a ``subprocess.run`` stand-in that materializes each stage's outputs.

    :param probe_dir: Probe directory whose ``audio/`` / ``metrics/`` outputs to fake.
    :param calls: Receives each invocation's ``(argv, kwargs)`` for later assertions.
    :param stderr: Stderr text attached to each successful completion.
    :returns: Callable with the ``subprocess.run`` signature the probe uses.
    """

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((list(argv), kwargs))
        if audio_probe._PREDICT_VST_AUDIO_MODULE in argv:
            sample_dir = probe_dir / "audio" / "sample_0"
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "pred.wav").write_bytes(b"RIFF")
            (sample_dir / "target.wav").write_bytes(b"RIFF")
        else:
            metrics_dir = probe_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            (metrics_dir / "aggregated_metrics.csv").write_text(
                ",mean,std\nmss,1.5,0.1\nrms,0.9,0.05\n"
            )
        return subprocess.CompletedProcess(argv, 0, stderr=stderr)

    return fake_run


def test_run_audio_probe_returns_namespaced_metrics_and_uploads(
    tmp_path: Path, fake_r2_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe renders, scores, uploads (excluding tensors), and returns val_audio/* keys.

    The subprocess stages are faked (each writes its real output files); the R2
    upload runs through the real rclone binary against the fake remote.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param fake_r2_remote: Backs ``r2:`` with the local filesystem.
    :param monkeypatch: Replaces the probe's ``subprocess.run``.
    """
    probe_dir = fake_r2_remote / "probe"
    _stage(probe_dir)
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(audio_probe.subprocess, "run", _fake_probe_subprocesses(probe_dir, calls))

    metrics = run_audio_probe(
        probe_dir, 5000, settings=_SETTINGS, upload_uri="r2://bucket/probes/run-1"
    )

    assert metrics == {
        "val_audio/mss_mean": 1.5,
        "val_audio/mss_std": 0.1,
        "val_audio/rms_mean": 0.9,
        "val_audio/rms_std": 0.05,
    }
    assert len(calls) == 2, "expected exactly a render call then a metrics call"
    assert audio_probe._PREDICT_VST_AUDIO_MODULE in calls[0][0]
    assert audio_probe._COMPUTE_AUDIO_METRICS_MODULE in calls[1][0]
    for _argv, kwargs in calls:
        # Contract pin for the failure-diagnosability fix (#1990): dropping either
        # kwarg would silently revert probe errors to stderr-less warnings.
        assert kwargs.get("stderr") is subprocess.PIPE
        assert kwargs.get("text") is True
        assert kwargs.get("errors") == "replace"

    landed = fake_r2_remote / "bucket" / "probes" / "run-1" / "step-5000"
    uploaded = sorted(p.relative_to(landed).as_posix() for p in landed.rglob("*") if p.is_file())
    assert "audio/sample_0/pred.wav" in uploaded
    assert "metrics/aggregated_metrics.csv" in uploaded
    assert not [p for p in uploaded if p.endswith(".pt")], (
        f"staged tensors must stay local, got {uploaded}"
    )


def test_run_audio_probe_skips_upload_when_uri_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``upload_uri=None`` keeps the probe local and never touches r2_io.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Replaces the probe's ``subprocess.run`` and upload helper.
    """
    _stage(tmp_path)
    monkeypatch.setattr(audio_probe.subprocess, "run", _fake_probe_subprocesses(tmp_path, []))

    def _fail_upload(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("upload_dir must not be called without an upload_uri")

    monkeypatch.setattr(audio_probe.r2_io, "upload_dir", _fail_upload)

    metrics = run_audio_probe(tmp_path, 1, settings=_SETTINGS, upload_uri=None)

    assert "val_audio/mss_mean" in metrics


def test_run_audio_probe_propagates_render_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing render subprocess surfaces as CalledProcessError for the caller to log.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Replaces the probe's ``subprocess.run`` with a failing one.
    """
    _stage(tmp_path)

    def failing_run(argv: list[str], **_kwargs: object) -> NoReturn:
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(audio_probe.subprocess, "run", failing_run)

    with pytest.raises(subprocess.CalledProcessError):
        run_audio_probe(tmp_path, 1, settings=_SETTINGS)


def test_run_audio_probe_render_failure_carries_subprocess_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing render's stderr rides the raised error so the caller can log it (#1990).

    The render argv is swapped for a tiny real command that dies with a traceback
    on stderr; ``run_audio_probe`` runs it through the real ``subprocess.run``.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Replaces the render argv builder.
    """
    _stage(tmp_path)
    argv = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('boom-traceback-marker'); sys.exit(1)",
    ]
    monkeypatch.setattr(audio_probe, "_render_argv", lambda *_args: argv)

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_audio_probe(tmp_path, 1, settings=_SETTINGS)

    assert "boom-traceback-marker" in (excinfo.value.stderr or "")


def test_run_audio_probe_render_failure_with_non_utf8_stderr_still_carries_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid bytes on stderr (e.g. a native VST crash) must not mask the diagnostic.

    With strict decoding the ``UnicodeDecodeError`` would replace the
    ``CalledProcessError`` this fix exists to enrich.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Replaces the render argv builder.
    """
    _stage(tmp_path)
    argv = [
        sys.executable,
        "-c",
        r"import sys; sys.stderr.buffer.write(b'\xff\xfe native-crash-marker'); sys.exit(1)",
    ]
    monkeypatch.setattr(audio_probe, "_render_argv", lambda *_args: argv)

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_audio_probe(tmp_path, 1, settings=_SETTINGS)

    assert "native-crash-marker" in (excinfo.value.stderr or "")


def test_run_audio_probe_metrics_failure_carries_subprocess_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing metrics subprocess's stderr rides the raised error, like the render stage's (#1990).

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Replaces the render argv builder and the metrics module name.
    """
    _stage(tmp_path)
    monkeypatch.setattr(audio_probe, "_render_argv", lambda *_args: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(
        audio_probe, "_COMPUTE_AUDIO_METRICS_MODULE", "synth_setter.nonexistent_probe_metrics"
    )

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_audio_probe(tmp_path, 1, settings=_SETTINGS)

    assert "No module named" in (excinfo.value.stderr or "")


@pytest.mark.slow
def test_run_audio_probe_render_timeout_carries_partial_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real stage timeout surfaces the child's pre-kill stderr on the raised error.

    CPython attaches the partial capture to ``TimeoutExpired`` as raw bytes even
    under ``text=True`` — the chain the warning path's bytes handling exists for.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Shrinks the render budget and replaces the argv builder.
    """
    _stage(tmp_path)
    monkeypatch.setattr(audio_probe, "RENDER_TIMEOUT_OVERHEAD_SECONDS", 0.5)
    monkeypatch.setattr(audio_probe, "RENDER_TIMEOUT_PER_SAMPLE_SECONDS", 0.0)
    argv = [
        sys.executable,
        "-c",
        "import sys, time; sys.stderr.write('pre-timeout-chatter'); "
        "sys.stderr.flush(); time.sleep(30)",
    ]
    monkeypatch.setattr(audio_probe, "_render_argv", lambda *_args: argv)

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_audio_probe(tmp_path, 1, settings=_SETTINGS)

    assert "pre-timeout-chatter" in str(excinfo.value.stderr)


def test_run_audio_probe_forwards_success_stderr_to_debug_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-fatal stderr chatter from a successful stage lands in the debug log, not nowhere.

    Capturing stderr for failure diagnosis (#1990) stops it streaming to the run
    log; on success the captured text must still be reachable for operators.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Replaces the probe's ``subprocess.run``.
    :param caplog: Pytest fixture capturing the debug records.
    """
    _stage(tmp_path)
    monkeypatch.setattr(
        audio_probe.subprocess,
        "run",
        _fake_probe_subprocesses(tmp_path, [], stderr="deprecation-chatter"),
    )

    with caplog.at_level("DEBUG", logger="synth_setter.evaluation.audio_probe"):
        run_audio_probe(tmp_path, 1, settings=_SETTINGS, upload_uri=None)

    assert "deprecation-chatter" in caplog.text


def test_run_audio_probe_success_stderr_debug_log_is_tail_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Oversized success-path chatter reaches the debug log tail-only, like failures.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param monkeypatch: Replaces the probe's ``subprocess.run``.
    :param caplog: Pytest fixture capturing the debug records.
    """
    _stage(tmp_path)
    chatter = "HEAD-marker\n" + "x" * 5000 + "\nTAIL-marker"
    monkeypatch.setattr(
        audio_probe.subprocess, "run", _fake_probe_subprocesses(tmp_path, [], stderr=chatter)
    )

    with caplog.at_level("DEBUG", logger="synth_setter.evaluation.audio_probe"):
        run_audio_probe(tmp_path, 1, settings=_SETTINGS, upload_uri=None)

    assert "TAIL-marker" in caplog.text
    assert "HEAD-marker" not in caplog.text
