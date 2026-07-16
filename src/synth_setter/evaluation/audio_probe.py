"""Render, score, and archive a staged sample of validation predictions.

Reuses the eval chain verbatim — ``predict_vst_audio`` then
``compute_audio_metrics`` — against a probe directory laid out the way
``PredictionWriter`` writes one. Runs on a worker thread off the training loop
(see :class:`synth_setter.utils.callbacks.ValAudioProbe`), so every step here
waits on a child process rather than holding the GIL.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch

from synth_setter.evaluation.compute_audio_metrics import load_aggregated_metrics
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.subprocess_stream import STDERR_TAIL_CHARS, scaled_timeout
from synth_setter.resources import as_file, vst_headless_wrapper

log = logging.getLogger(__name__)

_PREDICT_VST_AUDIO_MODULE = "synth_setter.evaluation.predict_vst_audio"
_COMPUTE_AUDIO_METRICS_MODULE = "synth_setter.evaluation.compute_audio_metrics"

# Postprocessing budgets shared with cli/eval.py: timeouts scale with sample
# count so a slow render host can't spuriously trip them. See scaled_timeout.
RENDER_TIMEOUT_OVERHEAD_SECONDS = 300.0
RENDER_TIMEOUT_PER_SAMPLE_SECONDS = 60.0
METRICS_TIMEOUT_OVERHEAD_SECONDS = 180.0
METRICS_TIMEOUT_PER_SAMPLE_SECONDS = 30.0

# Raw prediction tensors stay local; the R2 snapshot is for listening to.
_UPLOAD_EXCLUDE = "predictions/**"

PREDICTIONS_DIRNAME = "predictions"
AUDIO_DIRNAME = "audio"
METRICS_DIRNAME = "metrics"


@dataclass(frozen=True, kw_only=True)
class ProbeRenderSettings:
    """Render fields forwarded to ``predict_vst_audio``.

    Mirrors the ``render`` config group so the probe re-renders the way the dataset
    was generated rather than falling back to the CLI's own defaults. ``None``
    means "leave the CLI default", matching eval's gated forwarding.

    .. attribute :: param_spec_name

        Registry key naming the spec predictions decode against.

    .. attribute :: plugin_state_path

        Preset applied before each render.

    .. attribute :: plugin_path

        VST3 bundle path; ``None`` uses the CLI default.

    .. attribute :: sample_rate

        Render sample rate in Hz.

    .. attribute :: channels

        Output channel count.

    .. attribute :: velocity

        MIDI note velocity.

    .. attribute :: signal_duration_seconds

        Rendered clip length in seconds.
    """

    param_spec_name: str
    plugin_state_path: str
    plugin_path: str | None = None
    sample_rate: float | None = None
    channels: int | None = None
    velocity: int | None = None
    signal_duration_seconds: float | None = None


def _run_captured(argv: list[str], stage: str, timeout: float) -> None:  # noqa: DOC502 — raised by subprocess.run
    """Run one probe stage with stderr captured for failure diagnosis (#1990).

    A failure's ``CalledProcessError``/``TimeoutExpired`` carries the child's
    stderr for the caller's warning; ``errors="replace"`` keeps invalid bytes
    from masking that diagnostic with a ``UnicodeDecodeError``. A successful
    stage's chatter goes to the debug log, tail-capped, so it is not silently
    discarded.

    :param argv: Stage command line.
    :param stage: Label naming the subprocess stage (``render`` / ``metrics``).
    :param timeout: Stage budget in seconds.
    :raises subprocess.CalledProcessError: on a non-zero stage exit.
    :raises subprocess.TimeoutExpired: when the stage exceeds ``timeout``.
    """
    result = subprocess.run(  # noqa: S603
        argv, check=True, stderr=subprocess.PIPE, text=True, errors="replace", timeout=timeout
    )
    if result.stderr:
        log.debug("val audio probe: %s stderr tail\n%s", stage, result.stderr[-STDERR_TAIL_CHARS:])


def _staged_sample_count(probe_dir: Path) -> int:
    """Return the number of prediction rows staged in ``probe_dir``.

    :param probe_dir: Probe directory holding ``predictions/pred-0.pt``.
    :returns: Row count of the staged prediction tensor.
    """
    pred = torch.load(probe_dir / PREDICTIONS_DIRNAME / "pred-0.pt", weights_only=True)
    return int(pred.shape[0])


def _render_argv(probe_dir: Path, settings: ProbeRenderSettings, stack: ExitStack) -> list[str]:
    """Build the ``predict_vst_audio`` argv, Xvfb-wrapped on Linux.

    The wrapper must precede the interpreter so the VST3 has a display before
    pedalboard imports it.

    :param probe_dir: Probe directory to read predictions from and write audio to.
    :param settings: Render fields to forward.
    :param stack: Exit stack keeping the materialized wrapper script alive.
    :returns: Argv list for :func:`subprocess.run`.
    """
    argv: list[str] = []
    if sys.platform == "linux":
        argv.append(str(Path(stack.enter_context(as_file(vst_headless_wrapper())))))
    argv += [
        sys.executable,
        "-m",
        _PREDICT_VST_AUDIO_MODULE,
        str(probe_dir / PREDICTIONS_DIRNAME),
        str(probe_dir / AUDIO_DIRNAME),
        "--param_spec",
        settings.param_spec_name,
        "--plugin_state_path",
        settings.plugin_state_path,
        # Training val batches carry no raw audio, so the target is re-rendered
        # from the staged target-params.
        "--rerender_target",
    ]
    optional = (
        ("--plugin_path", settings.plugin_path),
        ("--sample_rate", settings.sample_rate),
        ("--channels", settings.channels),
        ("--velocity", settings.velocity),
        ("--signal_duration_seconds", settings.signal_duration_seconds),
    )
    for flag, value in optional:
        if value is not None:
            argv += [flag, str(value)]
    return argv


def run_audio_probe(  # noqa: DOC502 — raised by the subprocess.run calls
    probe_dir: Path,
    step: int,
    *,
    settings: ProbeRenderSettings,
    upload_uri: str | None = None,
    num_workers: int = 2,
) -> dict[str, float]:
    """Render the staged predictions, score them, archive them, and return the metrics.

    :param probe_dir: Directory holding a ``predictions/`` subdir in
        ``PredictionWriter`` layout; ``audio/`` and ``metrics/`` are written beside it.
    :param step: Training ``global_step`` the predictions were staged at; names the
        R2 snapshot prefix.
    :param settings: Render fields forwarded to ``predict_vst_audio``.
    :param upload_uri: ``r2://`` prefix to archive the snapshot under; ``None`` skips
        the upload and keeps the probe local.
    :param num_workers: ``compute_audio_metrics`` process-pool width.
    :returns: ``{"val_audio/<metric>_<stat>": value}`` for the rendered samples.
    :raises subprocess.CalledProcessError: propagated from a non-zero subprocess exit.
    :raises subprocess.TimeoutExpired: propagated when a stage exceeds its budget.
    :raises FileNotFoundError: when the metrics stage exits 0 without writing its CSV.
    :raises ValueError: when the metrics CSV is missing a required stat column.
    """
    n_samples = _staged_sample_count(probe_dir)

    with ExitStack() as stack:
        argv = _render_argv(probe_dir, settings, stack)
        log.info("val audio probe: rendering %s samples at step %s", n_samples, step)
        _run_captured(
            argv,
            "render",
            # 2× samples: --rerender_target renders both pred and target per sample.
            timeout=scaled_timeout(
                n_samples * 2,
                overhead_seconds=RENDER_TIMEOUT_OVERHEAD_SECONDS,
                per_sample_seconds=RENDER_TIMEOUT_PER_SAMPLE_SECONDS,
            ),
        )

    metrics_dir = probe_dir / METRICS_DIRNAME
    _run_captured(
        [
            sys.executable,
            "-m",
            _COMPUTE_AUDIO_METRICS_MODULE,
            str(probe_dir / AUDIO_DIRNAME),
            str(metrics_dir),
            "-w",
            str(num_workers),
        ],
        "metrics",
        timeout=scaled_timeout(
            n_samples,
            workers=num_workers,
            overhead_seconds=METRICS_TIMEOUT_OVERHEAD_SECONDS,
            per_sample_seconds=METRICS_TIMEOUT_PER_SAMPLE_SECONDS,
        ),
    )

    if upload_uri is not None:
        destination = f"{upload_uri.rstrip('/')}/step-{step}"
        log.info("val audio probe: uploading snapshot to %s", destination)
        r2_io.upload_dir(probe_dir, destination, exclude=_UPLOAD_EXCLUDE)

    return {
        f"val_audio/{name}": value
        for name, value in load_aggregated_metrics(metrics_dir / "aggregated_metrics.csv").items()
    }
