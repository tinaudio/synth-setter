"""Fast unit tests for ``synth_setter.cli.eval._run_predict_postprocessing``.

Assert the render / metrics subprocess argv per gate and the user-facing
validation errors. ``subprocess.run`` is monkeypatched so no real VST plugin
or Python subprocess is launched.
"""

import logging
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import wandb
from omegaconf import DictConfig, OmegaConf

from synth_setter.cli import eval as eval_mod
from synth_setter.cli.eval import (
    _COMPUTE_AUDIO_METRICS_MODULE,
    _PREDICT_VST_AUDIO_MODULE,
    _log_audio_metrics_to_wandb,
    _log_metrics_csv_to_wandb,
    _log_shuffle_permutation_to_wandb,
    _run_predict_postprocessing,
)

_FAKE_WRAPPER = "/fake/vst-headless-wrapper"

_AGGREGATED_METRICS_CSV = ",mean,std\nmss,0.5,0.1\nwmfcc,0.3,0.05\nsot,0.2,0.02\nrms,0.9,0.01\n"
_AGGREGATED_METRICS_SHUFFLED_CSV = (
    ",mean,std\nmss,0.6,0.12\nwmfcc,0.35,0.06\nsot,0.25,0.025\nrms,0.85,0.015\n"
)

_EXPECTED_AUDIO_METRICS = {
    "audio/mss_mean": pytest.approx(0.5),
    "audio/mss_std": pytest.approx(0.1),
    "audio/wmfcc_mean": pytest.approx(0.3),
    "audio/wmfcc_std": pytest.approx(0.05),
    "audio/sot_mean": pytest.approx(0.2),
    "audio/sot_std": pytest.approx(0.02),
    "audio/rms_mean": pytest.approx(0.9),
    "audio/rms_std": pytest.approx(0.01),
}


def _build_postprocess_cfg(
    output_dir: Path,
    *,
    render_vst: bool = True,
    compute_metrics: bool = True,
    rerender_target: bool = True,
    num_workers: int = 1,
    shuffle_seed: int = 0,
    metric_prefix: str = "",
    render: dict[str, Any] | None = None,
) -> DictConfig:
    """Build a minimal cfg accepted by ``_run_predict_postprocessing``.

    :param output_dir: Resolves to ``cfg.paths.output_dir``; the helper derives
        ``predictions/`` / ``audio/`` / ``metrics/`` under it.
    :param render_vst: Drives ``cfg.evaluation.render_vst``.
    :param compute_metrics: Drives ``cfg.evaluation.compute_metrics``.
    :param rerender_target: Drives ``cfg.evaluation.rerender_target``.
    :param num_workers: Drives ``cfg.evaluation.num_workers``.
    :param shuffle_seed: Drives ``cfg.evaluation.shuffle_seed``.
    :param metric_prefix: Drives ``cfg.evaluation.metric_prefix``; prepended to
        every returned audio metric key.
    :param render: Drives ``cfg.render``; pass ``None`` to test the unset-render branch.
    :returns: Minimal :class:`DictConfig` shaped the way the helper reads it.
    """
    return OmegaConf.create(  # type: ignore[no-any-return]
        {
            "paths": {"output_dir": str(output_dir)},
            "evaluation": {
                "render_vst": render_vst,
                "compute_metrics": compute_metrics,
                "rerender_target": rerender_target,
                "num_workers": num_workers,
                "shuffle_seed": shuffle_seed,
                "metric_prefix": metric_prefix,
            },
            "render": render,
        }
    )


def _writes_metrics_csv(audio_metrics_dir: Path, csv_body: str) -> Any:
    """Build a fake ``subprocess.run`` that materializes the aggregated CSV before returning.

    :param audio_metrics_dir: Destination directory for the synthesized ``aggregated_metrics.csv``.
    :param csv_body: Raw CSV content the fake subprocess "wrote".
    :returns: A no-arg-compatible stand-in for ``subprocess.run`` used by ``monkeypatch``.
    """

    def _fake_run(args: list[str], **_kwargs: Any) -> None:
        if _COMPUTE_AUDIO_METRICS_MODULE not in args:
            return
        audio_metrics_dir.mkdir(parents=True, exist_ok=True)
        (audio_metrics_dir / "aggregated_metrics.csv").write_text(csv_body)

    return _fake_run


@pytest.fixture
def captured_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Capture every ``subprocess.run`` argv from the eval module without launching it.

    The metrics subprocess also materializes a placeholder ``aggregated_metrics.csv``
    under the metrics output dir (``<audio_dir>/../metrics``) so the load step in
    :func:`_run_predict_postprocessing` finds a CSV after the fake "subprocess" returns.

    :param monkeypatch: Used to stub ``subprocess.run``, ``as_file``, and
        ``vst_headless_wrapper`` so the helper builds argv without touching
        the real VST subprocess or package-data extraction.
    :returns: List that grows by one entry per intercepted ``subprocess.run`` call.
    """
    captured: list[list[str]] = []

    def _fake_run(args: list[str], **_kwargs: Any) -> None:
        captured.append(list(args))
        if _COMPUTE_AUDIO_METRICS_MODULE not in args:
            return
        metrics_dir = Path(args[args.index(_COMPUTE_AUDIO_METRICS_MODULE) + 2])
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "aggregated_metrics.csv").write_text(_AGGREGATED_METRICS_CSV)

    monkeypatch.setattr(eval_mod.subprocess, "run", _fake_run)

    # The render branch wraps the VST3 host with an Xvfb wrapper extracted from
    # package resources; stub the extractor so we don't touch the package data.
    @contextmanager
    def _fake_as_file(_traversable: Any) -> Any:
        yield Path(_FAKE_WRAPPER)

    monkeypatch.setattr(eval_mod, "as_file", _fake_as_file)
    monkeypatch.setattr(eval_mod, "vst_headless_wrapper", lambda: object())
    return captured


@pytest.fixture
def predictions_tree(tmp_path: Path) -> Path:
    """Create a ``predictions/`` and ``audio/`` subtree so existence guards pass.

    :param tmp_path: Pytest-provided per-test temporary directory; receives the
        ``predictions/`` and ``audio/`` children the helper checks for.
    :returns: ``tmp_path`` itself — used as ``cfg.paths.output_dir`` by callers.
    """
    (tmp_path / "predictions").mkdir()
    (tmp_path / "audio").mkdir()
    return tmp_path


def test_postprocessing_linux_argv_has_wrapper_prefix(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """On Linux the render argv starts with the Xvfb wrapper path.

    :param monkeypatch: Pins ``sys.platform`` to ``linux`` so the wrapper branch fires.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param captured_argv: Captured argv list populated by the fixture.
    """
    monkeypatch.setattr(eval_mod.sys, "platform", "linux")
    cfg = _build_postprocess_cfg(
        predictions_tree,
        compute_metrics=False,
        rerender_target=False,
        render={"param_spec_name": "surge/fake_oracle", "preset_path": "preset.fxp"},
    )

    _run_predict_postprocessing(cfg)

    assert len(captured_argv) == 1
    render_argv = captured_argv[0]
    assert render_argv[0] == _FAKE_WRAPPER
    assert render_argv[1:3] == [sys.executable, "-m"]
    assert render_argv[3] == "synth_setter.evaluation.predict_vst_audio"


def test_postprocessing_non_linux_argv_omits_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """Off-Linux platforms must invoke the python entrypoint directly.

    :param monkeypatch: Pins ``sys.platform`` to ``darwin`` so the wrapper branch is skipped.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param captured_argv: Captured argv list populated by the fixture.
    """
    monkeypatch.setattr(eval_mod.sys, "platform", "darwin")
    cfg = _build_postprocess_cfg(
        predictions_tree,
        compute_metrics=False,
        rerender_target=False,
        render={"param_spec_name": "surge/fake_oracle", "preset_path": "preset.fxp"},
    )

    _run_predict_postprocessing(cfg)

    render_argv = captured_argv[0]
    assert render_argv[0] == sys.executable
    assert render_argv[1] == "-m"
    assert _FAKE_WRAPPER not in render_argv


def test_postprocessing_plugin_path_gate(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """``cfg.render.plugin_path`` adds ``--plugin_path <value>`` to the render argv only when set.

    :param monkeypatch: Pins ``sys.platform`` to ``darwin`` so the headless wrapper
        prefix doesn't shift argv indices the test asserts on.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created
        so the helper's existence guards pass.
    :param captured_argv: List of every ``subprocess.run`` argv the helper would have
        spawned; populated by the fixture's monkeypatch.
    """
    monkeypatch.setattr(eval_mod.sys, "platform", "darwin")
    plugin_path = str(predictions_tree / "Surge XT.vst3")
    cfg = _build_postprocess_cfg(
        predictions_tree,
        compute_metrics=False,
        rerender_target=False,
        render={
            "param_spec_name": "surge/fake_oracle",
            "preset_path": "preset.fxp",
            "plugin_path": plugin_path,
        },
    )

    _run_predict_postprocessing(cfg)

    render_argv = captured_argv[0]
    assert "--plugin_path" in render_argv
    plugin_idx = render_argv.index("--plugin_path")
    assert render_argv[plugin_idx + 1] == plugin_path


def test_postprocessing_forwards_render_audio_fields(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """Render fields predict_vst_audio renders with are forwarded from ``cfg.render``.

    sample_rate / channels / velocity / signal_duration_seconds must reach the render
    argv so the re-render matches the dataset's generation render instead of
    predict_vst_audio's CLI defaults.

    :param monkeypatch: Pins ``sys.platform`` to ``darwin`` so the wrapper prefix
        doesn't shift the argv the test inspects.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param captured_argv: Captured argv list populated by the fixture.
    """
    monkeypatch.setattr(eval_mod.sys, "platform", "darwin")
    cfg = _build_postprocess_cfg(
        predictions_tree,
        compute_metrics=False,
        rerender_target=False,
        render={
            "param_spec_name": "surge/fake_oracle",
            "preset_path": "preset.fxp",
            "sample_rate": 22050,
            "channels": 1,
            "velocity": 64,
            "signal_duration_seconds": 2.5,
        },
    )

    _run_predict_postprocessing(cfg)

    render_argv = captured_argv[0]
    for flag, value in (
        ("--sample_rate", "22050"),
        ("--channels", "1"),
        ("--velocity", "64"),
        ("--signal_duration_seconds", "2.5"),
    ):
        assert flag in render_argv, f"{flag} not forwarded"
        assert render_argv[render_argv.index(flag) + 1] == value


def test_postprocessing_rerender_target_gate(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """``evaluation.rerender_target`` appends ``-t`` only when truthy.

    :param monkeypatch: Pins ``sys.platform`` to ``darwin`` so the wrapper branch is skipped.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param captured_argv: Captured argv list populated by the fixture.
    """
    monkeypatch.setattr(eval_mod.sys, "platform", "darwin")
    render = {"param_spec_name": "surge/fake_oracle", "preset_path": "preset.fxp"}

    _run_predict_postprocessing(
        _build_postprocess_cfg(
            predictions_tree, compute_metrics=False, rerender_target=True, render=render
        )
    )
    _run_predict_postprocessing(
        _build_postprocess_cfg(
            predictions_tree, compute_metrics=False, rerender_target=False, render=render
        )
    )

    assert "-t" in captured_argv[0]
    assert "-t" not in captured_argv[1]


def test_postprocessing_metrics_argv_includes_num_workers(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """The metrics subprocess receives ``-w <num_workers>`` from ``cfg.evaluation``.

    :param monkeypatch: Pins ``sys.platform`` to ``darwin`` so render-branch monkeypatches
        (still installed by the fixture) don't influence the metrics-only argv.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param captured_argv: Captured argv list populated by the fixture.
    """
    monkeypatch.setattr(eval_mod.sys, "platform", "darwin")
    cfg = _build_postprocess_cfg(
        predictions_tree,
        render_vst=False,
        compute_metrics=True,
        num_workers=4,
    )

    _run_predict_postprocessing(cfg)

    assert len(captured_argv) == 1
    metrics_argv = captured_argv[0]
    assert metrics_argv[:3] == [
        sys.executable,
        "-m",
        "synth_setter.evaluation.compute_audio_metrics",
    ]
    assert "-w" in metrics_argv
    w_idx = metrics_argv.index("-w")
    assert metrics_argv[w_idx + 1] == "4"


def test_postprocessing_always_forwards_shuffle_seed_to_metrics_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """``--shuffle_seed`` is always forwarded so callers control the permutation seed.

    :param monkeypatch: Pins ``sys.platform`` to ``darwin`` so the metrics-only argv is asserted.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param captured_argv: Captured argv list populated by the fixture.
    """
    monkeypatch.setattr(eval_mod.sys, "platform", "darwin")
    cfg = _build_postprocess_cfg(
        predictions_tree,
        render_vst=False,
        compute_metrics=True,
        shuffle_seed=7,
    )

    _run_predict_postprocessing(cfg)

    metrics_argv = captured_argv[0]
    assert "--shuffle_seed" in metrics_argv
    seed_idx = metrics_argv.index("--shuffle_seed")
    assert metrics_argv[seed_idx + 1] == "7"


def test_postprocessing_forwards_default_shuffle_seed_zero(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """Default ``shuffle_seed=0`` is forwarded so the probe is reproducible without config.

    :param monkeypatch: Pins ``sys.platform`` to ``darwin``.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param captured_argv: Captured argv list populated by the fixture.
    """
    monkeypatch.setattr(eval_mod.sys, "platform", "darwin")
    cfg = _build_postprocess_cfg(predictions_tree, render_vst=False, compute_metrics=True)

    _run_predict_postprocessing(cfg)

    metrics_argv = captured_argv[0]
    assert "--shuffle_seed" in metrics_argv
    seed_idx = metrics_argv.index("--shuffle_seed")
    assert metrics_argv[seed_idx + 1] == "0"


def test_postprocessing_compute_metrics_off_fires_no_subprocess(
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """No subprocess fires when ``compute_metrics`` is off regardless of other settings.

    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param captured_argv: Captured argv list — asserted empty (no subprocess fires).
    """
    cfg = _build_postprocess_cfg(
        predictions_tree,
        render_vst=False,
        compute_metrics=False,
        render=None,
    )

    _run_predict_postprocessing(cfg)

    assert captured_argv == []


def test_postprocessing_no_op_when_both_gates_off(
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """No subprocess fires when ``render_vst`` and ``compute_metrics`` are both off.

    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param captured_argv: Captured argv list — asserted empty.
    """
    cfg = _build_postprocess_cfg(
        predictions_tree, render_vst=False, compute_metrics=False, render=None
    )

    _run_predict_postprocessing(cfg)

    assert captured_argv == []


def test_postprocessing_render_requires_render_cfg(
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """``render_vst=True`` with ``cfg.render is None`` raises a directed ``ValueError``.

    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param captured_argv: Captured argv list — asserted empty (helper fails before dispatch).
    """
    cfg = _build_postprocess_cfg(predictions_tree, compute_metrics=False, render=None)

    with pytest.raises(ValueError, match="render config group"):
        _run_predict_postprocessing(cfg)
    assert captured_argv == []


def test_postprocessing_render_requires_predictions_dir(
    tmp_path: Path,
    captured_argv: list[list[str]],
) -> None:
    """Missing ``predictions/`` dir surfaces a callback-pointing ``ValueError``.

    :param tmp_path: Pytest temp dir used as ``output_dir`` without pre-creating subtrees.
    :param captured_argv: Captured argv list — asserted empty (helper fails before dispatch).
    """
    cfg = _build_postprocess_cfg(
        tmp_path,
        compute_metrics=False,
        render={"param_spec_name": "surge/fake_oracle", "preset_path": "preset.fxp"},
    )

    with pytest.raises(ValueError, match="PredictionWriter"):
        _run_predict_postprocessing(cfg)
    assert captured_argv == []


def test_postprocessing_metrics_requires_audio_dir(
    tmp_path: Path,
    captured_argv: list[list[str]],
) -> None:
    """Missing ``audio/`` dir surfaces a render-pointing ``ValueError`` for metrics-only runs.

    :param tmp_path: Pytest temp dir used as ``output_dir`` without pre-creating subtrees.
    :param captured_argv: Captured argv list — asserted empty (helper fails before dispatch).
    """
    cfg = _build_postprocess_cfg(tmp_path, render_vst=False, compute_metrics=True)

    with pytest.raises(ValueError, match="render_vst"):
        _run_predict_postprocessing(cfg)
    assert captured_argv == []


def test_postprocessing_returns_empty_when_compute_metrics_off(
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """Helper returns ``{}`` so callers can unconditionally merge into ``callback_metrics``.

    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param captured_argv: Captured argv list — asserted untouched after the no-op return.
    """
    cfg = _build_postprocess_cfg(
        predictions_tree,
        render_vst=False,
        compute_metrics=False,
        render=None,
    )

    assert _run_predict_postprocessing(cfg) == {}


def test_postprocessing_returns_loaded_audio_metrics(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
) -> None:
    """Helper returns the parsed audio metrics so ``evaluate()`` can stash them.

    :param monkeypatch: Replaces ``subprocess.run`` with a fake that writes the
        aggregated CSV before returning, mimicking the real subprocess effect.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    """
    metrics_dir = predictions_tree / "metrics"
    monkeypatch.setattr(
        eval_mod.subprocess,
        "run",
        _writes_metrics_csv(metrics_dir, _AGGREGATED_METRICS_CSV),
    )
    cfg = _build_postprocess_cfg(
        predictions_tree,
        render_vst=False,
        compute_metrics=True,
    )

    result = _run_predict_postprocessing(cfg)

    assert result == _EXPECTED_AUDIO_METRICS


def test_postprocessing_returns_shuffled_audio_metrics_when_subprocess_writes_shuffled_csv(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
) -> None:
    """``shuffled_audio/*`` keys are returned when the metrics subprocess writes both CSVs.

    Verifies the ``_load_audio_metrics`` shuffled-CSV branch is wired through
    ``_run_predict_postprocessing``; deleting that branch would cause this test to fail.

    :param monkeypatch: Replaces ``subprocess.run`` with a fake that writes both
        ``aggregated_metrics.csv`` and ``aggregated_metrics_shuffled.csv``.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    """
    metrics_dir = predictions_tree / "metrics"

    def _writes_both_csvs(args: list[str], **_kwargs: object) -> None:
        if _COMPUTE_AUDIO_METRICS_MODULE not in args:
            return
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "aggregated_metrics.csv").write_text(_AGGREGATED_METRICS_CSV)
        (metrics_dir / "aggregated_metrics_shuffled.csv").write_text(
            _AGGREGATED_METRICS_SHUFFLED_CSV
        )

    monkeypatch.setattr(eval_mod.subprocess, "run", _writes_both_csvs)
    cfg = _build_postprocess_cfg(predictions_tree, render_vst=False, compute_metrics=True)

    result = _run_predict_postprocessing(cfg)

    assert "shuffled_audio/mss_mean" in result
    assert result["shuffled_audio/mss_mean"] == pytest.approx(0.6)
    assert result["shuffled_audio/mss_std"] == pytest.approx(0.12)
    assert "audio/mss_mean" in result
    assert result["audio/mss_mean"] == pytest.approx(0.5)


def test_postprocessing_prefixes_audio_metric_keys_when_metric_prefix_set(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
) -> None:
    """``metric_prefix`` is prepended to every returned key so per-split runs stay distinct.

    The inline oracle eval resumes one wandb run for all splits; without a
    prefix the bare ``audio/<name>_<stat>`` summary key is overwritten by the
    last split. A non-empty prefix namespaces every key (both ``audio/*`` and
    ``shuffled_audio/*``), e.g. ``train/``.

    :param monkeypatch: Replaces ``subprocess.run`` with a fake that writes both
        aggregated CSVs before returning.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    """
    metrics_dir = predictions_tree / "metrics"

    def _writes_both_csvs(args: list[str], **_kwargs: Any) -> None:
        if _COMPUTE_AUDIO_METRICS_MODULE not in args:
            return
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "aggregated_metrics.csv").write_text(_AGGREGATED_METRICS_CSV)
        (metrics_dir / "aggregated_metrics_shuffled.csv").write_text(
            _AGGREGATED_METRICS_SHUFFLED_CSV
        )

    monkeypatch.setattr(eval_mod.subprocess, "run", _writes_both_csvs)
    cfg = _build_postprocess_cfg(
        predictions_tree,
        render_vst=False,
        compute_metrics=True,
        metric_prefix="train/",
    )

    result = _run_predict_postprocessing(cfg)

    # Both groups are namespaced — the prefix applies to every loaded key.
    assert result["train/audio/mss_mean"] == pytest.approx(0.5)
    assert result["train/shuffled_audio/mss_mean"] == pytest.approx(0.6)
    # The bare keys must be gone: that is exactly the cross-split collision the prefix fixes.
    assert "audio/mss_mean" not in result
    assert "shuffled_audio/mss_mean" not in result


def test_postprocessing_render_subprocess_nonzero_exit_raises(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """A non-zero render subprocess exit propagates ``CalledProcessError`` to the caller.

    :param monkeypatch: Re-patches ``subprocess.run`` (over the fixture's stub) so the
        render module's invocation raises, and pins ``sys.platform`` to ``darwin``.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param captured_argv: Captured argv list — unused here but installs the fixture's
        ``as_file`` / ``vst_headless_wrapper`` stubs the render branch needs.
    """
    monkeypatch.setattr(eval_mod.sys, "platform", "darwin")

    def _raise_on_render(args: list[str], **_kwargs: object) -> None:
        if _PREDICT_VST_AUDIO_MODULE in args:
            raise subprocess.CalledProcessError(returncode=1, cmd=args)

    monkeypatch.setattr(eval_mod.subprocess, "run", _raise_on_render)
    cfg = _build_postprocess_cfg(
        predictions_tree,
        compute_metrics=False,
        rerender_target=False,
        render={"param_spec_name": "surge/fake_oracle", "preset_path": "preset.fxp"},
    )

    with pytest.raises(subprocess.CalledProcessError):
        _run_predict_postprocessing(cfg)


def test_postprocessing_metrics_subprocess_timeout_raises(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
    captured_argv: list[list[str]],
) -> None:
    """A metrics subprocess timeout propagates ``TimeoutExpired`` to the caller.

    The render call no-ops; only the compute-metrics module's invocation raises, so the
    propagation is attributed to the metrics stage.

    :param monkeypatch: Re-patches ``subprocess.run`` (over the fixture's stub) so the
        metrics module's invocation times out, and pins ``sys.platform`` to ``darwin``.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created
        so the metrics branch's ``audio/`` guard passes.
    :param captured_argv: Captured argv list — unused here but installs the fixture's
        ``as_file`` / ``vst_headless_wrapper`` stubs the render branch needs.
    """
    monkeypatch.setattr(eval_mod.sys, "platform", "darwin")

    def _timeout_on_metrics(args: list[str], **_kwargs: object) -> None:
        if _COMPUTE_AUDIO_METRICS_MODULE in args:
            raise subprocess.TimeoutExpired(cmd=args, timeout=1)

    monkeypatch.setattr(eval_mod.subprocess, "run", _timeout_on_metrics)
    cfg = _build_postprocess_cfg(
        predictions_tree,
        compute_metrics=True,
        rerender_target=False,
        render={"param_spec_name": "surge/fake_oracle", "preset_path": "preset.fxp"},
    )

    with pytest.raises(subprocess.TimeoutExpired):
        _run_predict_postprocessing(cfg)


def test_log_audio_metrics_to_wandb_log_raises_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``wandb.run.log`` exception is swallowed so metrics still flow back to the caller.

    :param monkeypatch: Pins ``wandb.run`` to a fake whose ``.log`` raises ``RuntimeError``.
    """

    class _RaisingRun:
        def log(self, _payload: dict[str, float]) -> None:
            raise RuntimeError("wandb backend unavailable")

    monkeypatch.setattr(eval_mod.wandb, "run", _RaisingRun())

    assert _log_audio_metrics_to_wandb({"audio/mss_mean": 0.5}) is None


def test_postprocessing_logs_audio_metrics_to_active_wandb_run(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
) -> None:
    """When ``wandb.run`` is set, the loaded audio metrics are forwarded to ``run.log`` once.

    :param monkeypatch: Stubs ``subprocess.run`` so the CSV materializes without launching
        the real subprocess, and stubs ``wandb.run`` so logging is observable.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    """
    metrics_dir = predictions_tree / "metrics"
    monkeypatch.setattr(
        eval_mod.subprocess,
        "run",
        _writes_metrics_csv(metrics_dir, _AGGREGATED_METRICS_CSV),
    )

    logged: list[dict[str, float]] = []

    class _FakeRun:
        def log(self, payload: dict[str, float]) -> None:
            logged.append(payload)

    monkeypatch.setattr(eval_mod.wandb, "run", _FakeRun())

    cfg = _build_postprocess_cfg(
        predictions_tree,
        render_vst=False,
        compute_metrics=True,
    )

    _run_predict_postprocessing(cfg)

    assert logged == [_EXPECTED_AUDIO_METRICS]


def test_postprocessing_skips_wandb_when_no_run(
    monkeypatch: pytest.MonkeyPatch,
    predictions_tree: Path,
) -> None:
    """No active wandb run → metrics are still returned but ``wandb.log`` is not touched.

    :param monkeypatch: Stubs ``subprocess.run`` so the CSV materializes, and pins
        ``wandb.run`` to ``None`` so the no-run branch is exercised.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    """
    metrics_dir = predictions_tree / "metrics"
    monkeypatch.setattr(
        eval_mod.subprocess,
        "run",
        _writes_metrics_csv(metrics_dir, _AGGREGATED_METRICS_CSV),
    )

    monkeypatch.setattr(eval_mod.wandb, "run", None)

    cfg = _build_postprocess_cfg(
        predictions_tree,
        render_vst=False,
        compute_metrics=True,
    )

    result = _run_predict_postprocessing(cfg)

    assert result == _EXPECTED_AUDIO_METRICS


_METRICS_CSV = ",mss,wmfcc,sot,rms\n0,0.1,0.2,0.3,0.4\n1,0.5,0.6,0.7,0.8\n"


def test_log_metrics_csv_to_wandb_logs_table_to_active_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With an active ``wandb.run`` and ``metrics.csv`` present, ``run.log`` receives a Table.

    :param monkeypatch: Pins ``wandb.run`` to a spy so the log payload is observable.
    :param tmp_path: Scratch metrics dir seeded with a minimal ``metrics.csv``.
    """

    (tmp_path / "metrics.csv").write_text(_METRICS_CSV)

    logged: list[dict[str, object]] = []

    class _FakeRun:
        """Spy stand-in for ``wandb.run`` that records every ``log`` payload."""

        def log(self, payload: dict[str, object]) -> None:
            """Append payload to the captured log list.

            :param payload: The wandb log payload to capture.
            """
            logged.append(payload)

    monkeypatch.setattr(eval_mod.wandb, "run", _FakeRun())

    _log_metrics_csv_to_wandb(tmp_path)

    assert len(logged) == 1
    assert "audio/per_sample_metrics" in logged[0]
    assert isinstance(logged[0]["audio/per_sample_metrics"], wandb.Table)


def test_log_metrics_csv_to_wandb_prepends_prefix_to_table_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-empty ``prefix`` namespaces the Table key so per-split runs stay distinct.

    :param monkeypatch: Pins ``wandb.run`` to a spy so the log payload is observable.
    :param tmp_path: Scratch metrics dir seeded with a minimal ``metrics.csv``.
    """

    (tmp_path / "metrics.csv").write_text(_METRICS_CSV)

    logged: list[dict[str, object]] = []

    class _FakeRun:
        """Spy stand-in for ``wandb.run`` that records every ``log`` payload."""

        def log(self, payload: dict[str, object]) -> None:
            """Append payload to the captured log list.

            :param payload: The wandb log payload to capture.
            """
            logged.append(payload)

    monkeypatch.setattr(eval_mod.wandb, "run", _FakeRun())

    _log_metrics_csv_to_wandb(tmp_path, prefix="train/")

    assert len(logged) == 1
    assert "train/audio/per_sample_metrics" in logged[0]
    assert isinstance(logged[0]["train/audio/per_sample_metrics"], wandb.Table)


def test_log_metrics_csv_to_wandb_noop_when_no_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``wandb.run is None`` → function returns without calling ``wandb.run.log``.

    :param monkeypatch: Pins ``wandb.run`` to ``None`` to exercise the early-exit branch.
    :param tmp_path: Scratch metrics dir — ``metrics.csv`` is present but must not be read.
    """
    (tmp_path / "metrics.csv").write_text(_METRICS_CSV)
    monkeypatch.setattr(eval_mod.wandb, "run", None)

    _log_metrics_csv_to_wandb(tmp_path)


def test_log_metrics_csv_to_wandb_missing_file_is_silent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing ``metrics.csv`` is silently skipped; ``run.log`` is never called.

    :param monkeypatch: Pins ``wandb.run`` to a spy that would fail the test if called.
    :param tmp_path: Empty scratch dir — no ``metrics.csv`` present.
    """

    class _NeverCalledRun:
        """Sentinel stand-in for ``wandb.run`` that asserts it is never called."""

        def log(self, _payload: object) -> None:
            """Raise to fail the test if called — ``run.log`` must stay silent.

            :param _payload: Unused; any call is a test failure.
            :raises AssertionError: Always, to signal an unexpected ``run.log`` call.
            """
            raise AssertionError("run.log must not be called when metrics.csv is absent")

    monkeypatch.setattr(eval_mod.wandb, "run", _NeverCalledRun())

    _log_metrics_csv_to_wandb(tmp_path)


def test_log_metrics_csv_to_wandb_log_exception_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An exception from ``wandb.run.log`` is swallowed and a warning is emitted.

    :param monkeypatch: Pins ``wandb.run`` to a fake whose ``.log`` raises ``RuntimeError``.
    :param tmp_path: Scratch metrics dir seeded with a minimal ``metrics.csv``.
    :param caplog: Captures log output to verify the warning is emitted.
    """

    (tmp_path / "metrics.csv").write_text(_METRICS_CSV)

    class _RaisingRun:
        """Stand-in for ``wandb.run`` whose ``log`` always raises to simulate a backend failure."""

        def log(self, _payload: object) -> None:
            """Raise to simulate a failing wandb backend.

            :param _payload: Unused; always raises.
            :raises RuntimeError: Always, to simulate a wandb backend failure.
            """
            raise RuntimeError("wandb backend unavailable")

    monkeypatch.setattr(eval_mod.wandb, "run", _RaisingRun())

    with caplog.at_level(logging.WARNING):
        _log_metrics_csv_to_wandb(tmp_path)

    assert any("RuntimeError" in r.message for r in caplog.records)


_SHUFFLE_PERMUTATION_CSV = "dest_idx,src_idx\n0,1\n1,0\n"


class _RecordingWandbRun:
    """Spy stand-in for ``wandb.run`` that appends every ``log`` payload to ``payloads``."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def log(self, payload: dict[str, object]) -> None:
        """Record one ``wandb.run.log`` call's argument.

        :param payload: The dict passed to ``wandb.run.log``.
        """
        self.payloads.append(payload)


@pytest.fixture
def wandb_log_spy(monkeypatch: pytest.MonkeyPatch) -> _RecordingWandbRun:
    """Pin ``wandb.run`` to a fresh recording spy for the test's duration.

    :param monkeypatch: Applies and reverts the ``eval_mod.wandb.run`` patch.
    :returns: The installed spy, whose ``payloads`` collects every logged dict.
    """
    spy = _RecordingWandbRun()
    monkeypatch.setattr(eval_mod.wandb, "run", spy)
    return spy


def test_log_shuffle_permutation_to_wandb_logs_table_to_active_run(
    wandb_log_spy: _RecordingWandbRun,
    tmp_path: Path,
) -> None:
    """An active run plus a present ``shuffle_permutation.csv`` logs exactly one Table.

    :param wandb_log_spy: Recording spy pinned to ``wandb.run``.
    :param tmp_path: Scratch metrics dir seeded with a minimal ``shuffle_permutation.csv``.
    """
    (tmp_path / "shuffle_permutation.csv").write_text(_SHUFFLE_PERMUTATION_CSV)

    _log_shuffle_permutation_to_wandb(tmp_path)

    assert len(wandb_log_spy.payloads) == 1
    table = wandb_log_spy.payloads[0]["shuffle/permutation"]
    assert isinstance(table, wandb.Table)
    assert table.columns == ["dest_idx", "src_idx"]
    # Rows must round-trip the CSV verbatim, not just carry the right header.
    assert table.data == [[0, 1], [1, 0]]


def test_log_shuffle_permutation_to_wandb_prepends_prefix_to_table_key(
    wandb_log_spy: _RecordingWandbRun,
    tmp_path: Path,
) -> None:
    """A non-empty ``prefix`` namespaces the Table key so per-split runs stay distinct.

    :param wandb_log_spy: Recording spy pinned to ``wandb.run``.
    :param tmp_path: Scratch metrics dir seeded with a minimal ``shuffle_permutation.csv``.
    """
    (tmp_path / "shuffle_permutation.csv").write_text(_SHUFFLE_PERMUTATION_CSV)

    _log_shuffle_permutation_to_wandb(tmp_path, prefix="train/")

    assert len(wandb_log_spy.payloads) == 1
    assert isinstance(wandb_log_spy.payloads[0]["train/shuffle/permutation"], wandb.Table)


def test_log_shuffle_permutation_to_wandb_noop_when_no_run(
    wandb_log_spy: _RecordingWandbRun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``wandb.run is None`` → returns without raising or logging, even with the CSV present.

    Overriding the spy with ``None`` exercises the early-exit guard: removing it would
    dereference ``None.log`` and raise instead of logging nothing.

    :param wandb_log_spy: Recording spy whose ``payloads`` must stay empty.
    :param monkeypatch: Overrides ``wandb.run`` with ``None`` after the fixture installs the spy.
    :param tmp_path: Scratch dir — ``shuffle_permutation.csv`` is present but must not be read.
    """
    (tmp_path / "shuffle_permutation.csv").write_text(_SHUFFLE_PERMUTATION_CSV)
    monkeypatch.setattr(eval_mod.wandb, "run", None)

    _log_shuffle_permutation_to_wandb(tmp_path)

    assert wandb_log_spy.payloads == []


def test_log_shuffle_permutation_to_wandb_missing_file_is_silent(
    wandb_log_spy: _RecordingWandbRun,
    tmp_path: Path,
) -> None:
    """A missing ``shuffle_permutation.csv`` is skipped; nothing is logged.

    The probe writes the file only for uniform-params datasets, so its absence is the
    common non-oracle case and must not raise.

    :param wandb_log_spy: Recording spy pinned to ``wandb.run``.
    :param tmp_path: Empty scratch dir — no ``shuffle_permutation.csv`` present.
    """
    _log_shuffle_permutation_to_wandb(tmp_path)

    assert wandb_log_spy.payloads == []


def test_log_shuffle_permutation_to_wandb_log_exception_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An exception from ``wandb.run.log`` is swallowed and a warning is emitted.

    :param monkeypatch: Pins ``wandb.run`` to a fake whose ``.log`` raises ``RuntimeError``.
    :param tmp_path: Scratch metrics dir seeded with a minimal ``shuffle_permutation.csv``.
    :param caplog: Captures log output to verify the warning is emitted.
    """
    (tmp_path / "shuffle_permutation.csv").write_text(_SHUFFLE_PERMUTATION_CSV)

    class _RaisingRun:
        """Stand-in for ``wandb.run`` whose ``log`` always raises to simulate a backend failure."""

        def log(self, _payload: object) -> None:
            """Simulate a failing wandb backend.

            :param _payload: The would-be log dict; discarded before raising.
            :raises RuntimeError: Always, to simulate a wandb backend failure.
            """
            raise RuntimeError("wandb backend unavailable")

    monkeypatch.setattr(eval_mod.wandb, "run", _RaisingRun())

    with caplog.at_level(logging.WARNING):
        _log_shuffle_permutation_to_wandb(tmp_path)

    assert any("RuntimeError" in r.message for r in caplog.records)


def test_postprocessing_logs_shuffle_permutation_table_when_subprocess_writes_csv(
    wandb_log_spy: _RecordingWandbRun,
    predictions_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe permutation Table reaches ``wandb.run.log`` end-to-end through postprocessing.

    Wires the producer→consumer contract: when the metrics subprocess writes
    ``shuffle_permutation.csv``, ``_run_predict_postprocessing`` logs it as a Table under
    the ``shuffle/permutation`` key. Deleting the log call would fail this test.

    :param wandb_log_spy: Recording spy pinned to ``wandb.run``.
    :param predictions_tree: ``tmp_path`` with ``predictions/`` + ``audio/`` pre-created.
    :param monkeypatch: Replaces ``subprocess.run`` with a fake that writes the aggregated
        metrics CSV and the permutation CSV.
    """
    metrics_dir = predictions_tree / "metrics"

    def _writes_metrics_and_permutation(args: list[str], **_kwargs: object) -> None:
        if _COMPUTE_AUDIO_METRICS_MODULE not in args:
            return
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "aggregated_metrics.csv").write_text(_AGGREGATED_METRICS_CSV)
        (metrics_dir / "shuffle_permutation.csv").write_text(_SHUFFLE_PERMUTATION_CSV)

    monkeypatch.setattr(eval_mod.subprocess, "run", _writes_metrics_and_permutation)
    cfg = _build_postprocess_cfg(predictions_tree, render_vst=False, compute_metrics=True)

    _run_predict_postprocessing(cfg)

    permutation_payloads = [p for p in wandb_log_spy.payloads if "shuffle/permutation" in p]
    assert len(permutation_payloads) == 1
    assert isinstance(permutation_payloads[0]["shuffle/permutation"], wandb.Table)
