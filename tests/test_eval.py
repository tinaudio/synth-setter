"""Tests for the ``synth-setter-eval`` CLI entrypoint."""

import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli import eval as eval_mod
from synth_setter.cli.eval import _run_predict_postprocessing, evaluate
from synth_setter.cli.train import train
from tests.helpers.run_if import RunIf


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Train for 1 epoch with ``train.py`` then evaluate the resulting checkpoint with ``eval.py``.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.test = True
    with open_dict(cfg_eval):
        cfg_eval.trainer.accelerator = "gpu"

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert math.isfinite(test_metric_dict["test/loss"].item())
    assert (
        abs(train_metric_dict["test/loss"].item() - test_metric_dict["test/loss"].item()) < 0.001
    )


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_validate(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Train one epoch then validate the checkpoint via ``eval.py`` ``mode=validate``.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.test = False
    with open_dict(cfg_eval):
        cfg_eval.trainer.accelerator = "gpu"

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_eval.mode = "validate"

    HydraConfig().set_config(cfg_eval)
    val_metric_dict, _ = evaluate(cfg_eval)

    assert math.isfinite(val_metric_dict["val/loss"].item())
    assert abs(train_metric_dict["val/loss"].item() - val_metric_dict["val/loss"].item()) < 0.001


# ---------------------------------------------------------------------------
# Fast unit tests for ``_run_predict_postprocessing`` — assert argv per gate
# and the user-facing validation errors. Monkeypatches ``subprocess.run`` so
# no real VST plugin or Python subprocess is launched.
# ---------------------------------------------------------------------------


_FAKE_WRAPPER = "/fake/vst-headless-wrapper"


def _build_postprocess_cfg(
    output_dir: Path,
    *,
    render_vst: bool = True,
    compute_metrics: bool = True,
    rerender_target: bool = True,
    num_workers: int = 1,
    render: dict[str, Any] | None = None,
) -> DictConfig:
    """Build a minimal cfg accepted by ``_run_predict_postprocessing``.

    :param output_dir: Resolves to ``cfg.paths.output_dir``; the helper derives
        ``predictions/`` / ``audio/`` / ``metrics/`` under it.
    :param render_vst: Drives ``cfg.evaluation.render_vst``.
    :param compute_metrics: Drives ``cfg.evaluation.compute_metrics``.
    :param rerender_target: Drives ``cfg.evaluation.rerender_target``.
    :param num_workers: Drives ``cfg.evaluation.num_workers``.
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
            },
            "render": render,
        }
    )


@pytest.fixture
def captured_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Capture every ``subprocess.run`` argv from the eval module without launching it.

    :param monkeypatch: Used to stub ``subprocess.run``, ``as_file``, and
        ``vst_headless_wrapper`` so the helper builds argv without touching
        the real VST subprocess or package-data extraction.
    :returns: List that grows by one entry per intercepted ``subprocess.run`` call.
    """
    captured: list[list[str]] = []

    def _fake_run(args: list[str], **_kwargs: Any) -> None:
        captured.append(list(args))

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
    assert metrics_argv[-2:] == ["-w", "4"]


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
