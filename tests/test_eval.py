"""Tests for the ``synth-setter-eval`` CLI entrypoint."""

import json
import math
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli import eval as eval_mod
from synth_setter.cli.eval import (
    _COMPUTE_AUDIO_METRICS_MODULE,
    _dump_metric_dict,
    _load_audio_metrics,
    _maybe_upload_output_dir,
    _run_predict_postprocessing,
    evaluate,
)
from synth_setter.cli.train import train
from synth_setter.data.vst import param_specs
from synth_setter.workspace import operator_workspace
from tests.helpers.run_if import RunIf


@pytest.mark.requires_vst
@pytest.mark.slow
def test_evaluate_runs_oracle_with_null_ckpt_path(
    tmp_path: Path,
    surge_xt_smoke_datasets: Path,
) -> None:
    """Fake oracle returns ``batch["params"]`` verbatim, so ``test/param_mse`` is exactly zero.

    The load-bearing invariant is that ``ckpt_path=null`` survives Hydra
    composition into ``evaluate()`` and the oracle's exact-zero MSE reaches
    the metric dict.

    :param tmp_path: Pinned as Hydra ``paths.output_dir`` / ``paths.log_dir``.
    :param surge_xt_smoke_datasets: Holds ``{train,val,test}.h5`` + ``stats.npz``.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/test-mps-fake-oracle",
                "trainer=cpu",
                # The experiment defaults to mode=predict; this invariant is test-mode.
                "mode=test",
                f"model.net.d_out={len(param_specs['surge_4'])}",
                "callbacks.log_per_param_mse.param_spec=surge_4",
            ],
        )

    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.dataset_root = str(surge_xt_smoke_datasets)
        cfg.datamodule.predict_file = str(surge_xt_smoke_datasets / "test.h5")
        cfg.datamodule.batch_size = 1
        cfg.datamodule.num_workers = 0
        cfg.ckpt_path = None

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    param_mse = metric_dict["test/param_mse"]
    assert isinstance(param_mse, torch.Tensor)
    assert param_mse.numel() == 1
    assert param_mse.dtype.is_floating_point
    assert torch.isfinite(param_mse), f"oracle test/param_mse must be finite; got {param_mse!r}"
    assert param_mse.item() == 0.0


def test_dump_metric_dict_writes_json_with_coerced_scalars(tmp_path: Path) -> None:
    """Lightning tensors and numpy arrays are coerced to native floats / lists in ``metrics.json``.

    Pins the artifact downstream gates (workflow asserter, CSV joiners) read from —
    a torch / numpy dependency in those gates would force imports just to deserialize.

    :param tmp_path: Hydra-style output dir; the ``metrics/`` subdir lands under it.
    """

    import numpy as np

    metric_dict = {
        "test/param_mse": torch.tensor(0.0),
        "test/per_param_mse": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "audio/mss_mean": np.float32(0.5),
        "raw/string": "v1",
    }
    out_path = _dump_metric_dict(metric_dict, tmp_path)

    assert out_path == tmp_path / "metrics" / "metrics.json"
    assert out_path.is_file()
    payload = json.loads(out_path.read_text())
    assert payload["test/param_mse"] == 0.0
    assert payload["test/per_param_mse"] == [0.0, 0.0, 0.0, 0.0]
    assert payload["audio/mss_mean"] == pytest.approx(0.5)
    assert payload["raw/string"] == "v1"


def _upload_cfg(output_dir: Path, upload_output_dir_uri: str | None) -> DictConfig:
    """Build the minimal cfg slice ``_maybe_upload_output_dir`` reads.

    :param output_dir: Resolves to ``cfg.paths.output_dir`` — the tree to copy.
    :param upload_output_dir_uri: Resolves to ``cfg.evaluation.upload_output_dir_uri``.
    :returns: A :class:`DictConfig` carrying only the two keys the helper reads.
    """
    return OmegaConf.create(  # type: ignore[no-any-return]
        {
            "paths": {"output_dir": str(output_dir)},
            "evaluation": {"upload_output_dir_uri": upload_output_dir_uri},
        }
    )


def test_maybe_upload_output_dir_noop_when_uri_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A null URI uploads nothing and never touches R2 credentials.

    :param monkeypatch: Stubs ``r2_io`` so any R2 call would be observable.
    :param tmp_path: Stands in for the output dir.
    """
    calls: list[str] = []
    monkeypatch.setattr(
        eval_mod.r2_io, "ensure_r2_env_loaded", lambda *a, **k: calls.append("env")
    )
    monkeypatch.setattr(eval_mod.r2_io, "upload_dir", lambda *a, **k: calls.append("upload"))

    _maybe_upload_output_dir(_upload_cfg(tmp_path, upload_output_dir_uri=None))

    assert calls == []


def test_maybe_upload_output_dir_uploads_tree_when_uri_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A set URI validates credentials, then uploads the output dir to that prefix.

    :param monkeypatch: Stubs ``r2_io`` so the credential check + upload are observable.
    :param tmp_path: Stands in for the output dir passed to ``upload_dir``.
    """
    order: list[str] = []
    monkeypatch.setattr(
        eval_mod.r2_io, "ensure_r2_env_loaded", lambda *a, **k: order.append("env")
    )
    captured: dict[str, object] = {}

    def _fake_upload_dir(local_dir: Path, r2_uri: str) -> None:
        order.append("upload")
        captured["local_dir"] = local_dir
        captured["r2_uri"] = r2_uri

    monkeypatch.setattr(eval_mod.r2_io, "upload_dir", _fake_upload_dir)

    _maybe_upload_output_dir(_upload_cfg(tmp_path, "r2://bucket/evals/run-1"))

    assert order == ["env", "upload"]
    assert captured["local_dir"] == tmp_path
    assert captured["r2_uri"] == "r2://bucket/evals/run-1"


@pytest.mark.requires_vst
@pytest.mark.slow
def test_eval_cli_downloads_dataset_from_r2_then_scores_oracle(
    tmp_path: Path, surge_xt_smoke_datasets: Path
) -> None:
    """End-to-end through the ``synth-setter-eval`` CLI: R2 prefetch then oracle scoring.

    No in-process shortcuts and no mocks — the real entrypoint runs with real
    ``rclone`` (local-backed remote). A dataset staged under an ``r2://`` prefix is
    downloaded into an initially-absent ``data.dataset_root``, and the fake oracle's
    exact-zero ``test/param_mse`` reaches ``metrics.json``. Proves the new
    ``data.download_dataset_root_uri`` gate composes with eval through ``main``.

    :param tmp_path: Root for the fake R2 remote, the download target, and the output dir.
    :param surge_xt_smoke_datasets: Source ``{train,val,test}.h5`` + ``stats.npz``.
    """
    if shutil.which("rclone") is None:
        pytest.skip("rclone binary not available on PATH")

    remote_root = tmp_path / "r2"
    staged = remote_root / "intermediate-data" / "dataset"
    staged.mkdir(parents=True)
    splits_and_stats = ("train.h5", "val.h5", "test.h5", "stats.npz")
    for name in splits_and_stats:
        shutil.copy(surge_xt_smoke_datasets / name, staged / name)

    dataset_root = tmp_path / "downloaded"
    output_dir = tmp_path / "out"

    env = {
        **os.environ,
        "RCLONE_CONFIG_R2_TYPE": "local",
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": "stub",
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "stub",
        "RCLONE_CONFIG_R2_ENDPOINT": "stub",
    }
    proc = subprocess.run(  # noqa: S603 — controlled argv
        [
            sys.executable,
            "-m",
            "synth_setter.cli.eval",
            "experiment=surge/test-mps-fake-oracle",
            "trainer=cpu",
            "mode=test",
            # render defaults to null and is read only in mode=predict's
            # postprocessing, so mode=test needs no render group.
            "hydra.job.chdir=false",
            f"model.net.d_out={len(param_specs['surge_4'])}",
            "callbacks.log_per_param_mse.param_spec=surge_4",
            "datamodule.download_dataset_root_uri=r2://intermediate-data/dataset",
            f"datamodule.dataset_root={dataset_root}",
            f"datamodule.predict_file={dataset_root}/test.h5",
            "datamodule.batch_size=1",
            "datamodule.num_workers=0",
            "ckpt_path=null",
            f"paths.output_dir={output_dir}",
            f"hydra.run.dir={output_dir}",
        ],
        cwd=remote_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    for name in splits_and_stats:
        assert (dataset_root / name).is_file(), f"{name} was not downloaded from R2"

    metrics = json.loads((output_dir / "metrics" / "metrics.json").read_text())
    assert metrics["test/param_mse"] == 0.0


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

_AGGREGATED_METRICS_CSV = ",mean,std\nmss,0.5,0.1\nwmfcc,0.3,0.05\nsot,0.2,0.02\nrms,0.9,0.01\n"

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


def test_load_audio_metrics_flattens_mean_and_std(tmp_path: Path) -> None:
    """``aggregated_metrics.csv`` becomes a flat ``audio/<name>_<stat>`` float dict.

    :param tmp_path: Scratch metrics dir seeded with the fixture CSV.
    """
    (tmp_path / "aggregated_metrics.csv").write_text(_AGGREGATED_METRICS_CSV)

    metrics = _load_audio_metrics(tmp_path)

    assert metrics == _EXPECTED_AUDIO_METRICS


def test_load_audio_metrics_returns_python_floats(tmp_path: Path) -> None:
    """Values are plain ``float`` — protects downstream wandb / Lightning logs from numpy scalars.

    :param tmp_path: Scratch metrics dir seeded with the fixture CSV.
    """
    (tmp_path / "aggregated_metrics.csv").write_text(_AGGREGATED_METRICS_CSV)

    metrics = _load_audio_metrics(tmp_path)

    assert all(type(value) is float for value in metrics.values())


def test_load_audio_metrics_missing_csv_raises(tmp_path: Path) -> None:
    """Missing aggregated CSV surfaces a directed FileNotFoundError naming the subprocess.

    :param tmp_path: Used as a metrics dir intentionally left empty to trigger the guard.
    """
    with pytest.raises(
        FileNotFoundError,
        match=r"aggregated_metrics\.csv.*compute_audio_metrics.*did not write",
    ):
        _load_audio_metrics(tmp_path)


def test_load_audio_metrics_missing_stat_column_raises(tmp_path: Path) -> None:
    """A CSV lacking a required stat column surfaces a directed ValueError naming the gap.

    :param tmp_path: Scratch metrics dir seeded with a mean-only CSV.
    """
    (tmp_path / "aggregated_metrics.csv").write_text(",mean\nmss,0.5\n")

    with pytest.raises(ValueError, match=r"missing required stat columns \['std'\]"):
        _load_audio_metrics(tmp_path)


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
