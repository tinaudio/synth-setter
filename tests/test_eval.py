"""Tests for the ``synth-setter-eval`` CLI entrypoint.

Every test composes a Hydra ``cfg`` and drives the in-process ``evaluate(cfg)``
entrypoint. Helper-level unit tests live in the sibling ``test_eval_*`` modules:
postprocessing argv in ``test_eval_postprocessing``, metric IO in
``test_eval_metrics``, and R2 upload / CLI e2e in ``test_eval_upload``.
``tests/_meta/test_entrypoint_test_modules.py`` enforces that no private
``synth_setter.cli`` helper is imported here.
"""

import math
import os
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from synth_setter.cli.eval import evaluate
from synth_setter.cli.train import train
from synth_setter.data.vst import param_specs, preset_paths
from synth_setter.workspace import operator_workspace
from tests.helpers.run_if import RunIf

# Public worker module names the predict-postprocessing subprocesses run; matched
# as argv substrings so the fake ``subprocess.run`` can route render vs metrics
# without importing eval.py's private ``_*_MODULE`` constants (forbidden here by
# ``tests/_meta/test_entrypoint_test_modules.py``).
_PREDICT_VST_AUDIO_FRAGMENT = "predict_vst_audio"
_COMPUTE_AUDIO_METRICS_FRAGMENT = "compute_audio_metrics"

# Aggregated audio-metrics CSV the fake metrics subprocess writes; one row per
# metric, columns mean/std — the shape ``_load_audio_metrics`` flattens.
_FAKE_AGGREGATED_METRICS_CSV = (
    ",mean,std\nmss,0.5,0.1\nwmfcc,0.3,0.05\nsot,0.2,0.02\nrms,0.9,0.01\n"
)


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


def _compose_fake_oracle_eval_cfg(
    tmp_path: Path,
    dataset_root: Path,
    *,
    mode: str,
    param_spec_name: str = "surge_4",
) -> DictConfig:
    """Compose ``eval.yaml`` with the CPU ``surge/fake_oracle`` experiment, pinned to a dataset.

    Drives the CPU production oracle config (``experiment/surge/fake_oracle.yaml``)
    rather than its MPS smoke sibling, so this composition is itself coverage of
    that config. The ``render`` group is set inline to ``param_spec_name`` so the
    oracle's ``${render.param_spec_name}``-keyed per-param-MSE callback matches the
    rendered dataset's encoding width.

    :param tmp_path: Pinned as ``paths.output_dir`` / ``paths.log_dir``; the
        predict-mode ``PredictionWriter`` writes ``predictions/`` beneath it.
    :param dataset_root: Holds ``{train,val,test}.h5`` + ``stats.npz``.
    :param mode: ``cfg.mode`` under test (``test`` / ``validate`` / ``val`` /
        ``predict`` / an unknown spelling).
    :param param_spec_name: Param spec the dataset was rendered with; drives the
        inline ``render`` group and the per-param-MSE callback's spec.
    :returns: Composed eval ``DictConfig`` ready for ``evaluate``.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=["experiment=surge/fake_oracle", f"mode={mode}"],
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.dataset_root = str(dataset_root)
        cfg.datamodule.predict_file = str(dataset_root / "test.h5")
        cfg.datamodule.batch_size = 1
        cfg.datamodule.num_workers = 0
        cfg.datamodule.use_saved_mean_and_variance = True
        cfg.ckpt_path = None
        # surge/base disables validation (limit_val_batches=0) since fake_oracle is
        # a predict-mode experiment; re-enable it so mode=val/validate actually runs.
        cfg.trainer.limit_val_batches = 1.0
        # Render group is null on fake_oracle; set it inline to the dataset's spec
        # so render_vst has a config and the per-param-MSE labels line up.
        cfg.render = {
            "param_spec_name": param_spec_name,
            "preset_path": str(preset_paths[param_spec_name]),
            "plugin_path": "plugins/fake.vst3",
        }
    return cfg


def _fake_postprocessing_subprocess(
    audio_metrics_csv: str,
) -> Callable[[list[str]], None]:
    """Build a fake ``subprocess.run`` that materializes the render/metrics outputs.

    Routes on the worker module name in argv: the render call creates the ``audio/``
    output dir (its second positional) so the metrics branch's existence check
    passes; the metrics call writes ``aggregated_metrics.csv`` under its output dir.
    No real VST or Python subprocess is launched.

    :param audio_metrics_csv: CSV body the fake metrics subprocess writes.
    :returns: A ``subprocess.run``-compatible callable for ``monkeypatch.setattr``.
    """

    def _fake_run(args: list[str], **_kwargs: object) -> None:
        is_render = any(_PREDICT_VST_AUDIO_FRAGMENT in a for a in args)
        is_metrics = any(_COMPUTE_AUDIO_METRICS_FRAGMENT in a for a in args)
        if not (is_render or is_metrics):
            return
        # Both worker argvs are ``[... -m <module> <in_dir> <out_dir> ...]``; the
        # output dir is 3 past ``-m`` (module, in_dir, out_dir), robust to the
        # Linux headless-wrapper prefix that shifts the leading entries.
        out_dir = Path(args[args.index("-m") + 3])
        out_dir.mkdir(parents=True, exist_ok=True)
        if is_metrics:
            (out_dir / "aggregated_metrics.csv").write_text(audio_metrics_csv)

    return _fake_run


@pytest.mark.fake_vst
def test_evaluate_predict_mode_merges_audio_metrics_into_metric_dict(
    tmp_path: Path,
    fake_surge_smoke_datasets: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mode=predict`` runs the oracle's predict + postprocessing and merges audio metrics.

    Exercises the predict branch of ``evaluate`` end-to-end on the fast loop: the
    fake-plugin dataset feeds ``trainer.predict`` (the ``PredictionWriter`` writes
    ``predictions/``), then the render + metrics subprocesses are faked so the
    aggregated ``audio/*`` values land in ``trainer.callback_metrics`` via
    ``metric_dict.update(audio_metrics)``. Pins that the rank-zero gate fires and
    the float audio metrics reach the returned dict alongside any tensor metrics.

    :param tmp_path: Hydra ``output_dir``; ``predictions/`` / ``audio/`` / ``metrics/``
        are derived beneath it.
    :param fake_surge_smoke_datasets: CPU-fast surge_4 dataset (no real VST).
    :param monkeypatch: Stubs the render/metrics subprocesses and the headless
        wrapper extraction so no real VST host or Python subprocess launches.
    """
    cfg = _compose_fake_oracle_eval_cfg(tmp_path, fake_surge_smoke_datasets, mode="predict")
    monkeypatch.setattr(
        "synth_setter.cli.eval.subprocess.run",
        _fake_postprocessing_subprocess(_FAKE_AGGREGATED_METRICS_CSV),
    )
    monkeypatch.setattr("synth_setter.cli.eval.vst_headless_wrapper", lambda: object())
    monkeypatch.setattr(
        "synth_setter.cli.eval.as_file",
        lambda _traversable: nullcontext(Path("/fake/headless-wrapper")),
    )

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert metric_dict["audio/mss_mean"] == pytest.approx(0.5)
    assert metric_dict["audio/rms_std"] == pytest.approx(0.01)
    for key in ("mss", "wmfcc", "sot", "rms"):
        for stat in ("mean", "std"):
            value = metric_dict[f"audio/{key}_{stat}"]
            assert isinstance(value, float) and math.isfinite(value)


@pytest.mark.fake_vst
def test_evaluate_validate_mode_legacy_val_spelling_runs_oracle(
    tmp_path: Path,
    fake_surge_smoke_datasets: Path,
) -> None:
    """``mode=val`` (legacy spelling) routes to ``trainer.validate`` and logs zero MSE.

    The ``evaluate`` mode branch accepts both ``val`` and ``validate``; only
    ``validate`` is otherwise covered (the GPU train→validate test). This pins the
    backward-compatible ``val`` alias on the fast loop: the oracle returns params
    verbatim, so ``val/param_mse`` is exactly zero.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``.
    :param fake_surge_smoke_datasets: CPU-fast surge_4 dataset (no real VST).
    """
    cfg = _compose_fake_oracle_eval_cfg(tmp_path, fake_surge_smoke_datasets, mode="val")

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    param_mse = metric_dict["val/param_mse"]
    assert isinstance(param_mse, torch.Tensor)
    assert param_mse.item() == 0.0


@pytest.mark.fake_vst
def test_evaluate_unknown_mode_returns_only_callback_metrics(
    tmp_path: Path,
    fake_surge_smoke_datasets: Path,
) -> None:
    """An unrecognized ``mode`` runs no trainer stage and returns the empty callback metrics.

    ``evaluate`` has no ``else``/raise on its mode branch: an unknown spelling is a
    silent no-op that skips test/validate/predict, so ``trainer.callback_metrics``
    is empty and no ``audio/*`` postprocessing runs. Pins that contract so a typo'd
    mode fails visibly (empty metrics) rather than masquerading as a passing run.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``.
    :param fake_surge_smoke_datasets: CPU-fast surge_4 dataset (no real VST).
    """
    cfg = _compose_fake_oracle_eval_cfg(tmp_path, fake_surge_smoke_datasets, mode="bogus-mode")

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    assert metric_dict == {}


@pytest.mark.parametrize("render_group", ["surge_simple", "surge_xt"])
def test_eval_render_group_exposes_postprocessing_keys(render_group: str) -> None:
    """Composing ``render=<group>`` into eval exposes the three keys postprocessing reads.

    ``_run_predict_postprocessing`` reads ``cfg.render.param_spec_name`` /
    ``preset_path`` / ``plugin_path`` to build the renderer argv. This composition
    test pins that both shipped render groups supply all three keys, so a future
    rename in a ``render/*.yaml`` surfaces here rather than mid-eval.

    :param render_group: Render config group composed into the eval cfg.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=["experiment=surge/fake_oracle", f"render={render_group}"],
        )
    try:
        assert cfg.render.param_spec_name
        assert cfg.render.preset_path
        assert cfg.render.plugin_path
    finally:
        GlobalHydra.instance().clear()
