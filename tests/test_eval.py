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
import shutil
import subprocess
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import wandb
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from synth_setter.cli.eval import evaluate
from synth_setter.cli.train import train
from synth_setter.data.vst import param_specs, preset_paths
from synth_setter.utils.utils import register_resolvers
from synth_setter.workspace import operator_workspace
from tests.helpers.run_if import RunIf
from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

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
# Per-sample metrics CSV the fake metrics subprocess writes alongside the aggregated
# CSV; one row per sample, columns matching ``compute_audio_metrics`` output.
_FAKE_METRICS_CSV = ",mss,wmfcc,sot,rms\n0,0.1,0.2,0.3,0.4\n1,0.5,0.6,0.7,0.8\n"


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


@pytest.mark.requires_vst
@pytest.mark.slow
def test_evaluate_predict_explicit_shuffle_seed_rejects_nonuniform_params_via_subprocess(
    cfg_surge_xt: DictConfig,
    cfg_surge_xt_eval: DictConfig,
) -> None:
    """Non-zero ``shuffle_seed`` with non-uniform params causes the metrics subprocess to fail.

    Drives the real train→eval roundtrip end-to-end with ``shuffle_seed=7``,
    exercising the ``evaluate()`` → ``_run_predict_postprocessing`` →
    metrics-subprocess wiring. The smoke dataset renders distinct params per
    sample, so the uniform-params guard inside ``compute_audio_metrics`` raises
    ``ValueError`` (non-zero seed + non-uniform = misconfiguration), the
    subprocess exits non-zero, and ``CalledProcessError`` surfaces at the
    ``evaluate()`` boundary — confirming the gate is wired through the real
    entrypoint (#489).

    :param cfg_surge_xt: Surge XT smoke-test training config.
    :param cfg_surge_xt_eval: Matching predict-mode eval config (render + metrics on),
        sharing ``tmp_path`` so eval reads the checkpoint training writes.
    """
    HydraConfig().set_config(cfg_surge_xt)
    train(cfg_surge_xt)
    assert Path(cfg_surge_xt_eval.ckpt_path).exists()

    with open_dict(cfg_surge_xt_eval):
        cfg_surge_xt_eval.evaluation.shuffle_seed = 7

    HydraConfig().set_config(cfg_surge_xt_eval)
    with pytest.raises(subprocess.CalledProcessError):
        evaluate(cfg_surge_xt_eval)


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
    datamodule: str | None = None,
) -> DictConfig:
    """Compose ``eval.yaml`` with the CPU ``surge/fake_oracle`` experiment, pinned to a dataset.

    Drives the CPU production oracle config (``experiment/surge/fake_oracle.yaml``)
    rather than its MPS smoke sibling, so this composition is itself coverage of
    that config. The ``render`` group is set inline to ``param_spec_name`` so the
    oracle's ``${render.param_spec_name}``-keyed per-param-MSE callback matches the
    rendered dataset's encoding width.

    :param tmp_path: Pinned as ``paths.output_dir`` / ``paths.log_dir``; the
        predict-mode ``PredictionWriter`` writes ``predictions/`` beneath it.
    :param dataset_root: Holds the ``{train,val,test}`` splits (``.h5`` or
        ``.lance`` per the selected ``datamodule``) + ``stats.npz``.
    :param mode: ``cfg.mode`` under test (``test`` / ``validate`` / ``val`` /
        ``predict`` / an unknown spelling).
    :param param_spec_name: Param spec the dataset was rendered with; drives the
        inline ``render`` group and the per-param-MSE callback's spec.
    :param datamodule: Optional datamodule group override (e.g. ``surge_lance``);
        ``None`` keeps the experiment's HDF5 ``surge`` group.
    :returns: Composed eval ``DictConfig`` ready for ``evaluate``.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=["experiment=surge/fake_oracle", f"mode={mode}"]
            + ([f"datamodule={datamodule}"] if datamodule else []),
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.dataset_root = str(dataset_root)
        # None lets the datamodule derive ``test.<its shard suffix>`` under dataset_root.
        cfg.datamodule.predict_file = None
        cfg.datamodule.batch_size = 1
        cfg.datamodule.num_workers = 0
        cfg.datamodule.use_saved_mean_and_variance = True
        cfg.ckpt_path = None
        # surge/base enables the wandb logger; null it so the fast loop never hits
        # wandb init/network/login (these tests don't assert on logging).
        cfg.logger = None
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
def test_evaluate_predict_mode_logs_per_sample_metrics_table_to_wandb(
    tmp_path: Path,
    fake_surge_smoke_datasets: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mode=predict`` with an active wandb run uploads ``metrics.csv`` as a wandb.Table.

    Exercises the ``_log_metrics_csv_to_wandb`` call-through via the real
    ``evaluate`` entrypoint: the fake metrics subprocess writes both
    ``aggregated_metrics.csv`` and ``metrics.csv``; a spy on ``wandb.run.log``
    verifies the per-sample Table arrives under ``audio/per_sample_metrics``.

    :param tmp_path: Hydra ``output_dir``; the fake subprocess writes CSVs beneath it.
    :param fake_surge_smoke_datasets: CPU-fast surge_4 dataset (no real VST).
    :param monkeypatch: Stubs subprocesses, headless wrapper, and ``wandb.run``.
    """
    logged: list[dict[str, object]] = []

    class _Spy:
        """Stand-in for ``wandb.run`` that records ``log`` payloads; no-ops SDK lifecycle calls.

        ``__getattr__`` absorbs wandb SDK cleanup methods (e.g. ``finish``,
        ``summary``) that Lightning triggers after predict — they are irrelevant to
        this test's contract.
        """

        def log(self, payload: dict[str, object]) -> None:
            """Append payload to the capture list.

            :param payload: The wandb log payload to record.
            """
            logged.append(payload)

        def __getattr__(self, _name: str) -> object:
            """Return a no-op callable for any wandb SDK method not explicitly defined.

            :param _name: Attribute name; unused — any undeclared attribute gets a no-op.
            :returns: A callable that accepts any arguments and returns ``None``.
            """
            return lambda *_args, **_kwargs: None

    def _fake_run_with_metrics(args: list[str], **_kwargs: object) -> None:
        is_render = any(_PREDICT_VST_AUDIO_FRAGMENT in a for a in args)
        is_metrics = any(_COMPUTE_AUDIO_METRICS_FRAGMENT in a for a in args)
        if not (is_render or is_metrics):
            return
        out_dir = Path(args[args.index("-m") + 3])
        out_dir.mkdir(parents=True, exist_ok=True)
        if is_metrics:
            (out_dir / "aggregated_metrics.csv").write_text(_FAKE_AGGREGATED_METRICS_CSV)
            (out_dir / "metrics.csv").write_text(_FAKE_METRICS_CSV)

    cfg = _compose_fake_oracle_eval_cfg(tmp_path, fake_surge_smoke_datasets, mode="predict")
    monkeypatch.setattr("synth_setter.cli.eval.subprocess.run", _fake_run_with_metrics)
    monkeypatch.setattr("synth_setter.cli.eval.vst_headless_wrapper", lambda: object())
    monkeypatch.setattr(
        "synth_setter.cli.eval.as_file",
        lambda _traversable: nullcontext(Path("/fake/headless-wrapper")),
    )
    monkeypatch.setattr(wandb, "run", _Spy())
    # task_wrapper's teardown calls module-level wandb.finish() while wandb.run is
    # truthy (utils.py); stub it so the spy run is the only wandb surface exercised.
    monkeypatch.setattr(wandb, "finish", lambda *_args, **_kwargs: None)

    HydraConfig().set_config(cfg)
    try:
        evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    table_payloads = [p for p in logged if "audio/per_sample_metrics" in p]
    assert len(table_payloads) == 1
    assert isinstance(table_payloads[0]["audio/per_sample_metrics"], wandb.Table)


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


def test_evaluate_unregistered_param_spec_name_raises_key_error_at_setup(
    tmp_path: Path,
) -> None:
    """An unregistered ``datamodule.param_spec_name`` fails fast through ``evaluate``.

    ``VSTDataModule.setup`` does a ``param_specs[param_spec_name]`` lookup to derive the
    fake/real param width; an unknown spec must surface as a ``KeyError`` at the
    ``evaluate`` entrypoint rather than a later opaque shape mismatch. Pins that the
    registry-lookup contract is wired through the real eval entrypoint. The lookup
    precedes any dataset open, so no dataset (and no fake plugin) is materialized.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``; the dataset root
        points at a nonexistent subdirectory that is never read.
    """
    cfg = _compose_fake_oracle_eval_cfg(tmp_path, tmp_path / "missing-datasets", mode="validate")
    with open_dict(cfg):
        cfg.datamodule.param_spec_name = "does_not_exist"

    HydraConfig().set_config(cfg)
    try:
        with pytest.raises(KeyError, match="does_not_exist"):
            evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()


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


@pytest.mark.fake_vst
def test_evaluate_predict_mode_includes_shuffled_audio_metrics_when_subprocess_writes_shuffled_csv(
    tmp_path: Path,
    fake_surge_smoke_datasets: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merges ``shuffled_audio/*`` keys when the metrics subprocess also writes the shuffled CSV.

    The fake subprocess writes both ``aggregated_metrics.csv`` and
    ``aggregated_metrics_shuffled.csv``, exercising the ``evaluate()`` →
    ``_run_predict_postprocessing`` → ``_load_audio_metrics`` path that merges
    the shuffled probe output into the returned metric dict under the
    ``shuffled_audio/`` prefix. Pins that the new branch in ``_load_audio_metrics``
    is wired through the real ``evaluate()`` entrypoint (#489).

    :param tmp_path: Hydra ``output_dir``; output files are derived beneath it.
    :param fake_surge_smoke_datasets: CPU-fast surge_4 dataset (no real VST).
    :param monkeypatch: Stubs render/metrics subprocesses; no real VST launches.
    """
    _SHUFFLED_CSV = ",mean,std\nmss,0.8,0.05\nwmfcc,0.4,0.03\nsot,0.3,0.02\nrms,0.7,0.01\n"

    def _fake_run_with_shuffled(args: list[str], **_kwargs: object) -> None:
        is_render = any(_PREDICT_VST_AUDIO_FRAGMENT in a for a in args)
        is_metrics = any(_COMPUTE_AUDIO_METRICS_FRAGMENT in a for a in args)
        if not (is_render or is_metrics):
            return
        out_dir = Path(args[args.index("-m") + 3])
        out_dir.mkdir(parents=True, exist_ok=True)
        if is_metrics:
            (out_dir / "aggregated_metrics.csv").write_text(_FAKE_AGGREGATED_METRICS_CSV)
            (out_dir / "aggregated_metrics_shuffled.csv").write_text(_SHUFFLED_CSV)

    cfg = _compose_fake_oracle_eval_cfg(tmp_path, fake_surge_smoke_datasets, mode="predict")
    monkeypatch.setattr("synth_setter.cli.eval.subprocess.run", _fake_run_with_shuffled)
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

    assert metric_dict["shuffled_audio/mss_mean"] == pytest.approx(0.8)
    assert metric_dict["shuffled_audio/rms_std"] == pytest.approx(0.01)
    for key in ("mss", "wmfcc", "sot", "rms"):
        for stat in ("mean", "std"):
            value = metric_dict[f"shuffled_audio/{key}_{stat}"]
            assert isinstance(value, float) and math.isfinite(value)


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


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("WANDB_API_KEY"),
    reason="real W&B round-trip needs WANDB_API_KEY (injected on trusted CI only)",
)
@pytest.mark.parametrize("experiment_name", ["surge/ffn_full"], indirect=True)
def test_evaluate_loads_wandb_resolved_checkpoint_and_runs_inference(
    tmp_path: Path,
    cfg_surge_xt: DictConfig,
    cfg_surge_xt_eval: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
) -> None:
    """Predict eval resolves ckpt_path via ``${wandb:...}`` from the live registry, loads it, and runs inference.

    The full wandb_checkpoint contract end to end: train a real checkpoint, publish it to
    ``tinaudio/synth-setter-citest``, then pin ``ckpt_path: ${wandb:...}`` (no CLI path, the form
    the ``surge/wandb_checkpoint/<id>`` overlay produces) and run ``evaluate()`` in predict mode.
    The resolver downloads the artifact, Lightning loads the weights, and predict-mode inference
    writes finite per-sample predictions — the contract a fake-stub test cannot prove.

    :param tmp_path: Shared output dir; also the workspace root the resolver caches under.
    :param cfg_surge_xt: Surge XT smoke training config — one step produces the checkpoint.
    :param cfg_surge_xt_eval: Matching predict-mode eval config; its ckpt_path is repinned here.
    :param monkeypatch: Pins ``SYNTH_SETTER_WORKSPACE`` so the download cache stays under tmp_path.
    :param experiment_name: Pinned to ``surge/ffn_full`` — the artifact id need only round-trip.
    """
    HydraConfig().set_config(cfg_surge_xt)
    train(cfg_surge_xt)
    ckpt = Path(cfg_surge_xt_eval.ckpt_path)
    assert ckpt.is_file(), "train step did not write the checkpoint"

    # Body runs inside the ``with`` so the resolver downloads before the artifact/run teardown.
    with publish_checkpoint_artifact(
        ckpt, "model-citest-ffn_full-eval", tmp_path / "wandb"
    ) as ref:
        # Contain the resolver's download cache under tmp_path so each run fetches fresh (a warm
        # self-hosted runner must not reuse a stale cached ckpt for the same :latest ref).
        monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
        operator_workspace.cache_clear()
        register_resolvers()
        with open_dict(cfg_surge_xt_eval):
            cfg_surge_xt_eval.ckpt_path = "${wandb:" + ref + "}"

        HydraConfig().set_config(cfg_surge_xt_eval)
        evaluate(cfg_surge_xt_eval)

    assert (tmp_path / ".cache" / "checkpoints").is_dir(), "resolver did not download the artifact"
    predictions_dir = tmp_path / "predictions"
    assert predictions_dir.is_dir()
    preds = sorted(predictions_dir.glob("pred-*.pt"))
    assert preds, "predict mode wrote no predictions"
    for pred_file in preds:
        tensor = torch.load(pred_file, weights_only=True)
        assert torch.isfinite(tensor).all(), f"{pred_file.name} contains NaN/Inf"


@pytest.fixture
def fake_surge_smoke_lance_datasets(fake_surge_smoke_datasets: Path, tmp_path: Path) -> Path:
    """Convert the fake-VST HDF5 smoke splits into single-file Lance shards under one root.

    :param fake_surge_smoke_datasets: HDF5 smoke dataset directory to convert.
    :param tmp_path: Per-test tmpdir holding the Lance copy.
    :return: Directory holding ``{train,val,test}.lance`` and ``stats.npz``.
    """
    # Local import: pulls in pyarrow, which the Docker VST CI images don't
    # install (no `data` dependency group) — module scope would break their
    # collection if this file is ever added to an in-image pytest run.
    from tests.helpers.lance_fixtures import write_lance_shard

    root = tmp_path / "lance-smoke"
    root.mkdir()
    for split in ("train", "val", "test"):
        columns: dict[str, np.ndarray] = {}
        with h5py.File(fake_surge_smoke_datasets / f"{split}.h5", "r") as f:
            for name in ("audio", "mel_spec", "param_array"):
                dataset = f[name]
                assert isinstance(dataset, h5py.Dataset)
                columns[name] = dataset[...]
        write_lance_shard(root / f"{split}.lance", columns)
    shutil.copy(fake_surge_smoke_datasets / "stats.npz", root / "stats.npz")
    return root


@pytest.mark.fake_vst
def test_evaluate_validate_mode_lance_datamodule_runs_oracle(
    tmp_path: Path,
    fake_surge_smoke_lance_datasets: Path,
) -> None:
    """``datamodule=surge_lance`` drives ``evaluate`` end-to-end over Lance splits.

    The oracle returns params verbatim, so ``val/param_mse`` is exactly zero —
    the same contract as the HDF5 leg, with every batch read from Lance.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``.
    :param fake_surge_smoke_lance_datasets: Lance conversion of the smoke dataset.
    """
    cfg = _compose_fake_oracle_eval_cfg(
        tmp_path, fake_surge_smoke_lance_datasets, mode="validate", datamodule="surge_lance"
    )

    HydraConfig().set_config(cfg)
    try:
        metric_dict, _ = evaluate(cfg)
    finally:
        GlobalHydra.instance().clear()

    param_mse = metric_dict["val/param_mse"]
    assert isinstance(param_mse, torch.Tensor)
    assert param_mse.item() == 0.0
