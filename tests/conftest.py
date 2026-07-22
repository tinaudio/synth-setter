"""Config fixtures and collection-time skip hooks for the test suite."""

import copy
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from synth_setter.data.vst import core, param_specs, plugin_state_paths
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig
from synth_setter.pipeline.subprocess_stream import scaled_timeout
from synth_setter.resources import vst_headless_wrapper
from synth_setter.utils.callbacks import LogPerParamMSE
from synth_setter.utils.utils import register_resolvers
from synth_setter.workspace import operator_workspace
from tests._baseline_worktree import worktree_for_ref  # noqa: F401 — pytest fixture re-export
from tests._vst import PLUGIN_PATH, VST_AVAILABLE
from tests.data.vst._fake_plugin import FakeVST3Plugin
from tests.pipeline.conftest import fake_r2_remote  # noqa: F401 — pytest fixture re-export

# These values must match the explicit RenderConfig fixture arguments.
_SURGE_FIXTURE_SAMPLE_RATE = 44100
_SURGE_FIXTURE_CHANNELS = 2
_SURGE_FIXTURE_DURATION_SECONDS = 4.0
_SURGE_FIXTURE_VELOCITY = 100
_SURGE_FIXTURE_MIN_LOUDNESS = -55.0
_SURGE_FIXTURE_RENDERER_VERSION = "1.3.4"
_SURGE_AUDIO_SAMPLES_PER_CLIP = int(_SURGE_FIXTURE_SAMPLE_RATE * _SURGE_FIXTURE_DURATION_SECONDS)
_SURGE_AUDIO_CHANNELS = _SURGE_FIXTURE_CHANNELS
_SURGE_MEL_SHAPE = (2, 128, 401)
# ~-80 dBFS — same threshold used by `test_train_eval_surge_xt` to catch
# silent renders that would later poison metric computation.
_SURGE_SILENCE_PEAK_THRESHOLD = 1e-4

NUM_FIXTURE_SAMPLES = 5


def assert_log_per_param_mse_wired(trainer: Any, param_spec_name: str) -> None:
    """Assert that a trainer's per-parameter MSE callback uses its active VST spec.

    :param trainer: Lightning trainer constructed by the entrypoint.
    :param param_spec_name: Registry key expected by the callback.
    """
    mse_callbacks = [
        callback for callback in trainer.callbacks if isinstance(callback, LogPerParamMSE)
    ]
    assert len(mse_callbacks) == 1
    assert mse_callbacks[0].param_spec is param_specs[param_spec_name]


def train_loss_keys(metric_dict: dict[str, torch.Tensor]) -> list[str]:
    """Collect the ``train/loss*`` keys, asserting at least one was emitted.

    Modules log ``train/loss`` with ``on_step=True, on_epoch=True``; with a single
    step only the step-level key is guaranteed, so scan the prefix instead of
    pinning one key.

    :param metric_dict: Train-metric mapping returned by ``train(cfg)``.
    :returns: The ``train/loss*`` keys present in the mapping.
    """
    loss_keys = [k for k in metric_dict if k.startswith("train/loss")]
    assert loss_keys, f"no train/loss* key in metric_dict: {sorted(metric_dict)}"
    return loss_keys


def assert_finite_train_loss(metric_dict: dict[str, torch.Tensor]) -> None:
    """Assert a ``train/loss*`` metric was emitted and every one is finite.

    ``trainer.global_step`` advances even past a NaN/Inf loss, so advancement asserts
    alone let silent numerical failures pass.

    :param metric_dict: Train-metric mapping returned by ``train(cfg)``.
    """
    for key in train_loss_keys(metric_dict):
        loss = metric_dict[key]
        assert torch.isfinite(loss).all(), f"{key} is not finite: {loss}"


def _scaled_vst_subprocess_timeout(num_samples: int = NUM_FIXTURE_SAMPLES) -> float:
    """Wall-clock budget for a fixture-building VST subprocess, scaled by sample count.

    A flat ceiling silently under-budgets the day ``NUM_FIXTURE_SAMPLES`` is
    raised; scaling on it keeps the budget honest as the fixture grows. Overhead
    covers fixed startup (plugin load, imports); the per-sample term is loose
    render margin.

    :param num_samples: Sample count the subprocess renders or reads.
    :returns: Timeout in seconds for the subprocess.
    """
    return scaled_timeout(num_samples, overhead_seconds=300.0, per_sample_seconds=60.0)


# Probed from the env var, no network hit — AGENTS.md's `rclone lsd r2:` is for
# interactive verification, not the skip criterion. VST presence lives in tests._vst.
_R2_AVAILABLE = bool(os.environ.get("RCLONE_CONFIG_R2_ACCESS_KEY_ID"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-skip requires_vst / integration_r2 tests when resources are absent.

    :param items: mutated in-place to insert skip markers for missing resources.
    """
    skip_vst = pytest.mark.skip(
        reason=f"VST plugin not found at {PLUGIN_PATH!r} "
        f"(set SYNTH_SETTER_PLUGIN_PATH or place plugin at that path)"
    )
    skip_r2 = pytest.mark.skip(
        reason="R2 credentials absent (RCLONE_CONFIG_R2_ACCESS_KEY_ID not set); "
        "run `rclone lsd r2:` to verify"
    )
    for item in items:
        if "requires_vst" in item.keywords and not VST_AVAILABLE:
            item.add_marker(skip_vst)
        if "integration_r2" in item.keywords and not _R2_AVAILABLE:
            item.add_marker(skip_r2)


# Bootstraps Xvfb + xsettingsd + dbus for VST3 plugin init; ships inside
# the ``synth_setter`` package via :mod:`synth_setter.resources`. X11
# wrapping lives at the audio-rendering boundary (the subprocess call),
# not at the container entrypoint — the click CLI stays X11-agnostic so
# idle and passthrough don't pay the Xvfb startup cost.
VST_HEADLESS_WRAPPER = str(vst_headless_wrapper())


def _validate_surge_dataset(path: Path, num_samples: int) -> None:
    """Assert the generated Surge XT Lance dataset is structurally sound.

    Verifies the three required columns exist with the expected shapes, that no NaN/Inf leaked in
    from the VST/mel pipeline, and that every audio clip is above the silence floor — surface those
    failures here rather than letting downstream training crash on opaque NaN losses.
    """
    import lance

    table = lance.dataset(str(path)).to_table()
    for name in ("audio", "mel_spec", "param_array"):
        assert name in table.schema.names, f"missing column {name!r} in {path}"

    def _col(name: str) -> np.ndarray:
        return table.column(name).combine_chunks().to_numpy_ndarray()

    audio_arr = _col("audio").astype(np.float32)
    mel_arr = _col("mel_spec")
    params_arr = _col("param_array")

    expected_audio_shape = (
        num_samples,
        _SURGE_AUDIO_CHANNELS,
        _SURGE_AUDIO_SAMPLES_PER_CLIP,
    )
    assert audio_arr.shape == expected_audio_shape, (
        f"audio shape {audio_arr.shape} != expected {expected_audio_shape}"
    )
    assert mel_arr.shape == (num_samples, *_SURGE_MEL_SHAPE), (
        f"mel_spec shape {mel_arr.shape} != expected {(num_samples, *_SURGE_MEL_SHAPE)}"
    )
    assert params_arr.shape[0] == num_samples, (
        f"param_array first dim {params_arr.shape[0]} != num_samples {num_samples}"
    )
    assert params_arr.ndim == 2, f"param_array must be 2D, got shape {params_arr.shape}"

    assert np.isfinite(audio_arr).all(), f"audio in {path} contains NaN/Inf"
    assert np.isfinite(mel_arr).all(), f"mel_spec in {path} contains NaN/Inf"
    assert np.isfinite(params_arr).all(), f"param_array in {path} contains NaN/Inf"

    per_clip_peak = np.abs(audio_arr).reshape(num_samples, -1).max(axis=1)
    silent = np.where(per_clip_peak <= _SURGE_SILENCE_PEAK_THRESHOLD)[0]
    assert silent.size == 0, (
        f"audio clips {silent.tolist()} in {path} are silent "
        f"(peaks={per_clip_peak[silent].tolist()})"
    )


# Register custom OmegaConf resolvers (mul, div) needed to parse Hydra configs.
# This import pulls in torch/lightning transitively via synth_setter.utils.utils, but every
# test in this suite already requires those dependencies, so there is no benefit to
# isolating resolver registration into a lighter module.
register_resolvers()


def reset_hydra_config_singleton() -> None:
    """Clear Hydra's ``HydraConfig`` singleton so its ``cfg`` reads as unset.

    ``HydraConfig`` is a Hydra ``Singleton`` distinct from ``GlobalHydra``; tests
    that call ``HydraConfig().set_config(...)`` populate a process-global
    singleton that ``GlobalHydra.instance().clear()`` leaves untouched. The stale
    ``runtime.choices.experiment`` then leaks into a later, Hydra-context-free
    test via :func:`synth_setter.utils.logging_utils.resolve_run_config_id`,
    which reads a stale experiment instead of falling back to ``task_name``.
    """
    HydraConfig.instance().cfg = None


@pytest.fixture(autouse=True)
def _clear_hydra_config_singleton() -> Iterator[None]:
    """Reset the ``HydraConfig`` singleton after every test.

    :yields None: Control to the test, then clears the singleton on teardown.
    """
    yield
    reset_hydra_config_singleton()


def _set_workspace_root(cfg: DictConfig) -> None:
    """Pin ``paths.root_dir`` to the operator workspace, in place.

    :param cfg: Composed config mutated in place under an open ``open_dict``.
    """
    cfg.paths.root_dir = str(operator_workspace())


def _apply_common_train_eval_overrides(cfg: DictConfig) -> None:
    """Apply the single-epoch smoke defaults ``cfg_train_global`` and ``cfg_eval_global`` share, in place.

    :param cfg: Composed config mutated in place under an open ``open_dict``.
    """
    cfg.trainer.check_val_every_n_epoch = 1
    cfg.trainer.val_check_interval = 1
    cfg.trainer.max_epochs = 1
    cfg.trainer.num_sanity_val_steps = 0
    cfg.trainer.log_every_n_steps = 1
    cfg.trainer.devices = 1
    cfg.trainer.deterministic = True
    cfg.datamodule.pin_memory = False
    cfg.datamodule.batch_size = 1
    cfg.datamodule.train_val_test_sizes = [2, 2, 2]
    cfg.datamodule.break_symmetry = True
    cfg.model.compile = False
    cfg.logger = None
    _set_workspace_root(cfg)


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """Build a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["datamodule=ksin", "model=ffn", "trainer=cpu"],
        )

        # set defaults for all tests
        with open_dict(cfg):
            _apply_common_train_eval_overrides(cfg)
            cfg.datamodule.num_workers = 0
            cfg.callbacks.model_checkpoint.save_top_k = -1
            cfg.callbacks.model_checkpoint.save_last = True
            callbacks = cfg.get("callbacks")
            if callbacks is not None and "lr_monitor" in callbacks:
                del callbacks.lr_monitor

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """Build a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "datamodule=ksin",
                "model=ffn",
                "trainer=cpu",
                "ckpt_path=.",
            ],
        )

        # set defaults for all tests
        with open_dict(cfg):
            _apply_common_train_eval_overrides(cfg)
            cfg.datamodule.num_workers = 0
    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """Build on top of ``cfg_train_global()`` and redirect logging into ``tmp_path``.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture
def cfg_torchsynth_train(tmp_path: Path) -> Iterator[DictConfig]:
    """Compose a CPU-cheap production TorchSynth config at the entrypoint boundary.

    Composes the production experiment through ``train.yaml``, then shrinks
    the online splits and trainer loop so the entrypoint test stays CPU-cheap.

    :param tmp_path: Pinned Hydra output and log directory.
    :yields: Ready-to-run training configuration.
    :ytype: DictConfig
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=torchsynth/ffn",
                "trainer=cpu",
                "+trainer.fast_dev_run=true",
                "datamodule.train_val_test_sizes=[2,2,2]",
                "datamodule.batch_size=1",
                "datamodule.num_workers=0",
                "logger=csv",
            ],
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(operator_workspace())
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
    yield cfg
    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """Build on top of ``cfg_eval_global()`` and redirect logging into ``tmp_path``.

    This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging path.

    :param cfg_eval_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="package")
def cfg_dataset_global() -> DictConfig:
    """Build a default Hydra DictConfig for ``generate_dataset``.

    Omits ``return_hydra_config=True`` so the ``hydra.*`` sub-tree (whose
    ``sweep.subdir`` interpolates the runtime-only ``${hydra.job.num}``) does
    not leak in and break ``spec_from_cfg``'s ``resolve=True`` round-trip.

    :return: A DictConfig composed from ``configs/dataset.yaml`` with
        ``experiment=generate_dataset/smoke-shard`` so every required (``???``)
        field is populated.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=["experiment=generate_dataset/smoke-shard"],
        )
        with open_dict(cfg):
            _set_workspace_root(cfg)
    return cfg


@pytest.fixture(scope="function")
def cfg_dataset(cfg_dataset_global: DictConfig, tmp_path: Path) -> Iterator[DictConfig]:
    """Build on top of ``cfg_dataset_global()`` and redirect paths into ``tmp_path``.

    :param cfg_dataset_global: The package-scoped dataset DictConfig to copy.
    :param tmp_path: The per-test temporary path used as output/work/log root.

    :yields DictConfig: ``paths.{output_dir,work_dir,log_dir}`` pinned to
        ``tmp_path``; teardown clears Hydra's global singleton.
    """
    cfg = cfg_dataset_global.copy()
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.work_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_dataset_obxf(tmp_path: Path) -> Iterator[DictConfig]:
    """Compose ``dataset.yaml`` with ``render=obxf`` for entrypoint OB-Xf coverage.

    The Hydra config-initializer lives here, not in ``tests/test_generate_dataset.py``,
    so that module stays free of the imports banned by
    ``tests/_meta/test_entrypoint_e2e_only.py`` while still carrying the
    second-synth entrypoint test ``synth-setter-project-standards`` P31 requires.

    :param tmp_path: Per-test output/work/log root.

    :yields DictConfig: ``render=obxf`` cfg with ``tmp_path``-pinned paths;
        teardown clears Hydra's global singleton.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=["experiment=generate_dataset/smoke-shard", "render=obxf"],
        )
        with open_dict(cfg):
            _set_workspace_root(cfg)
            cfg.paths.output_dir = str(tmp_path)
            cfg.paths.work_dir = str(tmp_path)
            cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_dataset_torchsynth(tmp_path: Path) -> Iterator[DictConfig]:
    """Compose the torchsynth smoke experiment with temporary local paths.

    :param tmp_path: Per-test output/work/log root.

    :yields DictConfig: torchsynth smoke cfg with ``tmp_path``-pinned paths;
        teardown clears Hydra's global singleton.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=["experiment=generate_dataset/torchsynth-smoke"],
        )
        with open_dict(cfg):
            _set_workspace_root(cfg)
            cfg.paths.output_dir = str(tmp_path)
            cfg.paths.work_dir = str(tmp_path)
            cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_dataset_default_cadence(tmp_path: Path) -> Iterator[DictConfig]:
    """Compose ``dataset.yaml`` with an experiment that sets no cadence keys.

    The Hydra config-initializer lives here, not in ``tests/test_generate_dataset.py``,
    so that module stays free of the imports banned by
    ``tests/_meta/test_entrypoint_e2e_only.py`` while still carrying the
    default-cadence entrypoint test ``synth-setter-project-standards`` P31 requires.

    :param tmp_path: Per-test output/work/log root.

    :yields DictConfig: No-cadence-override cfg with ``tmp_path``-pinned paths;
        teardown clears Hydra's global singleton.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=["experiment=generate_dataset/ci-materialize-test"],
        )
        with open_dict(cfg):
            _set_workspace_root(cfg)
            cfg.paths.output_dir = str(tmp_path)
            cfg.paths.work_dir = str(tmp_path)
            cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_dataset_dawdreamer(tmp_path: Path) -> Iterator[DictConfig]:
    """Compose the DawDreamer smoke experiment with temporary local paths.

    :param tmp_path: Per-test output/work/log root.

    :yields DictConfig: DawDreamer smoke cfg with ``tmp_path``-pinned paths;
        teardown clears Hydra's global singleton.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="dataset",
            overrides=["experiment=generate_dataset/surge-xt-dawdreamer-smoke"],
        )
        with open_dict(cfg):
            _set_workspace_root(cfg)
            cfg.paths.output_dir = str(tmp_path)
            cfg.paths.work_dir = str(tmp_path)
            cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="package")
def cfg_finalize_global() -> DictConfig:
    """Build a default Hydra DictConfig for ``finalize_dataset``.

    Composes with ``return_hydra_config=True`` so the ``hydra.run.dir`` /
    ``job_logging`` interpolations finalize relies on are present in the tree
    (the entrypoint overrides ``hydra.run.dir`` because the shared group
    references ``${run_name}``, which this cfg does not surface). Supplies the
    required ``dataset_root_uri`` so every ``???`` field is populated.

    :return: A DictConfig composed from ``configs/finalize_dataset.yaml`` with
        ``dataset_root_uri`` set and ``paths.root_dir`` pinned to the workspace.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="finalize_dataset",
            return_hydra_config=True,
            overrides=["dataset_root_uri=r2://bucket/run/"],
        )
        with open_dict(cfg):
            _set_workspace_root(cfg)
    return cfg


@pytest.fixture(scope="function")
def cfg_finalize(cfg_finalize_global: DictConfig, tmp_path: Path) -> Iterator[DictConfig]:
    """Build on top of ``cfg_finalize_global()`` and redirect paths into ``tmp_path``.

    :param cfg_finalize_global: The package-scoped finalize DictConfig to copy.
    :param tmp_path: The per-test temporary path used as output/log root.

    :yields DictConfig: ``paths.{output_dir,log_dir}`` pinned to ``tmp_path``;
        teardown clears Hydra's global singleton.
    """
    cfg = cfg_finalize_global.copy()
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(
    params=[
        pytest.param("cpu", id="cpu"),
        pytest.param("mps", id="mps", marks=[pytest.mark.mps]),
        pytest.param("gpu", id="gpu", marks=[pytest.mark.gpu]),
    ]
)
def accelerator(request: pytest.FixtureRequest) -> str:
    """Parametrized accelerator selector for Surge XT smoke tests.

    Generates one test ID per accelerator (``[cpu]`` / ``[mps]`` / ``[gpu]``) and attaches
    matching markers so CI runners can filter via ``-m``. Hardfails (rather than skips) when
    the requested accelerator isn't available on this host: a runner that asks for ``mps``
    or ``gpu`` should *have* it — silent skips would mask CI misconfiguration.

    :param request: The pytest fixture request carrying the parametrized accelerator name.

    :return: One of ``"cpu"``, ``"mps"``, or ``"gpu"`` — guaranteed available on the host.
    """
    acc = request.param
    if acc == "gpu" and not torch.cuda.is_available():
        pytest.fail("CUDA not available", pytrace=False)
    if acc == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.fail("MPS not available", pytrace=False)
    return acc


@pytest.fixture(scope="function")
def param_spec_name(request: pytest.FixtureRequest) -> str:
    """Param spec name driving the Surge XT smoke fixtures.

    Defaults to ``"surge_4"`` (the 4-continuous + 2-note mini-example spec used by
    the smoke-test fixture and the ``predict_vst_audio`` end-to-end test). Override
    per-test via indirect parametrization to exercise other specs::

        @pytest.mark.parametrize("param_spec_name", ["surge_simple"], indirect=True)
        def test_thing(cfg_surge_xt_global): ...

    :param request: Pytest fixture request — when parametrized indirectly, ``request.param``
        carries the spec name; otherwise the default ``"surge_4"`` is used.

    :return: A key into :data:`synth_setter.data.vst.param_specs` and :data:`synth_setter.data.vst.plugin_state_paths`.
    """
    return getattr(request, "param", "surge_4")


@pytest.fixture(scope="function")
def experiment_name(request: pytest.FixtureRequest) -> str:
    """Hydra experiment override driving the Surge XT smoke fixtures.

    Defaults to ``"surge/fake_oracle"`` (the oracle-baseline smoke experiment, kept in
    lockstep with ``configs/experiment/surge/test-mps-fake-oracle.yaml``). Override per-test via
    indirect parametrization to exercise other experiments::

        @pytest.mark.parametrize("experiment_name", ["surge/ffn_full"], indirect=True)
        def test_thing(cfg_surge_xt_global): ...

    :param request: Pytest fixture request — when parametrized indirectly, ``request.param``
        carries the experiment name; otherwise the default ``"surge/fake_oracle"`` is used.

    :return: The Hydra experiment override name (for example, ``"surge/ffn_full"``).
    :rtype: str
    """
    return getattr(request, "param", "surge/fake_oracle")


def _build_surge_xt_smoke_cfg(
    accelerator: str,
    param_spec_name: str,
    experiment: str,
    datamodule_group: Literal["surge", "surge_lance"] | None = "surge",
) -> DictConfig:
    """Construct the Surge XT smoke-test config without the accelerator availability gate.

    Composes ``train.yaml`` with ``experiment=<experiment>`` and bakes in the minimal
    overrides needed to train-smoke-test on the dataset generated by
    :func:`surge_xt_smoke_datasets`. The datamodule's ``param_spec_name`` drives model
    width and per-parameter callback labels. Used both by the
    :func:`cfg_surge_xt_global` fixture (where the parametrized ``accelerator`` is
    host-checked upstream) and by the ``configs/experiment/surge/test-mps*.yaml``
    equality test (where the cfg must be built on any host so the YAMLs never silently
    drift from this builder).

    :param accelerator: Lightning ``trainer.accelerator`` — ``"cpu"``, ``"mps"``, or ``"gpu"``.
    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` driving
        model width and per-parameter callback labels.
    :param experiment: Hydra ``experiment=...`` override (e.g. ``"surge/fake_oracle"``,
        ``"surge/ffn_full"``); selects which model the smoke cfg wires up.
    :param datamodule_group: Hydra datamodule group override, or ``None`` to retain the
        experiment's selection.

    :return: Resolved DictConfig with the smoke-test bake-ins applied.
    """
    overrides = [f"experiment={experiment}", "callbacks=[default_vst,eval_vst]"]
    if datamodule_group is not None:
        overrides.insert(1, f"datamodule={datamodule_group}")

    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=overrides,
        )
        TRAINING_STEPS = 1
        with open_dict(cfg):
            cfg.paths.root_dir = str(operator_workspace())

            cfg.trainer.accelerator = accelerator
            # MPS doesn't support float64 ops Lightning uses by default; pin to float32.
            if accelerator == "cpu":
                cfg.model.compile = False
                cfg.trainer.precision = "32-true"
            elif accelerator == "mps":
                cfg.trainer.precision = "32-true"
                cfg.model.compile = False
            elif accelerator == "gpu":
                cfg.model.compile = True
                cfg.trainer.precision = "16-mixed"

            # The smoke fixture writes one sample, so use a one-row training batch.
            cfg.datamodule.batch_size = 1
            cfg.datamodule.param_spec_name = param_spec_name
            cfg.datamodule.pin_memory = False
            cfg.datamodule.ot = False
            # Smoke fixture writes stats.npz via masked get_dataset_stats — see #1002.
            cfg.datamodule.use_saved_mean_and_variance = True
            cfg.datamodule.num_workers = 0

            cfg.trainer.devices = 1
            cfg.trainer.max_steps = TRAINING_STEPS
            cfg.trainer.check_val_every_n_epoch = 1  # validate at end of each epoch
            cfg.trainer.val_check_interval = 1.0  # default: end of (validating) epoch
            cfg.trainer.log_every_n_steps = TRAINING_STEPS
            cfg.trainer.enable_model_summary = False
            cfg.trainer.limit_val_batches = 1.0
            cfg.trainer.deterministic = True

            cfg.model.scheduler = None
            cfg.logger = None
            cfg.test = False
            mc = cfg.callbacks.model_checkpoint
            mc.save_last = True
            # Set rather than delete so the structural cfg matches the equivalent
            # ``configs/experiment/surge/test-mps-*.yaml`` (which use ``lr_monitor: null``
            # in YAML). ``instantiate_callbacks`` skips entries without ``_target_``, so
            # the runtime behavior is identical to a deletion.
            if cfg.get("callbacks") is not None and "lr_monitor" in cfg.callbacks:
                cfg.callbacks.lr_monitor = None

    return cfg


def build_fake_train_cfg(
    output_dir: Path,
    param_spec_name: str,
    model_group: str = "vst_fake_oracle",
    callbacks_group: str = "default_vst",
) -> DictConfig:
    """Compose a one-step CPU fake-mode train cfg wired to ``param_spec_name``.

    Drives ``datamodule.fake=true`` so no dataset is read; the fake batch width comes
    from ``param_specs[param_spec_name]``. Pinned to the width-agnostic
    ``surge/fake_oracle`` experiment so any registry width trains cleanly. Lives here
    (not inline in ``tests/test_train.py``) because that module is an entrypoint-only
    test file barred from importing Hydra config-initializers (see
    ``tests/_meta/test_entrypoint_e2e_only.py``).

    :param output_dir: Pinned as Hydra ``output_dir`` / ``log_dir``; no dataset is read.
    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` driving
        the fake param width and the per-param-MSE callback's spec.
    :param model_group: Hydra model group to compose.
    :param callbacks_group: Hydra callbacks group to compose.
    :returns: Resolved one-step fake-mode train DictConfig.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/fake_oracle",
                "trainer=cpu",
                f"model={model_group}",
                f"callbacks={callbacks_group}",
            ],
        )
        with open_dict(cfg):
            # Random-weight training on random fake data is irreproducible unseeded.
            cfg.seed = 1234
            cfg.paths.root_dir = str(operator_workspace())
            cfg.paths.output_dir = str(output_dir)
            cfg.paths.log_dir = str(output_dir)
            cfg.datamodule.fake = True
            cfg.datamodule.param_spec_name = param_spec_name
            cfg.datamodule.batch_size = 2
            cfg.datamodule.num_workers = 0
            cfg.datamodule.use_saved_mean_and_variance = False
            cfg.trainer.max_steps = 1
            cfg.trainer.limit_val_batches = 0
            cfg.logger = None
            if "lr_monitor" in cfg.callbacks:
                del cfg.callbacks.lr_monitor
            # log_per_param_mse keys its spec off ${render.param_spec_name}; pin it
            # concretely — this train path composes no render group.
            cfg.callbacks.log_per_param_mse.param_spec = param_spec_name
    return cfg


def build_fake_flow_ast_pretrained_train_cfg(output_dir: Path) -> DictConfig:
    """Compose an offline one-step config through the production Hydra selection.

    :param output_dir: Hydra output and log directory.
    :returns: Resolved fake-mode flow training config.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/flow_simple",
                "model/encoder=ast_pretrained",
                "trainer=cpu",
            ],
        )
        with open_dict(cfg):
            # Random-weight training on random fake data is irreproducible unseeded.
            cfg.seed = 1234
            cfg.paths.root_dir = str(operator_workspace())
            cfg.paths.output_dir = str(output_dir)
            cfg.paths.log_dir = str(output_dir)
            cfg.datamodule.fake = True
            cfg.datamodule.batch_size = 2
            cfg.datamodule.num_workers = 0
            cfg.datamodule.use_saved_mean_and_variance = False
            cfg.trainer.min_steps = 1
            cfg.trainer.max_steps = 1
            cfg.trainer.limit_val_batches = 0
            # ODE sampling dominates this encoder-wiring test even with a tiny backbone.
            cfg.test = False
            cfg.logger = None
            if "lr_monitor" in (cfg.get("callbacks") or {}):
                del cfg.callbacks.lr_monitor
            cfg.model.compile = False
            cfg.model.encoder.pretrained = False
            cfg.model.encoder.d_model = 32
            cfg.model.encoder.n_pool_heads = 2
            cfg.model.encoder.backbone_config = {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 64,
            }
    return cfg


@pytest.fixture(scope="function")
def cfg_surge_xt_global(
    accelerator: str, param_spec_name: str, experiment_name: str
) -> DictConfig:
    """Build a one-step Surge XT training config on the N-sample test fixture.

    Thin wrapper around :func:`_build_surge_xt_smoke_cfg`; the ``accelerator`` fixture
    enforces hardware availability before this fixture composes the config so MPS/GPU
    runs on hosts without the accelerator hardfail rather than producing a silent
    placeholder cfg.

    :param accelerator: Parametrized accelerator (``"cpu"`` / ``"mps"`` / ``"gpu"``) — drives
        Lightning's ``trainer.accelerator`` and applies device-specific config tweaks.
    :param param_spec_name: Name of the :mod:`synth_setter.data.vst` param spec the cfg is wired for —
        drives ``model.net.d_out`` and ``callbacks.log_per_param_mse.param_spec``.
    :param experiment_name: Hydra ``experiment=...`` override (e.g. ``"surge/fake_oracle"``,
        ``"surge/ffn_full"``); selects which model the smoke cfg wires up.

    :return: A DictConfig object configured for a one-step Surge XT smoke train.
    """
    return _build_surge_xt_smoke_cfg(
        accelerator=accelerator,
        param_spec_name=param_spec_name,
        experiment=experiment_name,
    )


def _render_smoke_train_subprocess(output_path: Path, param_spec_name: str) -> None:
    """Render the smoke ``train`` shard via the ``generate_vst_dataset`` subprocess (real VST), failing loud on timeout/non-zero exit/missing output.

    The writer is dispatched on ``output_path``'s ``.lance`` suffix by
    ``generate_vst_dataset``, backing the real-VST smoke fixture.

    :param output_path: Destination shard path (``train.lance``); its
        parent must already exist.
    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` and
        :data:`synth_setter.data.vst.plugin_state_paths` selecting spec and preset.
    """
    generate_dataset_args = []
    if sys.platform == "linux":
        generate_dataset_args.append(VST_HEADLESS_WRAPPER)

    generate_dataset_args += [
        sys.executable,
        "src/synth_setter/data/vst/generate_vst_dataset.py",
        str(output_path),
        f"--plugin_path={PLUGIN_PATH}",
        f"--plugin_state_path={plugin_state_paths[param_spec_name]}",
        f"--param_spec_name={param_spec_name}",
        f"--renderer_version={_SURGE_FIXTURE_RENDERER_VERSION}",
        f"--sample_rate={_SURGE_FIXTURE_SAMPLE_RATE}",
        f"--channels={_SURGE_FIXTURE_CHANNELS}",
        f"--velocity={_SURGE_FIXTURE_VELOCITY}",
        f"--signal_duration_seconds={_SURGE_FIXTURE_DURATION_SECONDS}",
        f"--min_loudness={_SURGE_FIXTURE_MIN_LOUDNESS}",
        f"--samples_per_render_batch={NUM_FIXTURE_SAMPLES}",
        f"--samples_per_shard={NUM_FIXTURE_SAMPLES}",
    ]

    # capture_output=False (default): child inherits parent's stdout/stderr, no pipe is
    # created. Avoids the `capture_output=True` deadlock where fork-inherited fds in
    # pytest/DataLoader workers keep the pipe's read end open and block `communicate()`
    # forever. Output flows to pytest's normal capture (visible with `-s` or on failure);
    # we lose `result.stdout/stderr` on the failure branch but keep the exit code, which
    # is what the failure branch needs to fail loud. See #695.
    timeout = _scaled_vst_subprocess_timeout()
    try:
        result = subprocess.run(  # noqa: S603
            generate_dataset_args,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"generate_vst_dataset timed out after {timeout}s\n"
            f"command: {generate_dataset_args}\n"
            f"(child stdout/stderr printed above; rerun with `pytest -s` if captured)",
            pytrace=False,
        )
    if result.returncode != 0:
        pytest.fail(
            f"generate_vst_dataset failed (exit {result.returncode})\n"
            f"command: {generate_dataset_args}\n"
            f"(child stdout/stderr printed above; rerun with `pytest -s` if captured)",
            pytrace=False,
        )
    assert output_path.exists(), "Dataset generation failed to produce train fixture"


def _smoke_fake_render_cfg(param_spec_name: str) -> RenderConfig:
    """Build the one-shard fake-plugin ``RenderConfig`` for the Lance smoke render.

    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` and
        :data:`synth_setter.data.vst.plugin_state_paths` selecting spec and preset.
    :returns: A CPU ``RenderConfig`` with the GUI toggle disabled.
    """
    return RenderConfig(
        plugin_path=PLUGIN_PATH,
        plugin_state_path=str(plugin_state_paths[param_spec_name]),
        param_spec_name=param_spec_name,
        renderer_version=_SURGE_FIXTURE_RENDERER_VERSION,
        sample_rate=_SURGE_FIXTURE_SAMPLE_RATE,
        channels=_SURGE_FIXTURE_CHANNELS,
        velocity=_SURGE_FIXTURE_VELOCITY,
        signal_duration_seconds=_SURGE_FIXTURE_DURATION_SECONDS,
        min_loudness=_SURGE_FIXTURE_MIN_LOUDNESS,
        samples_per_render_batch=NUM_FIXTURE_SAMPLES,
        samples_per_shard=NUM_FIXTURE_SAMPLES,
        gui_toggle_cadence="never",
    )


def _render_smoke_train_lance_fake(train_lance: Path, param_spec_name: str) -> None:
    """Render the smoke ``train.lance`` in-process via ``make_lance_dataset``; requires the caller to have installed ``FakeVST3Plugin``.

    :param train_lance: Destination ``train.lance`` file; its parent must already exist.
    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` and
        :data:`synth_setter.data.vst.plugin_state_paths` selecting spec and preset.
    """
    from synth_setter.data.vst.writers import make_lance_dataset

    make_lance_dataset(train_lance, _smoke_fake_render_cfg(param_spec_name))


def _build_surge_smoke_lance_datasets(
    tmp_path: Path,
    param_spec_name: str,
    render_train_lance: Callable[[Path, str], None],
) -> Path:
    """Render the N-sample Surge smoke dataset natively as single-file Lance shards.

    ``render_train_lance`` is the only difference between the real-VST and fake
    fixtures. It renders ``train.lance`` through the production
    :func:`make_lance_dataset` writer, then this folds the mel rows into
    ``stats.npz`` via :func:`fold_lance_shard_into_welford` and clones the split
    into ``val``/``test``. Every shard carries the exact on-disk format the
    pipeline's Lance finalize emits.

    :param tmp_path: Per-test temporary directory; the dataset is written under
        ``tmp_path / "data" / "smoke-lance"``.
    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` and
        :data:`synth_setter.data.vst.plugin_state_paths` selecting spec and preset.
    :param render_train_lance: Renders ``train.lance`` given ``(train_lance, param_spec_name)``.

    :return: Path to the directory holding ``{train,val,test}.lance`` and ``stats.npz``.
    """
    from synth_setter.pipeline.data.stats import finalize, fold_lance_shard_into_welford

    smoke_dataset_dir = tmp_path / "data" / "smoke-lance"
    smoke_dataset_dir.mkdir(parents=True, exist_ok=True)
    train_lance = smoke_dataset_dir / "train.lance"

    render_train_lance(train_lance, param_spec_name)
    _validate_surge_dataset(train_lance, NUM_FIXTURE_SAMPLES)

    # Sibling stats.npz folded straight from the Lance mel rows; mask degenerate
    # bins as the h5 path's --mask-degenerate-bins flag does for fake-plugin data.
    welford = fold_lance_shard_into_welford((0, 0, 0), train_lance)
    mean, std = finalize(welford, mask_degenerate=True)
    np.savez(smoke_dataset_dir / "stats.npz", mean=mean, std=std)

    shutil.copytree(train_lance, smoke_dataset_dir / "val.lance")
    shutil.copytree(train_lance, smoke_dataset_dir / "test.lance")
    return smoke_dataset_dir


@pytest.fixture(scope="function")
def surge_xt_smoke_datasets(tmp_path: Path, param_spec_name: str) -> Path:
    """Generate the N-sample Surge XT smoke dataset as native Lance shards via the real VST.

    ``generate_vst_dataset`` dispatches the ``.lance`` suffix to
    :func:`make_lance_dataset`, so the real Surge XT subprocess writes
    ``train.lance`` directly. Backs the real-VST half of the train/eval smoke matrix.

    :param tmp_path: Per-test temporary directory; the dataset is written under
        ``tmp_path / "data" / "smoke-lance"``.
    :param param_spec_name: Param spec name (key into :data:`synth_setter.data.vst.param_specs`
        and :data:`synth_setter.data.vst.plugin_state_paths`) — selects the matching ``--param_spec_name``
        and ``--plugin_state_path`` for ``generate_vst_dataset``.

    :return: Path to the directory holding ``{train,val,test}.lance`` and ``stats.npz``.
    """
    return _build_surge_smoke_lance_datasets(
        tmp_path, param_spec_name, _render_smoke_train_subprocess
    )


def augment_lance_splits_with_embeddings(dataset_root: Path) -> Path:
    """Add real ``m2l`` + ``clap`` columns to a rendered smoke dataset's splits.

    Composes the shipped ``add_embeddings.yaml`` pinned at ``train.lance`` and runs
    the real endpoint (real music2latent + CLAP encoders — no mocks), then clones
    the augmented ``train.lance`` over the identical ``val``/``test`` clones so the
    encoders load once. Backs the real clap/m2l conditioning e2e tests; requires
    network (HuggingFace CLAP download on first run).

    :param dataset_root: Dir holding ``{train,val,test}.lance`` from
        :func:`surge_xt_smoke_datasets`; each split is augmented in place.
    :returns: ``dataset_root`` for call-site chaining.
    """
    from synth_setter.pipeline.data.add_embeddings import add_embeddings
    from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig

    train_uri = dataset_root / "train.lance"
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="add_embeddings",
            overrides=[f"lance_uri={train_uri}", "build_index=false", "device=cpu"],
        )
        config = AddEmbeddingsConfig.from_hydra_cfg(cfg)
    GlobalHydra.instance().clear()
    add_embeddings(config)

    # Identical val/test clones are intentional: a plumbing smoke, not a
    # generalization check.
    for split in ("val", "test"):
        dest = dataset_root / f"{split}.lance"
        shutil.rmtree(dest)
        shutil.copytree(train_uri, dest)
    return dataset_root


def build_surge_xt_embedding_train_cfg(
    output_dir: Path,
    dataset_root: Path,
    *,
    param_spec_name: str,
    conditioning: str,
) -> DictConfig:
    """Compose a one-step CPU flow-training cfg wired to an embedding-conditioning profile.

    Composes ``experiment=surge/flow_simple`` with ``conditioning=<profile>`` over
    the map-style ``surge_lance`` datamodule, pinned to a real (``fake=False``)
    dataset augmented with the profile's Lance column. Lives here (not inline in
    ``tests/test_train.py``) because that module is barred from importing Hydra
    config-initializers (see ``tests/_meta/test_entrypoint_e2e_only.py``).

    :param output_dir: Pinned as Hydra ``output_dir`` / ``log_dir``; the checkpoint
        callback writes ``last.ckpt`` beneath it.
    :param dataset_root: Dir holding the augmented ``{train,val,test}.lance`` splits.
    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` driving
        model width and per-parameter callback labels.
    :param conditioning: Conditioning profile group (``"clap"`` / ``"m2l"``).
    :returns: Resolved one-step embedding-conditioning train DictConfig.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=surge/flow_simple",
                f"conditioning={conditioning}",
                "trainer=cpu",
                "datamodule=surge_lance",
                "callbacks=[default_vst,eval_vst]",
            ],
        )
        with open_dict(cfg):
            cfg.seed = 1234
            cfg.paths.root_dir = str(operator_workspace())
            cfg.paths.output_dir = str(output_dir)
            cfg.paths.log_dir = str(output_dir)

            cfg.trainer.accelerator = "cpu"
            cfg.trainer.precision = "32-true"
            cfg.trainer.devices = 1
            cfg.trainer.max_steps = 1
            cfg.trainer.min_steps = 1
            # Flow validation runs expensive ODE sampling; the train-loss e2e skips
            # it. The eval e2e drives validation through evaluate() instead.
            cfg.trainer.limit_val_batches = 0
            cfg.trainer.log_every_n_steps = 1
            cfg.trainer.enable_model_summary = False
            cfg.trainer.deterministic = True

            cfg.datamodule.fake = False
            cfg.datamodule.dataset_root = str(dataset_root)
            cfg.datamodule.predict_file = str(dataset_root / "test.lance")
            cfg.datamodule.param_spec_name = param_spec_name
            cfg.datamodule.batch_size = 1
            cfg.datamodule.num_workers = 0
            cfg.datamodule.pin_memory = False
            cfg.datamodule.ot = False
            cfg.datamodule.use_saved_mean_and_variance = True

            cfg.model.compile = False
            cfg.model.scheduler = None
            cfg.logger = None
            cfg.test = False
            cfg.callbacks.model_checkpoint.save_last = True
            if cfg.get("callbacks") is not None and "lr_monitor" in cfg.callbacks:
                cfg.callbacks.lr_monitor = None
    return cfg


@pytest.fixture(scope="function")
def fake_surge_smoke_datasets(
    tmp_path: Path, param_spec_name: str, install_fake_plugin: FakeVST3Plugin
) -> Path:
    """Render the N-sample Surge smoke dataset in-process as native Lance shards (no real VST/X11).

    The fast counterpart to :func:`surge_xt_smoke_datasets`: ``install_fake_plugin``
    swaps the loader for ``FakeVST3Plugin`` so :func:`make_lance_dataset` writes
    structurally-valid ``{train,val,test}.lance`` shards directly. Lets oracle-eval
    tests that only need a loadable dataset (not real audio fidelity) run on the
    CPU-fast loop.

    :param tmp_path: Per-test temporary directory; the dataset is written under
        ``tmp_path / "data" / "smoke-lance"``.
    :param param_spec_name: Param spec name (key into :data:`synth_setter.data.vst.param_specs`
        and :data:`synth_setter.data.vst.plugin_state_paths`); defaults to ``"surge_4"``.
    :param install_fake_plugin: Swaps ``core.load_plugin`` / ``core.VST3Plugin``
        for the fake so the render needs no real VST3 binary or display server.

    :return: Path to the directory holding ``{train,val,test}.lance`` and ``stats.npz``.
    """
    return _build_surge_smoke_lance_datasets(
        tmp_path, param_spec_name, _render_smoke_train_lance_fake
    )


@pytest.fixture(scope="function")
def cfg_surge_xt(
    cfg_surge_xt_global: DictConfig, tmp_path: Path, surge_xt_smoke_datasets: Path
) -> DictConfig:
    """Per-test wrapper around `cfg_surge_xt_global` that sets `tmp_path`-scoped output dirs.

    :param cfg_surge_xt_global: The Surge XT training config (parametrized over accelerator, param_spec_name, and experiment_name).
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with output and log dirs pointing at `tmp_path`.
    """
    cfg = cfg_surge_xt_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.dataset_root = str(surge_xt_smoke_datasets)
        cfg.datamodule.predict_file = str(surge_xt_smoke_datasets / "test.lance")

    yield cfg

    GlobalHydra.instance().clear()


# One dataset-format arm of the Surge XT smoke train/eval parity matrix. Private NamedTuple
# (no public docstring) matching the sibling ``_FakeOracleDataset`` in test_eval.py.
class _SurgeSmokeVariant(NamedTuple):
    dataset_fixture: str  # conftest fixture yielding the dataset root dir
    datamodule_group: str  # Hydra ``datamodule=`` group: "surge_lance"
    split_ext: str  # split file suffix: ".lance"
    plugin_path: str  # render plugin for eval postprocessing: real PLUGIN_PATH | fake.vst3


# Real- and fake-plugin Lance smoke fixtures share the same map-style datamodule.
REAL_VST_VARIANTS = [
    pytest.param(
        _SurgeSmokeVariant("surge_xt_smoke_datasets", "surge_lance", ".lance", PLUGIN_PATH),
        id="lance",
    )
]
FAKE_VST_VARIANTS = [
    pytest.param(
        _SurgeSmokeVariant(
            "fake_surge_smoke_datasets", "surge_lance", ".lance", "plugins/fake.vst3"
        ),
        id="lance",
    )
]


@pytest.fixture(scope="function")
def surge_smoke_variant(request: pytest.FixtureRequest) -> _SurgeSmokeVariant:
    """Indirect-parametrized dataset-format arm for the Surge smoke train/eval cfgs.

    :param request: Carries the the :class:`_SurgeSmokeVariant` arm under ``request.param``.
    :return: The dataset-format arm under test.
    """
    return request.param


def _apply_smoke_train_paths(
    cfg: DictConfig, dataset_root: Path, variant: _SurgeSmokeVariant, tmp_path: Path
) -> None:
    """Pin a smoke train cfg's output dirs and dataset/predict paths to a variant.

    :param cfg: Config composed by :func:`_build_surge_xt_smoke_cfg`.
    :param dataset_root: Directory holding the variant's ``{train,val,test}`` splits.
    :param variant: Dataset-format arm selecting the predict-split suffix.
    :param tmp_path: Shared output/log directory (and checkpoint parent for eval).
    """
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.dataset_root = str(dataset_root)
        cfg.datamodule.predict_file = str(dataset_root / f"test{variant.split_ext}")


@pytest.fixture(scope="function")
def cfg_surge_real_train(
    surge_smoke_variant: _SurgeSmokeVariant,
    request: pytest.FixtureRequest,
    accelerator: str,
    param_spec_name: str,
    experiment_name: str,
    tmp_path: Path,
) -> Iterator[DictConfig]:
    """Real-VST Surge smoke train cfg for the requested dataset-format arm.

    Composes over the ``accelerator`` fixture so the h5 and Lance arms each inherit the
    cpu/mps/gpu matrix. The dataset fixture named by the variant renders through the real
    Surge XT subprocess.

    :param surge_smoke_variant: Dataset-format arm (h5 or Lance) under test.
    :param request: Resolves the variant's dataset fixture via ``getfixturevalue``.
    :param accelerator: Parametrized accelerator driving ``trainer.accelerator``.
    :param param_spec_name: Param spec the cfg is wired for.
    :param experiment_name: Hydra ``experiment=...`` override.
    :param tmp_path: The temporary logging path.
    :yields DictConfig: One-step train cfg pinned to the variant's splits.
    """
    dataset_root = request.getfixturevalue(surge_smoke_variant.dataset_fixture)
    cfg = _build_surge_xt_smoke_cfg(
        accelerator=accelerator,
        param_spec_name=param_spec_name,
        experiment=experiment_name,
        datamodule_group=surge_smoke_variant.datamodule_group,
    )
    _apply_smoke_train_paths(cfg, dataset_root, surge_smoke_variant, tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_surge_fake_train(
    surge_smoke_variant: _SurgeSmokeVariant,
    request: pytest.FixtureRequest,
    param_spec_name: str,
    experiment_name: str,
    tmp_path: Path,
) -> Iterator[DictConfig]:
    """Fake-plugin CPU Surge smoke train cfg for the requested dataset-format arm.

    The CPU-fast counterpart to :func:`cfg_surge_real_train`: pins ``accelerator="cpu"``
    and resolves a fake-plugin dataset fixture, so the h5 and Lance arms run in the
    inner test loop with no real VST host.

    :param surge_smoke_variant: Dataset-format arm (h5 or Lance) under test.
    :param request: Resolves the variant's dataset fixture via ``getfixturevalue``.
    :param param_spec_name: Param spec the cfg is wired for.
    :param experiment_name: Hydra ``experiment=...`` override.
    :param tmp_path: The temporary logging path.
    :yields DictConfig: One-step CPU train cfg pinned to the variant's splits.
    """
    dataset_root = request.getfixturevalue(surge_smoke_variant.dataset_fixture)
    cfg = _build_surge_xt_smoke_cfg(
        accelerator="cpu",
        param_spec_name=param_spec_name,
        experiment=experiment_name,
        datamodule_group=surge_smoke_variant.datamodule_group,
    )
    _apply_smoke_train_paths(cfg, dataset_root, surge_smoke_variant, tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_surge_xt_eval(
    cfg_surge_xt_global: DictConfig,
    tmp_path: Path,
    surge_xt_smoke_datasets: Path,
    param_spec_name: str,
) -> DictConfig:
    """Eval config for the Surge XT train->eval smoke-test roundtrip.

    Inherits from `cfg_surge_xt_global` and points `ckpt_path` at the checkpoint
    that `cfg_surge_xt`'s training run will write under the same `tmp_path`.

    :param cfg_surge_xt_global: The Surge XT training config (parametrized over accelerator, param_spec_name, and experiment_name).
    :param tmp_path: The temporary logging path (shared with `cfg_surge_xt`).
    :param param_spec_name: Keys ``plugin_state_paths`` so ``cfg.render`` matches the
        spec the model was trained against.

    :return: A DictConfig configured to evaluate a Surge XT checkpoint on the smoke-test
        dataset.
    """
    cfg = cfg_surge_xt_global.copy()
    _configure_surge_xt_eval_cfg(
        cfg,
        tmp_path=tmp_path,
        dataset_root=surge_xt_smoke_datasets,
        predict_file=surge_xt_smoke_datasets / "test.lance",
        param_spec_name=param_spec_name,
        plugin_path=PLUGIN_PATH,
        rerender_target=True,
    )

    yield cfg

    GlobalHydra.instance().clear()


def _configure_surge_xt_eval_cfg(
    cfg: DictConfig,
    *,
    tmp_path: Path,
    dataset_root: Path,
    predict_file: Path,
    param_spec_name: str,
    plugin_path: str,
    rerender_target: bool,
) -> None:
    """Apply shared train-to-predict eval overrides to a Surge smoke config.

    :param cfg: Config copied from the matching train fixture.
    :param tmp_path: Shared output/log directory and checkpoint parent.
    :param dataset_root: Directory holding the eval split and stats.
    :param predict_file: Split consumed by ``trainer.predict``.
    :param param_spec_name: Param spec used by the rendered fixture.
    :param plugin_path: Plugin path forwarded to render postprocessing.
    :param rerender_target: Whether postprocessing should render target audio.
    """
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.datamodule.batch_size = 1
        cfg.datamodule.dataset_root = str(dataset_root)
        cfg.datamodule.predict_file = str(predict_file)
        cfg.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg.mode = "predict"
        cfg.evaluation = {
            "render_vst": True,
            "compute_metrics": True,
            "rerender_target": rerender_target,
            "num_workers": 1,
        }
        cfg.render = {
            "param_spec_name": param_spec_name,
            "plugin_state_path": plugin_state_paths[param_spec_name],
            "plugin_path": plugin_path,
        }


@pytest.fixture(scope="function")
def cfg_surge_real_eval(
    surge_smoke_variant: _SurgeSmokeVariant,
    request: pytest.FixtureRequest,
    accelerator: str,
    param_spec_name: str,
    experiment_name: str,
    tmp_path: Path,
) -> Iterator[DictConfig]:
    """Real-VST predict-mode eval cfg matching :func:`cfg_surge_real_train`.

    Shares ``tmp_path`` (and so the checkpoint), ``accelerator``, ``experiment_name`` and
    ``surge_smoke_variant`` with the train fixture, so the train->eval roundtrip reads the
    checkpoint training writes for the same dataset-format arm.

    :param surge_smoke_variant: Dataset-format arm (h5 or Lance) under test.
    :param request: Resolves the variant's dataset fixture via ``getfixturevalue``.
    :param accelerator: Parametrized accelerator driving ``trainer.accelerator``.
    :param param_spec_name: Param spec used by the rendered fixture.
    :param experiment_name: Hydra ``experiment=...`` override.
    :param tmp_path: The temporary logging path shared with training.
    :yields DictConfig: Config that predicts from ``last.ckpt`` over the variant's splits.
    """
    dataset_root = request.getfixturevalue(surge_smoke_variant.dataset_fixture)
    cfg = _build_surge_xt_smoke_cfg(
        accelerator=accelerator,
        param_spec_name=param_spec_name,
        experiment=experiment_name,
        datamodule_group=surge_smoke_variant.datamodule_group,
    )
    _configure_surge_xt_eval_cfg(
        cfg,
        tmp_path=tmp_path,
        dataset_root=dataset_root,
        predict_file=dataset_root / f"test{surge_smoke_variant.split_ext}",
        param_spec_name=param_spec_name,
        plugin_path=surge_smoke_variant.plugin_path,
        rerender_target=True,
    )

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_surge_fake_eval(
    surge_smoke_variant: _SurgeSmokeVariant,
    request: pytest.FixtureRequest,
    param_spec_name: str,
    experiment_name: str,
    tmp_path: Path,
) -> Iterator[DictConfig]:
    """Fake-plugin CPU predict-mode eval cfg matching :func:`cfg_surge_fake_train`.

    The CPU-fast counterpart to :func:`cfg_surge_real_eval`: the postprocessing subprocess
    is stubbed by the test body, so this only needs to point ``trainer.predict`` at the
    variant's fake-plugin splits.

    :param surge_smoke_variant: Dataset-format arm (h5 or Lance) under test.
    :param request: Resolves the variant's dataset fixture via ``getfixturevalue``.
    :param param_spec_name: Param spec used by the rendered fixture.
    :param experiment_name: Hydra ``experiment=...`` override.
    :param tmp_path: The temporary logging path shared with training.
    :yields DictConfig: Config that predicts from ``last.ckpt`` over the variant's splits.
    """
    dataset_root = request.getfixturevalue(surge_smoke_variant.dataset_fixture)
    cfg = _build_surge_xt_smoke_cfg(
        accelerator="cpu",
        param_spec_name=param_spec_name,
        experiment=experiment_name,
        datamodule_group=surge_smoke_variant.datamodule_group,
    )
    _configure_surge_xt_eval_cfg(
        cfg,
        tmp_path=tmp_path,
        dataset_root=dataset_root,
        predict_file=dataset_root / f"test{surge_smoke_variant.split_ext}",
        param_spec_name=param_spec_name,
        plugin_path=surge_smoke_variant.plugin_path,
        rerender_target=True,
    )

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture
def fake_vst3_plugin() -> FakeVST3Plugin:
    """Return a fresh ``FakeVST3Plugin`` instance per test (no shared state).

    :returns: Stand-in plugin whose ``plugin_path`` field is set but never
        read from disk; downstream production code receives this via
        ``install_fake_plugin``.
    """
    return FakeVST3Plugin("plugins/fake.vst3")


@pytest.fixture
def install_fake_plugin(
    monkeypatch: pytest.MonkeyPatch, fake_vst3_plugin: FakeVST3Plugin
) -> FakeVST3Plugin:
    """Patch ``core.load_plugin`` and ``core.VST3Plugin`` to yield the fake.

    Both seams are covered: ``load_plugin`` is the normal pipeline entry
    point; ``VST3Plugin`` is constructed directly by
    ``extract_renderer_version``'s fallback path.

    :param monkeypatch: Pytest fixture used to swap the two ``core``
        callables for the test's duration; teardown restores both.
    :param fake_vst3_plugin: The instance the patched callables return.
    :returns: The same ``fake_vst3_plugin`` instance, so tests asserting
        on it can compare by identity.
    """
    monkeypatch.setattr(core, "load_plugin", lambda _path, **_kw: fake_vst3_plugin)
    monkeypatch.setattr(core, "VST3Plugin", lambda _path: fake_vst3_plugin)
    return fake_vst3_plugin


def _base_dataset_spec_kwargs() -> dict[str, Any]:
    """Return the skeleton shared by hand-built ``DatasetSpec`` test specs.

    Carries the fields both the wandb-tracking and parallel-dispatch tests fix
    identically (deterministic ``created_at`` / ``git_sha``, lance output, the
    Darwin-portable ``gui_toggle_cadence``); per-test fields (``task_name``,
    ``run_id``, ``r2``, shard sizes, ``parallel``) come in as overrides.

    :returns: A fresh kwargs dict safe for in-place override per call.
    """
    return {
        "created_at": datetime(2026, 5, 20, 0, 0, 0, tzinfo=UTC),
        "git_sha": "0" * 40,
        "is_repo_dirty": False,
        "output_format": "lance",
        "base_seed": 42,
        "render": {
            "plugin_path": "plugins/fake.vst3",
            "plugin_state_path": "presets/fake.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "0.0.0-fake",
            "sample_rate": 44100,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 1.0,
            "min_loudness": -60.0,
            # Darwin-portable (#714).
            "gui_toggle_cadence": "never",
        },
    }


@pytest.fixture()
def dataset_spec_factory() -> Callable[..., DatasetSpec]:
    """Build a ``DatasetSpec`` from the shared skeleton plus per-test overrides.

    ``render`` overrides deep-merge into the skeleton's ``render`` block; all
    other keyword arguments replace top-level fields. Consolidates the
    previously duplicated ``_build_spec`` helpers in the wandb-tracking and
    parallel-dispatch tests.

    :returns: ``factory(*, render=None, **overrides) -> DatasetSpec``.
    """

    def factory(*, render: dict[str, Any] | None = None, **overrides: Any) -> DatasetSpec:
        kwargs = copy.deepcopy(_base_dataset_spec_kwargs())
        if render is not None:
            kwargs["render"].update(render)
        kwargs.update(overrides)
        return DatasetSpec(**kwargs)  # type: ignore[arg-type]

    return factory


def _cgroup_aware_cpu_count() -> int:
    """Return CPUs available to this process, honouring cgroup quota and affinity.

    Takes min(affinity, cgroup_quota) so ``-n auto`` doesn't over-subscribe the
    container — see #1490 for the "worker crashed" failure mode this fixes.

    :returns: Usable CPU count, always at least 1.
    """
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    else:
        affinity = os.cpu_count() or 1

    quota: float | None = None
    try:  # cgroup v2: unified hierarchy, kernel >= 4.5
        with open("/sys/fs/cgroup/cpu.max") as fh:
            parts = fh.read().split()
            if len(parts) >= 2 and parts[0] != "max":
                quota = int(parts[0]) / int(parts[1])
    except (OSError, ValueError, ZeroDivisionError):
        try:  # cgroup v1: legacy per-subsystem hierarchy
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
                quota_us = int(fh.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
                period_us = int(fh.read())
            if quota_us > 0 and period_us > 0:
                quota = quota_us / period_us
        except (OSError, ValueError, ZeroDivisionError):
            pass  # no cgroup limit; use affinity only

    cpus = affinity if quota is None else min(affinity, quota)
    return max(1, int(cpus))


# Per-worker resident-memory budget for the -n auto clamp (override via
# PYTEST_XDIST_WORKER_MEM_MB); 2 GiB = measured peak RSS per worker slot (#1646).
_DEFAULT_WORKER_MEM_MB = 2048
_LOCAL_DARWIN_XDIST_WORKERS = 4

# Wall-clock budget for the whole session (seconds); each Makefile lane pins its
# own value (#2274) so a silently degraded run fails loudly instead of crawling.
_SESSION_BUDGET_ENV = "PYTEST_SESSION_BUDGET_SECONDS"

# Stored on the session object (not a module global) so unit tests driving fake
# sessions through these hooks can't corrupt the live session's own budget.
_SESSION_BUDGET_START_ATTR = "_synth_setter_session_budget_start"


def _session_budget_seconds() -> float | None:
    """Parse the session budget env var, failing open on anything non-positive.

    :returns: Budget in seconds, or None when unset, malformed, or <= 0.
    """
    raw_budget = os.environ.get(_SESSION_BUDGET_ENV)
    if not raw_budget:
        return None
    try:
        budget = float(raw_budget)
    except ValueError:
        return None
    return budget if budget > 0 else None


def pytest_sessionstart(session: pytest.Session) -> None:
    """Record the controller's session start time for budget enforcement.

    :param session: The pytest session being started.
    """
    if hasattr(session.config, "workerinput"):
        return  # xdist worker: only the controller owns the wall clock
    setattr(session, _SESSION_BUDGET_START_ATTR, time.monotonic())


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail an otherwise-green session that blew its wall-clock budget.

    :param session: The finishing pytest session; its exitstatus may be raised to 1.
    :param exitstatus: Exit status the run would report; nonzero values are preserved.
    """
    if hasattr(session.config, "workerinput") or exitstatus != 0:
        return
    budget = _session_budget_seconds()
    budget_start = getattr(session, _SESSION_BUDGET_START_ATTR, None)
    if budget is None or budget_start is None:
        return
    elapsed = time.monotonic() - budget_start
    if elapsed > budget:
        sys.stderr.write(
            f"\nERROR: session budget exceeded: {elapsed:.1f}s > {_SESSION_BUDGET_ENV}={budget}s\n"
        )
        session.exitstatus = 1


# CPUs left free on local (non-CI) runs so the host stays responsive while the
# suite runs (#2274); override via PYTEST_XDIST_RESERVED_CPUS.
_DEFAULT_RESERVED_CPUS = 2


def _reserved_cpu_count() -> int:
    """Return how many CPUs the ``-n auto`` clamp must leave unused.

    ``PYTEST_XDIST_RESERVED_CPUS`` wins when set (invalid values fall back to
    the default); otherwise CI hosts reserve nothing — they are dedicated —
    and local runs reserve the default headroom.

    :returns: Non-negative CPU count to subtract from the CPU term.
    """
    raw_reserve = os.environ.get("PYTEST_XDIST_RESERVED_CPUS")
    if raw_reserve:
        try:
            return max(0, int(raw_reserve))
        except ValueError:
            pass  # non-integer override -> ignore and fall through to the defaults
    return 0 if os.environ.get("CI") else _DEFAULT_RESERVED_CPUS


# A v1 memory.limit_in_bytes at or above this is the kernel's unlimited sentinel.
_MEM_UNLIMITED_SENTINEL = 1 << 62


def _meminfo_available_bytes() -> int | None:
    """Read ``MemAvailable`` from ``/proc/meminfo`` as bytes.

    :returns: Host available memory in bytes, or None if the field is unreadable.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # field is in kibibytes
    except (OSError, ValueError, IndexError):
        pass
    return None


def _swap_in_use_bytes() -> int | None:
    """Read swap currently in use from ``/proc/meminfo`` (``SwapTotal - SwapFree``).

    :returns: Bytes of swap in use, or None if either field is unreadable.
    """
    total: int | None = None
    free: int | None = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("SwapTotal:"):
                    total = int(line.split()[1]) * 1024  # field is in kibibytes
                elif line.startswith("SwapFree:"):
                    free = int(line.split()[1]) * 1024
                if total is not None and free is not None:
                    break
    except (OSError, ValueError, IndexError):
        return None
    if total is None or free is None:
        return None
    return max(0, total - free)


def _cgroup_memory_limit_bytes() -> int | None:
    """Read the cgroup memory limit in bytes, honouring v2 then v1.

    :returns: The cgroup memory cap in bytes, or None when unset or unlimited.
    """
    try:  # cgroup v2: unified hierarchy, kernel >= 4.5
        with open("/sys/fs/cgroup/memory.max") as fh:
            limit_field = fh.read().split()[0]
            if limit_field != "max":
                return int(limit_field)
            return None
    except (OSError, ValueError, IndexError):
        pass
    try:  # cgroup v1: legacy per-subsystem hierarchy
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as fh:
            limit = int(fh.read())
        if 0 < limit < _MEM_UNLIMITED_SENTINEL:
            return limit
    except (OSError, ValueError):
        pass
    return None


def _available_memory_bytes() -> int | None:
    """Return usable memory as ``min(swap-adjusted host MemAvailable, cgroup limit)``.

    MemAvailable counts reclaimable cache as free, so it overstates headroom on a swapping host;
    used swap is debited from the host term before the min, a no-op when nothing is swapped.

    :returns: The tighter of the two figures in bytes, or None if neither is known.
    """
    host = _meminfo_available_bytes()
    if host is not None:
        swap_used = _swap_in_use_bytes()
        if swap_used is not None:
            host = max(0, host - swap_used)
    candidates = [value for value in (host, _cgroup_memory_limit_bytes()) if value is not None]
    return min(candidates) if candidates else None


def _memory_aware_worker_count() -> int | None:
    """Cap ``-n auto`` workers by available memory and a per-worker budget.

    Divides available memory by ``PYTEST_XDIST_WORKER_MEM_MB`` (default 2 GiB) so
    a busy shared host doesn't OOM-kill the run — the failure #1490's CPU clamp
    can't catch, since neither cpu.max nor memory.max is set on a shared host.

    :returns: Memory-bounded worker count (>=1), or None when memory is unknown.
    """
    available = _available_memory_bytes()
    if available is None:
        return None
    raw_budget = os.environ.get("PYTEST_XDIST_WORKER_MEM_MB")
    try:
        budget_mb = int(raw_budget) if raw_budget else _DEFAULT_WORKER_MEM_MB
    except ValueError:
        budget_mb = _DEFAULT_WORKER_MEM_MB
    if budget_mb <= 0:
        budget_mb = _DEFAULT_WORKER_MEM_MB
    return max(1, available // (budget_mb * 1024 * 1024))


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:  # noqa: ARG001
    """Override pytest-xdist ``-n auto`` to fit the host's CPU and memory headroom.

    Checks ``PYTEST_XDIST_AUTO_NUM_WORKERS`` first so the env-var escape hatch
    that xdist's built-in implementation honours is preserved even when this
    hook wins the ``firstresult`` race. A non-integer or empty pin is ignored,
    not fatal. Otherwise returns ``min(cpu - reserved, memory)`` and applies the
    local Darwin worker cap so nested ML multiprocessing retains interactive headroom.

    :param config: The pytest config object (unused; required by the hook signature).
    :returns: Worker count clamped to the host's real CPU and memory allocation.
    """
    env_override = os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS")
    if env_override:
        try:
            return max(1, int(env_override))
        except ValueError:
            pass  # non-integer pin -> ignore and fall through to the adaptive clamps
    cpu_workers = max(1, _cgroup_aware_cpu_count() - _reserved_cpu_count())
    mem_workers = _memory_aware_worker_count()
    allocated_workers = cpu_workers if mem_workers is None else min(cpu_workers, mem_workers)
    if sys.platform == "darwin" and not os.environ.get("CI"):
        return min(allocated_workers, _LOCAL_DARWIN_XDIST_WORKERS)
    return allocated_workers


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom CLI options for the test suite."""
    parser.addoption(
        "--compare-baseline-configs-keep-yaml-dir",
        action="store",
        default=None,
        metavar="DIR",
        help=(
            "Directory to capture resolved Hydra YAMLs into. When set, the "
            "python-shim harness writes <test-id>__<role>.yaml into DIR "
            "instead of pytest's tmp_path so the files survive between runs."
        ),
    )


# Lance datamodule smoke fixtures.

# vst_ffn's AST net hard-codes the production mel shape and channel count, so the
# Lance smoke fixture must carry production-shaped mel rows; everything else is tiny.
_LANCE_SMOKE_MEL_SHAPE = (2, 128, 401)
_LANCE_SMOKE_PARAM_SPEC = "surge_4"
_LANCE_SMOKE_NUM_PARAMS = len(param_specs[_LANCE_SMOKE_PARAM_SPEC])
_LANCE_SMOKE_ROWS = 4


def _write_lance_smoke_split(path: Path, num_rows: int, *, seed: int) -> None:
    """Write one surge-shaped ``.lance`` split readable by ``LanceVSTDataModule``.

    :param path: Output ``.lance`` shard file.
    :param num_rows: Rows in every column.
    :param seed: RNG seed so splits get distinguishable values.
    """
    # Local import: pulls in pyarrow, which the Docker VST CI images don't
    # install (no `data` dependency group) — module scope would break their
    # conftest collection.
    from tests.helpers.lance_fixtures import write_lance_shard

    rng = np.random.default_rng(seed)
    write_lance_shard(
        path,
        {
            # float16 mirrors the pipeline's on-disk audio dtype (DATASET_FIELD_DTYPES).
            "audio": rng.uniform(-1.0, 1.0, (num_rows, 2, 64)).astype(np.float16),
            "mel_spec": rng.standard_normal((num_rows, *_LANCE_SMOKE_MEL_SHAPE)).astype(
                np.float32
            ),
            "param_array": rng.random((num_rows, _LANCE_SMOKE_NUM_PARAMS)).astype(np.float32),
        },
    )


@pytest.fixture
def cfg_train_lance(tmp_path: Path) -> Iterator[DictConfig]:
    """Compose a ``datamodule=surge_lance`` training cfg over a generated Lance dataset.

    Writes tiny ``train/val/test.lance`` splits + ``stats.npz`` under
    ``tmp_path``, then composes the real ``train.yaml`` with
    ``datamodule=surge_lance`` — the same Hydra path a user takes — shrinking
    the AST net to a 1-layer toy so a ``fast_dev_run`` step stays CPU-cheap.

    :param tmp_path: Per-test tmpdir holding the dataset and output/log dirs.
    :yields: Resolved DictConfig ready for ``train(cfg)``.
    :ytype: DictConfig
    """
    dataset_root = tmp_path / "lance-data"
    dataset_root.mkdir()
    for seed, split in enumerate(("train", "val", "test")):
        _write_lance_smoke_split(dataset_root / f"{split}.lance", _LANCE_SMOKE_ROWS, seed=seed)
    np.savez(
        dataset_root / "stats.npz",
        mean=np.zeros(_LANCE_SMOKE_MEL_SHAPE, dtype=np.float32),
        std=np.ones(_LANCE_SMOKE_MEL_SHAPE, dtype=np.float32),
    )

    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["datamodule=surge_lance", "model=vst_ffn", "trainer=cpu"],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(operator_workspace())
            cfg.paths.output_dir = str(tmp_path)
            cfg.paths.log_dir = str(tmp_path)
            cfg.logger = None
            # lr_monitor requires an attached logger, which this smoke cfg disables.
            if "lr_monitor" in cfg.callbacks:
                del cfg.callbacks.lr_monitor
            cfg.trainer.fast_dev_run = True
            # Not a loop bound under fast_dev_run — vst_ffn's scheduler resolves
            # ${trainer.max_steps}, which trainer/cpu.yaml leaves undefined.
            cfg.trainer.max_steps = 1
            cfg.datamodule.dataset_root = str(dataset_root)
            cfg.datamodule.batch_size = 1
            cfg.datamodule.param_spec_name = _LANCE_SMOKE_PARAM_SPEC
            cfg.datamodule.ot = False
            cfg.datamodule.num_workers = 0
            cfg.datamodule.pin_memory = False
            cfg.model.compile = False
            cfg.model.net.d_model = 32
            cfg.model.net.n_heads = 2
            cfg.model.net.n_layers = 1

    yield cfg

    GlobalHydra.instance().clear()
