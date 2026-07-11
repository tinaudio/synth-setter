"""Config fixtures and collection-time skip hooks for the test suite."""

import copy
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import h5py
import hdf5plugin  # noqa: F401   side-effect import: registers HDF5_PLUGIN_PATH so h5py can load Blosc2 filters in fixtures
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from synth_setter.data.vst import core, param_specs, preset_paths
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig
from synth_setter.resources import vst_headless_wrapper
from synth_setter.utils.utils import register_resolvers
from synth_setter.workspace import operator_workspace
from tests._baseline_worktree import worktree_for_ref  # noqa: F401 — pytest fixture re-export
from tests._vst import PLUGIN_PATH, VST_AVAILABLE, VST_SUBPROCESS_TIMEOUT_SECONDS
from tests.data.vst._fake_plugin import FakeVST3Plugin
from tests.pipeline.conftest import fake_r2_remote  # noqa: F401 — pytest fixture re-export

# Per-clip dimensions for the smoke fixture's HDF5 output. ``RenderConfig`` in
# ``synth_setter.pipeline.schemas.spec`` declares no field defaults — the fixture passes
# explicit values for every flag below, so these constants must match the values
# the subprocess is invoked with.
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
    """Assert the generated Surge XT dataset is structurally sound.

    Verifies the three required datasets exist with the expected shapes, that no NaN/Inf leaked in
    from the VST/mel pipeline, and that every audio clip is above the silence floor — surface those
    failures here rather than letting downstream training crash on opaque NaN losses.
    """
    with h5py.File(path, "r") as f:
        for name in ("audio", "mel_spec", "param_array"):
            assert name in f, f"missing dataset {name!r} in {path}"

        audio = f["audio"]
        mel = f["mel_spec"]
        params = f["param_array"]
        # `h5py.File.__getitem__` returns `Group | Dataset | Datatype`; the
        # generator only writes Datasets, so narrow the type for shape access.
        assert isinstance(audio, h5py.Dataset), f"'audio' is not a Dataset in {path}"
        assert isinstance(mel, h5py.Dataset), f"'mel_spec' is not a Dataset in {path}"
        assert isinstance(params, h5py.Dataset), f"'param_array' is not a Dataset in {path}"

        expected_audio_shape = (
            num_samples,
            _SURGE_AUDIO_CHANNELS,
            _SURGE_AUDIO_SAMPLES_PER_CLIP,
        )
        assert audio.shape == expected_audio_shape, (
            f"audio shape {audio.shape} != expected {expected_audio_shape}"
        )
        assert mel.shape == (num_samples, *_SURGE_MEL_SHAPE), (
            f"mel_spec shape {mel.shape} != expected {(num_samples, *_SURGE_MEL_SHAPE)}"
        )
        assert params.shape[0] == num_samples, (
            f"param_array first dim {params.shape[0]} != num_samples {num_samples}"
        )
        assert params.ndim == 2, f"param_array must be 2D, got shape {params.shape}"

        audio_arr = audio[...].astype(np.float32)
        mel_arr = mel[...]
        params_arr = params[...]
        assert np.isfinite(audio_arr).all(), f"audio in {path} contains NaN/Inf"
        assert np.isfinite(mel_arr).all(), f"mel_spec in {path} contains NaN/Inf"
        assert np.isfinite(params_arr).all(), f"param_array in {path} contains NaN/Inf"

        per_clip_peak = np.abs(audio_arr).reshape(num_samples, -1).max(axis=1)
        silent = np.where(per_clip_peak <= _SURGE_SILENCE_PEAK_THRESHOLD)[0]
        assert silent.size == 0, (
            f"audio clips {silent.tolist()} in {path} are silent "
            f"(peaks={per_clip_peak[silent].tolist()})"
        )


def _write_smoke_stats_npz(train_h5: Path) -> None:
    """Write the sibling ``stats.npz`` for a smoke dataset via the stats CLI subprocess.

    Runs ``python -m synth_setter.pipeline.data.stats <train_h5> --mask-degenerate-bins``;
    the subprocess registers the hdf5plugin Blosc2 filter on import, which the
    in-process dask path does not surface to its workers. Fails loud on timeout
    or non-zero exit so a broken stats fold surfaces here, not as a downstream
    datamodule load error.

    :param train_h5: Path to the rendered ``train.h5``; ``stats.npz`` is written beside it.
    """
    stats_args = [
        sys.executable,
        "-m",
        "synth_setter.pipeline.data.stats",
        str(train_h5),
        "--mask-degenerate-bins",
    ]
    try:
        result = subprocess.run(  # noqa: S603
            stats_args, text=True, check=False, timeout=VST_SUBPROCESS_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"get_dataset_stats timed out after {VST_SUBPROCESS_TIMEOUT_SECONDS}s\n"
            f"command: {stats_args}\n"
            f"(child stdout/stderr printed above; rerun with `pytest -s` if captured)",
            pytrace=False,
        )
    if result.returncode != 0:
        pytest.fail(
            f"get_dataset_stats failed (exit {result.returncode})\n"
            f"command: {stats_args}\n"
            f"(child stdout/stderr printed above; rerun with `pytest -s` if captured)",
            pytrace=False,
        )
    assert train_h5.parent.joinpath("stats.npz").exists(), (
        "get_dataset_stats failed to produce stats.npz fixture"
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
            cfg.datamodule.num_workers = 4
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

    :return: A key into :data:`synth_setter.data.vst.param_specs` and :data:`synth_setter.data.vst.preset_paths`.
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
    datamodule_group: Literal["surge", "surge_lance"] = "surge",
) -> DictConfig:
    """Construct the Surge XT smoke-test config without the accelerator availability gate.

    Composes ``train.yaml`` with ``experiment=<experiment>`` and bakes in the minimal
    overrides needed to train-smoke-test on the dataset generated by
    :func:`surge_xt_smoke_datasets`. The ``model.net.d_out`` and
    ``callbacks.log_per_param_mse.param_spec`` bake-ins re-pin the per-experiment YAML
    defaults to the smoke fixture's ``param_spec_name``, so the YAML's production
    ``d_out`` / ``param_spec`` values do not leak into the smoke path. Used both by the
    :func:`cfg_surge_xt_global` fixture (where the parametrized ``accelerator`` is
    host-checked upstream) and by the ``configs/experiment/surge/test-mps*.yaml``
    equality test (where the cfg must be built on any host so the YAMLs never silently
    drift from this builder).

    :param accelerator: Lightning ``trainer.accelerator`` — ``"cpu"``, ``"mps"``, or ``"gpu"``.
    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs`; drives
        ``model.net.d_out`` and ``callbacks.log_per_param_mse.param_spec``.
    :param experiment: Hydra ``experiment=...`` override (e.g. ``"surge/fake_oracle"``,
        ``"surge/ffn_full"``); selects which model the smoke cfg wires up.
    :param datamodule_group: Hydra datamodule group override.

    :return: Resolved DictConfig with the smoke-test bake-ins applied.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                f"experiment={experiment}",
                f"datamodule={datamodule_group}",
                "callbacks=[default_surge,eval_surge]",
            ],
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

            # batch_size=1 is forced: ShiftedBatchSampler (used in the
            # VSTDataModule's train_dataloader) drops one batch per epoch,
            # so any batch_size > dataset_size // 2 leaves the dataloader empty
            # and Lightning aborts with "Trainer.fit stopped: No training batches."
            cfg.datamodule.batch_size = 1
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
            cfg.model.net.d_out = len(param_specs[param_spec_name])
            cfg.callbacks.log_per_param_mse.param_spec = param_spec_name
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


def build_fake_train_cfg(output_dir: Path, param_spec_name: str) -> DictConfig:
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
    :returns: Resolved one-step fake-mode train DictConfig.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["experiment=surge/fake_oracle", "trainer=cpu"],
        )
        with open_dict(cfg):
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
            # log_per_param_mse keys its spec off ${render.param_spec_name}; pin it
            # concretely — this train path composes no render group.
            cfg.callbacks.log_per_param_mse.param_spec = param_spec_name
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

    The writer is dispatched on ``output_path``'s suffix by ``generate_vst_dataset``
    (``.h5`` -> HDF5, ``.lance`` -> Lance), so this one renderer backs both the h5
    and Lance real-VST smoke fixtures.

    :param output_path: Destination shard path (``train.h5`` or ``train.lance``); its
        parent must already exist.
    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` and
        :data:`synth_setter.data.vst.preset_paths` selecting spec and preset.
    """
    generate_dataset_args = []
    if sys.platform == "linux":
        generate_dataset_args.append(VST_HEADLESS_WRAPPER)

    generate_dataset_args += [
        sys.executable,
        "src/synth_setter/data/vst/generate_vst_dataset.py",
        str(output_path),
        f"--plugin_path={PLUGIN_PATH}",
        f"--preset_path={preset_paths[param_spec_name]}",
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
    try:
        result = subprocess.run(  # noqa: S603
            generate_dataset_args,
            text=True,
            check=False,
            timeout=VST_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"generate_vst_dataset timed out after {VST_SUBPROCESS_TIMEOUT_SECONDS}s\n"
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
    """Build the one-shard fake-plugin ``RenderConfig`` shared by the h5 and Lance smoke renders.

    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` and
        :data:`synth_setter.data.vst.preset_paths` selecting spec and preset.
    :returns: A CPU ``RenderConfig`` with the GUI toggle disabled.
    """
    return RenderConfig(
        plugin_path=PLUGIN_PATH,
        preset_path=str(preset_paths[param_spec_name]),
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


def _render_smoke_train_h5_fake(train_h5: Path, param_spec_name: str) -> None:
    """Render the smoke ``train.h5`` in-process via ``make_hdf5_dataset``; requires the caller to have installed ``FakeVST3Plugin``.

    :param train_h5: Destination ``train.h5`` path; its parent must already exist.
    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` and
        :data:`synth_setter.data.vst.preset_paths` selecting spec and preset.
    """
    from synth_setter.data.vst.writers import make_hdf5_dataset

    make_hdf5_dataset(train_h5, _smoke_fake_render_cfg(param_spec_name))


def _render_smoke_train_lance_fake(train_lance: Path, param_spec_name: str) -> None:
    """Render the smoke ``train.lance`` in-process via ``make_lance_dataset``; requires the caller to have installed ``FakeVST3Plugin``.

    :param train_lance: Destination ``train.lance`` file; its parent must already exist.
    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` and
        :data:`synth_setter.data.vst.preset_paths` selecting spec and preset.
    """
    from synth_setter.data.vst.writers import make_lance_dataset

    make_lance_dataset(train_lance, _smoke_fake_render_cfg(param_spec_name))


def _build_surge_smoke_datasets(
    tmp_path: Path,
    param_spec_name: str,
    render_train_h5: Callable[[Path, str], None],
) -> Path:
    """Build the N-sample Surge smoke dataset; ``render_train_h5`` is the only difference between the real-VST and fake fixtures.

    :param tmp_path: Per-test temporary directory; the dataset is written under
        ``tmp_path / "data" / "smoke"``.
    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` and
        :data:`synth_setter.data.vst.preset_paths` selecting spec and preset.
    :param render_train_h5: Renders ``train.h5`` given ``(train_h5, param_spec_name)``.

    :return: Path to the directory holding ``{train,val,test}.h5`` and ``stats.npz``.
    """
    smoke_dataset_dir = tmp_path / "data" / "smoke"
    smoke_dataset_dir.mkdir(parents=True, exist_ok=True)
    train_h5 = smoke_dataset_dir / "train.h5"

    render_train_h5(train_h5, param_spec_name)
    _validate_surge_dataset(train_h5, NUM_FIXTURE_SAMPLES)

    # Sibling stats.npz; shared across train/val/test splits — see #1002. Subprocess
    # path: in-process dask workers miss the hdf5plugin Blosc2 filter.
    _write_smoke_stats_npz(train_h5)

    shutil.copy(train_h5, smoke_dataset_dir / "val.h5")
    shutil.copy(train_h5, smoke_dataset_dir / "test.h5")
    return smoke_dataset_dir


def _build_surge_smoke_lance_datasets(
    tmp_path: Path,
    param_spec_name: str,
    render_train_lance: Callable[[Path, str], None],
) -> Path:
    """Render the N-sample Surge smoke dataset natively as single-file Lance shards.

    The Lance counterpart to :func:`_build_surge_smoke_datasets`: ``render_train_lance``
    is the only difference between the real-VST and fake fixtures. It renders
    ``train.lance`` through the production :func:`make_lance_dataset` writer, then this
    folds the mel rows into ``stats.npz`` via :func:`stream_stats_lance` and clones the
    split into ``val``/``test``. No HDF5 file is produced — every shard carries the
    exact on-disk format the pipeline's Lance finalize emits.

    :param tmp_path: Per-test temporary directory; the dataset is written under
        ``tmp_path / "data" / "smoke-lance"``.
    :param param_spec_name: Key into :data:`synth_setter.data.vst.param_specs` and
        :data:`synth_setter.data.vst.preset_paths` selecting spec and preset.
    :param render_train_lance: Renders ``train.lance`` given ``(train_lance, param_spec_name)``.

    :return: Path to the directory holding ``{train,val,test}.lance`` and ``stats.npz``.
    """
    from synth_setter.pipeline.data.stats import stream_stats_lance

    smoke_dataset_dir = tmp_path / "data" / "smoke-lance"
    smoke_dataset_dir.mkdir(parents=True, exist_ok=True)
    train_lance = smoke_dataset_dir / "train.lance"

    render_train_lance(train_lance, param_spec_name)

    # Sibling stats.npz folded straight from the Lance mel rows; mask degenerate
    # bins as the h5 path's --mask-degenerate-bins flag does for fake-plugin data.
    mean, std = stream_stats_lance([train_lance], mask_degenerate=True)
    np.savez(smoke_dataset_dir / "stats.npz", mean=mean, std=std)

    shutil.copytree(train_lance, smoke_dataset_dir / "val.lance")
    shutil.copytree(train_lance, smoke_dataset_dir / "test.lance")
    return smoke_dataset_dir


@pytest.fixture(scope="function")
def surge_xt_smoke_datasets(tmp_path: Path, param_spec_name: str) -> Path:
    """Generate the N-sample Surge XT dataset used by the e2e smoke test.

    :param tmp_path: Per-test temporary directory; the dataset is written under
        ``tmp_path / "data" / "smoke"``.
    :param param_spec_name: Param spec name (key into :data:`synth_setter.data.vst.param_specs`
        and :data:`synth_setter.data.vst.preset_paths`) — selects the matching ``--param_spec_name``
        and ``--preset_path`` for ``generate_vst_dataset``.

    :return: A Path object pointing at the directory containing the N-sample Surge XT smoke-test
        dataset.
    """
    return _build_surge_smoke_datasets(tmp_path, param_spec_name, _render_smoke_train_subprocess)


@pytest.fixture(scope="function")
def surge_xt_smoke_lance_datasets(tmp_path: Path, param_spec_name: str) -> Path:
    """Generate the N-sample Surge XT smoke dataset as native Lance shards via the real VST.

    The Lance counterpart to :func:`surge_xt_smoke_datasets`: ``generate_vst_dataset``
    dispatches the ``.lance`` suffix to :func:`make_lance_dataset`, so the real Surge XT
    subprocess writes ``train.lance`` directly. Backs the real-VST half of the h5<->Lance
    train/eval parity matrix.

    :param tmp_path: Per-test temporary directory; the dataset is written under
        ``tmp_path / "data" / "smoke-lance"``.
    :param param_spec_name: Param spec name (key into :data:`synth_setter.data.vst.param_specs`
        and :data:`synth_setter.data.vst.preset_paths`) — selects the matching ``--param_spec_name``
        and ``--preset_path`` for ``generate_vst_dataset``.

    :return: Path to the directory holding ``{train,val,test}.lance`` and ``stats.npz``.
    """
    return _build_surge_smoke_lance_datasets(
        tmp_path, param_spec_name, _render_smoke_train_subprocess
    )


@pytest.fixture(scope="function")
def fake_surge_smoke_datasets(
    tmp_path: Path, param_spec_name: str, install_fake_plugin: FakeVST3Plugin
) -> Path:
    """Render the N-sample Surge dataset in-process via the fake plugin (no real VST/X11).

    The fast counterpart to :func:`surge_xt_smoke_datasets`: ``install_fake_plugin``
    swaps the loader for ``FakeVST3Plugin`` so ``make_hdf5_dataset`` produces a
    structurally-valid ``train.h5`` (audio/mel/param) with no Surge XT subprocess.
    Lets oracle-eval tests that only need a loadable dataset (not real audio fidelity)
    run on the CPU-fast loop.

    :param tmp_path: Per-test temporary directory; the dataset is written under
        ``tmp_path / "data" / "smoke"``.
    :param param_spec_name: Param spec name (key into :data:`synth_setter.data.vst.param_specs`
        and :data:`synth_setter.data.vst.preset_paths`); defaults to ``"surge_4"``.
    :param install_fake_plugin: Swaps ``core.load_plugin`` / ``core.VST3Plugin``
        for the fake so the render needs no real VST3 binary or display server.

    :return: Path to the directory holding ``{train,val,test}.h5`` and ``stats.npz``.
    """
    return _build_surge_smoke_datasets(tmp_path, param_spec_name, _render_smoke_train_h5_fake)


@pytest.fixture(scope="function")
def fake_surge_smoke_lance_datasets(
    tmp_path: Path, param_spec_name: str, install_fake_plugin: FakeVST3Plugin
) -> Path:
    """Render the N-sample Surge smoke dataset in-process as native Lance shards (no real VST/X11).

    The Lance counterpart to :func:`fake_surge_smoke_datasets`: ``install_fake_plugin``
    swaps the loader for ``FakeVST3Plugin`` so :func:`make_lance_dataset` writes
    structurally-valid ``{train,val,test}.lance`` shards directly — no HDF5
    intermediate to convert. Backs the Lance ``evaluate`` oracle smoke test.

    :param tmp_path: Per-test temporary directory; the dataset is written under
        ``tmp_path / "data" / "smoke-lance"``.
    :param param_spec_name: Param spec name (key into :data:`synth_setter.data.vst.param_specs`
        and :data:`synth_setter.data.vst.preset_paths`); defaults to ``"surge_4"``.
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
        cfg.datamodule.predict_file = str(surge_xt_smoke_datasets / "test.h5")

    yield cfg

    GlobalHydra.instance().clear()


# One dataset-format arm of the Surge XT smoke train/eval parity matrix. Private NamedTuple
# (no public docstring) matching the sibling ``_FakeOracleDataset`` in test_eval.py.
class _SurgeSmokeVariant(NamedTuple):
    dataset_fixture: str  # conftest fixture yielding the dataset root dir
    datamodule_group: str  # Hydra ``datamodule=`` group: "surge" (h5) | "surge_lance"
    split_ext: str  # split file suffix: ".h5" | ".lance"
    plugin_path: str  # render plugin for eval postprocessing: real PLUGIN_PATH | fake.vst3


# Dataset-format arms of the h5<->Lance smoke parity matrix, shared by the train and eval
# entrypoint tests as ``surge_smoke_variant`` parametrize values. The real-VST arms render
# through the Surge XT subprocess (slow); the fake arms render in-process via the fake
# plugin (CPU inner loop). Both feed the same test bodies so a Lance-datamodule regression
# cannot hide behind h5-only coverage.
REAL_VST_VARIANTS = [
    pytest.param(
        _SurgeSmokeVariant("surge_xt_smoke_datasets", "surge", ".h5", PLUGIN_PATH), id="h5"
    ),
    pytest.param(
        _SurgeSmokeVariant("surge_xt_smoke_lance_datasets", "surge_lance", ".lance", PLUGIN_PATH),
        id="lance",
    ),
]
FAKE_VST_VARIANTS = [
    pytest.param(
        _SurgeSmokeVariant("fake_surge_smoke_datasets", "surge", ".h5", "plugins/fake.vst3"),
        id="h5",
    ),
    pytest.param(
        _SurgeSmokeVariant(
            "fake_surge_smoke_lance_datasets", "surge_lance", ".lance", "plugins/fake.vst3"
        ),
        id="lance",
    ),
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
    :param param_spec_name: Keys ``preset_paths`` so ``cfg.render`` matches the
        spec the model was trained against.

    :return: A DictConfig configured to evaluate a Surge XT checkpoint on the smoke-test
        dataset.
    """
    cfg = cfg_surge_xt_global.copy()
    _configure_surge_xt_eval_cfg(
        cfg,
        tmp_path=tmp_path,
        dataset_root=surge_xt_smoke_datasets,
        predict_file=surge_xt_smoke_datasets / "test.h5",
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
            "preset_path": preset_paths[param_spec_name],
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
    identically (deterministic ``created_at`` / ``git_sha``, hdf5 output, the
    Darwin-portable ``gui_toggle_cadence``); per-test fields (``task_name``,
    ``run_id``, ``r2``, shard sizes, ``parallel``) come in as overrides.

    :returns: A fresh kwargs dict safe for in-place override per call.
    """
    return {
        "created_at": datetime(2026, 5, 20, 0, 0, 0, tzinfo=UTC),
        "git_sha": "0" * 40,
        "is_repo_dirty": False,
        "output_format": "hdf5",
        "base_seed": 42,
        "render": {
            "plugin_path": "plugins/fake.vst3",
            "preset_path": "presets/fake.vstpreset",
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
    not fatal. Otherwise returns ``min(cpu, memory)`` so neither resource is
    over-subscribed.

    :param config: The pytest config object (unused; required by the hook signature).
    :returns: Worker count clamped to the host's real CPU and memory allocation.
    """
    env_override = os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS")
    if env_override:
        try:
            return max(1, int(env_override))
        except ValueError:
            pass  # non-integer pin -> ignore and fall through to the adaptive clamps
    cpu_workers = _cgroup_aware_cpu_count()
    mem_workers = _memory_aware_worker_count()
    return cpu_workers if mem_workers is None else min(cpu_workers, mem_workers)


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

# surge_ffn's AST net hard-codes the production mel shape and channel count, so the
# Lance smoke fixture must carry production-shaped mel rows; everything else is tiny.
_LANCE_SMOKE_MEL_SHAPE = (2, 128, 401)
_LANCE_SMOKE_NUM_PARAMS = 16
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
            "audio": rng.standard_normal((num_rows, 2, 64)).astype(np.float16),
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
            overrides=["datamodule=surge_lance", "model=surge_ffn", "trainer=cpu"],
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
            # Not a loop bound under fast_dev_run — surge_ffn's scheduler resolves
            # ${trainer.max_steps}, which trainer/cpu.yaml leaves undefined.
            cfg.trainer.max_steps = 1
            cfg.datamodule.dataset_root = str(dataset_root)
            cfg.datamodule.batch_size = 1
            cfg.datamodule.ot = False
            cfg.datamodule.num_workers = 0
            cfg.datamodule.pin_memory = False
            cfg.model.compile = False
            cfg.model.net.d_model = 32
            cfg.model.net.n_heads = 2
            cfg.model.net.n_layers = 1
            cfg.model.net.d_out = _LANCE_SMOKE_NUM_PARAMS

    yield cfg

    GlobalHydra.instance().clear()
