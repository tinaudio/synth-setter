"""This file prepares config fixtures for other tests."""

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401   side-effect import: registers HDF5_PLUGIN_PATH so h5py can load Blosc2 filters in fixtures
import numpy as np
import pytest
import rootutils
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from synth_setter.data.vst import param_specs, preset_paths
from synth_setter.resources import vst_headless_wrapper
from synth_setter.utils.utils import register_resolvers
from tests._baseline_worktree import worktree_for_ref  # noqa: F401 — pytest fixture re-export
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
_SURGE_FIXTURE_PLUGIN_PATH = os.environ.get("SYNTH_SETTER_PLUGIN_PATH", "plugins/Surge XT.vst3")
_SURGE_AUDIO_SAMPLES_PER_CLIP = int(_SURGE_FIXTURE_SAMPLE_RATE * _SURGE_FIXTURE_DURATION_SECONDS)
_SURGE_AUDIO_CHANNELS = _SURGE_FIXTURE_CHANNELS
_SURGE_MEL_SHAPE = (2, 128, 401)
# ~-80 dBFS — same threshold used by `test_train_eval_surge_xt` to catch
# silent renders that would later poison metric computation.
_SURGE_SILENCE_PEAK_THRESHOLD = 1e-4

# Hard ceiling for VST subprocess calls (dataset generation, audio rendering).
# Picked at 10 minutes: comfortably above the observed runtime on the slowest
# CI runner (macOS with brew-installed cask), well below the workflow timeout
# so a hung VST surfaces as a clear test failure instead of a job kill. Eager
# constant on purpose — both call sites pass it directly to `subprocess.run`,
# no per-call tuning, no stack-distant default.
_VST_SUBPROCESS_TIMEOUT_SECONDS = 600

NUM_FIXTURE_SAMPLES = 5


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


# Register custom OmegaConf resolvers (mul, div) needed to parse Hydra configs.
# This import pulls in torch/lightning transitively via synth_setter.utils.utils, but every
# test in this suite already requires those dependencies, so there is no benefit to
# isolating resolver registration into a lighter module.
register_resolvers()


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """Build a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["data=ksin", "model=ffn", "trainer=cpu"],
        )

        # set defaults for all tests
        with open_dict(cfg):
            # Trainer defaults
            cfg.trainer.check_val_every_n_epoch = 1
            cfg.trainer.val_check_interval = 1
            cfg.trainer.max_epochs = 1
            cfg.trainer.num_sanity_val_steps = 0
            cfg.trainer.log_every_n_steps = 1
            cfg.trainer.devices = 1
            cfg.trainer.deterministic = True
            # DataLoader defaults
            cfg.data.num_workers = 4
            cfg.data.pin_memory = False
            cfg.data.batch_size = 1
            cfg.data.train_val_test_sizes = [2, 2, 2]
            cfg.data.break_symmetry = True
            # Other defaults
            cfg.model.compile = False
            cfg.logger = None
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
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
                "data=ksin",
                "model=ffn",
                "trainer=cpu",
                "ckpt_path=.",
            ],
        )

        # set defaults for all tests
        with open_dict(cfg):
            # Trainer defaults
            cfg.trainer.check_val_every_n_epoch = 1
            cfg.trainer.val_check_interval = 1
            cfg.trainer.max_epochs = 1
            cfg.trainer.num_sanity_val_steps = 0
            cfg.trainer.log_every_n_steps = 1
            cfg.trainer.devices = 1
            cfg.trainer.deterministic = True
            # DataLoader defaults
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.data.batch_size = 1
            cfg.data.train_val_test_sizes = [2, 2, 2]
            cfg.data.break_symmetry = True
            # Other defaults
            cfg.model.compile = False
            cfg.logger = None
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
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
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
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
    accelerator: str, param_spec_name: str, experiment: str
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

    :return: Resolved DictConfig with the smoke-test bake-ins applied.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                f"experiment={experiment}",
                "callbacks=[default_surge,eval_surge]",
            ],
        )
        TRAINING_STEPS = 1
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))

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
            # SurgeDataModule's train_dataloader) drops one batch per epoch,
            # so any batch_size > dataset_size // 2 leaves the dataloader empty
            # and Lightning aborts with "Trainer.fit stopped: No training batches."
            cfg.data.batch_size = 1
            cfg.data.pin_memory = False
            cfg.data.ot = False
            # Smoke fixture writes stats.npz via masked get_dataset_stats — see #1002.
            cfg.data.use_saved_mean_and_variance = True
            cfg.data.num_workers = 0

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
    smoke_dataset_dir = tmp_path / "data" / "smoke"

    Path(smoke_dataset_dir).mkdir(parents=True, exist_ok=True)

    generate_dataset_args = []
    if sys.platform == "linux":
        generate_dataset_args.append(VST_HEADLESS_WRAPPER)

    generate_dataset_args += [
        sys.executable,
        "src/synth_setter/data/vst/generate_vst_dataset.py",
        str(smoke_dataset_dir / "train.h5"),
        f"--plugin_path={_SURGE_FIXTURE_PLUGIN_PATH}",
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
            timeout=_VST_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"generate_vst_dataset timed out after {_VST_SUBPROCESS_TIMEOUT_SECONDS}s\n"
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
    assert (smoke_dataset_dir / "train.h5").exists(), (
        "Dataset generation failed to produce train.h5 fixture"
    )
    _validate_surge_dataset(smoke_dataset_dir / "train.h5", NUM_FIXTURE_SAMPLES)

    # Sibling stats.npz; shared across train/val/test splits — see #1002.
    stats_args = [
        sys.executable,
        "-m",
        "synth_setter.pipeline.data.stats",
        str(smoke_dataset_dir / "train.h5"),
        "--mask-degenerate-bins",
    ]
    try:
        result = subprocess.run(  # noqa: S603
            stats_args,
            text=True,
            check=False,
            timeout=_VST_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"get_dataset_stats timed out after {_VST_SUBPROCESS_TIMEOUT_SECONDS}s\n"
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
    assert (smoke_dataset_dir / "stats.npz").exists(), (
        "get_dataset_stats failed to produce stats.npz fixture"
    )

    shutil.copy(smoke_dataset_dir / "train.h5", smoke_dataset_dir / "val.h5")
    shutil.copy(smoke_dataset_dir / "train.h5", smoke_dataset_dir / "test.h5")
    return Path(smoke_dataset_dir)


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
        cfg.data.dataset_root = str(surge_xt_smoke_datasets)
        cfg.data.predict_file = str(surge_xt_smoke_datasets / "test.h5")

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_surge_xt_eval(
    cfg_surge_xt_global: DictConfig, tmp_path: Path, surge_xt_smoke_datasets: Path
) -> DictConfig:
    """Eval config for the Surge XT train->eval smoke-test roundtrip.

    Inherits from `cfg_surge_xt_global` and points `ckpt_path` at the checkpoint
    that `cfg_surge_xt`'s training run will write under the same `tmp_path`.

    :param cfg_surge_xt_global: The Surge XT training config (parametrized over accelerator, param_spec_name, and experiment_name).
    :param tmp_path: The temporary logging path (shared with `cfg_surge_xt`).

    :return: A DictConfig configured to evaluate a Surge XT checkpoint on the smoke-test
        dataset.
    """
    cfg = cfg_surge_xt_global.copy()
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.data.batch_size = 1
        cfg.data.dataset_root = str(surge_xt_smoke_datasets)
        cfg.data.predict_file = str(surge_xt_smoke_datasets / "test.h5")
        cfg.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg.mode = "predict"

    yield cfg

    GlobalHydra.instance().clear()


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
