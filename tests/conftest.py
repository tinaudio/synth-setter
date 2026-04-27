"""This file prepares config fixtures for other tests."""

import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.utils.utils import register_resolvers
from tests._baseline_worktree import worktree_for_ref  # noqa: F401 — pytest fixture re-export

# Defaults baked into `src/data/vst/generate_vst_dataset.py` (channels=2,
# sample_rate=44100, signal_duration_seconds=4.0, mel shape (2, 128, 401)).
# The fixture invokes the script with these defaults, so the generated H5
# file must match.
_SURGE_AUDIO_SAMPLES_PER_CLIP = int(44100 * 4.0)
_SURGE_AUDIO_CHANNELS = 2
_SURGE_MEL_SHAPE = (2, 128, 401)
# ~-80 dBFS — same threshold used by `test_train_eval_surge_xt` to catch
# silent renders that would later poison metric computation.
_SURGE_SILENCE_PEAK_THRESHOLD = 1e-4


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
# This import pulls in torch/lightning transitively via src.utils.utils, but every
# test in this suite already requires those dependencies, so there is no benefit to
# isolating resolver registration into a lighter module.
register_resolvers()


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize(version_base="1.3", config_path="../configs"):
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
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(version_base="1.3", config_path="../configs"):
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
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

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
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

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
def cfg_surge_xt_global() -> DictConfig:
    """A pytest fixture for a one-step Surge XT training config on the N-sample test fixture.

    Composes `train.yaml` with `experiment=surge/ffn_full` and bakes in the minimal overrides
    needed to train-smoke-test on the dataset generated by `surge_xt_smoke_datasets`.
    Tests can override any knob.

    :return: A DictConfig object configured for a one-step Surge XT smoke train.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["experiment=surge/ffn_full", "callbacks=eval_surge"],
        )

        TRAINING_STEPS = 5
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))

            # batch_size=1 is forced: ShiftedBatchSampler (used in the
            # SurgeDataModule's train_dataloader) drops one batch per epoch,
            # so any: batch_size >  dataset_size // 2
            # leaves the dataloader empty and Lightning aborts with:
            # "Trainer.fit stopped: No training batches."
            cfg.data.batch_size = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.data.ot = False
            # The 5-sample fixture's stats.npz has zero-std mel bins that poison the batch
            # with NaN via (mel - mean) / std.
            cfg.data.use_saved_mean_and_variance = False

            cfg.trainer.devices = 1
            cfg.trainer.max_steps = TRAINING_STEPS
            cfg.trainer.check_val_every_n_epoch = 1  # validate every 5 epochs
            cfg.trainer.val_check_interval = 1.0  # default: end of (validating) epoch
            cfg.trainer.log_every_n_steps = TRAINING_STEPS
            cfg.trainer.enable_model_summary = False
            cfg.trainer.limit_val_batches = 1.0
            cfg.trainer.accelerator = "cpu"
            cfg.model.compile = True
            cfg.model.scheduler = None
            cfg.logger = None
            cfg.test = False
            cfg.callbacks.model_checkpoint = {
                "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                "dirpath": "${paths.output_dir}/checkpoints",
                "save_last": True,
                "every_n_train_steps": TRAINING_STEPS,
            }
            callbacks = cfg.get("callbacks")
            if callbacks is not None and "lr_monitor" in callbacks:
                del callbacks.lr_monitor

    return cfg


@pytest.fixture(scope="function")
def surge_xt_smoke_datasets(tmp_path: Path) -> Path:
    """Generate the N-sample Surge XT dataset used by the e2e smoke test.

    :return: A Path object pointing at the directory containing the N-sample Surge XT smoke-test
        dataset.
    """
    NUM_FIXTURE_SAMPLES = 5
    # Bootstraps Xvfb + xsettingsd + dbus for VST3 plugin init; resolved relative
    # to the container WORKDIR (``/home/build/synth-setter``) baked in the image.
    # X11 wrapping lives at the audio-rendering boundary (this subprocess call),
    # not at the container entrypoint — the click CLI stays X11-agnostic so idle
    # and passthrough don't pay the Xvfb startup cost.
    VST_HEADLESS_WRAPPER = "scripts/run-linux-vst-headless.sh"

    smoke_dataset_dir = tmp_path / "data" / "smoke"

    Path(smoke_dataset_dir).mkdir(parents=True, exist_ok=True)

    generate_dataset_args = []
    if sys.platform == "linux":
        generate_dataset_args.append(VST_HEADLESS_WRAPPER)

    generate_dataset_args += [
        sys.executable,
        "src/data/vst/generate_vst_dataset.py",
        str(smoke_dataset_dir / "train.h5"),
        str(NUM_FIXTURE_SAMPLES),
    ]

    result = subprocess.run(  # noqa: S603
        generate_dataset_args,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"generate_vst_dataset failed (exit {result.returncode})\n"
            f"command: {generate_dataset_args}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}",
            pytrace=False,
        )
    assert (smoke_dataset_dir / "train.h5").exists(), (
        "Dataset generation failed to produce train.h5 fixture"
    )
    _validate_surge_dataset(smoke_dataset_dir / "train.h5", NUM_FIXTURE_SAMPLES)
    shutil.copy(smoke_dataset_dir / "train.h5", smoke_dataset_dir / "val.h5")
    shutil.copy(smoke_dataset_dir / "train.h5", smoke_dataset_dir / "test.h5")
    return Path(smoke_dataset_dir)


@pytest.fixture(scope="function")
def cfg_surge_xt(
    cfg_surge_xt_global: DictConfig, tmp_path: Path, surge_xt_smoke_datasets: Path
) -> DictConfig:
    """Per-test wrapper around `cfg_surge_xt_global` that sets `tmp_path`-scoped output dirs.

    :param cfg_surge_xt_global: The package-scoped Surge XT training config.
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

    :param cfg_surge_xt_global: The package-scoped Surge XT training config.
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
