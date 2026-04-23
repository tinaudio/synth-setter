"""This file prepares config fixtures for other tests."""

from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.utils.utils import register_resolvers

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
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.min_steps = None
            # Lightning 2.x uses ``-1`` as the sentinel for "unbounded" max_steps;
            # passing ``None`` triggers an internal ``None < int`` comparison. We
            # want the epoch-based ``max_epochs=1`` to drive termination.
            cfg.trainer.max_steps = -1
            cfg.trainer.val_check_interval = None
            cfg.trainer.check_val_every_n_epoch = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.model.compile = False
            cfg.logger = None
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
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.min_steps = None
            # Lightning 2.x uses ``-1`` as the sentinel for "unbounded" max_steps;
            # passing ``None`` triggers an internal ``None < int`` comparison. We
            # want the epoch-based ``max_epochs=1`` to drive termination.
            cfg.trainer.max_steps = -1
            cfg.trainer.val_check_interval = None
            cfg.trainer.check_val_every_n_epoch = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
            callbacks = cfg.get("callbacks")
            if callbacks is not None and "lr_monitor" in callbacks:
                del callbacks.lr_monitor

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

    :param cfg_train_global: The input DictConfig object to be modified.
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
    """A pytest fixture for a one-step Surge XT training config on the 5-sample test fixture.

    Composes `train.yaml` with `experiment=surge/flow_full` and bakes in the minimal overrides
    needed to train-smoke-test on `tests/fixtures/surge_xt/`. Tests can override any knob.

    :return: A DictConfig object configured for a one-step Surge XT smoke train.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["experiment=surge/flow_full"],
        )

        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))

            cfg.data.dataset_root = "tests/fixtures/surge_xt"
            cfg.data.batch_size = 2
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            # `configs/data/surge.yaml` hardcodes a researcher-local predict_file that
            # `SurgeDataModule.setup()` opens unconditionally.
            cfg.data.predict_file = "tests/fixtures/surge_xt/test.h5"
            # The 5-sample fixture's stats.npz has zero-std mel bins that poison the batch
            # with NaN via (mel - mean) / std.
            cfg.data.use_saved_mean_and_variance = False

            cfg.trainer.accelerator = "gpu"
            cfg.trainer.devices = 1
            cfg.trainer.min_steps = 1
            cfg.trainer.max_steps = 1
            cfg.trainer.max_epochs = -1
            cfg.trainer.limit_train_batches = 1
            cfg.trainer.limit_val_batches = 0
            cfg.trainer.limit_test_batches = 0

            cfg.model.compile = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
            cfg.test = False

            callbacks = cfg.get("callbacks")
            if callbacks is not None and "lr_monitor" in callbacks:
                del callbacks.lr_monitor

    return cfg


@pytest.fixture(scope="function")
def cfg_surge_xt(cfg_surge_xt_global: DictConfig, tmp_path: Path) -> DictConfig:
    """Per-test wrapper around `cfg_surge_xt_global` that sets `tmp_path`-scoped output dirs.

    :param cfg_surge_xt_global: The package-scoped Surge XT training config.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with output and log dirs pointing at `tmp_path`.
    """
    cfg = cfg_surge_xt_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_surge_xt_eval(cfg_surge_xt: DictConfig, tmp_path: Path) -> DictConfig:
    """Eval config for the Surge XT one-step train->eval roundtrip.

    Composes `eval.yaml` and copies `data`/`model`/`callbacks` from `cfg_surge_xt` so the
    evaluator loads the same LightningModule shape the trainer just saved. The test body
    must set `cfg.ckpt_path` before calling `evaluate`.

    :param cfg_surge_xt: The one-step Surge XT training config.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig configured to evaluate a Surge XT checkpoint on the test fixture.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=[])

        with open_dict(cfg):
            cfg.paths.root_dir = cfg_surge_xt.paths.root_dir
            cfg.paths.output_dir = str(tmp_path)
            cfg.paths.log_dir = str(tmp_path)

            # `configs/eval.yaml` defaults `data=surge_mini` (researcher-local path) and
            # `model=surge_flow` without the `num_params=300` override. Copy from the training
            # config so the evaluator loads the same datamodule and LightningModule shape the
            # trainer just saved. `callbacks=eval_surge` (= `prediction_writer`) is already the
            # eval default; keeping it avoids pulling in train-only callbacks like
            # `model_checkpoint` / `plot_projii`.
            cfg.data = cfg_surge_xt.data
            cfg.model = cfg_surge_xt.model

            cfg.trainer.accelerator = "gpu"
            cfg.trainer.devices = 1
            cfg.trainer.limit_test_batches = 1
            cfg.trainer.limit_predict_batches = 1

            # Flow-matching `test_step` runs `test_sample_steps` (default 200) sampling iterations
            # per batch. Shrink for smoke tests.
            cfg.model.test_sample_steps = 2

            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    yield cfg

    GlobalHydra.instance().clear()
