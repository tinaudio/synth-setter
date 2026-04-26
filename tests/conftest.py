"""This file prepares config fixtures for other tests."""

from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.utils.utils import register_resolvers
from tests._baseline_worktree import worktree_for_ref  # noqa: F401 — pytest fixture re-export

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
            overrides=["data=ksin", "model=ffn", "trainer=test", "logger=wandb"],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.data.batch_size = 4
            cfg.data.pin_memory = False
            cfg.data.train_val_test_sizes = [2, 2, 2]
            # cfg.data.signal_length = 64
            cfg.model.compile = False
            # cfg.logger = None
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
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
            overrides=["data=ksin", "model=ffn", "trainer=test", "ckpt_path=."],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.data.batch_size = 4
            cfg.data.pin_memory = False
            cfg.data.train_val_test_sizes = [2, 2, 2]
            # cfg.data.signal_length = 64
            cfg.model.compile = False
            cfg.logger = None
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
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
