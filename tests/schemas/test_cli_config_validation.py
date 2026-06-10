"""Safety net for the CLI-boundary config validation wired in ``schemas.validate``.

The headline guarantee: every config the train/eval entrypoints can compose
must pass :func:`validate_composed_config`. The parametrized matrix below
composes each shipped ``experiment=`` selection (plus the bare ``train.yaml`` /
``eval.yaml`` defaults) and asserts validation accepts it — so wiring the
schemas in cannot silently start rejecting a currently-valid run.

Rejection tests pin the other direction: malformed subtrees raise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.resources import configs_dir
from synth_setter.schemas.validate import validate_composed_config

_EXPERIMENT_DIR = Path(str(configs_dir() / "experiment"))

# Experiments that are NOT selectable standalone via ``experiment=<name>``: base
# fragments that still demand a ``model=``/``datamodule=`` override, Hydra
# sub-groups (``fm/algorithm/*``), and configs pinning a renamed model group
# (``ksin_flow``/``ksin_ff`` no longer exist). These fail Hydra *composition*
# — before validation runs — so they are out of scope for this safety net.
_NON_STANDALONE_EXPERIMENTS = frozenset(
    {
        "flow_size/base",
        "flow_size/bigenc",
        "flow_size/medenc",
        "flow_size/smallenc",
        "flow_size/tinyenc",
        "flow_size/vbigenc",
        "fm/algorithm/conditional",
        "fm/algorithm/hierarchical",
        "fm/algorithm/mixed",
        "fm/base",
        "kosc/base",
        "ksin/base",
        "ksin_ood/base",
        "ksin_ood/flow",
        "ksin_ood/mlp_chamfer",
        "ksin_ood/mlp_mse",
        "ksin_ood/mlp_sort",
        "surge/base",
        "time_weighting",
    }
)


def _all_experiment_names() -> list[str]:
    """Return every ``experiment=`` stem except the ``generate_dataset`` pipeline group.

    ``generate_dataset`` experiments drive a different entrypoint and never
    compose against ``train.yaml``/``eval.yaml``, so they are excluded here.

    :return: Sorted ``"<dir>/<stem>"`` selectors under ``configs/experiment/``.
    """
    names = sorted(
        str(p.relative_to(_EXPERIMENT_DIR).with_suffix(""))
        for p in _EXPERIMENT_DIR.glob("**/*.yaml")
        if "generate_dataset" not in p.parts
    )
    assert names, f"no experiment YAMLs found under {_EXPERIMENT_DIR} — layout changed?"
    return names


_STANDALONE_EXPERIMENTS = [
    name for name in _all_experiment_names() if name not in _NON_STANDALONE_EXPERIMENTS
]


def _compose(config_name: str, overrides: list[str]) -> dict[str, Any]:
    """Compose ``config_name`` with ``overrides`` and return it unresolved as a dict.

    ``resolve=False`` mirrors the CLI boundary: interpolations stay opaque
    strings, exactly what the schemas are written to accept.

    :param config_name: Root config (``train.yaml`` or ``eval.yaml``).
    :param overrides: Hydra overrides applied during composition.
    :return: The composed config as a plain ``dict[str, Any]``.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=False))


class TestEveryComposedConfigValidates:
    """No currently-composable run may be rejected by the new validation."""

    @pytest.mark.parametrize("experiment", _STANDALONE_EXPERIMENTS)
    def test_train_experiment_validates(self, experiment: str) -> None:
        """Each standalone ``experiment=`` composes and passes train validation.

        :param experiment: Parametrized ``"<dir>/<stem>"`` selector.
        """
        cfg = _compose("train.yaml", [f"experiment={experiment}"])
        validate_composed_config(cfg, include_train_config=True)

    def test_default_train_yaml_validates(self) -> None:
        """The bare ``train.yaml`` defaults validate (no experiment selected)."""
        cfg = _compose("train.yaml", ["datamodule=ksin", "model=ffn", "trainer=cpu"])
        validate_composed_config(cfg, include_train_config=True)

    def test_default_eval_yaml_validates(self) -> None:
        """The bare ``eval.yaml`` defaults validate over the shared subtrees."""
        cfg = _compose(
            "eval.yaml",
            ["datamodule=ksin", "model=ffn", "trainer=cpu", "ckpt_path=/tmp/x.ckpt"],  # noqa: S108
        )
        validate_composed_config(cfg, include_train_config=False)

    def test_eval_experiment_with_overridden_callbacks_validates(self) -> None:
        """An eval experiment that composes callbacks (fake_oracle) still validates."""
        cfg = _compose(
            "eval.yaml",
            ["experiment=surge/fake_oracle", "ckpt_path=/tmp/x.ckpt"],  # noqa: S108
        )
        validate_composed_config(cfg, include_train_config=False)


class TestValidationRejectsMalformedConfigs:
    """The wired-in validation must still reject genuinely-broken subtrees."""

    def test_blank_task_name_rejected(self) -> None:
        """A blank ``task_name`` (empty output dir) raises with a clear message."""
        cfg = _compose("train.yaml", ["datamodule=ksin", "model=ffn", "trainer=cpu"])
        cfg["task_name"] = "   "
        with pytest.raises(ValidationError, match="at least 1 character"):
            validate_composed_config(cfg, include_train_config=True)

    def test_negative_seed_rejected(self) -> None:
        """A negative ``seed`` raises (Lightning rejects it at runtime)."""
        cfg = _compose("train.yaml", ["datamodule=ksin", "model=ffn", "trainer=cpu"])
        cfg["seed"] = -1
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            validate_composed_config(cfg, include_train_config=True)

    def test_invalid_float32_matmul_precision_rejected(self) -> None:
        """An out-of-enum ``extras.float32_matmul_precision`` raises before torch sees it."""
        cfg = _compose("train.yaml", ["datamodule=ksin", "model=ffn", "trainer=cpu"])
        cfg["extras"]["float32_matmul_precision"] = "ultra"
        with pytest.raises(ValidationError):
            validate_composed_config(cfg, include_train_config=True)

    def test_non_positive_optimizer_lr_rejected(self) -> None:
        """A non-positive optimizer ``lr`` raises (model subtree is validated)."""
        cfg = _compose("train.yaml", ["datamodule=ksin", "model=ffn", "trainer=cpu"])
        cfg["model"]["optimizer"]["lr"] = 0
        with pytest.raises(ValidationError):
            validate_composed_config(cfg, include_train_config=True)
