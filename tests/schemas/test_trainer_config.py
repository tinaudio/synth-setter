"""Behavioural tests for the ``TrainerConfig`` pydantic model.

Every YAML under ``configs/trainer/`` must validate against ``TrainerConfig``
once Hydra finishes composing it onto ``default.yaml``. Variant-specific
``Trainer`` kwargs (``strategy``, ``precision``, ``min_steps``, ``max_steps``,
...) live under ``extra="allow"``; the typed surface here covers the keys
all shipped variants set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.schemas.trainer_config import TrainerConfig

_TRAINER_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "trainer"


def _all_trainer_config_names() -> list[str]:  # noqa: DOC201,DOC203
    """Return the YAML stem of every direct trainer config under ``configs/trainer/``."""
    names = sorted(p.stem for p in _TRAINER_CONFIG_DIR.glob("*.yaml"))
    assert names, f"no trainer YAMLs found under {_TRAINER_CONFIG_DIR} — has the layout changed?"
    return names


def _compose_trainer_cfg(trainer_name: str) -> dict[str, Any]:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Compose a full train config with ``trainer=<trainer_name>`` selected."""
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[f"trainer={trainer_name}", "data=ksin", "model=ffn"],
        )
    trainer_subtree = OmegaConf.to_container(cfg.trainer, resolve=False)
    assert isinstance(trainer_subtree, dict)
    return cast("dict[str, Any]", trainer_subtree)


class TestTrainerConfigAcceptsEveryConfig:
    """Every shipped trainer YAML must validate against ``TrainerConfig``."""

    @pytest.mark.parametrize("trainer_name", _all_trainer_config_names())
    def test_trainer_yaml_validates(self, trainer_name: str) -> None:  # noqa: DOC101,DOC103
        """The composed ``trainer`` subtree validates as ``TrainerConfig``."""
        trainer_subtree = _compose_trainer_cfg(trainer_name)
        TrainerConfig.model_validate(trainer_subtree)


class TestTrainerConfigCommonFields:
    """The typed scalar fields must land on the parsed model."""

    def test_default_cpu_fields_typed(self) -> None:
        """``cpu.yaml`` carries the expected ``_target_`` / accelerator / devices."""
        trainer_subtree = _compose_trainer_cfg("cpu")
        parsed = TrainerConfig.model_validate(trainer_subtree)
        assert parsed.target_ == "lightning.pytorch.trainer.Trainer"
        assert parsed.accelerator == "cpu"
        assert parsed.devices == 1
        assert parsed.log_every_n_steps == 100
        assert parsed.val_check_interval == 10_000
        assert parsed.gradient_clip_val == 1.0
        assert parsed.check_val_every_n_epoch is None
        assert parsed.deterministic is False

    def test_mps_variant_keeps_overrides(self) -> None:
        """``mps_32_true_non_deterministic.yaml`` overrides accelerator + precision."""
        trainer_subtree = _compose_trainer_cfg("mps_32_true_non_deterministic")
        parsed = TrainerConfig.model_validate(trainer_subtree)
        assert parsed.accelerator == "mps"
        assert parsed.deterministic is False


_VALID_TRAINER = {
    "_target_": "lightning.pytorch.trainer.Trainer",
    "default_root_dir": "/tmp/out",  # noqa: S108
    "accelerator": "cpu",
    "devices": 1,
    "log_every_n_steps": 100,
    "val_check_interval": 10_000,
    "gradient_clip_val": 1.0,
}


class TestTrainerConfigRejectsBadInputs:
    """Validators must catch obvious mistakes on the typed fields."""

    def test_blank_accelerator_rejected(self) -> None:
        """A blank ``accelerator`` would crash Lightning's backend lookup."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            TrainerConfig.model_validate({**_VALID_TRAINER, "accelerator": "   "})

    def test_zero_devices_rejected(self) -> None:
        """``devices=0`` is meaningless for any accelerator; reject up front."""
        with pytest.raises(ValidationError, match="greater than 0"):
            TrainerConfig.model_validate({**_VALID_TRAINER, "devices": 0})

    def test_negative_log_every_n_steps_rejected(self) -> None:
        """Lightning rejects non-positive logging cadence; surface it here too."""
        with pytest.raises(ValidationError, match="greater than 0"):
            TrainerConfig.model_validate({**_VALID_TRAINER, "log_every_n_steps": -1})

    def test_string_deterministic_rejected(self) -> None:
        """``deterministic`` is ``StrictBool``; ``"yes"`` must not silently coerce."""
        with pytest.raises(ValidationError, match="bool"):
            TrainerConfig.model_validate({**_VALID_TRAINER, "deterministic": "yes"})
