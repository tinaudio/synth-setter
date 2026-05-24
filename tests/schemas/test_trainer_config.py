"""Behavioural tests for the ``TrainerConfig`` pydantic model.

Every YAML under ``configs/trainer/`` must validate; variant ``Trainer``
kwargs ride ``extra="allow"``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_setter.resources import configs_dir
from synth_setter.schemas.trainer_config import TrainerConfig
from tests.schemas.conftest import compose_subtree

_TRAINER_CONFIG_DIR = configs_dir() / "trainer"


def _all_trainer_config_names() -> list[str]:
    """Return the YAML stem of every direct trainer config under ``configs/trainer/``.

    :return: Sorted list of YAML stems found in ``configs/trainer/``.
    """
    names = sorted(
        p.name.removesuffix(".yaml")
        for p in _TRAINER_CONFIG_DIR.iterdir()
        if p.is_file() and p.name.endswith(".yaml")
    )
    assert names, f"no trainer YAMLs found under {_TRAINER_CONFIG_DIR} — has the layout changed?"
    return names


class TestTrainerConfigAcceptsEveryConfig:
    """Every shipped trainer YAML must validate against ``TrainerConfig``."""

    @pytest.mark.parametrize("trainer_name", _all_trainer_config_names())
    def test_trainer_yaml_validates(self, trainer_name: str) -> None:
        """The composed ``trainer`` subtree validates as ``TrainerConfig``.

        :param trainer_name: Parametrized YAML stem under ``configs/trainer/``.
        """
        trainer_subtree = compose_subtree("trainer", trainer_name)
        parsed = TrainerConfig.model_validate(trainer_subtree)
        assert parsed.target_


class TestTrainerConfigCommonFields:
    """The typed scalar fields must land on the parsed model with the right types."""

    def test_default_cpu_fields_typed(self) -> None:
        """``cpu.yaml`` carries typed fields with sensible values; assert types/ranges."""
        trainer_subtree = compose_subtree("trainer", "cpu")
        parsed = TrainerConfig.model_validate(trainer_subtree)
        assert parsed.target_ == "lightning.pytorch.trainer.Trainer"
        assert isinstance(parsed.accelerator, str) and parsed.accelerator
        assert isinstance(parsed.devices, int) and parsed.devices > 0
        assert isinstance(parsed.log_every_n_steps, int) and parsed.log_every_n_steps > 0
        assert isinstance(parsed.val_check_interval, int) and parsed.val_check_interval > 0
        assert isinstance(parsed.gradient_clip_val, float) and parsed.gradient_clip_val > 0
        assert parsed.check_val_every_n_epoch is None or parsed.check_val_every_n_epoch > 0
        assert isinstance(parsed.deterministic, bool)

    def test_mps_variant_keeps_overrides(self) -> None:
        """``mps_32_true_non_deterministic.yaml`` overrides accelerator + precision."""
        trainer_subtree = compose_subtree("trainer", "mps_32_true_non_deterministic")
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

    def test_blank_default_root_dir_rejected(self) -> None:
        """``default_root_dir`` is ``NonBlankStr`` — whitespace overrides must fail."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            TrainerConfig.model_validate({**_VALID_TRAINER, "default_root_dir": "   "})

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
