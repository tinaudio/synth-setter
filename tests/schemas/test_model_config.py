"""Behavioural tests for the ``ModelConfig`` pydantic model.

Every YAML under ``configs/model/`` must validate against ``ModelConfig`` —
that's the contract the published docs assert. Variant-specific keys live
under ``extra="allow"`` so a new model module can ship its own fields
without re-touching the schema; the common shape stays typed and
documented.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_setter.schemas.model_config import ModelConfig, OptimizerConfig
from tests.schemas.conftest import compose_subtree

_MODEL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "model"

# Reused as a placeholder ``_target_`` across the negative-path cases — the
# exact dotted path doesn't matter for the validators under test, but
# centralising it stops the negative tests from looking like they each
# probe a different module.
_VALID_TARGET = "synth_setter.models.X"

# Minimal optimizer-config fixture reused by the negative-path cases; defined
# once so a new constraint adds one row, not a new boilerplate dict.
_VALID_OPTIMIZER = {
    "_target_": "torch.optim.Adam",
    "_partial_": True,
    "lr": 1e-4,
}


def _all_model_config_names() -> list[str]:  # noqa: DOC201,DOC203
    """Return the YAML stem of every direct model config under ``configs/model/``.

    Subgroups (``configs/model/encoder/`` and similar) compose into a parent
    via ``defaults:`` and are not top-level model configs; they're excluded.
    """
    names = sorted(p.stem for p in _MODEL_CONFIG_DIR.glob("*.yaml"))
    assert names, f"no model YAMLs found under {_MODEL_CONFIG_DIR} — has the layout changed?"
    return names


class TestModelConfigAcceptsEveryConfig:
    """Every shipped model YAML must validate against ``ModelConfig``."""

    @pytest.mark.parametrize("model_name", _all_model_config_names())
    def test_model_yaml_validates(self, model_name: str) -> None:  # noqa: DOC101,DOC103
        """The composed ``model`` subtree validates as ``ModelConfig``."""
        model_subtree = compose_subtree("model", model_name)
        parsed = ModelConfig.model_validate(model_subtree)
        assert parsed.target_


class TestModelConfigCommonFields:
    """The typed scalar / sub-model fields must land on the parsed model."""

    def test_common_fields_typed(self) -> None:
        """``_target_``, ``optimizer``, ``compile`` come through with the right types."""
        model_subtree = compose_subtree("model", "ffn")
        parsed = ModelConfig.model_validate(model_subtree)
        assert parsed.target_.endswith("KSinFeedForwardModule")
        assert isinstance(parsed.optimizer, OptimizerConfig)
        assert parsed.compile is True

    def test_scheduler_optional_none(self) -> None:
        """``scheduler: null`` in YAML parses to ``None`` on the model."""
        model_subtree = compose_subtree("model", "ffn")
        parsed = ModelConfig.model_validate(model_subtree)
        assert parsed.scheduler is None


class TestModelConfigRejectsBadInputs:
    """Validators must catch obvious mistakes on the typed fields."""

    def test_missing_target_rejected(self) -> None:
        """Hydra needs ``_target_`` to instantiate the module — required."""
        with pytest.raises(ValidationError):
            ModelConfig.model_validate({"optimizer": _VALID_OPTIMIZER})

    def test_blank_target_rejected(self) -> None:
        """A blank ``_target_`` would crash ``hydra.utils.instantiate`` mid-run."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            ModelConfig.model_validate({"_target_": "  ", "optimizer": _VALID_OPTIMIZER})

    def test_missing_lr_rejected(self) -> None:
        """``OptimizerConfig.lr`` has no default; omitting it must fail validation."""
        with pytest.raises(ValidationError):
            ModelConfig.model_validate(
                {
                    "_target_": _VALID_TARGET,
                    "optimizer": {"_target_": "torch.optim.Adam", "_partial_": True},
                }
            )

    def test_negative_lr_rejected(self) -> None:
        """Negative learning rate is always a bug; surface it at config time."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ModelConfig.model_validate(
                {
                    "_target_": _VALID_TARGET,
                    "optimizer": {**_VALID_OPTIMIZER, "lr": -1.0},
                }
            )

    def test_negative_weight_decay_rejected(self) -> None:
        """Negative weight decay is invalid for torch optimizers; reject it."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ModelConfig.model_validate(
                {
                    "_target_": _VALID_TARGET,
                    "optimizer": {**_VALID_OPTIMIZER, "weight_decay": -0.1},
                }
            )

    def test_blank_scheduler_target_rejected(self) -> None:
        """A blank scheduler ``_target_`` would crash instantiation mid-fit."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            ModelConfig.model_validate(
                {
                    "_target_": _VALID_TARGET,
                    "optimizer": _VALID_OPTIMIZER,
                    "scheduler": {"_target_": "  ", "_partial_": True},
                }
            )
