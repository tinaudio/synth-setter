"""Behavioural tests for the ``TrainConfig`` pydantic model.

Pins both directions: the live composed ``train.yaml`` validates, and
obvious mistakes (wrong types, blank ``task_name``, negative ``seed``) fail.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_setter.schemas.train_config import TrainConfig
from tests.schemas.conftest import compose_train_cfg


class TestTrainConfigAcceptsLiveCompose:
    """The pydantic model must accept the configs Hydra actually composes."""

    def test_default_composition_validates(self) -> None:
        """The default ``train.yaml`` composition validates without error."""
        cfg_dict = compose_train_cfg()
        parsed = TrainConfig.model_validate(cfg_dict)
        assert isinstance(parsed, TrainConfig)

    def test_default_composition_with_hydra_subtree_validates(self) -> None:
        """``return_hydra_config=True`` adds a ``hydra`` key; the model accepts it."""
        cfg_dict = compose_train_cfg(return_hydra_config=True)
        parsed = TrainConfig.model_validate(cfg_dict)
        assert isinstance(parsed, TrainConfig)

    def test_typed_scalars_survive_round_trip(self) -> None:
        """Every typed scalar lands on the model with the right type / shape."""
        cfg_dict = compose_train_cfg()
        model = TrainConfig.model_validate(cfg_dict)
        # Property-based asserts so this test doesn't shadow train.yaml's defaults.
        assert isinstance(model.task_name, str) and model.task_name
        assert all(isinstance(t, str) for t in model.tags)
        assert isinstance(model.train, bool)
        assert isinstance(model.test, bool)
        assert model.ckpt_path is None or isinstance(model.ckpt_path, str)
        assert model.seed is None or (isinstance(model.seed, int) and model.seed >= 0)
        assert model.optimized_metric is None or isinstance(model.optimized_metric, str)
        assert model.watch_gradients is None or isinstance(model.watch_gradients, bool)
        assert model.consumed_dataset_config_id is None or isinstance(
            model.consumed_dataset_config_id, str
        )
        assert isinstance(model.consumed_artifact_alias, str) and model.consumed_artifact_alias


class TestTrainConfigRejectsBadInputs:
    """Validators must reject obvious mistakes on the scalar fields."""

    def test_blank_task_name_rejected(self) -> None:
        """A blank ``task_name`` would produce an empty output dir; reject it."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            TrainConfig.model_validate({"task_name": "   ", "tags": ["dev"]})

    def test_string_train_flag_rejected_by_strict_bool(self) -> None:
        """``train`` is ``StrictBool``; ``"yes"`` would otherwise coerce silently."""
        with pytest.raises(ValidationError, match="bool"):
            TrainConfig.model_validate({"task_name": "train", "train": "yes"})

    def test_negative_seed_rejected(self) -> None:
        """Lightning's ``seed_everything`` rejects negative seeds; reject up front."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            TrainConfig.model_validate({"task_name": "train", "seed": -1})

    def test_string_tags_rejected(self) -> None:
        """``tags`` is consumed as ``list[str]`` by the run-id helpers."""
        with pytest.raises(ValidationError):
            TrainConfig.model_validate({"task_name": "train", "tags": "dev"})

    def test_blank_ckpt_path_rejected(self) -> None:
        """``ckpt_path`` is ``NonBlankStr | None`` — whitespace must not pass."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            TrainConfig.model_validate({"task_name": "train", "ckpt_path": "   "})

    def test_blank_optimized_metric_rejected(self) -> None:
        """``optimized_metric`` is ``NonBlankStr | None`` — whitespace must not pass."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            TrainConfig.model_validate({"task_name": "train", "optimized_metric": "   "})

    def test_blank_consumed_dataset_config_id_rejected(self) -> None:
        """``consumed_dataset_config_id`` is ``NonBlankStr | None`` — no whitespace id."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            TrainConfig.model_validate({"task_name": "train", "consumed_dataset_config_id": "   "})

    def test_blank_consumed_artifact_alias_rejected(self) -> None:
        """``consumed_artifact_alias`` is ``NonBlankStr`` — an empty alias is invalid."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            TrainConfig.model_validate({"task_name": "train", "consumed_artifact_alias": "   "})
