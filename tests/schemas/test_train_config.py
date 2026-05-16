"""Behavioural tests for the ``TrainConfig`` pydantic model.

The pydantic model documents the shape of ``configs/train.yaml`` plus its
``defaults:`` composition. The tests assert two things:

1. The model accepts the live composed config — so the published docs stay
   honest about what the entrypoint receives at runtime.
2. The model rejects obvious mistakes (wrong types, blank ``task_name``,
   negative ``seed``) so the doc-vs-reality contract is enforced at
   validation time, not just at import.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.schemas.train_config import TrainConfig


def _composed_train_cfg_dict(  # noqa: DOC101,DOC103,DOC201,DOC203
    *,
    return_hydra_config: bool = False,
) -> dict[str, Any]:
    """Compose ``configs/train.yaml`` for testing and return it as a plain dict."""
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=return_hydra_config,
            overrides=["data=ksin", "model=ffn", "trainer=cpu"],
        )
    container = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(container, dict)
    return cast("dict[str, Any]", container)


class TestTrainConfigAcceptsLiveCompose:
    """The pydantic model must accept the configs Hydra actually composes."""

    def test_default_composition_validates(self) -> None:
        """The default ``train.yaml`` composition validates without error."""
        cfg_dict = _composed_train_cfg_dict()
        TrainConfig.model_validate(cfg_dict)

    def test_default_composition_with_hydra_subtree_validates(self) -> None:
        """``return_hydra_config=True`` adds a ``hydra`` key; the model accepts it."""
        cfg_dict = _composed_train_cfg_dict(return_hydra_config=True)
        TrainConfig.model_validate(cfg_dict)

    def test_typed_scalars_survive_round_trip(self) -> None:
        """Every typed scalar lands on the model with the value the YAML declares."""
        cfg_dict = _composed_train_cfg_dict()
        model = TrainConfig.model_validate(cfg_dict)
        assert model.task_name == "train"
        assert model.tags == ["dev"]
        assert model.train is True
        assert model.test is True
        assert model.ckpt_path is None
        assert model.seed is None
        assert model.optimized_metric is None
        assert model.watch_gradients is None


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
