"""Behavioural tests for the ``ExtrasConfig`` pydantic model.

``configs/extras/default.yaml`` is the only composition in the repo. The
positive case asserts it validates and the typed scalars survive the round
trip; the negative cases pin the ``StrictBool`` / ``Literal`` constraints
so a stray ``"yes"`` or unknown precision value fails at compose time
instead of producing a confusing runtime error inside
``synth_setter.utils.extras``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.schemas.extras_config import ExtrasConfig


def _compose_extras_cfg() -> dict[str, Any]:  # noqa: DOC201,DOC203
    """Compose a full train config and return its ``extras`` subtree as a dict."""
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=ksin", "model=ffn", "trainer=cpu"],
        )
    extras_subtree = OmegaConf.to_container(cfg.extras, resolve=False)
    assert isinstance(extras_subtree, dict)
    return cast("dict[str, Any]", extras_subtree)


class TestExtrasConfigAcceptsDefault:
    """The shipped ``extras/default.yaml`` composition validates."""

    def test_default_validates(self) -> None:
        """All four toggles land on the parsed model with the YAML values."""
        extras_subtree = _compose_extras_cfg()
        parsed = ExtrasConfig.model_validate(extras_subtree)
        assert parsed.ignore_warnings is False
        assert parsed.enforce_tags is True
        assert parsed.print_config is True
        assert parsed.float32_matmul_precision == "high"


class TestExtrasConfigRejectsBadInputs:
    """Validators must catch obvious mistakes on the typed fields."""

    def test_string_bool_rejected_by_strict(self) -> None:
        """``ignore_warnings`` is ``StrictBool``; ``"yes"`` must not silently coerce."""
        with pytest.raises(ValidationError, match="bool"):
            ExtrasConfig.model_validate({"ignore_warnings": "yes"})

    def test_unknown_precision_rejected(self) -> None:
        """``float32_matmul_precision`` is a ``Literal`` of the three values torch accepts."""
        with pytest.raises(ValidationError):
            ExtrasConfig.model_validate({"float32_matmul_precision": "extreme"})
