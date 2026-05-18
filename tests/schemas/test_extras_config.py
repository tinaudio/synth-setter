"""Behavioural tests for the ``ExtrasConfig`` pydantic model.

Pins both directions on the only shipped composition
(``configs/extras/default.yaml``): valid YAML round-trips, and
``StrictBool`` / ``Literal`` reject ``"yes"`` / unknown precisions.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_setter.schemas.extras_config import ExtrasConfig
from tests.schemas.conftest import compose_subtree


class TestExtrasConfigAcceptsDefault:
    """The shipped ``extras/default.yaml`` composition validates."""

    def test_default_validates(self) -> None:
        """All four toggles land on the parsed model with the YAML values."""
        extras_subtree = compose_subtree("extras", "default")
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
