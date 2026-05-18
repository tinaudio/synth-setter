"""Behavioural tests for the ``PathsConfig`` pydantic model.

Pins that ``paths/default.yaml`` validates and that whitespace overrides on
any field are rejected by ``NonBlankStr``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_setter.schemas.paths_config import PathsConfig
from tests.schemas.conftest import compose_subtree


class TestPathsConfigAcceptsDefault:
    """The shipped ``paths/default.yaml`` composition validates."""

    def test_default_validates(self) -> None:
        """All five string fields land on the parsed model."""
        paths_subtree = compose_subtree("paths", "default")
        parsed = PathsConfig.model_validate(paths_subtree)
        # Values here are unresolved interpolation templates; assert non-blank only.
        assert parsed.root_dir
        assert parsed.data_dir
        assert parsed.log_dir
        assert parsed.output_dir
        assert parsed.work_dir


_VALID_PATHS = {
    "root_dir": "/proj",
    "data_dir": "/proj/data",
    "log_dir": "/proj/logs",
    "output_dir": "/proj/out",
    "work_dir": "/proj",
}


class TestPathsConfigRejectsBadInputs:
    """Validators must reject blank overrides on every path field."""

    @pytest.mark.parametrize("field", list(_VALID_PATHS))
    def test_blank_field_rejected(self, field: str) -> None:  # noqa: DOC101,DOC103
        """Each path field rejects whitespace-only overrides."""
        bad = {**_VALID_PATHS, field: "   "}
        with pytest.raises(ValidationError, match="at least 1 character"):
            PathsConfig.model_validate(bad)

    def test_missing_root_dir_rejected(self) -> None:
        """``root_dir`` has no default — omitting it must fail."""
        with pytest.raises(ValidationError):
            PathsConfig.model_validate({k: v for k, v in _VALID_PATHS.items() if k != "root_dir"})
