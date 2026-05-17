"""Behavioural tests for the ``PathsConfig`` pydantic model.

The shipped ``paths: default`` composition is the only one in the repo and
it must validate against ``PathsConfig``. The negative cases pin the
``NonBlankStr`` contract — empty / whitespace overrides would propagate
broken paths into half a dozen downstream YAMLs and must fail early.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.schemas.paths_config import PathsConfig


def _compose_paths_cfg() -> dict[str, Any]:  # noqa: DOC201,DOC203
    """Compose a full train config and return its ``paths`` subtree as a dict."""
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=ksin", "model=ffn", "trainer=cpu"],
        )
    paths_subtree = OmegaConf.to_container(cfg.paths, resolve=False)
    assert isinstance(paths_subtree, dict)
    return cast("dict[str, Any]", paths_subtree)


class TestPathsConfigAcceptsDefault:
    """The shipped ``paths/default.yaml`` composition validates."""

    def test_default_validates(self) -> None:
        """All five string fields land on the parsed model."""
        paths_subtree = _compose_paths_cfg()
        parsed = PathsConfig.model_validate(paths_subtree)
        # The values are unresolved interpolation templates; we only assert
        # they're non-blank strings, which is what the schema actually
        # enforces.
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
