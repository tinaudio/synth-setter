"""Behavioural tests for the ``PathsConfig`` pydantic model.

Pins that ``paths/default.yaml`` validates and that whitespace overrides on
any field are rejected by ``NonBlankStr``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
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


class TestPathsConfigResolvedInterpolation:
    """A real resolved ``PROJECT_ROOT`` must round-trip through ``NonBlankStr``."""

    def test_paths_resolved_with_env_var(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``${oc.env:PROJECT_ROOT}`` resolves to a real path and validates as non-blank.

        :param monkeypatch: Pytest fixture used to set ``PROJECT_ROOT``.
        """
        monkeypatch.setenv("PROJECT_ROOT", "/tmp/x")  # noqa: S108
        # output_dir / work_dir overridden because ``${hydra:runtime.*}`` is
        # only populated at fit time, not at compose time.
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="train.yaml",
                return_hydra_config=True,
                overrides=[
                    "datamodule=ksin",
                    "model=ffn",
                    "trainer=cpu",
                    "paths.output_dir=/tmp/x/out",  # noqa: S108
                    "paths.work_dir=/tmp/x",  # noqa: S108
                ],
            )
            HydraConfig.instance().set_config(cfg)
            resolved_paths = cast(
                "dict[str, Any]", OmegaConf.to_container(cfg.paths, resolve=True)
            )
        parsed = PathsConfig.model_validate(resolved_paths)
        assert parsed.root_dir == "/tmp/x"  # noqa: S108
        assert parsed.data_dir.startswith("/tmp/x")  # noqa: S108


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
    def test_blank_field_rejected(self, field: str) -> None:
        """Each path field rejects whitespace-only overrides.

        :param field: Parametrized name of the ``PathsConfig`` attribute under test.
        """
        bad = {**_VALID_PATHS, field: "   "}
        with pytest.raises(ValidationError, match="at least 1 character"):
            PathsConfig.model_validate(bad)

    def test_missing_root_dir_rejected(self) -> None:
        """``root_dir`` has no default — omitting it must fail."""
        with pytest.raises(ValidationError):
            PathsConfig.model_validate({k: v for k, v in _VALID_PATHS.items() if k != "root_dir"})
