"""Behavioural tests for the ``CallbacksConfig`` / ``CallbackInstance`` models.

Every YAML under ``configs/callbacks/`` that composes into a non-empty dict
must validate against ``CallbacksConfig`` (a RootModel wrapping
``dict[str, CallbackInstance]``). ``none.yaml`` is intentionally empty —
Hydra resolves it to ``None`` and ``instantiate_callbacks`` short-circuits
on a falsy config, so it's outside the schema's responsibility.

Callback-class-specific kwargs (``monitor``, ``dirpath``, ``patience``, ...)
live under ``extra="allow"`` on ``CallbackInstance``; only ``_target_`` is
typed at this layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.schemas.callbacks_config import CallbackInstance, CallbacksConfig

_CALLBACKS_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "callbacks"

# ``none.yaml`` is empty and composes to ``None``; ``instantiate_callbacks``
# handles that pathway outside the schema. Skip it here.
_NON_EMPTY_CALLBACK_CONFIGS = frozenset({"default", "default_surge", "eval_surge"})


def _composable_callback_config_names() -> list[str]:  # noqa: DOC201,DOC203
    """Return callback-group YAML stems that compose into a non-empty dict.

    Only the multi-callback compositions (``default``, ``default_surge``,
    ``eval_surge``) are valid top-level selections under ``callbacks=...``;
    the individual callback YAMLs (``model_checkpoint``, ``lr_monitor``,
    ``rich_progress_bar``, ...) are leaves consumed by ``defaults:`` and
    aren't themselves group-selectable.
    """
    available = {p.stem for p in _CALLBACKS_CONFIG_DIR.glob("*.yaml")}
    selected = sorted(_NON_EMPTY_CALLBACK_CONFIGS & available)
    assert selected, (
        f"none of {_NON_EMPTY_CALLBACK_CONFIGS} found under {_CALLBACKS_CONFIG_DIR} — "
        "has the callbacks composition layout changed?"
    )
    return selected


def _compose_callbacks_cfg(callbacks_name: str) -> dict[str, Any]:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Compose a full train config with ``callbacks=<callbacks_name>`` selected."""
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"callbacks={callbacks_name}",
                "data=ksin",
                "model=ffn",
                "trainer=cpu",
            ],
        )
    callbacks_subtree = OmegaConf.to_container(cfg.callbacks, resolve=False)
    assert isinstance(callbacks_subtree, dict)
    return cast("dict[str, Any]", callbacks_subtree)


class TestCallbacksConfigAcceptsEveryComposition:
    """Every non-empty callbacks group must validate against ``CallbacksConfig``."""

    @pytest.mark.parametrize("callbacks_name", _composable_callback_config_names())
    def test_callbacks_yaml_validates(self, callbacks_name: str) -> None:  # noqa: DOC101,DOC103
        """The composed ``callbacks`` subtree validates as ``CallbacksConfig``."""
        callbacks_subtree = _compose_callbacks_cfg(callbacks_name)
        CallbacksConfig.model_validate(callbacks_subtree)

    def test_default_yields_expected_callback_names(self) -> None:
        """Spot-check that ``default.yaml`` composes the callbacks ``train.py`` expects."""
        callbacks_subtree = _compose_callbacks_cfg("default")
        parsed = CallbacksConfig.model_validate(callbacks_subtree)
        names = set(parsed.root.keys())
        # Names known to be in default.yaml per its defaults: list.
        assert {"model_checkpoint", "lr_monitor", "rich_progress_bar"}.issubset(names)


_VALID_CALLBACK = {
    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
    "dirpath": "/tmp/ckpt",  # noqa: S108
}


class TestCallbacksConfigRejectsBadInputs:
    """Validators must catch obvious mistakes on the typed fields."""

    def test_missing_target_in_instance_rejected(self) -> None:
        """Each callback instance must carry ``_target_``; reject if absent."""
        with pytest.raises(ValidationError):
            CallbacksConfig.model_validate({"model_checkpoint": {"dirpath": "/tmp"}})  # noqa: S108

    def test_blank_target_rejected(self) -> None:
        """A blank ``_target_`` would crash ``hydra.utils.instantiate`` mid-fit."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            CallbacksConfig.model_validate({"cb": {"_target_": "  "}})

    def test_blank_callback_name_rejected(self) -> None:
        """RootModel key is ``NonBlankStr`` — empty callback names are rejected."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            CallbacksConfig.model_validate({"   ": _VALID_CALLBACK})


class TestCallbackInstanceDirect:
    """Direct ``CallbackInstance`` validation works for individual entries."""

    def test_valid_instance_parses(self) -> None:
        """A minimal valid callback dict parses cleanly."""
        parsed = CallbackInstance.model_validate(_VALID_CALLBACK)
        assert parsed.target_ == "lightning.pytorch.callbacks.ModelCheckpoint"
