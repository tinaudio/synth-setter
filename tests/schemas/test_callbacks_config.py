"""Behavioural tests for ``CallbacksConfig`` / ``CallbackInstance``.

Every ``callbacks=<name>`` selection that composes to a non-empty dict must
validate. ``none.yaml`` composes to ``None`` and is handled outside the
schema (``instantiate_callbacks`` short-circuits on falsy).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_setter.resources import configs_dir
from synth_setter.schemas.callbacks_config import CallbackInstance, CallbacksConfig
from tests.schemas.conftest import compose_train_cfg

_CALLBACKS_CONFIG_DIR = Path(str(configs_dir() / "callbacks"))


def _all_callback_config_names() -> list[str]:
    """Return every callback YAML stem under ``configs/callbacks/``.

    :return: Sorted list of YAML stems found in ``configs/callbacks/``.
    """
    names = sorted(p.stem for p in _CALLBACKS_CONFIG_DIR.glob("*.yaml"))
    assert names, (
        f"no callback YAMLs found under {_CALLBACKS_CONFIG_DIR} — "
        "has the callbacks composition layout changed?"
    )
    return names


def _compose_callbacks_subtree(
    callbacks_name: str,
) -> dict[str, object] | None:
    """Compose with ``callbacks=<name>``; returns dict, or ``None`` for ``none.yaml``.

    :param callbacks_name: Name of the callbacks YAML under ``configs/callbacks/``.
    :return: Composed ``callbacks`` subtree as a dict, or ``None`` for ``none.yaml``.
    """
    cfg_dict = compose_train_cfg(overrides=[f"callbacks={callbacks_name}"])
    return cfg_dict["callbacks"]


class TestCallbacksConfigAcceptsEveryComposition:
    """Every ``callbacks=<name>``-selectable YAML must validate against ``CallbacksConfig``.

    Non-dict composes (today only ``none.yaml`` → ``None``) emit pytest skips
    so a future addition surfaces visibly rather than as a deep ValidationError.
    """

    @pytest.mark.parametrize("callbacks_name", _all_callback_config_names())
    def test_callbacks_yaml_validates(self, callbacks_name: str) -> None:
        """The composed ``callbacks`` subtree validates as ``CallbacksConfig``.

        :param callbacks_name: Parametrized YAML stem under ``configs/callbacks/``.
        """
        callbacks_subtree = _compose_callbacks_subtree(callbacks_name)
        if not isinstance(callbacks_subtree, dict):
            pytest.skip(
                f"callbacks={callbacks_name} composes to non-dict "
                f"({type(callbacks_subtree).__name__}); schema covers dict form only"
            )
        parsed = CallbacksConfig.model_validate(callbacks_subtree)
        assert isinstance(parsed, CallbacksConfig)


_VALID_CALLBACK = {
    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
    "dirpath": "/tmp/ckpt",  # noqa: S108
}


class TestCallbacksConfigAcceptsDisabledAndPartialEntries:
    """Composed callbacks legitimately omit ``_target_`` or disable an entry."""

    def test_partial_override_without_target_accepted(self) -> None:
        """A partial override (no ``_target_``) is skipped at instantiation, not rejected."""
        parsed = CallbacksConfig.model_validate({"model_checkpoint": {"monitor": "val/lsd"}})
        entry = parsed.root["model_checkpoint"]
        assert entry is not None and entry.target_ is None

    def test_disabled_entry_none_accepted(self) -> None:
        """An ``<name>: null`` disabled callback validates as a ``None`` value."""
        parsed = CallbacksConfig.model_validate({"early_stopping": None})
        assert parsed.root["early_stopping"] is None


class TestCallbacksConfigRejectsBadInputs:
    """Validators must catch obvious mistakes on the typed fields."""

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
