"""Behavioural tests for the ``CallbacksConfig`` / ``CallbackInstance`` models.

Every YAML under ``configs/callbacks/`` is a valid ``callbacks=<name>``
selection — both the multi-callback compositions (``default``,
``default_surge``, ``eval_surge``) and the individual callback leaves
(``model_checkpoint``, ``early_stopping``, ``plot_pos_enc``, ...). Every
one of those that composes into a non-empty dict must validate against
``CallbacksConfig`` (a RootModel wrapping ``dict[str, CallbackInstance]``).
``none.yaml`` is intentionally empty — Hydra resolves it to ``None`` and
``instantiate_callbacks`` short-circuits on a falsy config, so it's
outside the schema's responsibility.

Callback-class-specific kwargs (``monitor``, ``dirpath``, ``patience``, ...)
live under ``extra="allow"`` on ``CallbackInstance``; only ``_target_`` is
typed at this layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_setter.schemas.callbacks_config import CallbackInstance, CallbacksConfig
from tests.schemas.conftest import compose_train_cfg

_CALLBACKS_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "callbacks"


def _all_callback_config_names() -> list[str]:  # noqa: DOC201,DOC203
    """Return every callback YAML stem under ``configs/callbacks/``."""
    names = sorted(p.stem for p in _CALLBACKS_CONFIG_DIR.glob("*.yaml"))
    assert names, (
        f"no callback YAMLs found under {_CALLBACKS_CONFIG_DIR} — "
        "has the callbacks composition layout changed?"
    )
    return names


def _compose_callbacks_subtree(  # noqa: DOC101,DOC103,DOC201,DOC203
    callbacks_name: str,
) -> dict[str, object] | None:
    """Compose a full train config with ``callbacks=<callbacks_name>`` selected.

    Returns the raw composed subtree (typically a dict, but for the
    intentionally-empty ``none.yaml`` it composes to ``None``).
    """
    cfg_dict = compose_train_cfg(overrides=[f"callbacks={callbacks_name}"])
    return cfg_dict["callbacks"]


class TestCallbacksConfigAcceptsEveryComposition:
    """Every ``callbacks=<name>``-selectable YAML must validate against ``CallbacksConfig``.

    Any callback YAML that composes to a non-dict (today only ``none.yaml``,
    which Hydra resolves to ``None``) is *recorded as a skip with rationale*
    so a future ``none2.yaml`` surfaces as a noticed skip rather than a
    silent ``ValidationError`` deep in an assertion stack.
    """

    @pytest.mark.parametrize("callbacks_name", _all_callback_config_names())
    def test_callbacks_yaml_validates(self, callbacks_name: str) -> None:  # noqa: DOC101,DOC103
        """The composed ``callbacks`` subtree validates as ``CallbacksConfig``."""
        callbacks_subtree = _compose_callbacks_subtree(callbacks_name)
        if not isinstance(callbacks_subtree, dict):
            pytest.skip(
                f"callbacks={callbacks_name} composes to non-dict "
                f"({type(callbacks_subtree).__name__}); schema covers dict form only"
            )
        parsed = CallbacksConfig.model_validate(callbacks_subtree)
        # ``none.yaml`` composes to ``{}`` and validates as an empty RootModel —
        # that's the intended ``instantiate_callbacks`` no-op pathway and we
        # don't want to assert truthiness here.
        assert isinstance(parsed, CallbacksConfig)


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
