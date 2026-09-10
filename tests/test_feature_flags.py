"""Behavioral tests for runtime feature-flag resolution and activation."""

from __future__ import annotations

import os
from dataclasses import asdict

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.feature_flags import FeatureFlagConfig, apply_feature_flags

_FEATURE_FLAG_NAME = "SYNTH_SETTER_FF_3160_CORRECT_AST_PATCH_PADDING"


def test_feature_flag_config_known_number_resolves_metadata() -> None:
    """A configured integer ID resolves to its complete registry record."""
    config = FeatureFlagConfig.model_validate({"feature_flags": [3160]})

    assert asdict(config.feature_flags[0]) == {
        "number": 3160,
        "name": _FEATURE_FLAG_NAME,
        "description": "Use the corrected AST patch-padding axis order.",
    }


def test_feature_flag_config_dump_round_trips_as_integer_ids() -> None:
    """Serialized config keeps Hydra's integer-ID representation."""
    config = FeatureFlagConfig.model_validate({"feature_flags": [3160]})

    dumped = config.model_dump()

    assert dumped == {"feature_flags": [3160]}
    assert FeatureFlagConfig.model_validate(dumped) == config
    assert FeatureFlagConfig.model_validate_json(config.model_dump_json()) == config


def test_feature_flag_config_unknown_number_rejected() -> None:
    """An unregistered ID must fail before an endpoint starts work."""
    with pytest.raises(ValidationError, match="unknown feature flag number: 9999"):
        FeatureFlagConfig.model_validate({"feature_flags": [9999]})


def test_feature_flag_config_duplicate_number_rejected() -> None:
    """Repeated IDs are configuration mistakes rather than duplicate activation."""
    with pytest.raises(ValidationError, match="duplicate feature flag number: 3160"):
        FeatureFlagConfig.model_validate({"feature_flags": [3160, 3160]})


def test_feature_flag_config_boolean_rejected() -> None:
    """Booleans must not pass as integer IDs despite Python's bool subclassing."""
    with pytest.raises(ValidationError, match="feature flag numbers must be integers"):
        FeatureFlagConfig.model_validate({"feature_flags": [True]})


def test_feature_flag_config_string_number_rejected() -> None:
    """Strict resolution must not coerce string IDs from quoted Hydra values."""
    with pytest.raises(ValidationError, match="feature flag numbers must be integers"):
        FeatureFlagConfig.model_validate({"feature_flags": ["3160"]})


def test_apply_feature_flags_empty_selection_clears_registered_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each application replaces flags left by an earlier config in the process.

    :param monkeypatch: Isolates the process environment modified by activation.
    """
    monkeypatch.delenv(_FEATURE_FLAG_NAME, raising=False)
    apply_feature_flags(OmegaConf.create({"feature_flags": [3160]}))

    apply_feature_flags(OmegaConf.create({"feature_flags": []}))

    assert _FEATURE_FLAG_NAME not in os.environ


def test_apply_feature_flags_sets_selected_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Activation exposes the selected full flag name to downstream consumers.

    :param monkeypatch: Isolates the process environment modified by activation.
    """
    monkeypatch.delenv(_FEATURE_FLAG_NAME, raising=False)
    cfg = OmegaConf.create({"feature_flags": [3160]})

    resolved = apply_feature_flags(cfg)

    assert os.environ[_FEATURE_FLAG_NAME] == "1"
    assert resolved.feature_flags[0].number == 3160
