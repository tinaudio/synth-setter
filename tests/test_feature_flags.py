"""Behavioral tests for runtime feature-flag resolution and activation."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.feature_flags import FeatureFlagConfig, apply_feature_flags


def test_feature_flag_config_removed_ast_flag_rejected() -> None:
    """The superseded AST runtime flag must not remain selectable."""
    with pytest.raises(ValidationError, match="unknown feature flag number: 3160"):
        FeatureFlagConfig.model_validate({"feature_flags": [3160]})


def test_feature_flag_config_empty_selection_round_trips() -> None:
    """Serialized empty selections keep Hydra's integer-list representation."""
    config = FeatureFlagConfig.model_validate({"feature_flags": []})

    dumped = config.model_dump()

    assert dumped == {"feature_flags": []}
    assert FeatureFlagConfig.model_validate(dumped) == config
    assert FeatureFlagConfig.model_validate_json(config.model_dump_json()) == config


def test_feature_flag_config_unknown_number_rejected() -> None:
    """An unregistered ID must fail before an endpoint starts work."""
    with pytest.raises(ValidationError, match="unknown feature flag number: 9999"):
        FeatureFlagConfig.model_validate({"feature_flags": [9999]})


def test_feature_flag_config_boolean_rejected() -> None:
    """Booleans must not pass as integer IDs despite Python's bool subclassing."""
    with pytest.raises(ValidationError, match="feature flag numbers must be integers"):
        FeatureFlagConfig.model_validate({"feature_flags": [True]})


def test_feature_flag_config_string_number_rejected() -> None:
    """Strict resolution must not coerce string IDs from quoted Hydra values."""
    with pytest.raises(ValidationError, match="feature flag numbers must be integers"):
        FeatureFlagConfig.model_validate({"feature_flags": ["3160"]})


def test_apply_feature_flags_empty_selection_returns_validated_config() -> None:
    """Activation accepts the empty registry selection used by endpoints."""
    resolved = apply_feature_flags(OmegaConf.create({"feature_flags": []}))

    assert resolved.feature_flags == []
