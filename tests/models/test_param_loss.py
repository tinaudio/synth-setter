"""Tests for per-parameter VAE loss aggregation."""

from __future__ import annotations

import pytest
import torch

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.components.vae import param_loss

_PARAM_LOSS_ENCODED_SLICES_ENV = "SYNTH_SETTER_PARAM_LOSS_ENCODED_SLICES"


@pytest.mark.parametrize("value", ["1", "on", "true", "yes"])
def test_param_loss_env_enabled_uses_encoded_slices(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Select encoded-slice aggregation for each truthy gate value.

    :param monkeypatch: Pytest environment and attribute patcher.
    :param value: Truthy gate value under test.
    """

    def encoded_slices() -> None:
        raise RuntimeError("encoded slices used")

    spec = param_specs["surge_4"]
    monkeypatch.setattr(spec, "encoded_slices", encoded_slices)
    monkeypatch.setenv(_PARAM_LOSS_ENCODED_SLICES_ENV, value)
    x = torch.rand(2, spec.encoded_width)

    with pytest.raises(RuntimeError, match="encoded slices used"):
        param_loss(x.clone(), x, "surge_4")


def test_param_loss_env_unset_keeps_legacy_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep legacy aggregation when the environment gate is absent.

    :param monkeypatch: Pytest environment and attribute patcher.
    """

    def encoded_slices() -> None:
        raise AssertionError("default path used encoded slices")

    spec = param_specs["surge_4"]
    monkeypatch.setattr(spec, "encoded_slices", encoded_slices)
    monkeypatch.delenv(_PARAM_LOSS_ENCODED_SLICES_ENV, raising=False)
    x = torch.rand(2, spec.encoded_width)

    loss = param_loss(x.clone(), x, "surge_4")

    assert torch.isfinite(loss)


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "unexpected"])
def test_param_loss_env_false_value_keeps_legacy_aggregation(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Keep legacy aggregation for false and unrecognized gate values.

    :param monkeypatch: Pytest environment and attribute patcher.
    :param value: Disabled gate value under test.
    """

    def encoded_slices() -> None:
        raise AssertionError("disabled path used encoded slices")

    spec = param_specs["surge_4"]
    monkeypatch.setattr(spec, "encoded_slices", encoded_slices)
    monkeypatch.setenv(_PARAM_LOSS_ENCODED_SLICES_ENV, value)
    x = torch.rand(2, spec.encoded_width)

    loss = param_loss(x.clone(), x, "surge_4")

    assert torch.isfinite(loss)


@pytest.mark.parametrize("param_spec", ["surge_4", "surge_simple", "surge_xt", "obxf"])
def test_param_loss_encoded_slices_perfect_reconstruction_is_minimal(
    monkeypatch: pytest.MonkeyPatch, param_spec: str
) -> None:
    """Score a perfect reconstruction below a corrupted encoded row.

    :param monkeypatch: Pytest environment patcher.
    :param param_spec: Registered parameter spec under test.
    """
    monkeypatch.setenv(_PARAM_LOSS_ENCODED_SLICES_ENV, "1")
    width = param_specs[param_spec].encoded_width
    x = torch.linspace(0.1, 0.9, width).repeat(4, 1)

    perfect = param_loss(x.clone(), x, param_spec)
    corrupted = param_loss(1.0 - x, x, param_spec)

    assert perfect < corrupted


def test_param_loss_encoded_slices_uses_categorical_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply cross-entropy to one-hot parameters on the enabled path.

    :param monkeypatch: Pytest environment patcher.
    """
    monkeypatch.setenv(_PARAM_LOSS_ENCODED_SLICES_ENV, "1")
    spec = param_specs["surge_xt"]
    x = torch.zeros(2, spec.encoded_width)

    loss = param_loss(x.clone(), x, "surge_xt")

    assert loss > 0


def test_param_loss_encoded_slices_rejects_row_wider_than_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject rows wider than the registered spec on the enabled path.

    :param monkeypatch: Pytest environment patcher.
    """
    monkeypatch.setenv(_PARAM_LOSS_ENCODED_SLICES_ENV, "1")
    x = torch.rand(2, param_specs["surge_4"].encoded_width + 3)

    with pytest.raises(AssertionError):
        param_loss(x.clone(), x, "surge_4")


def test_param_loss_encoded_slices_scores_final_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Include the final encoded column in the enabled-path loss.

    :param monkeypatch: Pytest environment patcher.
    """
    monkeypatch.setenv(_PARAM_LOSS_ENCODED_SLICES_ENV, "1")
    spec = param_specs["surge_4"]
    x = torch.full((4, spec.encoded_width), 0.25)
    x_hat = x.clone()
    x_hat[:, -1] = 0.75

    assert param_loss(x_hat, x, "surge_4") != param_loss(x.clone(), x, "surge_4")
