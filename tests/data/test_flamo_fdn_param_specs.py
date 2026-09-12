"""Basic-spec classification must preserve the entire native effect."""

import json

import matplotlib
import numpy as np
import pytest
from pyFDN import plot_flamo_graph

from synth_setter.data.pyfdn_instrument import PyFDNRenderer
from synth_setter.data.pyfdn_param_spec import (
    PYFDN_KRONECKER_ANGLES_NAME,
    PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC,
    PYFDN_KRONECKER_REFLECT_NAME,
    FlamoFDNParamSpec,
    kronecker_feedback_matrix,
    pyfdn_param_spec_json,
    pyfdn_param_spec_sha256,
)
from synth_setter.data.vst.param_spec_registry import param_specs, resolve_param_spec
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.synth_spec import SYNTHS, SynthName

matplotlib.use("Agg")

PYFDN_PARAM_SPEC_NAMES = tuple(name for name in param_specs if name.startswith("pyfdn_"))


@pytest.mark.parametrize("name", PYFDN_PARAM_SPEC_NAMES)
def test_every_pyfdn_spec_compiles_and_plots_complete_flamo_graph(name: str) -> None:
    """Every retained pyFDN identity is a complete FLAMO-compatible FDN.

    :param name: Registered pyFDN parameter specification.
    """
    spec = resolve_param_spec(ParamSpecName(name))
    assert isinstance(spec, FlamoFDNParamSpec)
    native, _ = spec.sample(np.random.default_rng(7))

    model = spec.to_flamo_fdn(native).to_flamo(nfft=4096)
    ax = plot_flamo_graph(model, name=name)

    assert ax.patches


@pytest.mark.parametrize("name", PYFDN_PARAM_SPEC_NAMES)
def test_every_pyfdn_spec_json_matches_registered_digest(name: str) -> None:
    """The stored identity digest detects parameter-schema drift.

    :param name: Registered pyFDN parameter specification.
    """
    spec = resolve_param_spec(ParamSpecName(name))
    assert isinstance(spec, FlamoFDNParamSpec)

    assert pyfdn_param_spec_sha256(spec) == SYNTHS[SynthName(name)].param_spec_sha256


def test_pyfdn_param_spec_json_is_canonical_and_complete() -> None:
    """Canonical JSON identifies the spec class, feedback rule, and ordered parameters."""
    payload = json.loads(pyfdn_param_spec_json(PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC))

    assert payload["type"] == "FlamoFDNParamSpec"
    assert payload["feedback_matrix"] == "_householder_feedback"
    assert [parameter["name"] for parameter in payload["synth_params"]] == [
        "delays",
        "input_matrix",
        "output_matrix",
        "direct_matrix",
        "post_delay.rt_dc_seconds",
        "post_delay.rt_nyquist_seconds",
    ]


@pytest.mark.parametrize(
    "name",
    ["pyfdn_n8_mono_householder", "pyfdn_n8_mono_householder_vector", "pyfdn_n8_mono_kronecker"],
)
def test_flamo_fdn_spec_build_preserves_complete_offline_effect(name: str) -> None:
    """No processing exists outside the FLAMO FDN build for a compatible spec.

    :param name: Registered FLAMO FDN parameterization.
    """
    spec = resolve_param_spec(ParamSpecName(name))
    assert isinstance(spec, FlamoFDNParamSpec)
    native, _ = spec.sample(np.random.default_rng(7))

    actual = spec.to_flamo_fdn(native).impulse_response(4096)[:, 0, 0]
    expected = PyFDNRenderer(param_spec_name=ParamSpecName(name)).render(native)[0, :4096]

    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_flamo_fdn_spec_missing_control_rejected() -> None:
    """A partial native mapping cannot construct a FLAMO FDN."""
    spec = resolve_param_spec(ParamSpecName("pyfdn_n8_mono_householder"))
    assert isinstance(spec, FlamoFDNParamSpec)
    native, _ = spec.sample(np.random.default_rng(3))
    del native["delays"]

    with pytest.raises(ValueError, match="exactly"):
        spec.to_flamo_fdn(native)


def test_flamo_fdn_spec_unexpected_control_rejected() -> None:
    """An unknown native field cannot be silently ignored."""
    spec = resolve_param_spec(ParamSpecName("pyfdn_n8_mono_householder"))
    assert isinstance(spec, FlamoFDNParamSpec)
    native, _ = spec.sample(np.random.default_rng(3))
    native["delayz"] = native["delays"]

    with pytest.raises(ValueError, match="exactly"):
        spec.to_flamo_fdn(native)


def test_flamo_fdn_spec_out_of_domain_control_rejected_when_feedback_matches() -> None:
    """A matching derived matrix cannot legitimize an invalid declared control."""
    spec = resolve_param_spec(ParamSpecName("pyfdn_n8_mono_kronecker"))
    assert isinstance(spec, FlamoFDNParamSpec)
    native, _ = spec.sample(np.random.default_rng(3))
    native[PYFDN_KRONECKER_REFLECT_NAME] = np.full(3, 2)
    native["feedback_matrix"] = kronecker_feedback_matrix(
        np.asarray(native[PYFDN_KRONECKER_ANGLES_NAME]),
        np.asarray(native[PYFDN_KRONECKER_REFLECT_NAME]),
    )

    with pytest.raises(ValueError, match=PYFDN_KRONECKER_REFLECT_NAME):
        spec.to_flamo_fdn(native)


def test_flamo_fdn_spec_feedback_controls_disagree_with_matrix_rejected() -> None:
    """The build cannot silently use feedback different from the encoded controls."""
    spec = resolve_param_spec(ParamSpecName("pyfdn_n8_mono_kronecker"))
    assert isinstance(spec, FlamoFDNParamSpec)
    native, _ = spec.sample(np.random.default_rng(3))
    native["feedback_matrix"] = np.eye(8)

    with pytest.raises(ValueError, match="feedback_matrix"):
        spec.to_flamo_fdn(native)
