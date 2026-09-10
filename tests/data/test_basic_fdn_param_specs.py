"""Basic-spec classification must preserve the entire native effect."""

import numpy as np
import pytest

from synth_setter.data.pyfdn_instrument import PyFDNRenderer
from synth_setter.data.pyfdn_param_spec import BasicFDNParamSpec
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.param_spec_name import ParamSpecName


@pytest.mark.parametrize(
    "name",
    ["pyfdn_n8_mono_householder", "pyfdn_n8_mono_householder_vector", "pyfdn_n8_mono_kronecker"],
)
def test_basic_spec_build_preserves_complete_offline_effect(name: str) -> None:
    """No processing exists outside the basic build for a compatible spec.

    :param name: Registered basic FDN parameterization.
    """
    spec = resolve_param_spec(ParamSpecName(name))
    assert isinstance(spec, BasicFDNParamSpec)
    native, _ = spec.sample(np.random.default_rng(7))

    actual = spec.to_basic_fdn(native).impulse_response(4096)[:, 0, 0]
    expected = PyFDNRenderer(param_spec_name=ParamSpecName(name)).render(native)[0, :4096]

    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize(
    "name",
    [
        "pyfdn_gotz_n8_mono_fixed_delays",
        "pyfdn_gotz_n8_mono_learned_delays",
        "pyfdn_gotz_n8_mono_fixed_delays_givens",
        "pyfdn_gotz_n8_mono_learned_delays_givens",
        "pyfdn_pitchshift_n8_mono_householder",
        "pyfdn_diffvox",
    ],
)
def test_advanced_spec_does_not_claim_complete_basic_build(name: str) -> None:
    """A renderable FDN core does not describe an effect's custom outer processing.

    :param name: Registered advanced FDN effect.
    """
    assert not isinstance(resolve_param_spec(ParamSpecName(name)), BasicFDNParamSpec)


def test_basic_spec_feedback_controls_disagree_with_matrix_rejected() -> None:
    """The build cannot silently use feedback different from the encoded controls."""
    spec = resolve_param_spec(ParamSpecName("pyfdn_n8_mono_kronecker"))
    assert isinstance(spec, BasicFDNParamSpec)
    native, _ = spec.sample(np.random.default_rng(3))
    native["feedback_matrix"] = np.eye(8)

    with pytest.raises(ValueError, match="feedback_matrix"):
        spec.to_basic_fdn(native)
