"""Behavior tests for field-aligned parameter-token projection."""

import pytest
import torch

from synth_setter.data.vst import param_spec_registry
from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousArrayParameter,
    ContinuousParameter,
    ParamSpec,
)
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    GroupedParameterProjection,
    LearntProjection,
    PositionalEncoding,
)
from synth_setter.param_spec_name import ParamSpecName

_SPEC_NAME = "test_grouped_projection"


@pytest.fixture
def grouped_spec(monkeypatch: pytest.MonkeyPatch) -> ParamSpec:
    """Register a mixed-width spec in scalar, categorical, array order.

    :param monkeypatch: Restores the process-local registry after each test.
    :returns: Registered three-field parameter specification.
    """
    spec = ParamSpec(
        synth_params=[
            ContinuousParameter("gain"),
            CategoricalParameter("mode", ["low", "band", "high"], encoding="onehot"),
            ContinuousArrayParameter("matrix", shape=(2, 2), min=-1.0, max=1.0),
        ],
        note_params=[],
    )
    monkeypatch.setitem(param_spec_registry._param_specs, ParamSpecName(_SPEC_NAME), spec)
    return spec


def _projection(d_model: int = 8) -> GroupedParameterProjection:
    return GroupedParameterProjection(d_model=d_model, param_spec_name=_SPEC_NAME)


def test_grouped_projection_uses_one_token_per_field_in_spec_order(
    grouped_spec: ParamSpec,
) -> None:
    """Map mixed scalar, categorical, and array spans to ordered field tokens.

    :param grouped_spec: Registered mixed-width parameter specification.
    """
    projection = _projection()

    tokens = projection.param_to_token(torch.randn(2, grouped_spec.encoded_width))

    assert projection.num_tokens == 3
    assert tokens.shape == (2, 3, 8)


@pytest.mark.parametrize(
    ("changed_span", "changed_token"),
    [(slice(0, 1), 0), (slice(1, 4), 1), (slice(4, 8), 2)],
)
def test_grouped_projection_changed_field_changes_only_its_token(
    grouped_spec: ParamSpec,
    changed_span: slice,
    changed_token: int,
) -> None:
    """Keep each field's encoded columns isolated from every other token.

    :param grouped_spec: Registered mixed-width parameter specification.
    :param changed_span: Encoded columns changed from the baseline.
    :param changed_token: Sole token expected to change.
    """
    projection = _projection()
    baseline = torch.zeros(1, grouped_spec.encoded_width)
    changed = baseline.clone()
    changed[:, changed_span] = 1.0

    token_delta = projection.param_to_token(changed) - projection.param_to_token(baseline)

    assert torch.count_nonzero(token_delta[:, changed_token])
    token_delta[:, changed_token] = 0
    assert torch.count_nonzero(token_delta) == 0


def test_grouped_projection_rejects_empty_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a spec that cannot provide any parameter tokens.

    :param monkeypatch: Restores the process-local registry after the test.
    """
    empty_spec_name = "test_empty_grouped_projection"
    monkeypatch.setitem(
        param_spec_registry._param_specs,
        ParamSpecName(empty_spec_name),
        ParamSpec(synth_params=[], note_params=[]),
    )

    with pytest.raises(ValueError, match="at least one field"):
        GroupedParameterProjection(d_model=8, param_spec_name=empty_spec_name)


def test_grouped_projection_rejects_wrong_parameter_width(grouped_spec: ParamSpec) -> None:
    """Reject flat rows wider or narrower than the registered encoded spec.

    :param grouped_spec: Registered mixed-width parameter specification.
    """
    projection = _projection()

    with pytest.raises(ValueError, match="parameter width"):
        projection.param_to_token(torch.randn(2, grouped_spec.encoded_width + 1))


def test_grouped_projection_rejects_wrong_token_count(grouped_spec: ParamSpec) -> None:
    """Reject token sequences that do not contain exactly one token per field.

    :param grouped_spec: Registered mixed-width parameter specification.
    """
    projection = _projection()

    with pytest.raises(ValueError, match="token count"):
        projection.token_to_param(torch.randn(2, projection.num_tokens + 1, 8))


def test_grouped_projection_decodes_fields_in_encoded_spec_order(
    grouped_spec: ParamSpec,
) -> None:
    """Concatenate scalar, categorical, and array decoder outputs in spec order.

    :param grouped_spec: Registered mixed-width parameter specification.
    """
    projection = _projection()
    scalar_decoder, categorical_decoder, array_decoder = projection.decoders
    assert isinstance(scalar_decoder, torch.nn.Linear)
    assert isinstance(categorical_decoder, torch.nn.Linear)
    assert isinstance(array_decoder, torch.nn.Linear)
    with torch.no_grad():
        scalar_decoder.weight.zero_()
        scalar_decoder.bias.copy_(torch.tensor([1.0]))
        categorical_decoder.weight.zero_()
        categorical_decoder.bias.copy_(torch.tensor([2.0, 3.0, 4.0]))
        array_decoder.weight.zero_()
        array_decoder.bias.copy_(torch.tensor([5.0, 6.0, 7.0, 8.0]))

    decoded = projection.token_to_param(torch.randn(1, 3, 8))

    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    torch.testing.assert_close(decoded, expected)


def test_grouped_projection_reconstruction_trains_every_head(grouped_spec: ParamSpec) -> None:
    """Route reconstruction gradients through every field encoder and decoder.

    :param grouped_spec: Registered mixed-width parameter specification.
    """
    projection = _projection()

    projection.token_to_param(projection.param_to_token(torch.randn(2, 8))).sum().backward()

    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in projection.parameters()
    )


def test_grouped_projection_penalty_is_scalar_tensor(grouped_spec: ParamSpec) -> None:
    """Return a device-compatible scalar zero without an assignment penalty.

    :param grouped_spec: Registered mixed-width parameter specification.
    """
    projection = _projection()

    penalty = projection.penalty()

    assert penalty.shape == ()
    assert penalty.item() == 0.0
    assert not hasattr(projection, "assignment")


def test_transformer_grouped_projection_forward_backward_and_pe(grouped_spec: ParamSpec) -> None:
    """Size positional encoding from grouped fields and train the full vector field.

    :param grouped_spec: Registered mixed-width parameter specification.
    """
    projection = _projection(d_model=8)
    transformer = ApproxEquivTransformer(
        projection=projection,
        num_layers=1,
        d_model=8,
        conditioning_dim=4,
        num_heads=2,
        d_ff=8,
        learn_projection=True,
        pe_type="initial",
        zero_init=False,
    )
    params = torch.randn(2, grouped_spec.encoded_width, requires_grad=True)

    output = transformer(params, torch.rand(2, 1), torch.randn(2, 4))
    output.sum().backward()

    assert output.shape == params.shape
    assert isinstance(transformer.pe, PositionalEncoding)
    assert transformer.pe.pe.shape == (1, projection.num_tokens, 8)
    assert params.grad is not None and torch.count_nonzero(params.grad)


def test_transformer_grouped_projection_runs_under_torch_compile(grouped_spec: ParamSpec) -> None:
    """Run the typed grouped projection through the production compilation boundary.

    :param grouped_spec: Registered mixed-width parameter specification.
    """
    transformer = ApproxEquivTransformer(
        projection=_projection(),
        num_layers=1,
        d_model=8,
        conditioning_dim=4,
        num_heads=2,
        d_ff=8,
        learn_projection=True,
        pe_type="none",
        zero_init=False,
    )
    compiled = torch.compile(transformer, backend="eager")

    params = torch.randn(2, 8, requires_grad=True)
    output = compiled(params, torch.rand(2, 1), torch.randn(2, 4))
    output.sum().backward()

    assert output.shape == (2, 8)
    assert params.grad is not None and torch.count_nonzero(params.grad)


def test_transformer_rejects_legacy_num_tokens_mismatch(grouped_spec: ParamSpec) -> None:
    """Reject an explicit legacy token count that disagrees with the projection.

    :param grouped_spec: Registered mixed-width parameter specification.
    """
    with pytest.raises(ValueError, match="num_tokens"):
        ApproxEquivTransformer(projection=_projection(), d_model=8, num_tokens=4)


def test_transformer_freezes_whole_projection_when_disabled(grouped_spec: ParamSpec) -> None:
    """Freeze grouped encoder and decoder heads for a fixed projection.

    :param grouped_spec: Registered mixed-width parameter specification.
    """
    transformer = ApproxEquivTransformer(
        projection=_projection(), d_model=8, num_layers=1, num_heads=2, d_ff=8
    )

    assert all(not parameter.requires_grad for parameter in transformer.projection.parameters())


def test_transformer_freezes_learnt_projection_when_disabled(grouped_spec: ParamSpec) -> None:
    """Freeze every legacy learnt-projection parameter without using a missing attribute.

    :param grouped_spec: Registered spec supplying a representative encoded width.
    """
    projection = LearntProjection(
        d_model=8,
        d_token=8,
        num_params=grouped_spec.encoded_width,
        num_tokens=3,
    )

    transformer = ApproxEquivTransformer(
        projection=projection, d_model=8, num_layers=1, num_heads=2, d_ff=8
    )

    assert all(not parameter.requires_grad for parameter in transformer.projection.parameters())


def test_projection_checkpoints_reject_cross_architecture(grouped_spec: ParamSpec) -> None:
    """Reject learnt weights when strict-loading a grouped projection and vice versa.

    :param grouped_spec: Registered mixed-width parameter specification.
    """
    learnt = LearntProjection(
        d_model=8,
        d_token=8,
        num_params=grouped_spec.encoded_width,
        num_tokens=3,
    )
    grouped = _projection()

    with pytest.raises(RuntimeError):
        grouped.load_state_dict(learnt.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        learnt.load_state_dict(grouped.state_dict(), strict=True)
