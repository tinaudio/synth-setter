"""The vst_flow_fourier arm instantiates and round-trips real parameter coordinates."""

import hydra
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

from synth_setter.models.components.fourier_number import FourierNumberEmbedder
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)


def compose_vector_field(model_name: str) -> ApproxEquivTransformer:
    """Compose one model group and instantiate its vector field.

    :param model_name: Hydra model group to select.
    :returns: Instantiated vector-field module.
    """
    GlobalHydra.instance().clear()
    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(
                config_name="train.yaml",
                overrides=[
                    "datamodule=surge_simple",
                    f"model={model_name}",
                    "trainer=cpu",
                    # A narrow field keeps the round trip fast; widths stay interpolated.
                    "model.vector_field.num_layers=1",
                    "model.vector_field.d_model=32",
                    "model.vector_field.projection.num_tokens=4",
                    "model.encoder.d_model=32",
                ],
            )
    finally:
        GlobalHydra.instance().clear()
    vector_field = hydra.utils.instantiate(cfg.model.vector_field)
    assert isinstance(vector_field, ApproxEquivTransformer)
    return vector_field


def projection_of(vector_field: ApproxEquivTransformer) -> LearntProjection:
    """Narrow the injected projection to the learnt implementation under test.

    :param vector_field: Instantiated vector field.
    :returns: The vector field's learnt projection.
    """
    projection = vector_field.projection
    assert isinstance(projection, LearntProjection)
    return projection


def test_vst_flow_config_leaves_value_encoder_unset() -> None:
    """The baseline arm keeps the linear value path."""
    assert projection_of(compose_vector_field("vst_flow")).value_encoder is None


def test_vst_flow_fourier_config_selects_fourier_value_encoder() -> None:
    """The opt-in arm wires the Fourier embedder through Hydra."""
    projection = projection_of(compose_vector_field("vst_flow_fourier"))

    assert isinstance(projection.value_encoder, FourierNumberEmbedder)


@pytest.mark.parametrize("model_name", ["vst_flow", "vst_flow_fourier"])
def test_vst_flow_vector_field_returns_velocity_in_parameter_space(model_name: str) -> None:
    """Both arms map (B, P) coordinates back to a finite (B, P) velocity.

    :param model_name: Hydra model group selected for the arm under test.
    """
    vector_field = compose_vector_field(model_name)
    num_params = int(projection_of(vector_field).in_projection.shape[0])

    velocity = vector_field(torch.rand(2, num_params), torch.rand(2, 1), torch.rand(2, 32))

    assert velocity.shape == (2, num_params)
    assert torch.isfinite(velocity).all()


def test_vst_flow_fourier_backward_reaches_the_fourier_projection() -> None:
    """The configured encoder trains through the full vector field."""
    vector_field = compose_vector_field("vst_flow_fourier")
    projection = projection_of(vector_field)
    encoder = projection.value_encoder
    assert isinstance(encoder, FourierNumberEmbedder)
    num_params = int(projection.in_projection.shape[0])

    vector_field(torch.rand(2, num_params), torch.rand(2, 1), torch.rand(2, 32)).sum().backward()

    assert encoder.projection.weight.grad is not None
    assert torch.isfinite(encoder.projection.weight.grad).all()
