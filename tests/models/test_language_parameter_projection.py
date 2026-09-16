"""Language fusion preserves the grouped numeric contract."""

import shutil
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch

from synth_setter.data.vst.param_spec_registry import param_specs
from synth_setter.models.components.language_projection import LanguageParameterProjection
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    GroupedParameterProjection,
    ParamTokenEmbed,
)
from synth_setter.pipeline.data.add_embeddings import embedding_field_metadata
from synth_setter.pipeline.data.param_language import (
    PARAM_NAME_EMBEDDING_FIELD,
    PARAM_NAME_EMBEDDING_REGISTRY_KEY,
    describe_fields,
    prepare_param_name_embeddings,
    write_param_name_dataset,
)
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig


def _write_artifact(path: Path, vectors: np.ndarray) -> None:
    write_param_name_dataset(path, "surge_4", "surge_4", dimension=vectors.shape[1])
    metadata = embedding_field_metadata(
        PARAM_NAME_EMBEDDING_REGISTRY_KEY,
        AddEmbeddingsConfig(
            lance_uri=str(path),
            embeddings=(PARAM_NAME_EMBEDDING_REGISTRY_KEY,),
            build_index=False,
            param_name_embedding_dimension=vectors.shape[1],
        ),
    )
    field = pa.field(
        PARAM_NAME_EMBEDDING_FIELD,
        pa.list_(pa.float32(), vectors.shape[1]),
        metadata=metadata,
    )
    table = (
        lance.dataset(path)
        .to_table()
        .append_column(
            field,
            pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)), vectors.shape[1]),
        )
    )
    lance.write_dataset(table, path, mode="overwrite")


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    """Create an aligned numeric fixture for projection-only tests.

    :param tmp_path: Isolated artifact directory.
    :returns: Pickle-free field metadata path.
    """
    count = len(describe_fields("surge_4", "surge_4"))
    vectors = np.random.default_rng(7).normal(size=(count, 128)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    path = tmp_path / "params.lance"
    _write_artifact(path, vectors)
    return path


def test_projection_zero_residual_matches_grouped(artifact: Path) -> None:
    """Initialization adds no correction to matching numeric heads.

    :param artifact: Aligned field metadata.
    """
    torch.manual_seed(12)
    grouped = GroupedParameterProjection(16, "surge_4")
    torch.manual_seed(12)
    projection = LanguageParameterProjection(
        16, "surge_4", "surge_4", embedding_path=str(artifact)
    )
    projection.initialize_embeddings()
    x = torch.randn(2, param_specs["surge_4"].encoded_width)
    torch.testing.assert_close(projection.param_to_token(x), grouped.param_to_token(x))


def test_projection_checkpoint_without_artifact_preserves_tokens(artifact: Path) -> None:
    """State restoration supplies embeddings without reopening their source.

    :param artifact: Aligned field metadata removed before restoration.
    """
    projection = LanguageParameterProjection(
        16, "surge_4", "surge_4", embedding_path=str(artifact)
    )
    projection.initialize_embeddings()
    x = torch.randn(2, param_specs["surge_4"].encoded_width)
    expected = projection.param_to_token(x)
    state = projection.state_dict()
    shutil.rmtree(artifact)
    restored = LanguageParameterProjection(16, "surge_4", "surge_4")
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.param_to_token(x), expected)


def test_projection_encoder_wrapper_freezes_only_decoders(artifact: Path) -> None:
    """SLAP's encoder wrapper leaves language adaptation trainable.

    :param artifact: Aligned field metadata.
    """
    projection = LanguageParameterProjection(
        16, "surge_4", "surge_4", embedding_path=str(artifact)
    )
    projection.initialize_embeddings()
    ParamTokenEmbed(projection)
    assert all(not parameter.requires_grad for parameter in projection.decoders.parameters())
    assert all(parameter.requires_grad for parameter in projection.text_adapter.parameters())
    assert not projection.language_embeddings.requires_grad


def test_projection_optimizer_step_opens_semantic_gradient_path(artifact: Path) -> None:
    """The zero final layer learns first, then propagates gradients to the adapter.

    :param artifact: Aligned field metadata.
    """
    projection = LanguageParameterProjection(
        16, "surge_4", "surge_4", embedding_path=str(artifact)
    )
    projection.initialize_embeddings()
    optimizer = torch.optim.Adam(projection.parameters(), lr=0.01)
    x = torch.randn(2, param_specs["surge_4"].encoded_width)
    projection.param_to_token(x).square().mean().backward()
    final_layer = projection.fusion[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    assert final_layer.weight.grad is not None
    assert final_layer.weight.grad.abs().sum() > 0
    assert projection.text_adapter.weight.grad is not None
    assert projection.text_adapter.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad()
    projection.param_to_token(x).square().mean().backward()
    assert projection.text_adapter.weight.grad is not None
    assert projection.text_adapter.weight.grad.abs().sum() > 0


def test_projection_semantic_change_affects_only_its_field(artifact: Path) -> None:
    """Active residual fusion responds locally to a changed field description.

    :param artifact: Aligned field metadata.
    """
    projection = LanguageParameterProjection(
        16, "surge_4", "surge_4", embedding_path=str(artifact)
    )
    projection.initialize_embeddings()
    final_layer = projection.fusion[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    torch.nn.init.constant_(final_layer.weight, 0.1)
    x = torch.randn(2, param_specs["surge_4"].encoded_width)
    before = projection.param_to_token(x)
    projection.language_embeddings[0].add_(10)
    after = projection.param_to_token(x)
    assert not torch.allclose(before[:, 0], after[:, 0])
    torch.testing.assert_close(before[:, 1:], after[:, 1:])


def test_projection_numeric_gradient_is_field_and_batch_local(artifact: Path) -> None:
    """One token depends only on its own example and encoded field span.

    :param artifact: Aligned field metadata.
    """
    projection = LanguageParameterProjection(
        16, "surge_4", "surge_4", embedding_path=str(artifact)
    )
    projection.initialize_embeddings()
    x = torch.randn(2, param_specs["surge_4"].encoded_width, requires_grad=True)
    projection.param_to_token(x)[0, 0].sum().backward()
    _, span = next(param_specs["surge_4"].encoded_slices())
    assert x.grad is not None
    assert x.grad[0, span].abs().sum() > 0
    assert x.grad[0, span.stop :].abs().sum() == 0
    assert x.grad[1].abs().sum() == 0


def test_projection_native_width_metadata_produces_model_width(tmp_path: Path) -> None:
    """A 768-dimensional artifact maps to the same transformer token width.

    :param tmp_path: Isolated artifact directory.
    """
    count = len(describe_fields("surge_4", "surge_4"))
    path = tmp_path / "params.lance"
    vectors = np.full((count, 768), 1 / np.sqrt(768), dtype=np.float32)
    _write_artifact(path, vectors)
    projection = LanguageParameterProjection(
        16, "surge_4", "surge_4", embedding_dim=768, embedding_path=str(path)
    )
    projection.initialize_embeddings()
    tokens = projection.param_to_token(torch.zeros(2, param_specs["surge_4"].encoded_width))
    assert tokens.shape == (2, count, 16)


def test_projection_decoder_allows_negative_velocities() -> None:
    """Output heads do not clamp signed flow velocities."""
    projection = LanguageParameterProjection(16, "surge_4", "surge_4")
    decoder = projection.decoders[0]
    assert isinstance(decoder, torch.nn.Linear)
    assert decoder.bias is not None
    torch.nn.init.constant_(decoder.bias, -2)
    values = projection.token_to_param(torch.zeros(2, projection.num_tokens, 16))
    assert values[0, 0].item() == -2


def test_projection_missing_artifact_fails_before_tokens() -> None:
    """Fresh models cannot silently train with empty language vectors."""
    projection = LanguageParameterProjection(16, "surge_4", "surge_4")
    with pytest.raises(ValueError, match="embedding_path"):
        projection.param_to_token(torch.zeros(2, param_specs["surge_4"].encoded_width))


def test_projection_same_width_wrong_semantics_rejects_checkpoint(artifact: Path) -> None:
    """Identical tensor shapes do not permit another synth's field semantics.

    :param artifact: Aligned field metadata.
    """
    projection = LanguageParameterProjection(
        16, "surge_4", "surge_4", embedding_path=str(artifact)
    )
    projection.initialize_embeddings()
    restored = LanguageParameterProjection(16, "surge_4", "different-synth")
    with pytest.raises(ValueError, match="spec"):
        restored.load_state_dict(projection.state_dict())


@pytest.mark.slow
def test_real_language_projection_checkpoint_reload_preserves_trained_tokens(
    tmp_path: Path,
) -> None:
    """Real encoder vectors survive optimization and a disk checkpoint without the source artifact.

    :param tmp_path: Isolated embedding and checkpoint directory.
    """
    path = prepare_param_name_embeddings(
        tmp_path, "surge_4", "surge_4", dimension=128, device="cpu"
    )
    projection = LanguageParameterProjection(16, "surge_4", "surge_4", embedding_path=str(path))
    projection.initialize_embeddings()
    x = torch.randn(2, param_specs["surge_4"].encoded_width)
    optimizer = torch.optim.Adam(projection.parameters(), lr=0.01)
    projection.param_to_token(x).square().mean().backward()
    optimizer.step()
    expected = projection.param_to_token(x).detach()
    checkpoint = tmp_path / "projection.pt"
    torch.save(projection.state_dict(), checkpoint)
    shutil.rmtree(path)
    restored = LanguageParameterProjection(16, "surge_4", "surge_4")
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored.param_to_token(x), expected)


@pytest.mark.gpu
@pytest.mark.slow
def test_projection_cuda_compilation_preserves_forward_backward(artifact: Path) -> None:
    """Real Inductor execution preserves the vector-field contract around projection graph breaks.

    :param artifact: Aligned field metadata.
    """
    projection = LanguageParameterProjection(
        16, "surge_4", "surge_4", embedding_path=str(artifact)
    )
    projection.initialize_embeddings()
    model = ApproxEquivTransformer(
        projection,
        d_model=16,
        conditioning_dim=4,
        num_heads=1,
        d_ff=16,
        num_layers=1,
        learn_projection=True,
        zero_init=False,
    ).cuda()
    x = torch.randn(2, param_specs["surge_4"].encoded_width, device="cuda", requires_grad=True)
    t = torch.rand(2, 1, device="cuda")
    conditioning = torch.randn(2, 4, device="cuda")
    expected = model(x, t, conditioning)
    actual = torch.compile(model)(x, t, conditioning)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
