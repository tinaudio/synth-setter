"""Regression guard: every public field on every config schema is documented.

Asserts ``model_fields[name].description`` is non-empty for every field on
``DatasetSpec``, ``RenderConfig``, ``ShardSpec``, ``ImageConfig``, and
``ShardMetadata``. Pins the contract that fields are documented (rendered on
the mkdocs config-reference site) without pinning the wording, so a future
refactor that strips a ``Field(description=...)`` fails here instead of
silently regressing the rendered docs.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from synth_setter.pipeline.schemas.image_config import ImageConfig
from synth_setter.pipeline.schemas.shard_metadata import ShardMetadata
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig, ShardSpec

_DOCUMENTED_MODELS: tuple[type[BaseModel], ...] = (
    DatasetSpec,
    RenderConfig,
    ShardSpec,
    ImageConfig,
    ShardMetadata,
)


@pytest.mark.parametrize(
    ("model", "field_name"),
    [(model, name) for model in _DOCUMENTED_MODELS for name in model.model_fields],
    ids=lambda x: x if isinstance(x, str) else x.__name__,
)
def test_field_has_nonempty_description(model: type[BaseModel], field_name: str) -> None:
    """Every public Pydantic field carries a non-empty ``Field(description=...)``.

    :param model: The Pydantic config schema under test.
    :param field_name: The field on ``model`` whose description must be non-empty.
    """
    description = model.model_fields[field_name].description
    assert description, (
        f"{model.__name__}.{field_name} is missing a Field(description=...); "
        "the mkdocs config-reference page will render it without a one-liner."
    )
